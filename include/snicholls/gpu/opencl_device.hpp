//
//  opencl_device.hpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::gpu::opencl_device_backend`: core's forward passes with every operand in device memory on an
/// OpenCL 1.2 device, through `opencl_arena`, device memory addressed by host-shaped spans.
/// \ingroup gpu
///
/// **Overview.** `opencl_backend` copies its operands to the device for each product; a model's weights then cross
/// the bus on every call, and everything but the products runs on the host. This backend keeps everything on the
/// device: the weights, the activations, the key-value cache. OpenCL 1.2 (all macOS offers its GPUs, and what PoCL
/// runs) has no pointers a host can hold, so `opencl_arena` reserves a range of virtual addresses that no page backs
/// and gives out spans inside it; a span's address is its place in a device buffer (`locate`). Core's passes take
/// spans, slice them and hand them to the backend's operations, which are overloaded here for
/// `opencl_device_backend` (found by argument-dependent lookup), as `cuda_ops.hpp` overloads them for CUDA's device
/// pointers: the same core code runs on both. A host read of device memory faults at once rather than reading garbage.
///
/// The operations, forward only (inference and evaluation), from one kernel source written for OpenCL and CUDA
/// (`detail/rows_source.hpp`):
///
/// - products: `gemm` (and through it `ml::linear_sequence`, `ml::linear_heads`) with `opencl_context`'s kernel over
///   device buffers (`gemm_buffers`), `gemv` as one column;
/// - rows: `fill_rows`, `copy_rows`, `add_rows`, `add_bias_rows`, `embed_rows`, `block_norm_rows` (RMS and layer norms),
///   `mlp_activate_rows` (SwiGLU), `attention_position_rows` (QK norm and RoPE), `attention_attend_rows` (each
///   sequence's causal attention, windowed or not), `cross_entropy_rows`, `sum_values`, `sum_squares`;
/// - the key-value cache: `kv_store_rows` and `attention_attend_cached_rows`, the two operations a forward pass over
///   new tokens needs beside core's (their host reference is in the consumer's model, `training/llama.hpp`, until
///   core has them).
///
/// \snippet opencl_device_demo.cpp device
///
/// **Relationships.** `opencl.hpp` (the context and its product kernel), `cuda_ops.hpp` and `cuda_rows.hpp` (the same
/// operations on CUDA), core's `ml/` passes. Group: \ref gpu.
///
/// **Implementations.** One. RoPE's table stays the caller's host memory (a `rope_source`): its rows for a call are
/// copied up, as `cuda_ops.hpp` does.
///
/// **Guarantees.** Products: core's reference bits (the kernel of `opencl.hpp`, each output summed in index order),
/// except products of at most `matvec_max` rows (a decode step), which run a work-group an output and add 64 partial
/// sums by a fixed tree: within `|Δ| ≤ k 2^-23 Σ_p |x_p w_p|`, as gpu's quantised matrix-vector kernel.
/// Norms, QK norm, RoPE and the cache's writes: core's operations in core's order, one work-item a row or head; they
/// differ from the host only by the device's `exp`, `log` and `sqrt` (OpenCL's are within a few ulp, not correctly
/// rounded). Attention and the cross-entropy share their reductions across a work-group in a fixed order (lanes merged
/// in order, a fixed tree), so they differ from core's sequential sums by rounding; every result is deterministic.
/// Measured against the host on real models in snicholls-ml's `PLAN.md`. The backend is trivially copyable (three
/// pointers); `opencl_arena` and `opencl_device_kernels` serve any number of threads (launches are serialised).
///
/// **Errors.** `opencl_exception` for a failed OpenCL call; `std::invalid_argument` for a span not in the arena;
/// `std::bad_alloc` when the arena's reservation or the device is exhausted; `std::logic_error` for the BLAS
/// operations a forward pass does not use (`dot`, `axpy`, `syrk`, `triangular_solve`).
///
/// **Performance.** Measured in snicholls-ml's `PLAN.md` (prefill and decode against llama.cpp on the same machine).
///
/// **Tests and demos.** `tests/opencl_device_test.cpp` (every operation against core's host reference, a whole
/// block's forward pass, the cache), `demos/opencl_device_demo.cpp`.
///
/// **Dependencies.** OpenCL 1.2, `<sys/mman.h>` (the reservation), core's `ml/` headers.
///
/// **Known issues.** Heads of at most 128 channels in attention. The backward operations are not here (training on
/// OpenCL keeps `opencl_backend`).
///
/// **Status.** Experimental.

#ifndef snicholls_gpu_opencl_device_hpp
#define snicholls_gpu_opencl_device_hpp

#include <snicholls/gpu/detail/quant_source.hpp>
#include <snicholls/gpu/detail/rows_source.hpp>
#include <snicholls/gpu/opencl.hpp>

#include <snicholls/core/ml/attention_layer.hpp>
#include <snicholls/core/ml/mlp.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <sys/mman.h>

#include <algorithm>
#include <iterator>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace snicholls::gpu {

/// \brief Device memory addressed by spans: a reserved range of host addresses, each allocation a place in a device
/// buffer. Allocation reuses a freed block that fits (first fit), else bumps within the current buffer (a new one when
/// full); `deallocate` returns one allocation, `release` everything after a mark.
class opencl_arena {
public:
    /// \brief Where a span's first element is. \see locate
    struct location {
        cl_mem mem = nullptr;    ///< The buffer.
        std::size_t offset = 0;  ///< The byte offset in it.
    };

    /// \brief An arena on a context. \param context The device. \param reserve The bytes of addresses reserved, the
    /// most it can ever hold. \param chunk The bytes of each device buffer (a larger allocation gets its own).
    /// \return The arena. \throws std::bad_alloc if the addresses cannot be reserved.
    [[nodiscard]] static std::unique_ptr<opencl_arena> create(const opencl_context& context, std::size_t reserve = std::size_t{64} << 30,
                                                              std::size_t chunk = std::size_t{1} << 30)
    {
        cl_ulong max_alloc = 0;
        clGetDeviceInfo(context.native_device(), CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, nullptr);
        void* base = ::mmap(nullptr, reserve, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (base == MAP_FAILED)
            throw std::bad_alloc();
        return std::unique_ptr<opencl_arena>(new opencl_arena(context, static_cast<std::byte*>(base), reserve,
                                                              std::min<std::size_t>(chunk, std::size_t(max_alloc)), std::size_t(max_alloc)));
    }

    opencl_arena(const opencl_arena&) = delete;
    opencl_arena& operator=(const opencl_arena&) = delete;
    ~opencl_arena()
    {
        for (const auto& c : chunks_)
            clReleaseMemObject(c.mem);
        ::munmap(base_, reserve_);
    }

    /// \brief `n` values of device memory, 256-byte aligned. \tparam T A trivially copyable type. \param n The values.
    /// \return Their span (never to be read on the host). \throws std::bad_alloc, opencl_exception.
    template <typename T>
    [[nodiscard]] std::span<T> allocate(std::size_t n)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        const std::size_t bytes = round(std::max<std::size_t>(n * sizeof(T), 1));
        const std::lock_guard lock(mutex_);
        for (auto it = free_.begin(); it != free_.end(); ++it)   // first fit among the freed blocks
            if (it->size >= bytes) {
                const std::size_t at = it->start;
                if (it->size == bytes)
                    free_.erase(it);
                else
                    it->start += bytes, it->size -= bytes;
                return {reinterpret_cast<T*>(base_ + at), n};
            }
        if (chunks_.empty() || round(top_) + bytes > chunks_.back().start + chunks_.back().size) {
            const std::size_t start = chunks_.empty() ? 0 : round(chunks_.back().start + chunks_.back().size);
            const std::size_t size = std::max(chunk_, round(bytes));
            if (size > max_alloc_ || start + size > reserve_)
                throw std::bad_alloc();
            cl_int code = CL_SUCCESS;
            cl_mem mem = clCreateBuffer(context_->native_context(), CL_MEM_READ_WRITE, size, nullptr, &code);
            detail::check(code, "clCreateBuffer (the arena)");
            chunks_.push_back({mem, start, size});
            top_ = start;
        }
        const std::size_t at = round(top_);
        top_ = at + bytes;
        return {reinterpret_cast<T*>(base_ + at), n};
    }

    /// \brief Returns an allocation for reuse (neighbouring free blocks are joined). \param s A span `allocate` gave, whole.
    template <typename T>
    void deallocate(std::span<T> s)
    {
        if (!owns(s.data()))
            return;
        const std::size_t at = std::size_t(reinterpret_cast<const std::byte*>(s.data()) - base_), bytes = round(std::max<std::size_t>(s.size_bytes(), 1));
        const std::lock_guard lock(mutex_);
        auto it = free_.begin();
        while (it != free_.end() && it->start < at)
            ++it;
        it = free_.insert(it, {at, bytes});
        if (std::next(it) != free_.end() && it->start + it->size == std::next(it)->start && same_chunk(it->start, std::next(it)->start))
            it->size += std::next(it)->size, free_.erase(std::next(it));
        if (it != free_.begin() && std::prev(it)->start + std::prev(it)->size == it->start && same_chunk(std::prev(it)->start, it->start)) {
            std::prev(it)->size += it->size;
            free_.erase(it);
        }
    }

    /// \brief Where a device address is. \param p An address inside an allocation. \return Its buffer and offset.
    /// \throws std::invalid_argument for an address outside the arena.
    [[nodiscard]] location locate(const void* p) const
    {
        const auto* b = static_cast<const std::byte*>(p);
        if (b < base_ || b >= base_ + reserve_)
            throw std::invalid_argument("snicholls::gpu: a span that is not in the OpenCL arena");
        const std::size_t at = std::size_t(b - base_);
        const std::lock_guard lock(mutex_);
        for (const auto& c : chunks_)
            if (at >= c.start && at < c.start + c.size)
                return {c.mem, at - c.start};
        throw std::invalid_argument("snicholls::gpu: a span in the arena's reservation but in no buffer");
    }

    /// \brief Whether an address is the arena's. \param p The address. \return True if it is.
    [[nodiscard]] bool owns(const void* p) const noexcept
    {
        const auto* b = static_cast<const std::byte*>(p);
        return b >= base_ && b < base_ + reserve_;
    }

    /// \brief Host values into device memory. \param to The device span. \param from As many host values.
    template <typename T>
    void write(std::span<T> to, std::span<const T> from) const
    {
        if (from.empty())
            return;
        const auto l = locate(to.data());
        detail::check(clEnqueueWriteBuffer(context_->native_queue(), l.mem, CL_TRUE, l.offset, from.size_bytes(), from.data(), 0, nullptr, nullptr),
                      "clEnqueueWriteBuffer");
    }

    /// \brief Device values to the host, after everything enqueued before. \param from The device span. \param to As many host values.
    template <typename T>
    void read(std::span<const T> from, std::span<std::remove_const_t<T>> to) const
    {
        if (to.empty())
            return;
        const auto l = locate(from.data());
        detail::check(clEnqueueReadBuffer(context_->native_queue(), l.mem, CL_TRUE, l.offset, to.size_bytes(), to.data(), 0, nullptr, nullptr),
                      "clEnqueueReadBuffer");
    }

    /// \brief The allocation's high-water mark, for `release`. \return It.
    [[nodiscard]] std::size_t mark() const
    {
        const std::lock_guard lock(mutex_);
        return top_;
    }
    /// \brief Frees everything allocated after `m` (the buffers are kept for reuse). \param m A `mark()`.
    void release(std::size_t m)
    {
        const std::lock_guard lock(mutex_);
        top_ = std::min(top_, m);
        std::erase_if(free_, [&](const block& b) { return b.start >= top_; });
        while (!chunks_.empty() && chunks_.back().start > top_) {
            clReleaseMemObject(chunks_.back().mem);
            chunks_.pop_back();
        }
    }
    /// \brief The bytes of device buffers held. \return They.
    [[nodiscard]] std::size_t capacity() const
    {
        const std::lock_guard lock(mutex_);
        std::size_t n = 0;
        for (const auto& c : chunks_)
            n += c.size;
        return n;
    }
    /// \brief The context. \return It.
    [[nodiscard]] const opencl_context& context() const noexcept { return *context_; }

private:
    struct chunk {
        cl_mem mem;
        std::size_t start, size;
    };
    struct block {
        std::size_t start, size;
    };
    bool same_chunk(std::size_t a, std::size_t b) const noexcept
    {
        for (const auto& c : chunks_)
            if (a >= c.start && a < c.start + c.size)
                return b >= c.start && b < c.start + c.size;
        return false;
    }
    opencl_arena(const opencl_context& c, std::byte* base, std::size_t reserve, std::size_t chunk_bytes, std::size_t max_alloc)
        : context_(&c), base_(base), reserve_(reserve), chunk_(chunk_bytes), max_alloc_(max_alloc)
    {
    }
    static constexpr std::size_t round(std::size_t n) noexcept { return (n + 255) / 256 * 256; }

    const opencl_context* context_;
    std::byte* base_;
    std::size_t reserve_, chunk_, max_alloc_;
    std::vector<chunk> chunks_;
    std::vector<block> free_;
    std::size_t top_ = 0;
    mutable std::mutex mutex_;
};

/// \brief The row kernels of `detail/rows_source.hpp`, built for one context.
class opencl_device_kernels {
public:
    /// \brief Builds the program. \param context The device. \return The kernels, or why the program did not build.
    [[nodiscard]] static result<std::unique_ptr<opencl_device_kernels>, opencl_failure> create(const opencl_context& context)
    {
        std::unique_ptr<opencl_device_kernels> k(new opencl_device_kernels(context));
        const std::string source = std::string(detail::quant::opencl_prelude) + detail::rows::math_opencl + detail::rows::source;
        const char* text = source.c_str();
        cl_int code = CL_SUCCESS;
        k->program_ = clCreateProgramWithSource(context.native_context(), 1, &text, nullptr, &code);
        if (code != CL_SUCCESS)
            return err(opencl_failure{opencl_error::call_failed, code, {}});
        cl_device_id device = context.native_device();
        if (const cl_int built = clBuildProgram(k->program_, 1, &device, "", nullptr, nullptr); built != CL_SUCCESS) {
            std::size_t size = 0;
            clGetProgramBuildInfo(k->program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
            std::string log(size, '\0');
            clGetProgramBuildInfo(k->program_, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
            return err(opencl_failure{opencl_error::build_failed, built, std::move(log)});
        }
        for (std::size_t i = 0; i < kernel_count; ++i) {
            k->kernels_[i] = clCreateKernel(k->program_, names[i], &code);
            if (code != CL_SUCCESS)
                return err(opencl_failure{opencl_error::call_failed, code, {}});
        }
        return ok(std::move(k));
    }

    opencl_device_kernels(const opencl_device_kernels&) = delete;
    opencl_device_kernels& operator=(const opencl_device_kernels&) = delete;
    ~opencl_device_kernels()
    {
        for (cl_kernel kernel : kernels_)
            if (kernel)
                clReleaseKernel(kernel);
        if (program_)
            clReleaseProgram(program_);
    }

    /// \brief The kernels, by name in the source.
    enum class id : std::size_t { fill, copy, add, add_bias, embed, norm, swiglu, position, attend, store, cross_entropy, sum, matvec };

    /// \brief Enqueues a kernel over `groups` work-groups of `local` items. \param k The kernel. \param groups The
    /// work-groups. \param local Their size. \param args Each argument as (size, address).
    template <typename... A>
    void launch(id k, std::size_t groups, std::size_t local, const A&... args) const
    {
        if (groups == 0)
            return;
        const std::lock_guard lock(mutex_);
        cl_kernel kernel = kernels_[std::size_t(k)];
        cl_uint i = 0;
        ((detail::check(clSetKernelArg(kernel, i++, sizeof(A), &args), "clSetKernelArg")), ...);
        const std::size_t global = groups * local;
        detail::check(clEnqueueNDRangeKernel(context_->native_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                      "clEnqueueNDRangeKernel");
    }

    /// \brief The context. \return It.
    [[nodiscard]] const opencl_context& context() const noexcept { return *context_; }

private:
    static constexpr std::size_t kernel_count = 13;
    static constexpr const char* names[kernel_count] = {"rows_fill",   "rows_copy",     "rows_add",    "rows_add_bias", "rows_embed",
                                                        "rows_norm",   "rows_swiglu",   "rows_position", "rows_attend", "rows_store",
                                                        "rows_cross_entropy", "rows_sum", "rows_matvec"};
    explicit opencl_device_kernels(const opencl_context& c) : context_(&c) {}
    const opencl_context* context_;
    cl_program program_ = nullptr;
    cl_kernel kernels_[kernel_count] = {};
    mutable std::mutex mutex_;
};

/// \brief `linalg::blas_backend` over `opencl_arena` memory: the products on the device, operands where they are.
/// Trivially copyable; pass it by value to core's free functions.
struct opencl_device_backend {
    const opencl_context* context = nullptr;          ///< The device; must outlive the backend.
    const opencl_arena* arena = nullptr;              ///< The memory every operand is in.
    const opencl_device_kernels* kernels = nullptr;   ///< The row kernels.
    std::size_t matvec_max = 4;                       ///< Products of at most this many rows by a weight matrix: a work-group an output
                                                      ///< (within the bound above, not the reference's bits); 0 for the tiled product.

    /// \brief Not on the device (a forward pass does not use it). \throws std::logic_error.
    template <linalg::scalar T>
    [[nodiscard]] T dot(linalg::vector_view<const T>, linalg::vector_view<const T>) const
    {
        throw std::logic_error("snicholls::gpu: opencl_device_backend has no dot");
    }
    /// \brief Not on the device. \throws std::logic_error.
    template <linalg::scalar T>
    void axpy(T, linalg::vector_view<const T>, linalg::vector_view<T>) const
    {
        throw std::logic_error("snicholls::gpu: opencl_device_backend has no axpy");
    }
    /// \brief `y = alpha A x + beta y` as a product with one column. \param alpha The scale. \param a `m x k`.
    /// \param x `k`. \param beta The scale of `y`. \param y `m`.
    template <linalg::scalar T>
    void gemv(T alpha, linalg::matrix_view<const T> a, linalg::vector_view<const T> x, T beta, linalg::vector_view<T> y) const
    {
        gemm(alpha, a, linalg::as_column(x), beta, linalg::as_column(y));
    }
    /// \brief `C = alpha A B + beta C` with every operand in the arena. \param alpha The scale. \param a `m x k`.
    /// \param b `k x n`. \param beta The scale of `C`. \param c `m x n`. \throws std::invalid_argument, opencl_exception.
    template <linalg::scalar T>
    void gemm(T alpha, linalg::matrix_view<const T> a, linalg::matrix_view<const T> b, T beta, linalg::matrix_view<T> c) const
    {
        static_assert(std::is_same_v<T, float>, "snicholls::gpu: opencl_device_backend multiplies float");
        if (c.extent(0) == 0 || c.extent(1) == 0)
            return;
        const auto la = arena->locate(a.data_handle()), lb = arena->locate(b.data_handle()), lc = arena->locate(c.data_handle());
        // A few rows times a transposed row-major matrix (ml::linear_sequence's decode step): one item an output
        if (c.extent(0) <= matvec_max && alpha == T(1) && beta == T(0) && a.stride(1) == 1 && b.stride(0) == 1 && b.stride(1) == a.extent(1)
            && c.stride(1) == 1) {
            kernels->launch(opencl_device_kernels::id::matvec, c.extent(1), 64, la.mem, cl_ulong(la.offset / sizeof(T)), lb.mem,
                            cl_ulong(lb.offset / sizeof(T)), lc.mem, cl_ulong(lc.offset / sizeof(T)), cl_ulong(c.extent(0)), cl_ulong(a.extent(1)),
                            cl_ulong(c.extent(1)), cl_ulong(a.stride(0)), cl_ulong(c.stride(0)));
            return;
        }
        context->gemm_buffers(alpha, c.extent(0), c.extent(1), a.extent(1), la.mem, la.offset / sizeof(T), a.stride(0), a.stride(1), lb.mem,
                              lb.offset / sizeof(T), b.stride(0), b.stride(1), beta, lc.mem, lc.offset / sizeof(T), c.stride(0), c.stride(1));
    }
    /// \brief Not on the device. \throws std::logic_error.
    template <linalg::scalar T>
    void syrk(T, linalg::matrix_view<const T>, T, linalg::matrix_view<T>, linalg::triangle) const
    {
        throw std::logic_error("snicholls::gpu: opencl_device_backend has no syrk");
    }
    /// \brief Not on the device. \throws std::logic_error.
    template <linalg::scalar T>
    [[nodiscard]] result<void, linalg::error> triangular_solve(linalg::matrix_view<const T>, linalg::matrix_view<T>, linalg::triangle,
                                                               linalg::diagonal) const
    {
        throw std::logic_error("snicholls::gpu: opencl_device_backend has no triangular_solve");
    }
};

static_assert(linalg::blas_backend<opencl_device_backend, float> && std::is_trivially_copyable_v<opencl_device_backend>);

namespace detail::opencl_device {
    // A device span as the kernels take it: its buffer and its element offset
    struct arg {
        cl_mem mem;
        cl_ulong offset;
    };
    template <typename T>
    arg at(const opencl_device_backend& b, std::span<T> s)
    {
        const auto l = b.arena->locate(s.data());
        return {l.mem, cl_ulong(l.offset / sizeof(T))};
    }
    // An optional span: a valid buffer either way (the kernel reads it only when told it is there)
    template <typename T>
    arg at_or(const opencl_device_backend& b, std::span<T> s, std::span<T> other)
    {
        return s.empty() ? at(b, other) : at(b, s);
    }
    inline std::size_t groups(std::size_t n, std::size_t local) noexcept { return (n + local - 1) / local; }
    using id = opencl_device_kernels::id;
} // namespace detail::opencl_device

/// \name core's row operations on the device, for `opencl_device_backend` (found by argument-dependent lookup)
/// @{

/// \brief `x = value`. \param b The backend. \param x Device values. \param value The value.
inline void fill_rows(const opencl_device_backend& b, std::span<float> x, std::type_identity_t<float> value)
{
    namespace d = detail::opencl_device;
    if (x.empty())
        return;
    const auto p = d::at(b, x);
    b.kernels->launch(d::id::fill, d::groups(x.size(), 256), 256, p.mem, p.offset, cl_ulong(x.size()), value);
}

/// \brief `to = from`. \param b The backend. \param from Device values. \param to As many.
inline void copy_rows(const opencl_device_backend& b, std::type_identity_t<std::span<const float>> from, std::span<float> to)
{
    namespace d = detail::opencl_device;
    if (to.empty())
        return;
    const auto f = d::at(b, from), t = d::at(b, to);
    b.kernels->launch(d::id::copy, d::groups(to.size(), 256), 256, f.mem, f.offset, t.mem, t.offset, cl_ulong(to.size()));
}

/// \brief `x += a`. \param b The backend. \param a Device values. \param x As many, added into.
inline void add_rows(const opencl_device_backend& b, std::type_identity_t<std::span<const float>> a, std::span<float> x)
{
    namespace d = detail::opencl_device;
    if (x.empty())
        return;
    const auto pa = d::at(b, a), px = d::at(b, x);
    b.kernels->launch(d::id::add, d::groups(x.size(), 256), 256, pa.mem, pa.offset, px.mem, px.offset, cl_ulong(x.size()));
}

/// \brief `y_t += bias` for every row. \param b The backend. \param ys Rows of `bias.size()`. \param bias The bias.
inline void add_bias_rows(const opencl_device_backend& b, std::span<float> ys, std::type_identity_t<std::span<const float>> bias)
{
    namespace d = detail::opencl_device;
    if (ys.empty())
        return;
    const auto py = d::at(b, ys), pb = d::at(b, bias);
    b.kernels->launch(d::id::add_bias, d::groups(ys.size(), 256), 256, py.mem, py.offset, pb.mem, pb.offset, cl_ulong(ys.size()),
                      cl_ulong(bias.size()));
}

/// \brief `out_t = table[ids_t] + positions_t`. \param b The backend. \param table `V` rows. \param positions One row a
/// token, or empty. \param ids `n` device tokens. \param out `n` rows.
inline void embed_rows(const opencl_device_backend& b, std::type_identity_t<std::span<const float>> table,
                       std::type_identity_t<std::span<const float>> positions, std::span<const std::uint32_t> ids, std::span<float> out)
{
    namespace d = detail::opencl_device;
    if (ids.empty())
        return;
    const auto pt = d::at(b, table), pp = d::at_or(b, positions, table), pi = d::at(b, ids), po = d::at(b, out);
    const std::size_t dm = out.size() / ids.size();
    b.kernels->launch(d::id::embed, d::groups(out.size(), 256), 256, pt.mem, pt.offset, pp.mem, pp.offset, cl_int(positions.empty() ? 0 : 1), pi.mem,
                      pi.offset, po.mem, po.offset, cl_ulong(ids.size()), cl_ulong(dm));
}

/// \brief The configuration's norm over every row. \tparam C The configuration. \param b The backend. \param x Rows of
/// `d_model`. \param weight The scale, or empty. \param bias The shift (layer norm), or empty. \param out As `x`.
template <ml::transformer_config C>
void block_norm_rows(const opencl_device_backend& b, std::type_identity_t<std::span<const float>> x, std::type_identity_t<std::span<const float>> weight,
                     std::type_identity_t<std::span<const float>> bias, std::span<float> out)
{
    namespace d = detail::opencl_device;
    if (x.empty())
        return;
    const std::size_t rows = x.size() / C.d_model;
    const auto px = d::at(b, x), pw = d::at_or(b, weight, x), pb = d::at_or(b, bias, x), po = d::at(b, out);
    b.kernels->launch(d::id::norm, d::groups(rows, 64), 64, px.mem, px.offset, pw.mem, pw.offset, cl_int(weight.empty() ? 0 : 1), pb.mem, pb.offset,
                      cl_int(bias.empty() ? 0 : 1), po.mem, po.offset, cl_ulong(rows), cl_ulong(C.d_model), ml::norm_eps_v<C, float>,
                      cl_int(C.norm == ml::norm_kind::rms ? 1 : 0));
}

/// \brief The MLP's activation over its saved rows: `hidden = silu(gate) · up`. \tparam C The configuration (SwiGLU).
/// \param b The backend. \param saved The rows.
template <ml::transformer_config C>
void mlp_activate_rows(const opencl_device_backend& b, const ml::mlp_activations<float>& saved)
{
    static_assert(C.mlp == ml::mlp_kind::swiglu, "snicholls::gpu: opencl_device_backend's MLP activation is SwiGLU");
    namespace d = detail::opencl_device;
    if (saved.up.empty())
        return;
    const auto pg = d::at(b, saved.gate), pu = d::at(b, saved.up), ph = d::at(b, saved.hidden);
    b.kernels->launch(d::id::swiglu, d::groups(saved.up.size(), 256), 256, pg.mem, pg.offset, pu.mem, pu.offset, ph.mem, ph.offset,
                      cl_ulong(saved.up.size()));
}

/// \brief QK norm and RoPE of every head in place (the raw heads copied first when the activations keep them).
/// \tparam C The configuration. \param b The backend. \param w The attention's weights (its norms on the device).
/// \param act The activations. \param n The tokens of each sequence. \param rope_src The cosines and sines, host
/// memory, by position within the sequence. \param sequences The sequences.
template <ml::transformer_config C, typename W, typename R>
void attention_position_rows(const opencl_device_backend& b, const ml::attention_weights<W, float>& w, const ml::attention_activations<float>& act,
                             std::size_t n, const R& rope_src, std::size_t sequences = 1)
{
    if constexpr (C.qk_norm == ml::qk_norm_kind::none && C.rope == ml::rope_kind::none) {
        (void)b, (void)w, (void)act, (void)n, (void)rope_src, (void)sequences;
    } else {
        namespace d = detail::opencl_device;
        constexpr std::size_t half = C.head_dim / 2;
        // RoPE's rows for positions 0 .. n - 1 of the source, up for this call (the source is the same for every sequence)
        std::vector<float> cs(C.rope == ml::rope_kind::none ? 1 : n * half), sn(cs.size());
        if constexpr (C.rope != ml::rope_kind::none)
            for (std::size_t t = 0; t < n; ++t) {
                const auto c = rope_src.cos_at(t), s = rope_src.sin_at(t);
                std::copy(c.begin(), c.end(), cs.begin() + std::ptrdiff_t(t * half));
                std::copy(s.begin(), s.end(), sn.begin() + std::ptrdiff_t(t * half));
            }
        const cl_context ctx = b.context->native_context();
        detail::buffer dcs(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cs.size() * sizeof(float), cs.data());
        detail::buffer dsn(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sn.size() * sizeof(float), sn.data());
        const bool raw = !act.q_raw.empty();
        const auto pq = d::at(b, act.q), pk = d::at(b, act.k), pqr = d::at_or(b, act.q_raw, act.q), pkr = d::at_or(b, act.k_raw, act.k);
        const auto pqn = d::at_or(b, w.q_norm, std::span<const float>(act.q)), pkn = d::at_or(b, w.k_norm, std::span<const float>(act.k));
        const int qk = C.qk_norm == ml::qk_norm_kind::before_rope ? 1 : C.qk_norm == ml::qk_norm_kind::after_rope ? 2 : 0;
        const std::size_t items = n * sequences * (C.n_heads + C.n_kv_heads);
        b.kernels->launch(d::id::position, d::groups(items, 64), 64, pq.mem, pq.offset, pk.mem, pk.offset, pqr.mem, pqr.offset, pkr.mem, pkr.offset,
                          cl_int(raw ? 1 : 0), pqn.mem, pqn.offset, cl_int(w.q_norm.empty() ? 0 : 1), pkn.mem, pkn.offset,
                          cl_int(w.k_norm.empty() ? 0 : 1), dcs.mem, dsn.mem, cl_ulong(n), cl_ulong(sequences), cl_int(C.n_heads),
                          cl_int(C.n_kv_heads), cl_ulong(C.head_dim), cl_int(qk), cl_int(C.rope == ml::rope_kind::none ? 0 : 1),
                          cl_int(C.layout() == ml::rope_layout::half ? 1 : 0), ml::norm_eps_v<C, float>);
        // The table's buffers are released on return; OpenCL frees them once the kernel that reads them has run
    }
}

/// \brief Attention of `n` new queries at positions `first ..` over a cache of `capacity` rows a key-value head
/// (`[kv_head][capacity][head_dim]`, positions `0 .. first + n - 1` filled), causal and windowed as the configuration
/// says. \tparam C The configuration. \param b The backend. \param q `n` rows of `q_size`. \param keys The keys.
/// \param values The values. \param first The first query's position. \param n The queries. \param capacity The rows
/// a head. \param y `n` rows of `q_size`. \param lse `n · n_heads` log-sum-exps.
template <ml::transformer_config C>
void attention_attend_cached_rows(const opencl_device_backend& b, std::type_identity_t<std::span<const float>> q,
                                  std::type_identity_t<std::span<const float>> keys, std::type_identity_t<std::span<const float>> values,
                                  std::size_t first, std::size_t n, std::size_t capacity, std::span<float> y, std::span<float> lse)
{
    static_assert(C.head_dim <= 128, "snicholls::gpu: opencl_device_backend's attention takes heads of at most 128 channels");
    static_assert(C.attention_softcap == 0, "snicholls::gpu: opencl_device_backend's attention has no soft cap");
    namespace d = detail::opencl_device;
    if (n == 0)
        return;
    if (first + n > capacity)
        throw std::invalid_argument("snicholls::gpu: attention past the cache's capacity");
    const auto pq = d::at(b, q), pk = d::at(b, keys), pv = d::at(b, values), py = d::at(b, y), pl = d::at(b, lse);
    const float scale = 1.0f / snicholls::math::sqrt(float(C.head_dim));
    b.kernels->launch(d::id::attend, n * C.n_heads, 64, pq.mem, pq.offset, pk.mem, pk.offset, pv.mem, pv.offset, cl_ulong(first), cl_ulong(n),
                      cl_ulong(capacity), cl_ulong(C.window), cl_int(C.n_heads), cl_int(C.n_kv_heads), cl_ulong(C.head_dim), scale, py.mem,
                      py.offset, pl.mem, pl.offset);
}

/// \brief The keys and values of `n` new tokens (`[kv_head][n][head_dim]`) into a cache's rows `first ..`.
/// \tparam C The configuration. \param b The backend. \param k The keys. \param v The values. \param keys The cache's
/// keys of one layer. \param values Its values. \param first The first position. \param n The tokens. \param capacity
/// The rows a head.
template <ml::transformer_config C>
void kv_store_rows(const opencl_device_backend& b, std::type_identity_t<std::span<const float>> k, std::type_identity_t<std::span<const float>> v,
                   std::span<float> keys, std::span<float> values, std::size_t first, std::size_t n, std::size_t capacity)
{
    namespace d = detail::opencl_device;
    if (n == 0)
        return;
    if (first + n > capacity)
        throw std::invalid_argument("snicholls::gpu: a cache write past its capacity");
    const auto pk = d::at(b, k), pv = d::at(b, v), pks = d::at(b, keys), pvs = d::at(b, values);
    const std::size_t items = C.n_kv_heads * n * C.head_dim;
    b.kernels->launch(d::id::store, d::groups(items, 256), 256, pk.mem, pk.offset, pv.mem, pv.offset, pks.mem, pks.offset, pvs.mem, pvs.offset,
                      cl_ulong(first), cl_ulong(n), cl_ulong(capacity), cl_int(C.n_kv_heads), cl_ulong(C.head_dim));
}

/// \brief Every sequence's causal attention: the sequence's own keys as the cache. \tparam C The configuration.
/// \param b The backend. \param act The activations. \param n The tokens of each sequence. \param sequences The sequences.
template <ml::transformer_config C>
void attention_attend_rows(const opencl_device_backend& b, const ml::attention_activations<float>& act, std::size_t n, std::size_t sequences = 1)
{
    static_assert(C.mask == ml::attention_mask::causal, "snicholls::gpu: opencl_device_backend's attention is causal");
    if (!act.lengths.empty())
        throw std::invalid_argument("snicholls::gpu: opencl_device_backend's attention takes whole sequences (no lengths)");
    constexpr std::size_t qn = C.q_size(), kn = C.kv_size();
    for (std::size_t s = 0; s < sequences; ++s)
        attention_attend_cached_rows<C>(b, std::span<const float>(act.q).subspan(s * n * qn, n * qn), std::span<const float>(act.k).subspan(s * n * kn, n * kn),
                                        std::span<const float>(act.v).subspan(s * n * kn, n * kn), 0, n, n, act.y.subspan(s * n * qn, n * qn),
                                        act.lse.subspan(s * n * C.n_heads, n * C.n_heads));
}

/// \brief Each row's cross-entropy over its first `classes` columns, the row replaced by `scale · (softmax - onehot)`
/// and the padding by 0. \param b The backend. \param logits `n` rows. \param targets `n` device targets. \param losses
/// `n`. \param scale The gradient's factor. \param classes The classes.
inline void cross_entropy_rows(const opencl_device_backend& b, std::span<float> logits, std::type_identity_t<std::span<const std::uint32_t>> targets,
                               std::span<float> losses, std::type_identity_t<float> scale, std::size_t classes)
{
    namespace d = detail::opencl_device;
    if (targets.empty())
        return;
    const auto px = d::at(b, logits), pt = d::at(b, targets), pl = d::at(b, losses);
    b.kernels->launch(d::id::cross_entropy, targets.size(), 256, px.mem, px.offset, pt.mem, pt.offset, pl.mem, pl.offset,
                      cl_ulong(logits.size() / targets.size()), cl_ulong(classes), scale);
}

/// \brief `cross_entropy_rows` over whole rows. \param b The backend. \param logits `n` rows. \param targets `n`.
/// \param losses `n`. \param scale The gradient's factor.
inline void cross_entropy_rows(const opencl_device_backend& b, std::span<float> logits, std::type_identity_t<std::span<const std::uint32_t>> targets,
                               std::span<float> losses, std::type_identity_t<float> scale)
{
    cross_entropy_rows(b, logits, targets, losses, scale, targets.empty() ? 0 : logits.size() / targets.size());
}

namespace detail::opencl_device {
    inline float reduce(const opencl_device_backend& b, std::span<const float> x, int squares)
    {
        float out = 0;
        if (x.empty())
            return out;
        cl_int code = CL_SUCCESS;
        cl_mem result = clCreateBuffer(b.context->native_context(), CL_MEM_WRITE_ONLY, sizeof(float), nullptr, &code);
        check(code, "clCreateBuffer");
        const auto px = at(b, x);
        b.kernels->launch(id::sum, 1, 256, px.mem, px.offset, cl_ulong(x.size()), cl_int(squares), result);
        code = clEnqueueReadBuffer(b.context->native_queue(), result, CL_TRUE, 0, sizeof(float), &out, 0, nullptr, nullptr);
        clReleaseMemObject(result);
        check(code, "clEnqueueReadBuffer");
        return out;
    }
} // namespace detail::opencl_device

/// \brief The sum of device values, to the host. \param b The backend. \param x The values. \return Their sum.
template <typename E>
    requires std::is_same_v<std::remove_const_t<E>, float>
float sum_values(const opencl_device_backend& b, std::span<E> x)
{
    return detail::opencl_device::reduce(b, std::span<const float>(x), 0);
}

/// \brief The sum of squares of device values, to the host. \param b The backend. \param x The values. \return It.
template <typename E>
    requires std::is_same_v<std::remove_const_t<E>, float>
float sum_squares(const opencl_device_backend& b, std::span<E> x)
{
    return detail::opencl_device::reduce(b, std::span<const float>(x), 1);
}

/// @}

} // namespace snicholls::gpu

#endif // snicholls_gpu_opencl_device_hpp
