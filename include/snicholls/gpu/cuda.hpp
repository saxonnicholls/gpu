//
//  cuda.hpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::gpu::cuda_backend`: a `snicholls::linalg::blas_backend` on an NVIDIA GPU through cuBLAS, over device memory in place or host memory staged.
/// \ingroup gpu
///
/// **Overview.** The NVIDIA backend, beside the portable OpenCL one: cuBLAS for the matrix products, and device memory
/// that core's views point at directly. CUDA tells a device pointer from a host pointer at run time, so one backend
/// takes both:
///
/// - **device memory** (`device_buffer<T>`, whose `span()` is a `std::span` over it) is used in place: a training step's
///   weights and activations stay on the GPU between products;
/// - **host memory** is staged: copied up, multiplied, the result copied back, as the OpenCL backend does.
///
/// - `cuda_context::create(device)`: the device, its cuBLAS handle and stream; `result` with a `cuda_failure` when there
///   is no device.
/// - `cuda_backend{&context}`: the backend, a pointer and `cuda_math` (float, or TF32 on the tensor cores), trivially
///   copyable as the concept requires. `gemm` and `gemv` run on the device; `dot`, `axpy`, `syrk` and
///   `triangular_solve` run on the host through core's reference, for host memory (device memory there throws, until
///   those move to the device).
/// - `device_buffer<T>`: `n` elements of device memory; `upload`, `download`, `span()`.
/// - `cuda_graph::capture(context, f)`: the work `f` queues on the context's stream as a CUDA graph, `launch()`ed
///   again as one launch (a training step's thousands of kernels).
///
/// \snippet cuda_demo.cpp cuda
///
/// **Relationships.** `snicholls/core/linalg/backend.hpp` (the concept), core's `linalg::reference` (the reference the
/// results are checked against) and its conformance suite; `opencl.hpp` (the portable backend, the reference's bits).
/// Group: \ref gpu.
///
/// **Implementations.** `gemm` maps each view onto cuBLAS's column-major storage without copying when it is row- or
/// column-major (a row-major `C` is computed as `Cᵀ = Bᵀ Aᵀ`); any other strides are packed into a contiguous copy on
/// the device. cuBLAS's default math in float and double (no TF32): its own order of summation, so results agree with
/// the reference to rounding, not to the bit.
///
/// **Guarantees.** One context serves any number of threads (one mutex per context around the handle and the stream).
/// A product whose operands and result are all device memory is queued on the context's stream and returns at once:
/// it is ordered before everything later on that stream (`cuda_ops.hpp`'s kernels, the next product) and before a
/// `device_buffer::download`; `cuda_context::synchronize()` waits for it, for anything else (another context's stream)
/// that reads it. A product that touches host memory has finished, its result in place, when it returns. The context
/// owns the handle and the stream; a backend must not outlive it, nor a `device_buffer` its context.
///
/// **Errors.** `cuda_context::create` returns a `cuda_failure` (no device, or a CUDA or cuBLAS call failed, with its
/// code). A call that fails on the device (out of memory, a lost device) throws `cuda_exception`, a
/// `std::runtime_error` with the code; so does a host-only operation handed device memory.
///
/// **Performance.** `benchmarks/cuda_benchmark.cpp`, against the OpenCL backend and core's reference, transfers
/// included for host memory and none for device memory: see `BENCHMARKS.md`.
///
/// **Tests and demos.** `tests/cuda_test.cpp` (core's linalg BLAS conformance suite, the reference's results to
/// rounding for every shape and layout, device memory in place, strided views, threads), `demos/cuda_demo.cpp`. Built
/// only where CUDA is (`SNICHOLLS_GPU_WITH_CUDA`).
///
/// **Dependencies.** `<cstddef>`, `<memory>`, `<mutex>`, `<stdexcept>`, `<string>`, `<type_traits>`, `<vector>`,
/// `<cuda_runtime.h>`, `<cublas_v2.h>`, `snicholls/core/linalg/reference.hpp`, `snicholls/core/result.hpp`. Link
/// `CUDA::cudart` and `CUDA::cublas` (the target `snicholls::gpu_cuda`).
///
/// **Known issues.** Tensor cores (TF32, bfloat16) are not used yet; they are the next step, with a stated tolerance.
///
/// **Status.** Experimental, new in 0.2.0.

#ifndef snicholls_gpu_cuda_hpp
#define snicholls_gpu_cuda_hpp

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/result.hpp>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls::gpu {

/// \brief Why `cuda_context::create` failed.
enum class cuda_error {
    no_device,     ///< No CUDA device, or none with that index.
    call_failed,   ///< A CUDA or cuBLAS call failed; `cuda_failure::code` holds its code.
};

/// \brief A failure of `cuda_context::create`: what failed and the CUDA or cuBLAS code.
struct cuda_failure {
    cuda_error kind = cuda_error::call_failed;   ///< What failed.
    int code = 0;                                ///< The code.
};

/// \brief A call that failed on the device: an unexpected failure (out of device memory, a lost device).
class cuda_exception : public std::runtime_error {
public:
    /// \brief An exception for a failed call. \param what The call. \param code Its code.
    cuda_exception(const std::string& what, int code)
        : std::runtime_error("snicholls::gpu: " + what + " failed with CUDA code " + std::to_string(code)), code_(code)
    {
    }
    /// \brief The code. \return The code.
    [[nodiscard]] int code() const noexcept { return code_; }

private:
    int code_;
};

namespace detail::cuda {

    inline void check(cudaError_t e, const char* what)
    {
        if (e != cudaSuccess)
            throw cuda_exception(what, static_cast<int>(e));
    }

    inline void check(cublasStatus_t s, const char* what)
    {
        if (s != CUBLAS_STATUS_SUCCESS)
            throw cuda_exception(what, static_cast<int>(s));
    }

    // Whether CUDA knows the pointer as device (or managed) memory
    inline bool on_device(const void* p) noexcept
    {
        if (p == nullptr)
            return false;
        cudaPointerAttributes a{};
        if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
            (void)cudaGetLastError();   // a plain host pointer on older drivers: clear the sticky error
            return false;
        }
        return a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged;
    }

    struct device_bytes {
        void* p = nullptr;
        explicit device_bytes(std::size_t n)
        {
            if (n != 0)
                check(cudaMalloc(&p, n), "cudaMalloc");
        }
        device_bytes(const device_bytes&) = delete;
        device_bytes& operator=(const device_bytes&) = delete;
        ~device_bytes()
        {
            if (p != nullptr)
                (void)cudaFree(p);
        }
    };

    inline cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const float* alpha,
                               const float* a, int lda, const float* b, int ldb, const float* beta, float* c, int ldc, bool bf16 = false)
    {
        if (bf16)   // float in and out, bfloat16 on the tensor cores inside, the sums in float
            return cublasGemmEx(h, ta, tb, m, n, k, alpha, a, CUDA_R_32F, lda, b, CUDA_R_32F, ldb, beta, c, CUDA_R_32F, ldc,
                                CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT);
        return cublasSgemm(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    inline cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const double* alpha,
                               const double* a, int lda, const double* b, int ldb, const double* beta, double* c, int ldc, bool = false)
    {
        return cublasDgemm(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    // A matrix as cuBLAS reads it: a column-major array and its leading dimension, transposed if the view is row-major
    template <typename T>
    struct operand {
        const T* data = nullptr;
        int ld = 1;
        bool row_major = false;   // the column-major array is the view's transpose
        std::unique_ptr<device_bytes> owned{};
    };

    // Whether a view is row- or column-major as it stands, and its leading dimension
    template <typename V>
    bool blas_layout(const V& v, bool& row_major, std::size_t& ld)
    {
        const std::size_t r = v.extent(0), c = v.extent(1);
        if (v.stride(1) == 1 && (r <= 1 || v.stride(0) >= c)) {
            row_major = true;
            ld = std::max<std::size_t>({std::size_t{1}, c, r <= 1 ? c : v.stride(0)});
            return true;
        }
        if (v.stride(0) == 1 && (c <= 1 || v.stride(1) >= r)) {
            row_major = false;
            ld = std::max<std::size_t>({std::size_t{1}, r, c <= 1 ? r : v.stride(1)});
            return true;
        }
        return false;
    }

    // A view's elements, row-major and contiguous, read from wherever they live
    template <typename T>
    std::vector<T> packed(linalg::matrix_view<const T> v)
    {
        const std::size_t r = v.extent(0), c = v.extent(1), span = v.mapping().required_span_size();
        std::vector<T> whole(span), out(r * c);
        if (span != 0)
            check(cudaMemcpy(whole.data(), v.data_handle(), span * sizeof(T), cudaMemcpyDefault), "cudaMemcpy");
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t j = 0; j < c; ++j)
                out[i * c + j] = whole[i * v.stride(0) + j * v.stride(1)];
        return out;
    }

    // An input on the device: in place when it is device memory in a BLAS layout; host memory in a BLAS layout is
    // copied up as it is (its span, one transfer); any other strides, packed into a contiguous row-major copy
    template <typename T>
    operand<T> input(linalg::matrix_view<const T> v)
    {
        operand<T> o;
        std::size_t ld = 0;
        if (blas_layout(v, o.row_major, ld)) {
            o.ld = static_cast<int>(ld);
            if (on_device(v.data_handle())) {
                o.data = v.data_handle();
                return o;
            }
            const std::size_t span = v.mapping().required_span_size();
            o.owned = std::make_unique<device_bytes>(span * sizeof(T));
            if (span != 0)
                check(cudaMemcpy(o.owned->p, v.data_handle(), span * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
            o.data = static_cast<const T*>(o.owned->p);
            return o;
        }
        const auto p = packed(v);
        o.owned = std::make_unique<device_bytes>(p.size() * sizeof(T));
        if (!p.empty())
            check(cudaMemcpy(o.owned->p, p.data(), p.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
        o.data = static_cast<const T*>(o.owned->p);
        o.row_major = true;
        o.ld = static_cast<int>(std::max<std::size_t>(1, v.extent(1)));
        return o;
    }

} // namespace detail::cuda

/// \brief How the products of `float` are computed. \ingroup gpu
enum class cuda_math {
    float32,   ///< In float throughout (cuBLAS's default): the reference's rounding, to the order of the sums.
    tf32,      ///< Tensor cores, the inputs rounded to TF32 (10 bits of mantissa), the sums in float: several times
               ///< faster, each product within about 1e-3 of the float result, relative to its scale.
    bf16       ///< Tensor cores in bfloat16 (7 bits of mantissa), the sums in float: about twice TF32's rate. Through
               ///< `cuda_training_backend` the operands are rounded to bfloat16 by a kernel and multiplied as such
               ///< (`cuda_context::gemm_bf16`); through `cuda_backend` alone (no kernels) cuBLAS is only allowed
               ///< bfloat16 (`CUBLAS_COMPUTE_32F_FAST_16BF`) and may use TF32.
};

/// \brief One CUDA device with its cuBLAS handle and stream.
///
/// Create it once and hand `cuda_backend{&context}` to core's free functions. Neither copyable nor movable: backends
/// and buffers point at it.
class cuda_context {
public:
    /// \brief Opens a device and creates its cuBLAS handle and stream.
    /// \param device The device's index.
    /// \return The context, or why there is none.
    [[nodiscard]] static result<std::unique_ptr<cuda_context>, cuda_failure> create(int device = 0)
    {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || device < 0 || device >= count) {
            (void)cudaGetLastError();
            return err(cuda_failure{cuda_error::no_device, 0});
        }
        std::unique_ptr<cuda_context> c(new cuda_context(device));
        if (const cudaError_t e = cudaSetDevice(device); e != cudaSuccess)
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(e)});
        if (const cudaError_t e = cudaStreamCreate(&c->stream_); e != cudaSuccess)
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(e)});
        if (const cublasStatus_t s = cublasCreate(&c->blas_); s != CUBLAS_STATUS_SUCCESS)
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(s)});
        if (const cublasStatus_t s = cublasSetStream(c->blas_, c->stream_); s != CUBLAS_STATUS_SUCCESS)
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(s)});
        return ok(std::move(c));
    }

    /// \brief Not copyable: backends and buffers point at it.
    cuda_context(const cuda_context&) = delete;
    /// \brief Not assignable. \return Nothing; deleted.
    cuda_context& operator=(const cuda_context&) = delete;
    /// \brief Releases the handle and the stream.
    ~cuda_context() { release(); }

    /// \brief The device's name. \return The name.
    [[nodiscard]] std::string device_name() const
    {
        cudaDeviceProp p{};
        return cudaGetDeviceProperties(&p, device_) == cudaSuccess ? std::string(p.name) : std::string();
    }
    /// \brief The device's index. \return The index.
    [[nodiscard]] int device() const noexcept { return device_; }
    /// \brief The context's stream, on which its products run (and `cuda_ops.hpp`'s kernels). \return The stream.
    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
    /// \brief Waits for everything queued on the context's stream. \throws cuda_exception if the device reports an error.
    void synchronize() const { detail::cuda::check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); }

    /// \brief `C = alpha A B + beta C` on the device, any layout, host or device memory. Called by `cuda_backend::gemm`.
    /// \param alpha The scale of the product. \param a `m x k`. \param b `k x n`. \param beta The scale of `C`.
    /// \param c `m x n`; may not overlap `a` or `b`. \param math How a product of `float` is computed.
    /// \throws cuda_exception if a CUDA or cuBLAS call fails.
    template <typename T>
        requires(std::is_same_v<T, float> || std::is_same_v<T, double>)
    void gemm(T alpha, linalg::matrix_view<const T> a, linalg::matrix_view<const T> b, T beta, linalg::matrix_view<T> c,
              cuda_math math = cuda_math::float32) const
    {
        namespace dc = detail::cuda;
        const std::size_t m = c.extent(0), n = c.extent(1), k = a.extent(1);
        if (m == 0 || n == 0)
            return;
        const std::lock_guard lock(mutex_);
        dc::check(cudaSetDevice(device_), "cudaSetDevice");
        if (const cublasStatus_t s = cublasSetMathMode(blas_, math == cuda_math::tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
            s != CUBLAS_STATUS_SUCCESS)
            throw cuda_exception("cublasSetMathMode", static_cast<int>(s));
        const auto A = dc::input(a), B = dc::input(b);
        // The output: in place when it is device memory in a BLAS layout; host memory in a BLAS layout, its span copied
        // (up only when it is read or has gaps to keep) and back; any other strides, a contiguous row-major device copy
        bool c_row = false;
        std::size_t c_ld = 0;
        const bool blas_c = dc::blas_layout(c, c_row, c_ld);
        const bool in_place = blas_c && dc::on_device(c.data_handle());
        const bool span_copy = blas_c && !in_place;
        const std::size_t c_span = c.mapping().required_span_size();
        std::unique_ptr<dc::device_bytes> staged;
        T* cp = c.data_handle();
        if (span_copy) {
            staged = std::make_unique<dc::device_bytes>(c_span * sizeof(T));
            cp = static_cast<T*>(staged->p);
            if (beta != T(0) || !c.mapping().is_exhaustive())
                dc::check(cudaMemcpy(cp, c.data_handle(), c_span * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
        } else if (!in_place) {
            staged = std::make_unique<dc::device_bytes>(m * n * sizeof(T));
            cp = static_cast<T*>(staged->p);
            c_row = true;
            c_ld = n;
            if (beta != T(0)) {
                const auto p = dc::packed(linalg::matrix_view<const T>(c));
                dc::check(cudaMemcpy(cp, p.data(), p.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
            }
        }
        const int ik = static_cast<int>(k);
        auto op = [](bool row_major, bool want_transposed) {
            return row_major != want_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
        };
        if (!c_row)   // C column-major: C = alpha op(A) op(B) + beta C
            dc::check(dc::gemm(blas_, op(A.row_major, false), op(B.row_major, false), static_cast<int>(m), static_cast<int>(n), ik, &alpha,
                               A.data, A.ld, B.data, B.ld, &beta, cp, static_cast<int>(c_ld), math == cuda_math::bf16),
                      "cublas gemm");
        else          // C row-major: its column-major array is Cᵀ = alpha Bᵀ Aᵀ + beta Cᵀ
            dc::check(dc::gemm(blas_, op(B.row_major, true), op(A.row_major, true), static_cast<int>(n), static_cast<int>(m), ik, &alpha,
                               B.data, B.ld, A.data, A.ld, &beta, cp, static_cast<int>(c_ld), math == cuda_math::bf16),
                      "cublas gemm");
        // Wait only for a staged copy: it is copied back or freed below. Device memory in place stays queued on the
        // stream, so the device runs the next step while the host enqueues it
        if (staged || A.owned || B.owned)
            dc::check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
        if (span_copy) {
            dc::check(cudaMemcpy(c.data_handle(), cp, c_span * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
        } else if (!in_place) {   // back into the view, wherever it lives, keeping the gaps of its span
            std::vector<T> result(m * n);
            dc::check(cudaMemcpy(result.data(), cp, result.size() * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
            const std::size_t span = c.mapping().required_span_size();
            std::vector<T> whole(span);
            dc::check(cudaMemcpy(whole.data(), c.data_handle(), span * sizeof(T), cudaMemcpyDefault), "cudaMemcpy");
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    whole[i * c.stride(0) + j * c.stride(1)] = result[i * n + j];
            dc::check(cudaMemcpy(c.data_handle(), whole.data(), span * sizeof(T), cudaMemcpyDefault), "cudaMemcpy");
        }
    }

    /// \brief `C = alpha A B + beta C` with `A` and `B` in bfloat16 (their bits as `std::uint16_t`) and `C` in float, on
    /// the tensor cores with float sums; queued on the stream. `cuda_training_backend` converts float operands into
    /// these under `cuda_math::bf16`.
    /// \param alpha The scale. \param a `m x k`. \param b `k x n`. \param beta The scale of `C`. \param c `m x n`.
    /// \throws std::invalid_argument unless every operand is device memory in a BLAS layout; cuda_exception if a call
    /// fails.
    void gemm_bf16(float alpha, linalg::matrix_view<const std::uint16_t> a, linalg::matrix_view<const std::uint16_t> b, float beta,
                   linalg::matrix_view<float> c) const
    {
        namespace dc = detail::cuda;
        const std::size_t m = c.extent(0), n = c.extent(1), k = a.extent(1);
        if (m == 0 || n == 0)
            return;
        bool ar = false, br = false, cr = false;
        std::size_t lda = 0, ldb = 0, ldc = 0;
        if (!dc::blas_layout(a, ar, lda) || !dc::blas_layout(b, br, ldb) || !dc::blas_layout(c, cr, ldc) || !dc::on_device(a.data_handle())
            || !dc::on_device(b.data_handle()) || !dc::on_device(c.data_handle()))
            throw std::invalid_argument("snicholls::gpu::cuda_context::gemm_bf16: every operand must be device memory in a BLAS layout");
        const std::lock_guard lock(mutex_);
        dc::check(cudaSetDevice(device_), "cudaSetDevice");
        auto op = [](bool row_major, bool want_transposed) { return row_major != want_transposed ? CUBLAS_OP_T : CUBLAS_OP_N; };
        const auto ex = [&](cublasOperation_t ta, cublasOperation_t tb, std::size_t mm, std::size_t nn, const void* x, std::size_t ldx, const void* y,
                            std::size_t ldy) {
            return cublasGemmEx(blas_, ta, tb, int(mm), int(nn), int(k), &alpha, x, CUDA_R_16BF, int(ldx), y, CUDA_R_16BF, int(ldy), &beta,
                                c.data_handle(), CUDA_R_32F, int(ldc), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        };
        if (!cr)
            dc::check(ex(op(ar, false), op(br, false), m, n, a.data_handle(), lda, b.data_handle(), ldb), "cublas gemm bf16");
        else   // C row-major: Cᵀ = Bᵀ Aᵀ, as in gemm
            dc::check(ex(op(br, true), op(ar, true), n, m, b.data_handle(), ldb, a.data_handle(), lda), "cublas gemm bf16");
    }

private:
    explicit cuda_context(int device) : device_(device) {}

    void release() noexcept
    {
        if (blas_ != nullptr)
            cublasDestroy(blas_);
        if (stream_ != nullptr)
            cudaStreamDestroy(stream_);
    }

    int device_ = 0;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t blas_ = nullptr;
    mutable std::mutex mutex_;
};

/// \brief `n` elements of device memory, released when it goes out of scope; `span()` is a view core's functions take.
/// \tparam T The element type.
template <typename T>
    requires std::is_trivially_copyable_v<T>
class device_buffer {
public:
    /// \brief Allocates `n` elements on the context's device. \param context The device. \param n The elements.
    /// \throws cuda_exception if the allocation fails.
    device_buffer(const cuda_context& context, std::size_t n) : size_(n)
    {
        detail::cuda::check(cudaSetDevice(context.device()), "cudaSetDevice");
        if (n != 0)
            detail::cuda::check(cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(T)), "cudaMalloc");
    }
    /// \brief Not copyable.
    device_buffer(const device_buffer&) = delete;
    /// \brief Not assignable. \return Nothing; deleted.
    device_buffer& operator=(const device_buffer&) = delete;
    /// \brief Releases the memory.
    ~device_buffer()
    {
        if (data_ != nullptr)
            (void)cudaFree(data_);
    }

    /// \brief A view of the device memory, for core's functions (never dereference it on the host). \return The span.
    [[nodiscard]] std::span<T> span() const noexcept { return {data_, size_}; }
    /// \brief The elements. \return The count.
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    /// \brief Copies host memory to the device. \param from `size()` elements. \throws cuda_exception on failure.
    void upload(std::span<const T> from)
    {
        if (from.size() != size_)
            throw std::invalid_argument("snicholls::gpu::device_buffer::upload: the sizes differ");
        if (size_ != 0)
            detail::cuda::check(cudaMemcpy(data_, from.data(), size_ * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    }
    /// \brief Copies the device memory to the host. \param to `size()` elements. \throws cuda_exception on failure.
    void download(std::span<T> to) const
    {
        if (to.size() != size_)
            throw std::invalid_argument("snicholls::gpu::device_buffer::download: the sizes differ");
        if (size_ != 0)
            detail::cuda::check(cudaMemcpy(to.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
    }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

/// \brief Work queued on a context's stream, captured once as a CUDA graph and launched again as a whole: one launch
/// in place of each of its kernels and products.
///
/// A training step is thousands of small launches; their overhead and the gaps between them are a large part of a
/// step. `capture` runs a function that queues work on the context's stream (`cuda_ops.hpp`'s steps, products over
/// device memory) without executing it, and `launch` replays exactly that work: the same kernels on the same device
/// addresses with the same arguments. So whatever changes between replays (a micro-batch's tokens, the learning rate)
/// must live in device memory the captured work reads, written before each `launch` (`device_buffer::upload` into the
/// same buffer).
///
/// The captured function must not allocate, free, wait for the stream or read device memory on the host: CUDA refuses
/// such a call during capture (the capture mode is thread-local), and `capture` returns the failure. Run the work once
/// uncaptured first, so that every buffer it grows (`cuda_kernels::scratch`) has its size. A replay gives the bits of
/// the uncaptured run: it is the same work.
///
/// \snippet cuda_demo.cpp cuda_graph
///
/// **Guarantees.** Movable, not copyable; owns the graph and its executable. `launch` is queued on the context's stream
/// and ordered with everything else on it. The context must outlive the graph.
class cuda_graph {
public:
    /// \brief Captures the work `enqueue` queues on the context's stream.
    /// \param context The device and stream. \param enqueue Called once; queues work on `context.stream()`.
    /// \return The graph, or the failure of the capture or of its instantiation (with CUDA's code). An exception from
    /// `enqueue` ends the capture and propagates.
    template <typename F>
    [[nodiscard]] static result<cuda_graph, cuda_failure> capture(const cuda_context& context, F&& enqueue)
    {
        cudaStream_t s = context.stream();
        if (const cudaError_t e = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal); e != cudaSuccess)
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(e)});
        cudaGraph_t graph = nullptr;
        try {
            std::forward<F>(enqueue)();
        } catch (...) {
            (void)cudaStreamEndCapture(s, &graph);
            if (graph != nullptr)
                (void)cudaGraphDestroy(graph);
            (void)cudaGetLastError();
            throw;
        }
        if (const cudaError_t e = cudaStreamEndCapture(s, &graph); e != cudaSuccess) {
            if (graph != nullptr)
                (void)cudaGraphDestroy(graph);
            (void)cudaGetLastError();
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(e)});
        }
        cuda_graph out(context.stream());
        out.graph_ = graph;
        if (const cudaError_t e = cudaGraphInstantiate(&out.exec_, graph, 0); e != cudaSuccess)
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(e)});
        return ok(std::move(out));
    }

    /// \brief Moves the graph. \param o Left empty.
    cuda_graph(cuda_graph&& o) noexcept : stream_(o.stream_), graph_(std::exchange(o.graph_, nullptr)), exec_(std::exchange(o.exec_, nullptr)) {}
    /// \brief Moves the graph, releasing this one's. \param o Left empty. \return This.
    cuda_graph& operator=(cuda_graph&& o) noexcept
    {
        if (this != &o) {
            release();
            stream_ = o.stream_, graph_ = std::exchange(o.graph_, nullptr), exec_ = std::exchange(o.exec_, nullptr);
        }
        return *this;
    }
    cuda_graph(const cuda_graph&) = delete;              ///< Not copyable.
    cuda_graph& operator=(const cuda_graph&) = delete;   ///< Not copyable. \return Nothing; deleted.
    /// \brief Releases the graph and its executable.
    ~cuda_graph() { release(); }

    /// \brief Queues the captured work on the context's stream, once more. \throws cuda_exception if the launch fails.
    void launch() const { detail::cuda::check(cudaGraphLaunch(exec_, stream_), "cudaGraphLaunch"); }

    /// \brief The kernels, copies and other nodes the graph holds. \return Their count.
    [[nodiscard]] std::size_t nodes() const
    {
        std::size_t n = 0;
        detail::cuda::check(cudaGraphGetNodes(graph_, nullptr, &n), "cudaGraphGetNodes");
        return n;
    }

private:
    explicit cuda_graph(cudaStream_t s) noexcept : stream_(s) {}
    void release() noexcept
    {
        if (exec_ != nullptr)
            (void)cudaGraphExecDestroy(exec_);
        if (graph_ != nullptr)
            (void)cudaGraphDestroy(graph_);
        exec_ = nullptr, graph_ = nullptr;
    }
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
};

/// \brief The `linalg::blas_backend` on a `cuda_context`: `gemm` and `gemv` on the device, the rest on the host.
///
/// Trivially copyable (a pointer and the choice of `cuda_math`); pass it by value. `float` and `double` products run on
/// the device, over device memory in place or host memory staged; any other scalar type, and the four host operations,
/// on the host. `math = cuda_math::tf32` puts the `float` products on the tensor cores.
struct cuda_backend {
    const cuda_context* context = nullptr;   ///< The device; must outlive the backend.
    cuda_math math = cuda_math::float32;     ///< How products of `float` are computed: in float, or on tensor cores.

    /// \brief `x · y`, on the host. \param x A vector. \param y Its partner. \return The dot product.
    /// \throws cuda_exception for device memory.
    template <linalg::scalar T>
    [[nodiscard]] T dot(linalg::vector_view<const T> x, linalg::vector_view<const T> y) const
    {
        host_only(x.data_handle(), y.data_handle(), "dot");
        return linalg::reference.dot(x, y);
    }

    /// \brief `y += alpha x`, on the host. \param alpha The scale. \param x The input. \param y The output.
    /// \throws cuda_exception for device memory.
    template <linalg::scalar T>
    void axpy(T alpha, linalg::vector_view<const T> x, linalg::vector_view<T> y) const
    {
        host_only(x.data_handle(), y.data_handle(), "axpy");
        linalg::reference.axpy(alpha, x, y);
    }

    /// \brief `y = alpha A x + beta y`, on the device as a product with one column.
    /// \param alpha The scale. \param a `m x k`. \param x `k`. \param beta The scale of `y`. \param y `m`.
    /// \throws cuda_exception if a call fails.
    template <linalg::scalar T>
        requires(std::is_same_v<T, float> || std::is_same_v<T, double>)
    void gemv(T alpha, linalg::matrix_view<const T> a, linalg::vector_view<const T> x, T beta, linalg::vector_view<T> y) const
    {
        context->gemm(alpha, a, linalg::as_column(x), beta, linalg::as_column(y), math);
    }

    /// \brief `y = alpha A x + beta y` for a type cuBLAS does not take, on the host.
    /// \param alpha The scale. \param a `m x k`. \param x `k`. \param beta The scale of `y`. \param y `m`.
    template <linalg::scalar T>
    void gemv(T alpha, linalg::matrix_view<const T> a, linalg::vector_view<const T> x, T beta, linalg::vector_view<T> y) const
    {
        linalg::reference.gemv(alpha, a, x, beta, y);
    }

    /// \brief `C = alpha A B + beta C`, on the device.
    /// \param alpha The scale. \param a `m x k`. \param b `k x n`. \param beta The scale of `C`. \param c `m x n`.
    /// \throws cuda_exception if a call fails.
    template <linalg::scalar T>
        requires(std::is_same_v<T, float> || std::is_same_v<T, double>)
    void gemm(T alpha, linalg::matrix_view<const T> a, linalg::matrix_view<const T> b, T beta, linalg::matrix_view<T> c) const
    {
        context->gemm(alpha, a, b, beta, c, math);
    }

    /// \brief `C = alpha A B + beta C` for a type cuBLAS does not take, on the host.
    /// \param alpha The scale. \param a `m x k`. \param b `k x n`. \param beta The scale of `C`. \param c `m x n`.
    template <linalg::scalar T>
    void gemm(T alpha, linalg::matrix_view<const T> a, linalg::matrix_view<const T> b, T beta, linalg::matrix_view<T> c) const
    {
        linalg::reference.gemm(alpha, a, b, beta, c);
    }

    /// \brief `C = alpha A Aᵀ + beta C` in one triangle, on the host.
    /// \param alpha The scale. \param a `n x k`. \param beta The scale of `C`. \param c `n x n`. \param t The triangle.
    /// \throws cuda_exception for device memory.
    template <linalg::scalar T>
    void syrk(T alpha, linalg::matrix_view<const T> a, T beta, linalg::matrix_view<T> c, linalg::triangle t) const
    {
        host_only(a.data_handle(), c.data_handle(), "syrk");
        linalg::reference.syrk(alpha, a, beta, c, t);
    }

    /// \brief Solves `T X = B` in place for a triangular `T`, on the host.
    /// \param t The triangular matrix. \param b The right-hand sides. \param which Its triangle. \param diag Its diagonal.
    /// \return Failure for a singular `T`. \throws cuda_exception for device memory.
    template <linalg::scalar T>
    [[nodiscard]] result<void, linalg::error> triangular_solve(linalg::matrix_view<const T> t, linalg::matrix_view<T> b,
                                                               linalg::triangle which, linalg::diagonal diag) const
    {
        host_only(t.data_handle(), b.data_handle(), "triangular_solve");
        return linalg::reference.triangular_solve(t, b, which, diag);
    }

private:
    static void host_only(const void* a, const void* b, const char* what)
    {
        if (detail::cuda::on_device(a) || detail::cuda::on_device(b))
            throw cuda_exception(std::string(what) + " on device memory (not on the device yet)", 0);
    }
};

static_assert(linalg::blas_backend<cuda_backend, float> && linalg::blas_backend<cuda_backend, double>);

} // namespace snicholls::gpu

#endif /* snicholls_gpu_cuda_hpp */
