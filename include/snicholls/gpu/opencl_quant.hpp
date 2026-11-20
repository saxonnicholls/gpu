//
//  opencl_quant.hpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::gpu::opencl_quant_kernels`: matrix products whose weights stay in core's quantised formats on an OpenCL device, dequantised in registers, forward and backward to the input (inference and QLoRA fine-tuning).
/// \ingroup gpu
///
/// **Overview.** A quantised model keeps its weights in blocks of a few bits (ggml's Q8_0, Q4_0 and MXFP4,
/// `ml/quantize.hpp`; QLoRA's NF4, `ml/nf4.hpp`; or dense 8- and 16-bit floats, `ml/minifloat.hpp`). Dequantising a
/// whole matrix before each product costs its float copy in memory and in bandwidth; these kernels read the blocks
/// themselves and dequantise each weight in registers as the product needs it, as llama.cpp's do. Core's operations
/// are overloaded for `opencl_quant_backend` (found by argument-dependent lookup), so core's `ml::lora_sequence` and
/// `ml::lora_sequence_backward` run a QLoRA layer with its frozen base on the device:
///
/// - `linear_sequence(b, w, xs, ys, bias)`: `Y = X Wᵀ + b` for `n` rows of float activations.
/// - `linear_sequence_backward_input(b, w, dys, dxs)`: `dX += dY W`, the input's gradient through the frozen base
///   (there is none to the weights: they are frozen).
/// - `linear_sequence_q8(b, w, xs, ys, bias)`: the integer path: each row of `X` quantised on the device to the 8-bit
///   blocks core's dot product pairs the format with (Q8_0 for Q8_0, Q4_0, MXFP4 and Q5_0; Q8_1 for Q4_1 and Q5_1; Q8_K
///   for Q2_K to Q6_K, TQ1_0 and TQ2_0), each output
///   core's block dot product (`ml::dot`), as ggml's CPU matrix-vector products and core's `ml::linear` do.
///
/// `w` is one of core's views over host memory (`ml::quantized_weights<Block>`, `ml::nf4_weights`,
/// `ml::dense_weights<S>` for `bfloat16`, `float16`, `float8_e4m3` or `float8_e5m2`) or a `block_weights<Block>` of
/// Q4_1, Q5_0, Q5_1, NVFP4, Q1_0, Q2_0, Q2_K to Q6_K, TQ1_0, TQ2_0, IQ4_NL or IQ4_XS (where the pinned core has them), copied to the device for the
/// call, or an `opencl_quant_matrix`, the same weights kept on the device by `opencl_quant_kernels::upload` (frozen
/// weights: upload once, multiply many times). The activations are host memory, copied for each call, as
/// `opencl_backend`'s products do.
///
/// \snippet opencl_quant_demo.cpp quant
///
/// **Relationships.** `opencl.hpp` (the context, and `opencl_backend`, whose dense products the LoRA adapters take),
/// `cuda_quant.hpp` (the same kernels on CUDA, from the same source), core's `ml/linear.hpp`, `ml/quantize.hpp`,
/// `ml/nf4.hpp`, `ml/lora.hpp` (the operations overloaded and the references). Group: \ref gpu.
///
/// **Implementations.** One source for OpenCL and CUDA (`detail/quant_source.hpp`): a format is one device function
/// giving the weight at (row, column), decoded with core's arithmetic.
///
/// - **Matrix-vector** (decode, at most `opencl_quant_kernels::matvec_max` rows, 8 by default): a work-group of 64
///   per four outputs, sharing each vector load of `x`; each item decodes eight consecutive columns at a time (a block's
///   scale once), then the 64 partial sums of each output are added by a fixed tree.
/// - **Tiled** (prefill, the backward): 64 x 64 tiles, 16 deep, the weights dequantised into local memory as the tile
///   is loaded, 16 x 16 items of 4 x 4 outputs, each output summed in index order: the bits of core's reference
///   product over the dequantised matrix (`linalg::reference`, then `add_bias_rows`, as `ml::linear_sequence` does).
/// - **Q8_0 activations**: a thread per block of 32 quantises (core's `quantize<block_q8_0>`); a work-group per
///   (output, row) computes core's per-block terms in integers and adds them in block order.
///
/// **Guarantees.** Deterministic: no atomics, every sum in a fixed order, so two runs give the same bits.
///
/// | operation | against | bound |
/// |---|---|---|
/// | tiled forward and backward | core's reference product over `dequantize`d weights | to the bit |
/// | matrix-vector | the same | `|Δ| ≤ k 2^-23 Σ_j |w_j x_j|` per output (the classical bound of a reordered sum) |
/// | Q8_0 activations | `ml::linear(w, prepare_input(w, x), y, bias)` | to the bit where the device divides correctly rounded (`exact_q8()`) |
///
/// On a device that flushes subnormal floats, the bits hold away from them. Weights and activations are not modified.
/// One `opencl_quant_kernels` serves any number of threads (its kernel arguments are set under a mutex); it and every
/// `opencl_quant_matrix` must not outlive the context.
///
/// **Errors.** `create` returns an `opencl_failure` when the program does not build. A shape the format does not allow
/// (columns not a multiple of the block; spans of the wrong size) throws `std::invalid_argument`; Q8_0 activations
/// with weights of another format, `std::invalid_argument`; a backend without kernels, `std::logic_error`; a failed
/// OpenCL call, `opencl_exception`.
///
/// **Performance.** `benchmarks/opencl_quant_benchmark.cpp`, transfers included. First numbers, an AMD Radeon Pro Vega
/// II (32 GB HBM2, macOS 26, Apple OpenCL 1.2, Release, 2026-11-20): see `docs/performance.md`.
///
/// **Tests and demos.** `tests/opencl_quant_test.cpp` (every format, every path, against core's references; the edge
/// cases; determinism; a QLoRA layer through core's `lora_sequence`), `demos/opencl_quant_demo.cpp`.
///
/// **Dependencies.** `opencl.hpp`, `detail/quant_source.hpp`, core's `ml/linear.hpp`, `ml/nf4.hpp`, `ml/quantize.hpp`,
/// `ml/minifloat.hpp`, `<algorithm>`, `<array>`, `<cstddef>`, `<memory>`, `<mutex>`, `<span>`, `<stdexcept>`, `<string>`, `<vector>`.
///
/// **Known issues.** OpenCL 1.2 has no integer dot instructions or tensor cores: everything is in float and int32 on
/// the vector units. The device's division is correctly rounded only where it says so
/// (`CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT`); elsewhere a Q8_0 scale can differ from core's in its last bit.
///
/// **Status.** Experimental, new in 0.9.0.

#ifndef snicholls_gpu_opencl_quant_hpp
#define snicholls_gpu_opencl_quant_hpp

#include <snicholls/gpu/detail/quant_source.hpp>
#include <snicholls/gpu/opencl.hpp>

#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/nf4.hpp>
#include <snicholls/core/ml/quantize.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace snicholls::gpu {

class opencl_quant_kernels;

/// \brief A weight matrix in one of core's quantised formats, kept in an OpenCL device's memory: made by
/// `opencl_quant_kernels::upload`, read by the products of `opencl_quant_backend`.
///
/// A handle: copies share the device memory, which is released with the last of them. `rows` and `cols` are members,
/// as on core's views, so `ml::lora_weights` takes it as its frozen base.
class opencl_quant_matrix {
public:
    std::size_t rows = 0;   ///< The outputs.
    std::size_t cols = 0;   ///< The inputs.

    /// \brief The bytes it holds on the device: the weights and their scales. \return The bytes.
    [[nodiscard]] std::size_t bytes() const noexcept { return s_ ? s_->bytes : 0; }
    /// \brief Whether it holds weights (a default-constructed matrix does not). \return True if it does.
    [[nodiscard]] bool has_weights() const noexcept { return s_ != nullptr; }

private:
    friend class opencl_quant_kernels;
    struct storage {
        detail::quant::format fmt = detail::quant::format::q8_0;
        cl_mem w = nullptr, sb = nullptr, gs = nullptr;
        float mean = 0.0f;
        std::size_t bytes = 0;
        storage() = default;
        storage(const storage&) = delete;
        storage& operator=(const storage&) = delete;
        ~storage()
        {
            for (cl_mem m : {w, sb, gs})
                if (m)
                    clReleaseMemObject(m);
        }
    };
    std::shared_ptr<const storage> s_;
};

/// \brief The quantised-weight kernels, compiled for an `opencl_context`'s device.
///
/// Create once (the program takes a moment to build) and hand `opencl_quant_backend{{&context}, kernels.get()}` to core's
/// operations. Neither copyable nor movable: backends point at it.
class opencl_quant_kernels {
public:
    /// \brief Builds the program for the context's device.
    /// \param context The device; must outlive the kernels.
    /// \return The kernels, or why the program did not build (with its log).
    [[nodiscard]] static result<std::unique_ptr<opencl_quant_kernels>, opencl_failure> create(const opencl_context& context)
    {
        std::unique_ptr<opencl_quant_kernels> k(new opencl_quant_kernels(context));
        cl_device_fp_config fp = 0;
        clGetDeviceInfo(context.native_device(), CL_DEVICE_SINGLE_FP_CONFIG, sizeof(fp), &fp, nullptr);
        k->exact_q8_ = (fp & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) != 0;
        const std::string source = detail::quant::program(detail::quant::opencl_prelude);
        const char* text = source.c_str();
        cl_int code = CL_SUCCESS;
        k->program_ = clCreateProgramWithSource(context.native_context(), 1, &text, nullptr, &code);
        if (code != CL_SUCCESS)
            return err(opencl_failure{opencl_error::call_failed, code, {}});
        cl_device_id device = context.native_device();
        const char* options = k->exact_q8_ ? "-cl-fp32-correctly-rounded-divide-sqrt" : "";
        if (const cl_int built = clBuildProgram(k->program_, 1, &device, options, nullptr, nullptr); built != CL_SUCCESS) {
            std::size_t size = 0;
            clGetProgramBuildInfo(k->program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
            std::string log(size, '\0');
            clGetProgramBuildInfo(k->program_, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
            return err(opencl_failure{opencl_error::build_failed, built, std::move(log)});
        }
        const auto make = [&](const std::string& name, cl_kernel& slot) {
            slot = clCreateKernel(k->program_, name.c_str(), &code);
            return code == CL_SUCCESS;
        };
        for (std::size_t f = 0; f < detail::quant::format_count; ++f) {
            const std::string n(detail::quant::names[f]);
            if (!make("qmv_" + n, k->mv_[f]) || !make("qmm_" + n, k->mm_[f]) || !make("qmmt_" + n, k->mmt_[f])
                || ((detail::quant::has_q8_0_dot(detail::quant::format(f)) || detail::quant::has_q8_1_dot(detail::quant::format(f)))
                    && !make("qdot_" + n, k->dot_[f]))
                || (detail::quant::has_q8_k_dot(detail::quant::format(f)) && !make("qkdot_" + n, k->dot_[f])))
                return err(opencl_failure{opencl_error::call_failed, code, {}});
        }
        if (!make("quantize_q8", k->quantize_) || !make("quantize_q8_1", k->quantize_1_) || !make("quantize_q8k", k->quantize_k_))
            return err(opencl_failure{opencl_error::call_failed, code, {}});
        return ok(std::move(k));
    }

    /// \brief Not copyable: backends point at it.
    opencl_quant_kernels(const opencl_quant_kernels&) = delete;
    /// \brief Not assignable. \return Nothing; deleted.
    opencl_quant_kernels& operator=(const opencl_quant_kernels&) = delete;
    /// \brief Releases the kernels and the program.
    ~opencl_quant_kernels()
    {
        for (auto* set : {&mv_, &mm_, &mmt_, &dot_})
            for (cl_kernel kernel : *set)
                if (kernel)
                    clReleaseKernel(kernel);
        for (cl_kernel kernel : {quantize_, quantize_1_, quantize_k_})
            if (kernel)
                clReleaseKernel(kernel);
        if (program_)
            clReleaseProgram(program_);
    }

    /// \brief The most rows of activations a forward product runs as matrix-vector products (8 rows a launch); more
    /// take the tiled kernel. 0 sends every product to the tiled kernel, which gives the reference's bits.
    std::size_t matvec_max = 8;

    /// \brief Whether the device divides correctly rounded, so `linear_sequence_q8` quantises the activations to
    /// core's bits. \return True if it does.
    [[nodiscard]] bool exact_q8() const noexcept { return exact_q8_; }
    /// \brief The context. \return It.
    [[nodiscard]] const opencl_context& context() const noexcept { return *context_; }

    /// \brief Copies a weight storage to the device.
    /// \param w `ml::quantized_weights<Block>`, `ml::nf4_weights`, or `ml::dense_weights<S>` of a minifloat `S`, over
    ///   host memory.
    /// \return The matrix on the device.
    /// \throws std::invalid_argument for a shape the format does not allow; opencl_exception if a copy fails.
    template <typename W>
        requires detail::quant::weights<W>
    [[nodiscard]] opencl_quant_matrix upload(const W& w) const
    {
        const detail::quant::parts p = detail::quant::traits<W>::of(w);
        if (!detail::quant::valid(p))
            throw std::invalid_argument("snicholls::gpu::opencl_quant_kernels::upload: the weights' size does not match their shape and format");
        auto s = std::make_shared<opencl_quant_matrix::storage>();
        s->fmt = p.fmt;
        s->mean = p.mean;
        s->bytes = p.weights.size() + p.block_scales.size() + p.group_scales.size();
        s->w = make_buffer(p.weights);
        s->sb = p.block_scales.empty() ? nullptr : make_buffer(p.block_scales);
        s->gs = p.group_scales.empty() ? nullptr : make_buffer(p.group_scales);
        opencl_quant_matrix m;
        m.rows = p.rows;
        m.cols = p.cols;
        m.s_ = std::move(s);
        return m;
    }

    /// \brief `Y = X Wᵀ + b`: matrix-vector kernels for at most `matvec_max` rows, else the tiled kernel.
    /// \param w The weights. \param xs `n · cols`. \param ys Written, `n · rows`. \param bias `rows`, or empty.
    /// \throws std::invalid_argument if the sizes do not fit; opencl_exception if a call fails.
    void forward(const opencl_quant_matrix& w, std::span<const float> xs, std::span<float> ys, std::span<const float> bias) const
    {
        const std::size_t n = rows_of(w, xs, w.cols, ys, w.rows, bias);
        if (n == 0 || w.rows == 0)
            return;
        detail::buffer db(context_->native_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bias.size_bytes(), const_cast<float*>(bias.data()));
        const cl_int has_bias = bias.empty() ? 0 : 1;
        const auto& s = *w.s_;
        const std::size_t f = std::size_t(s.fmt);
        if (n <= matvec_max) {
            for (std::size_t t0 = 0; t0 < n; t0 += detail::quant::mv_tokens) {
                const cl_ulong count = cl_ulong(std::min(detail::quant::mv_tokens, n - t0));
                detail::buffer part_x(context_->native_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, std::size_t(count) * w.cols * sizeof(float),
                                      const_cast<float*>(xs.data() + t0 * w.cols));
                detail::buffer part_y(context_->native_context(), CL_MEM_WRITE_ONLY, std::size_t(count) * w.rows * sizeof(float), nullptr);
                const cl_ulong rows = w.rows;
                const std::lock_guard lock(mutex_);
                cl_kernel k = mv_[f];
                cl_uint i = weight_args(k, s, w.cols);
                set(k, i, part_x.mem), set(k, i, count), set(k, i, rows), set(k, i, db.mem), set(k, i, has_bias), set(k, i, part_y.mem);
                const std::size_t global = (w.rows + detail::quant::mv_rows - 1) / detail::quant::mv_rows * 64, local = 64;
                detail::check(clEnqueueNDRangeKernel(context_->native_queue(), k, 1, nullptr, &global, &local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
                detail::check(clEnqueueReadBuffer(context_->native_queue(), part_y.mem, CL_TRUE, 0, std::size_t(count) * w.rows * sizeof(float),
                                                  ys.data() + t0 * w.rows, 0, nullptr, nullptr),
                              "clEnqueueReadBuffer");
            }
            return;
        }
        detail::buffer dx(context_->native_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, xs.size_bytes(), const_cast<float*>(xs.data()));
        detail::buffer dy(context_->native_context(), CL_MEM_WRITE_ONLY, ys.size_bytes(), nullptr);
        tiled(mm_[f], s, w.cols, n, w.rows, w.cols, dx.mem, db.mem, has_bias, 0.0f, dy.mem);
        detail::check(clEnqueueReadBuffer(context_->native_queue(), dy.mem, CL_TRUE, 0, ys.size_bytes(), ys.data(), 0, nullptr, nullptr),
                      "clEnqueueReadBuffer");
    }

    /// \brief `dX += dY W`, the tiled kernel.
    /// \param w The weights. \param dys `n · rows`. \param dxs Added into, `n · cols`.
    /// \throws std::invalid_argument if the sizes do not fit; opencl_exception if a call fails.
    void backward_input(const opencl_quant_matrix& w, std::span<const float> dys, std::span<float> dxs) const
    {
        const std::size_t n = rows_of(w, dys, w.rows, dxs, w.cols, {});
        if (n == 0 || w.cols == 0)
            return;
        detail::buffer dy(context_->native_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dys.size_bytes(), const_cast<float*>(dys.data()));
        detail::buffer dx(context_->native_context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dxs.size_bytes(), dxs.data());
        const auto& s = *w.s_;
        tiled(mmt_[std::size_t(s.fmt)], s, w.cols, n, w.cols, w.rows, dy.mem, dy.mem, 0, 1.0f, dx.mem);
        detail::check(clEnqueueReadBuffer(context_->native_queue(), dx.mem, CL_TRUE, 0, dxs.size_bytes(), dxs.data(), 0, nullptr, nullptr),
                      "clEnqueueReadBuffer");
    }

    /// \brief `Y = X Wᵀ + b` with `X` quantised on the device as core's dot product pairs the weights (Q8_0, Q8_1 or
    /// Q8_K) and each output core's block dot product.
    /// \param w Weights of one of those formats. \param xs `n · cols`. \param ys Written, `n · rows`. \param bias `rows`, or empty.
    /// \throws std::invalid_argument for weights of another format or sizes that do not fit; opencl_exception if a call fails.
    void forward_q8(const opencl_quant_matrix& w, std::span<const float> xs, std::span<float> ys, std::span<const float> bias) const
    {
        const std::size_t n = rows_of(w, xs, w.cols, ys, w.rows, bias);
        if (!detail::quant::has_q8_dot(w.s_->fmt))
            throw std::invalid_argument("snicholls::gpu: 8-bit activations need weights of a format with core's integer dot product");
        if (n == 0 || w.rows == 0)
            return;
        const auto ctx = context_->native_context();
        const auto queue = context_->native_queue();
        const bool k_quant = detail::quant::has_q8_k_dot(w.s_->fmt), q8_1 = detail::quant::has_q8_1_dot(w.s_->fmt);
        const cl_ulong blocks = n * w.cols / (k_quant ? 256 : 32);
        cl_kernel quantize = k_quant ? quantize_k_ : q8_1 ? quantize_1_ : quantize_;
        const std::size_t block_bytes = k_quant ? detail::quant::q8k_bytes : q8_1 ? detail::quant::q8_1_bytes : detail::quant::q8_bytes;
        detail::buffer dx(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, xs.size_bytes(), const_cast<float*>(xs.data()));
        detail::buffer dq(ctx, CL_MEM_READ_WRITE, std::size_t(blocks) * block_bytes, nullptr);
        detail::buffer dy(ctx, CL_MEM_WRITE_ONLY, ys.size_bytes(), nullptr);
        detail::buffer db(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bias.size_bytes(), const_cast<float*>(bias.data()));
        const cl_int has_bias = bias.empty() ? 0 : 1;
        const cl_ulong cols = w.cols, rows = w.rows, t0 = 0;
        const std::lock_guard lock(mutex_);
        if (blocks > 0) {
            cl_uint i = 0;
            set(quantize, i, dx.mem), set(quantize, i, blocks), set(quantize, i, dq.mem);
            const std::size_t global = (std::size_t(blocks) + 63) / 64 * 64, local = 64;
            detail::check(clEnqueueNDRangeKernel(queue, quantize, 1, nullptr, &global, &local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
        }
        cl_kernel k = dot_[std::size_t(w.s_->fmt)];
        cl_uint i = 0;
        set(k, i, w.s_->w), set(k, i, cols), set(k, i, dq.mem), set(k, i, rows), set(k, i, t0), set(k, i, db.mem), set(k, i, has_bias), set(k, i, dy.mem);
        const std::size_t global[2] = {w.rows * 64, n}, local[2] = {64, 1};
        detail::check(clEnqueueNDRangeKernel(queue, k, 2, nullptr, global, local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
        detail::check(clEnqueueReadBuffer(queue, dy.mem, CL_TRUE, 0, ys.size_bytes(), ys.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
    }

private:
    explicit opencl_quant_kernels(const opencl_context& c) : context_(&c) {}

    cl_mem make_buffer(std::span<const std::byte> bytes) const
    {
        cl_int code = CL_SUCCESS;
        const bool empty = bytes.empty();
        cl_mem m = clCreateBuffer(context_->native_context(), CL_MEM_READ_ONLY | (empty ? 0 : CL_MEM_COPY_HOST_PTR), empty ? 1 : bytes.size(),
                                  empty ? nullptr : const_cast<std::byte*>(bytes.data()), &code);
        detail::check(code, "clCreateBuffer");
        return m;
    }

    template <typename T>
    static void set(cl_kernel k, cl_uint& i, const T& v)
    {
        detail::check(clSetKernelArg(k, i++, sizeof(T), &v), "clSetKernelArg");
    }

    // The weight arguments every format's kernels start with (WARGS); a format without scales reads none, so its
    // weights stand in for them
    static cl_uint weight_args(cl_kernel k, const opencl_quant_matrix::storage& s, std::size_t cols)
    {
        cl_uint i = 0;
        const cl_ulong c = cols;
        set(k, i, s.w), set(k, i, s.sb ? s.sb : s.w), set(k, i, s.gs ? s.gs : s.w), set(k, i, s.mean), set(k, i, c);
        return i;
    }

    void tiled(cl_kernel k, const opencl_quant_matrix::storage& s, std::size_t cols, std::size_t m, std::size_t nn, std::size_t kk, cl_mem a,
               cl_mem bias, cl_int has_bias, float beta, cl_mem c) const
    {
        const cl_ulong mm = m, n = nn, kd = kk;
        const std::lock_guard lock(mutex_);
        cl_uint i = weight_args(k, s, cols);
        set(k, i, mm), set(k, i, n), set(k, i, kd), set(k, i, a), set(k, i, bias), set(k, i, has_bias), set(k, i, beta), set(k, i, c);
        const std::size_t global[2] = {(nn + 63) / 64 * 16, (m + 63) / 64 * 16}, local[2] = {16, 16};
        detail::check(clEnqueueNDRangeKernel(context_->native_queue(), k, 2, nullptr, global, local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    }

    // The rows of a product, after checking every size against the matrix
    static std::size_t rows_of(const opencl_quant_matrix& w, std::span<const float> in, std::size_t in_width, std::span<const float> out,
                               std::size_t out_width, std::span<const float> bias)
    {
        if (!w.s_)
            throw std::invalid_argument("snicholls::gpu: an opencl_quant_matrix without weights");
        const std::size_t n = in_width == 0 ? (out_width == 0 ? 0 : out.size() / out_width) : in.size() / in_width;
        if ((in_width != 0 && in.size() % in_width != 0) || out.size() != n * out_width || (!bias.empty() && bias.size() != w.rows))
            throw std::invalid_argument("snicholls::gpu: the activations' sizes do not fit the weights");
        return n;
    }

    const opencl_context* context_;
    cl_program program_ = nullptr;
    std::array<cl_kernel, detail::quant::format_count> mv_{}, mm_{}, mmt_{}, dot_{};
    cl_kernel quantize_ = nullptr, quantize_1_ = nullptr, quantize_k_ = nullptr;
    bool exact_q8_ = false;
    mutable std::mutex mutex_;
};

/// \brief `opencl_backend` with the quantised-weight kernels: core's operations over quantised weights run on the
/// device, everything else as `opencl_backend` does. Trivially copyable (two pointers).
struct opencl_quant_backend : opencl_backend {
    const opencl_quant_kernels* quant = nullptr;   ///< The kernels; must outlive the backend.
};

static_assert(linalg::blas_backend<opencl_quant_backend, float> && std::is_trivially_copyable_v<opencl_quant_backend>);

namespace detail::opencl_quant {
    inline const opencl_quant_kernels& kernels(const opencl_quant_backend& b)
    {
        if (b.quant == nullptr)
            throw std::logic_error("snicholls::gpu: a quantised product needs opencl_quant_backend::quant (opencl_quant_kernels)");
        return *b.quant;
    }
} // namespace detail::opencl_quant

/// \name core's quantised products on the device, for `opencl_quant_backend` (found by argument-dependent lookup)
/// @{

/// \brief `Y = X Wᵀ + b` over weights kept on the device. \param b The backend. \param w The weights. \param xs `n · cols`.
/// \param ys Written, `n · rows`. \param bias `rows`, or empty. \throws std::invalid_argument, opencl_exception.
inline void linear_sequence(const opencl_quant_backend& b, const opencl_quant_matrix& w, std::span<const float> xs, std::span<float> ys,
                            std::span<const float> bias = {})
{
    detail::opencl_quant::kernels(b).forward(w, xs, ys, bias);
}

/// \brief `dX += dY W` over weights kept on the device. \param b The backend. \param w The weights. \param dys `n · rows`.
/// \param dxs Added into, `n · cols`. \throws std::invalid_argument, opencl_exception.
inline void linear_sequence_backward_input(const opencl_quant_backend& b, const opencl_quant_matrix& w, std::span<const float> dys, std::span<float> dxs)
{
    detail::opencl_quant::kernels(b).backward_input(w, dys, dxs);
}

/// \brief `Y = X Wᵀ + b` with Q8_0 activations, over weights kept on the device. \param b The backend. \param w Q8_0,
/// Q4_0 or MXFP4 weights. \param xs `n · cols`. \param ys Written, `n · rows`. \param bias `rows`, or empty.
/// \throws std::invalid_argument, opencl_exception.
inline void linear_sequence_q8(const opencl_quant_backend& b, const opencl_quant_matrix& w, std::span<const float> xs, std::span<float> ys,
                               std::span<const float> bias = {})
{
    detail::opencl_quant::kernels(b).forward_q8(w, xs, ys, bias);
}

/// \brief `Y = X Wᵀ + b` over ggml blocks in host memory, copied for the call. \param b The backend. \param w The weights.
/// \param xs `n · cols`. \param ys Written, `n · rows`. \param bias `rows`, or empty. \throws std::invalid_argument, opencl_exception.
template <ml::quant_block Block>
void linear_sequence(const opencl_quant_backend& b, const ml::quantized_weights<Block>& w, std::span<const float> xs, std::span<float> ys,
                     std::span<const float> bias = {})
{
    linear_sequence(b, detail::opencl_quant::kernels(b).upload(w), xs, ys, bias);
}

/// \brief `Y = X Wᵀ + b` over dense minifloat weights in host memory. \param b The backend. \param w `bfloat16`,
/// `float16`, `float8_e4m3` or `float8_e5m2` weights. \param xs `n · cols`. \param ys Written, `n · rows`.
/// \param bias `rows`, or empty. \throws std::invalid_argument, opencl_exception.
template <typename S>
    requires detail::quant::weights<ml::dense_weights<S>>
void linear_sequence(const opencl_quant_backend& b, const ml::dense_weights<S>& w, std::span<const float> xs, std::span<float> ys,
                     std::span<const float> bias = {})
{
    linear_sequence(b, detail::opencl_quant::kernels(b).upload(w), xs, ys, bias);
}

/// \brief `Y = X Wᵀ + b` over any other ggml blocks in host memory (Q4_1, Q5_0, Q5_1, the K-quants, the ternary, where
/// core has them). \param b The backend. \param w The weights. \param xs `n · cols`. \param ys Written, `n · rows`.
/// \param bias `rows`, or empty. \throws std::invalid_argument, opencl_exception.
template <typename Block>
    requires detail::quant::weights<block_weights<Block>>
void linear_sequence(const opencl_quant_backend& b, const block_weights<Block>& w, std::span<const float> xs, std::span<float> ys,
                     std::span<const float> bias = {})
{
    linear_sequence(b, detail::opencl_quant::kernels(b).upload(w), xs, ys, bias);
}

/// \brief `Y = X Wᵀ + b` with 8-bit activations over other ggml blocks in host memory (Q8_0 for Q5_0, Q8_1 for Q4_1
/// and Q5_1, Q8_K for Q2_K to Q6_K, TQ1_0 and TQ2_0): core's `dot`.
/// \param b The backend. \param w The weights. \param xs `n · cols`. \param ys Written, `n · rows`. \param bias `rows`, or
/// empty. \throws std::invalid_argument, opencl_exception.
template <typename Block>
    requires detail::quant::weights<block_weights<Block>>
void linear_sequence_q8(const opencl_quant_backend& b, const block_weights<Block>& w, std::span<const float> xs, std::span<float> ys,
                        std::span<const float> bias = {})
{
    linear_sequence_q8(b, detail::opencl_quant::kernels(b).upload(w), xs, ys, bias);
}

/// \brief `dX += dY W` over any other ggml blocks in host memory. \param b The backend. \param w The weights.
/// \param dys `n · rows`. \param dxs Added into, `n · cols`. \throws std::invalid_argument, opencl_exception.
template <typename Block>
    requires detail::quant::weights<block_weights<Block>>
void linear_sequence_backward_input(const opencl_quant_backend& b, const block_weights<Block>& w, std::span<const float> dys, std::span<float> dxs)
{
    linear_sequence_backward_input(b, detail::opencl_quant::kernels(b).upload(w), dys, dxs);
}

/// \brief `Y = X Wᵀ + b` over NF4 weights in host memory: QLoRA's frozen base. \param b The backend. \param w The weights.
/// \param xs `n · cols`. \param ys Written, `n · rows`. \param bias `rows`, or empty. \throws std::invalid_argument, opencl_exception.
inline void linear_sequence(const opencl_quant_backend& b, const ml::nf4_weights& w, std::span<const float> xs, std::span<float> ys,
                            std::span<const float> bias = {})
{
    linear_sequence(b, detail::opencl_quant::kernels(b).upload(w), xs, ys, bias);
}

/// \brief `dX += dY W` over ggml blocks in host memory. \param b The backend. \param w The weights. \param dys `n · rows`.
/// \param dxs Added into, `n · cols`. \throws std::invalid_argument, opencl_exception.
template <ml::quant_block Block>
void linear_sequence_backward_input(const opencl_quant_backend& b, const ml::quantized_weights<Block>& w, std::span<const float> dys, std::span<float> dxs)
{
    linear_sequence_backward_input(b, detail::opencl_quant::kernels(b).upload(w), dys, dxs);
}

/// \brief `dX += dY W` over dense minifloat weights in host memory. \param b The backend. \param w The weights.
/// \param dys `n · rows`. \param dxs Added into, `n · cols`. \throws std::invalid_argument, opencl_exception.
template <typename S>
    requires detail::quant::weights<ml::dense_weights<S>>
void linear_sequence_backward_input(const opencl_quant_backend& b, const ml::dense_weights<S>& w, std::span<const float> dys, std::span<float> dxs)
{
    linear_sequence_backward_input(b, detail::opencl_quant::kernels(b).upload(w), dys, dxs);
}

/// \brief `dX += dY W` over NF4 weights in host memory: QLoRA's gradient through the frozen base. \param b The backend.
/// \param w The weights. \param dys `n · rows`. \param dxs Added into, `n · cols`. \throws std::invalid_argument, opencl_exception.
inline void linear_sequence_backward_input(const opencl_quant_backend& b, const ml::nf4_weights& w, std::span<const float> dys, std::span<float> dxs)
{
    linear_sequence_backward_input(b, detail::opencl_quant::kernels(b).upload(w), dys, dxs);
}

/// \brief `Y = X Wᵀ + b` with Q8_0 activations over ggml blocks in host memory. \param b The backend. \param w The weights.
/// \param xs `n · cols`. \param ys Written, `n · rows`. \param bias `rows`, or empty. \throws std::invalid_argument, opencl_exception.
template <ml::quant_block Block>
void linear_sequence_q8(const opencl_quant_backend& b, const ml::quantized_weights<Block>& w, std::span<const float> xs, std::span<float> ys,
                        std::span<const float> bias = {})
{
    linear_sequence_q8(b, detail::opencl_quant::kernels(b).upload(w), xs, ys, bias);
}

/// @}

} // namespace snicholls::gpu

#endif /* snicholls_gpu_opencl_quant_hpp */
