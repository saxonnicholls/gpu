//
//  cuda_quant.hpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::gpu::cuda_quant_kernels`: matrix products whose weights stay in core's quantised formats in an NVIDIA GPU's memory, dequantised in registers, forward and backward to the input, on the CUDA cores or the tensor cores.
/// \ingroup gpu
///
/// **Overview.** The CUDA counterpart of `opencl_quant.hpp`, from the same kernel source: weights in ggml's Q8_0, Q4_0
/// or MXFP4 blocks (`ml/quantize.hpp`), QLoRA's NF4 (`ml/nf4.hpp`) or dense bfloat16, float16, FP8 E4M3 or E5M2
/// (`ml/minifloat.hpp`), read in place from device memory and dequantised as the product needs each weight. Core's
/// operations are overloaded for `cuda_training_backend` (found by argument-dependent lookup), with every span device
/// memory, so core's `ml::lora_sequence` and `ml::lora_sequence_backward` run a QLoRA layer entirely on the GPU:
///
/// - `linear_sequence(b, w, xs, ys, bias)`: `Y = X Wᵀ + b`, float activations.
/// - `linear_sequence_backward_input(b, w, dys, dxs)`: `dX += dY W` (the base is frozen: no weight gradient).
/// - `linear_sequence_q8(b, w, xs, ys, bias)`: the activations quantised on the device to the 8-bit blocks core's dot
///   product pairs the weights with (Q8_0 for Q8_0, Q4_0, MXFP4, Q5_0; Q8_1 for Q4_1, Q5_1; Q8_K for Q2_K to Q6_K, TQ1_0 and TQ2_0), and
///   core's integer block dot products.
///
/// `w` is core's view (`ml::quantized_weights<Block>`, `ml::nf4_weights`, `ml::dense_weights<S>`), or a
/// `block_weights<Block>` of Q4_1, Q5_0, Q5_1, NVFP4, Q1_0, Q2_0, Q2_K to Q6_K, TQ1_0, TQ2_0, IQ4_NL or IQ4_XS where the pinned core has them, over device memory:
/// upload the blocks into a `device_buffer` and point the view at it. NF4's `scale_mean` is a host float, as in core.
///
/// \snippet cuda_quant_demo.cpp quant
///
/// **Relationships.** `cuda_ops.hpp` (`cuda_training_backend`, which carries `quant`), `cuda.hpp` (the context and
/// `device_buffer`), `opencl_quant.hpp` (the same kernels and tests on OpenCL), core's `ml/linear.hpp`, `ml/nf4.hpp`,
/// `ml/quantize.hpp`, `ml/lora.hpp`. Group: \ref gpu.
///
/// **Implementations.** As `opencl_quant.hpp`: matrix-vector kernels for at most `cuda_quant_kernels::matvec_max` rows
/// (a block of 64 threads per four outputs, a fixed reduction tree), the tiled kernel (64 x 64 tiles, each output summed in
/// index order), the Q8_0 path (core's terms added in block order). Under `cuda_math::bf16` the tiled products run
/// on the tensor cores instead: each 64 x 64 tile's weights dequantised and rounded to bfloat16 into shared memory as
/// they are loaded, the activations rounded likewise, `mma.sync m16n8k16` with float accumulators.
///
/// **Guarantees.** Every kernel is queued on the context's stream and none waits; nothing is atomic, so every result is
/// the same every run.
///
/// | operation | against | bound |
/// |---|---|---|
/// | tiled, `cuda_math::float32` or `tf32` | core's reference product over the dequantised weights | to the bit |
/// | tiled, `cuda_math::bf16` | the same | `|Δ| ≤ 2^-7 Σ_j |w_j x_j|` per output (both operands rounded to bfloat16) |
/// | matrix-vector | the same | `|Δ| ≤ k 2^-23 Σ_j |w_j x_j|` per output |
/// | Q8_0 activations | `ml::linear(w, prepare_input(w, x), y, bias)` | to the bit |
///
/// The Q8_0 path keeps its quantised activations in working space of the kernels object between calls, so one
/// `cuda_quant_kernels` serves one stream at a time, as `cuda_moe_kernels` does.
///
/// **Errors.** `create` returns a `cuda_failure` if the kernels do not compile (the log on stderr). A shape the format
/// does not allow or spans of the wrong size throw `std::invalid_argument`; Q8_0 activations over another format,
/// `std::invalid_argument`; a backend without `quant`, `std::logic_error`; a failed launch, `cuda_exception`.
///
/// **Performance.** `benchmarks/cuda_quant_benchmark.cpp`: decode bandwidth and prefill rate against dequantising to
/// float and multiplying with cuBLAS. Not yet measured: the numbers wait for a free GPU.
///
/// **Tests and demos.** `tests/cuda_quant_test.cpp` (the suite `opencl_quant_test.cpp` passes, `support/quant_suite.hpp`,
/// under each math; a QLoRA layer through core's `lora_sequence`), `demos/cuda_quant_demo.cpp`.
///
/// **Dependencies.** `cuda_ops.hpp`, `detail/quant_source.hpp`, `<nvrtc.h>`, core's `ml/linear.hpp`, `ml/nf4.hpp`,
/// `ml/quantize.hpp`, `ml/minifloat.hpp`.
///
/// **Known issues.** The integer path uses int32 multiply-adds, not `dp4a` (the blocks are 2-byte aligned) or the
/// integer tensor cores; the matrix-vector kernels read bytes. Both are the next steps for speed.
///
/// **Status.** Experimental, new in 0.9.0.

#ifndef snicholls_gpu_cuda_quant_hpp
#define snicholls_gpu_cuda_quant_hpp

#include <snicholls/gpu/cuda_ops.hpp>
#include <snicholls/gpu/detail/quant_source.hpp>

#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/nf4.hpp>
#include <snicholls/core/ml/quantize.hpp>

#include <nvrtc.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace snicholls::gpu {

namespace detail::cuda_quant {

    // The tensor-core kernels, CUDA only: helpers once, then per format with FMT and GET defined
    inline constexpr const char* mma_helpers = R"CU(
// float to bfloat16, to nearest even (ml/minifloat.hpp to_bfloat16)
DEVICE ushort f2bf(float x)
{
    const uint u = AS_UINT(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (ushort)((u >> 16) | 0x40u);
    return (ushort)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
DEVICE void mma_bf16(float* c, const uint* a, uint b0, uint b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#define TC_BM 64
#define TC_BN 64
#define TC_BK 32
#define TC_LD 40
)CU";

    // C (m x nn) = A (m x kk) B + beta C + bias, B the weights (BW 0: B(p, c) = W(c, p), the forward; 1: W(p, c)). A block
    // of 4 warps makes a 64 x 64 tile, each warp 32 x 32 (2 x 4 fragments of m16n8k16). Both tiles are stored in shared
    // memory as bfloat16 rows along k (A by token, B by output), so a lane's fragment registers are 32-bit loads of two
    // consecutive k: a0 (row g, k 2q..2q+1), a1 (row g + 8), a2 (k + 8), a3 (row g + 8, k + 8); b0 (column g, k 2q..2q+1),
    // b1 (k + 8); c0..c3 (row g, columns 2q, 2q + 1; row g + 8).
    inline constexpr const char* mma_source = R"CU(
#define TCORE(NAME, BW)                                                                                                \
KERNEL GROUP1(128) void NAME(WARGS, u64 m, u64 nn, u64 kk, const float* a, const float* bias, int has_bias, float beta, float* cm) \
{                                                                                                                      \
    __shared__ __align__(16) ushort ta[TC_BM * TC_LD];                                                                 \
    __shared__ __align__(16) ushort tb[TC_BN * TC_LD];                                                                 \
    const int tid = LX, warp = tid / 32, lane = tid % 32, g = lane >> 2, q = lane & 3;                                 \
    const int wm = (warp / 2) * 32, wn = (warp % 2) * 32;                                                              \
    const u64 n0 = GX * TC_BN, m0 = GY * TC_BM;                                                                        \
    float acc[2][4][4];                                                                                                \
    for (int i = 0; i < 2; ++i)                                                                                        \
        for (int j = 0; j < 4; ++j)                                                                                    \
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;                                                           \
    for (u64 k0 = 0; k0 < kk; k0 += TC_BK) {                                                                           \
        for (int i = 0; i < 16; ++i) {                                                                                 \
            const int e = tid + i * 128;                                                                               \
            const int ar = e / 32, ak = e % 32;                                                                        \
            const u64 gr = m0 + (u64)ar, gk = k0 + (u64)ak;                                                            \
            ta[ar * TC_LD + ak] = f2bf(gr < m && gk < kk ? a[gr * kk + gk] : 0.0f);                                    \
            const int bc = BW ? e % 64 : e / 32, bk = BW ? e / 64 : e % 32;                                            \
            const u64 gc = n0 + (u64)bc, gq = k0 + (u64)bk;                                                            \
            tb[bc * TC_LD + bk] = f2bf(gc < nn && gq < kk ? (BW ? GET(gq, gc) : GET(gc, gq)) : 0.0f);                  \
        }                                                                                                              \
        BARRIER();                                                                                                     \
        for (int s = 0; s < 2; ++s) {                                                                                  \
            const int kb = s * 16 + 2 * q;                                                                             \
            uint af[2][4], bf[4][2];                                                                                   \
            for (int i = 0; i < 2; ++i) {                                                                              \
                const int r = wm + 16 * i + g;                                                                         \
                af[i][0] = *(const uint*)&ta[r * TC_LD + kb];                                                          \
                af[i][1] = *(const uint*)&ta[(r + 8) * TC_LD + kb];                                                    \
                af[i][2] = *(const uint*)&ta[r * TC_LD + kb + 8];                                                      \
                af[i][3] = *(const uint*)&ta[(r + 8) * TC_LD + kb + 8];                                                \
            }                                                                                                          \
            for (int j = 0; j < 4; ++j) {                                                                              \
                const int c = wn + 8 * j + g;                                                                          \
                bf[j][0] = *(const uint*)&tb[c * TC_LD + kb];                                                          \
                bf[j][1] = *(const uint*)&tb[c * TC_LD + kb + 8];                                                      \
            }                                                                                                          \
            for (int i = 0; i < 2; ++i)                                                                                \
                for (int j = 0; j < 4; ++j) mma_bf16(acc[i][j], af[i], bf[j][0], bf[j][1]);                            \
        }                                                                                                              \
        BARRIER();                                                                                                     \
    }                                                                                                                  \
    for (int i = 0; i < 2; ++i)                                                                                        \
        for (int j = 0; j < 4; ++j)                                                                                    \
            for (int e = 0; e < 4; ++e) {                                                                              \
                const u64 r = m0 + (u64)(wm + 16 * i + g + 8 * (e / 2)), c = n0 + (u64)(wn + 8 * j + 2 * q + (e % 2)); \
                if (r < m && c < nn) {                                                                                 \
                    float* out = cm + r * nn + c;                                                                      \
                    float v = acc[i][j][e];                                                                            \
                    if (beta != 0.0f) { const float old = beta * *out; v = v + old; }                                  \
                    if (has_bias) v = v + bias[c];                                                                     \
                    *out = v;                                                                                          \
                }                                                                                                      \
            }                                                                                                          \
}
TCORE(CAT(qmma, FMT), 0)
TCORE(CAT(qmmat, FMT), 1)
#undef TCORE
)CU";

    inline std::string program()
    {
        std::string extra = mma_helpers;
        for (const auto name : quant::names) {
            const std::string n(name);
            extra += "#define FMT " + n + "\n#define GET(r, c) get_" + n + "(WPASS, r, c)\n" + mma_source + "#undef GET\n#undef FMT\n";
        }
        return quant::program(quant::cuda_prelude, extra.c_str());
    }

} // namespace detail::cuda_quant

/// \brief The quantised-weight kernels, compiled (NVRTC) for a `cuda_context`'s device.
///
/// Create once and set `cuda_training_backend::quant` to it. Neither copyable nor movable: backends point at it.
class cuda_quant_kernels {
public:
    /// \brief Compiles the kernels for the context's device and loads them.
    /// \param context The device; must outlive the kernels.
    /// \return The kernels, or why they did not compile (the log on stderr).
    [[nodiscard]] static result<std::unique_ptr<cuda_quant_kernels>, cuda_failure> create(const cuda_context& context)
    {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, context.device());
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, context.device());
        const std::string source = detail::cuda_quant::program();
        nvrtcProgram prog = nullptr;
        if (const nvrtcResult r = nvrtcCreateProgram(&prog, source.c_str(), "snicholls_gpu_quant.cu", 0, nullptr, nullptr); r != NVRTC_SUCCESS) {
            std::fprintf(stderr, "snicholls::gpu::cuda_quant_kernels: nvrtcCreateProgram: %s\n", nvrtcGetErrorString(r));
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(r)});
        }
        // Division and square root correctly rounded (NVRTC's default, stated): the Q8_0 scales are core's bits
        const std::string arch = "--gpu-architecture=sm_" + std::to_string(major * 10 + minor);
        const char* options[] = {arch.c_str(), "--fmad=false", "--prec-div=true", "--ftz=false", "--std=c++17"};
        const nvrtcResult compiled = nvrtcCompileProgram(prog, 5, options);
        if (compiled != NVRTC_SUCCESS) {
            std::size_t size = 0;
            nvrtcGetProgramLogSize(prog, &size);
            std::string log(size, '\0');
            nvrtcGetProgramLog(prog, log.data());
            std::fprintf(stderr, "snicholls::gpu::cuda_quant_kernels: %s\n", log.c_str());
            nvrtcDestroyProgram(&prog);
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(compiled)});
        }
        std::size_t size = 0;
        nvrtcGetCUBINSize(prog, &size);
        std::string cubin(size, '\0');
        nvrtcGetCUBIN(prog, cubin.data());
        nvrtcDestroyProgram(&prog);
        std::unique_ptr<cuda_quant_kernels> k(new cuda_quant_kernels(context));
        if (const cudaError_t e = cudaLibraryLoadData(&k->library_, cubin.data(), nullptr, nullptr, 0, nullptr, nullptr, 0); e != cudaSuccess) {
            std::fprintf(stderr, "snicholls::gpu::cuda_quant_kernels: cudaLibraryLoadData: %s\n", cudaGetErrorString(e));
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(e)});
        }
        std::vector<std::pair<std::string, cudaKernel_t*>> all{{"quantize_q8", &k->quantize_}, {"quantize_q8_1", &k->quantize_1_}, {"quantize_q8k", &k->quantize_k_}};
        for (std::size_t f = 0; f < detail::quant::format_count; ++f) {
            const std::string n(detail::quant::names[f]);
            all.emplace_back("qmv_" + n, &k->mv_[f]);
            all.emplace_back("qmm_" + n, &k->mm_[f]);
            all.emplace_back("qmmt_" + n, &k->mmt_[f]);
            all.emplace_back("qmma_" + n, &k->mma_[f]);
            all.emplace_back("qmmat_" + n, &k->mmat_[f]);
            if (detail::quant::has_q8_0_dot(detail::quant::format(f)) || detail::quant::has_q8_1_dot(detail::quant::format(f)))
                all.emplace_back("qdot_" + n, &k->dot_[f]);
            if (detail::quant::has_q8_k_dot(detail::quant::format(f)))
                all.emplace_back("qkdot_" + n, &k->dot_[f]);
        }
        for (const auto& [name, slot] : all)
            if (const cudaError_t e = cudaLibraryGetKernel(slot, k->library_, name.c_str()); e != cudaSuccess) {
                std::fprintf(stderr, "snicholls::gpu::cuda_quant_kernels: cudaLibraryGetKernel(%s): %s\n", name.c_str(), cudaGetErrorString(e));
                return err(cuda_failure{cuda_error::call_failed, static_cast<int>(e)});
            }
        return ok(std::move(k));
    }

    /// \brief Not copyable.
    cuda_quant_kernels(const cuda_quant_kernels&) = delete;
    /// \brief Not assignable. \return Nothing; deleted.
    cuda_quant_kernels& operator=(const cuda_quant_kernels&) = delete;
    /// \brief Unloads the kernels.
    ~cuda_quant_kernels()
    {
        if (library_ != nullptr)
            (void)cudaLibraryUnload(library_);
    }

    /// \brief The most rows of activations a forward product runs as matrix-vector products (8 a launch); more take the
    /// tiled kernel. 0 sends every product to the tiled kernel.
    std::size_t matvec_max = 8;

    /// \brief The context. \return It.
    [[nodiscard]] const cuda_context& context() const noexcept { return *context_; }

    /// \brief `Y = X Wᵀ + b` over device memory. \param math The backend's math (bf16: the tiled products on the tensor
    /// cores). \param w The weights. \param xs `n · cols`. \param ys Written, `n · rows`. \param bias `rows`, or empty.
    /// \throws std::invalid_argument if the sizes do not fit; cuda_exception if a launch fails.
    void forward(cuda_math math, const detail::quant::parts& w, std::span<const float> xs, std::span<float> ys, std::span<const float> bias) const
    {
        const std::size_t n = rows_of(w, xs, w.cols, ys, w.rows, bias);
        if (n == 0 || w.rows == 0)
            return;
        const std::size_t f = std::size_t(w.fmt);
        const float* bp = bias.empty() ? nullptr : bias.data();
        int has_bias = bias.empty() ? 0 : 1;
        if (n <= matvec_max) {
            for (std::size_t t0 = 0; t0 < n; t0 += detail::quant::mv_tokens) {
                u64 count = std::min(detail::quant::mv_tokens, n - t0), rows = w.rows;
                const float* x = xs.data() + t0 * w.cols;
                float* y = ys.data() + t0 * w.rows;
                auto a = weight_args(w);
                void* args[] = {&a.w, &a.sb, &a.gs, &a.mean, &a.cols, &x, &count, &rows, &bp, &has_bias, &y};
                launch(mv_[f], dim3(unsigned((w.rows + detail::quant::mv_rows - 1) / detail::quant::mv_rows)), dim3(64), args);
            }
            return;
        }
        tiled(math == cuda_math::bf16 ? mma_[f] : mm_[f], math == cuda_math::bf16, w, n, w.rows, w.cols, xs.data(), bp, has_bias, 0.0f, ys.data());
    }

    /// \brief `dX += dY W` over device memory. \param math The backend's math. \param w The weights. \param dys `n · rows`.
    /// \param dxs Added into, `n · cols`. \throws std::invalid_argument if the sizes do not fit; cuda_exception if a launch fails.
    void backward_input(cuda_math math, const detail::quant::parts& w, std::span<const float> dys, std::span<float> dxs) const
    {
        const std::size_t n = rows_of(w, dys, w.rows, dxs, w.cols, {});
        if (n == 0 || w.cols == 0)
            return;
        const std::size_t f = std::size_t(w.fmt);
        tiled(math == cuda_math::bf16 ? mmat_[f] : mmt_[f], math == cuda_math::bf16, w, n, w.cols, w.rows, dys.data(), nullptr, 0, 1.0f, dxs.data());
    }

    /// \brief `Y = X Wᵀ + b` with `X` quantised as core's dot product pairs the weights (Q8_0, Q8_1 or Q8_K) and core's
    /// block dot products, over device memory.
    /// \param w Weights of one of those formats. \param xs `n · cols`. \param ys Written, `n · rows`. \param bias `rows`, or empty.
    /// \throws std::invalid_argument for another format or sizes that do not fit; cuda_exception if a launch fails.
    void forward_q8(const detail::quant::parts& w, std::span<const float> xs, std::span<float> ys, std::span<const float> bias) const
    {
        const std::size_t n = rows_of(w, xs, w.cols, ys, w.rows, bias);
        if (!detail::quant::has_q8_dot(w.fmt))
            throw std::invalid_argument("snicholls::gpu: 8-bit activations need weights of a format with core's integer dot product");
        if (n == 0 || w.rows == 0)
            return;
        const bool k_quant = detail::quant::has_q8_k_dot(w.fmt), q8_1 = detail::quant::has_q8_1_dot(w.fmt);
        const std::size_t block_bytes = k_quant ? detail::quant::q8k_bytes : q8_1 ? detail::quant::q8_1_bytes : detail::quant::q8_bytes;
        u64 blocks = n * w.cols / (k_quant ? 256 : 32), cols = w.cols, rows = w.rows;
        if (!q8_ || q8_->size() < blocks * block_bytes)
            q8_ = std::make_unique<device_buffer<std::uint8_t>>(*context_, std::size_t(blocks) * block_bytes);
        const float* x = xs.data();
        std::uint8_t* q = q8_->span().data();
        if (blocks > 0) {
            void* args[] = {&x, &blocks, &q};
            launch(k_quant ? quantize_k_ : q8_1 ? quantize_1_ : quantize_, dim3(unsigned((blocks + 63) / 64)), dim3(64), args);
        }
        const auto* wp = reinterpret_cast<const std::uint8_t*>(w.weights.data());
        const float* bp = bias.empty() ? nullptr : bias.data();
        int has_bias = bias.empty() ? 0 : 1;
        float* y = ys.data();
        for (std::size_t t = 0; t < n; t += 65535) {   // the grid's y is at most 65535
            u64 t0 = t;
            void* args[] = {&wp, &cols, &q, &rows, &t0, &bp, &has_bias, &y};
            launch(dot_[std::size_t(w.fmt)], dim3(unsigned(w.rows), unsigned(std::min<std::size_t>(65535, n - t))), dim3(64), args);
        }
    }

private:
    using u64 = unsigned long long;
    explicit cuda_quant_kernels(const cuda_context& c) : context_(&c) {}

    struct wargs {
        const std::uint8_t *w, *sb, *gs;
        float mean;
        u64 cols;
    };
    static wargs weight_args(const detail::quant::parts& p)
    {
        const auto* w = reinterpret_cast<const std::uint8_t*>(p.weights.data());
        const auto* sb = p.block_scales.empty() ? w : reinterpret_cast<const std::uint8_t*>(p.block_scales.data());
        const auto* gs = p.group_scales.empty() ? w : reinterpret_cast<const std::uint8_t*>(p.group_scales.data());
        return {w, sb, gs, p.mean, p.cols};
    }

    void launch(cudaKernel_t k, dim3 grid, dim3 block, void** args) const
    {
        if (grid.x == 0 || grid.y == 0)
            return;
        if (const cudaError_t e = cudaLaunchKernel(reinterpret_cast<const void*>(k), grid, block, args, 0, context_->stream()); e != cudaSuccess)
            throw cuda_exception(std::string("cudaLaunchKernel(quant) [") + cudaGetErrorString(e) + "]", static_cast<int>(e));
    }

    void tiled(cudaKernel_t k, bool tensor, const detail::quant::parts& w, std::size_t m, std::size_t nn, std::size_t kk, const float* a,
               const float* bias, int has_bias, float beta, float* c) const
    {
        auto wa = weight_args(w);
        u64 mm = m, n = nn, kd = kk;
        void* args[] = {&wa.w, &wa.sb, &wa.gs, &wa.mean, &wa.cols, &mm, &n, &kd, &a, &bias, &has_bias, &beta, &c};
        for (std::size_t done = 0; done < m;) {   // the grid's y is at most 65535 tiles of 64 rows
            const std::size_t part = std::min<std::size_t>(m - done, std::size_t(65535) * 64);
            mm = part;
            const float* ap = a + done * kk;
            float* cp = c + done * nn;
            args[8] = &ap;
            args[12] = &cp;
            launch(k, dim3(unsigned((nn + 63) / 64), unsigned((part + 63) / 64)), tensor ? dim3(128) : dim3(16, 16), args);
            done += part;
        }
    }

    static std::size_t rows_of(const detail::quant::parts& w, std::span<const float> in, std::size_t in_width, std::span<const float> out,
                               std::size_t out_width, std::span<const float> bias)
    {
        if (!detail::quant::valid(w))
            throw std::invalid_argument("snicholls::gpu: the weights' size does not match their shape and format");
        const std::size_t n = in_width == 0 ? (out_width == 0 ? 0 : out.size() / out_width) : in.size() / in_width;
        if ((in_width != 0 && in.size() % in_width != 0) || out.size() != n * out_width || (!bias.empty() && bias.size() != w.rows))
            throw std::invalid_argument("snicholls::gpu: the activations' sizes do not fit the weights");
        return n;
    }

    const cuda_context* context_;
    cudaLibrary_t library_ = nullptr;
    std::array<cudaKernel_t, detail::quant::format_count> mv_{}, mm_{}, mmt_{}, mma_{}, mmat_{}, dot_{};
    cudaKernel_t quantize_ = nullptr, quantize_1_ = nullptr, quantize_k_ = nullptr;
    mutable std::unique_ptr<device_buffer<std::uint8_t>> q8_;
};

namespace detail::cuda_quant {
    inline const cuda_quant_kernels& kernels(const cuda_training_backend& b)
    {
        if (b.quant == nullptr)
            throw std::logic_error("snicholls::gpu: a quantised product needs cuda_training_backend::quant (cuda_quant_kernels)");
        return *b.quant;
    }
} // namespace detail::cuda_quant

/// \name core's quantised products on the device, for `cuda_training_backend` (found by argument-dependent lookup)
/// Every span is device memory.
/// @{

/// \brief `Y = X Wᵀ + b` over ggml blocks. \param b The backend. \param w The weights. \param xs `n · cols`.
/// \param ys Written, `n · rows`. \param bias `rows`, or empty. \throws std::invalid_argument, cuda_exception.
template <ml::quant_block Block>
void linear_sequence(const cuda_training_backend& b, const ml::quantized_weights<Block>& w, std::span<const float> xs, std::span<float> ys,
                     std::span<const float> bias = {})
{
    detail::cuda_quant::kernels(b).forward(b.math, detail::quant::traits<ml::quantized_weights<Block>>::of(w), xs, ys, bias);
}

/// \brief `Y = X Wᵀ + b` over dense minifloat weights. \param b The backend. \param w `bfloat16`, `float16`,
/// `float8_e4m3` or `float8_e5m2` weights. \param xs `n · cols`. \param ys Written, `n · rows`. \param bias `rows`, or
/// empty. \throws std::invalid_argument, cuda_exception.
template <typename S>
    requires detail::quant::weights<ml::dense_weights<S>>
void linear_sequence(const cuda_training_backend& b, const ml::dense_weights<S>& w, std::span<const float> xs, std::span<float> ys,
                     std::span<const float> bias = {})
{
    detail::cuda_quant::kernels(b).forward(b.math, detail::quant::traits<ml::dense_weights<S>>::of(w), xs, ys, bias);
}

/// \brief `Y = X Wᵀ + b` over any other ggml blocks (Q4_1, Q5_0, Q5_1, the K-quants, the ternary, where core has them).
/// \param b The backend. \param w The weights. \param xs `n · cols`. \param ys Written, `n · rows`. \param bias `rows`,
/// or empty. \throws std::invalid_argument, cuda_exception.
template <typename Block>
    requires detail::quant::weights<block_weights<Block>>
void linear_sequence(const cuda_training_backend& b, const block_weights<Block>& w, std::span<const float> xs, std::span<float> ys,
                     std::span<const float> bias = {})
{
    detail::cuda_quant::kernels(b).forward(b.math, detail::quant::traits<block_weights<Block>>::of(w), xs, ys, bias);
}

/// \brief `Y = X Wᵀ + b` with 8-bit activations over other ggml blocks (Q8_0 for Q5_0, Q8_1 for Q4_1 and Q5_1, Q8_K
/// for Q2_K to Q6_K): core's `dot`. \param b The backend.
/// \param w The weights. \param xs `n · cols`. \param ys Written, `n · rows`. \param bias `rows`, or empty.
/// \throws std::invalid_argument, cuda_exception.
template <typename Block>
    requires detail::quant::weights<block_weights<Block>>
void linear_sequence_q8(const cuda_training_backend& b, const block_weights<Block>& w, std::span<const float> xs, std::span<float> ys,
                        std::span<const float> bias = {})
{
    detail::cuda_quant::kernels(b).forward_q8(detail::quant::traits<block_weights<Block>>::of(w), xs, ys, bias);
}

/// \brief `dX += dY W` over any other ggml blocks. \param b The backend. \param w The weights. \param dys `n · rows`.
/// \param dxs Added into, `n · cols`. \throws std::invalid_argument, cuda_exception.
template <typename Block>
    requires detail::quant::weights<block_weights<Block>>
void linear_sequence_backward_input(const cuda_training_backend& b, const block_weights<Block>& w, std::span<const float> dys, std::span<float> dxs)
{
    detail::cuda_quant::kernels(b).backward_input(b.math, detail::quant::traits<block_weights<Block>>::of(w), dys, dxs);
}

/// \brief `Y = X Wᵀ + b` over NF4 weights: QLoRA's frozen base. \param b The backend. \param w The weights.
/// \param xs `n · cols`. \param ys Written, `n · rows`. \param bias `rows`, or empty. \throws std::invalid_argument, cuda_exception.
inline void linear_sequence(const cuda_training_backend& b, const ml::nf4_weights& w, std::span<const float> xs, std::span<float> ys,
                            std::span<const float> bias = {})
{
    detail::cuda_quant::kernels(b).forward(b.math, detail::quant::traits<ml::nf4_weights>::of(w), xs, ys, bias);
}

/// \brief `dX += dY W` over ggml blocks. \param b The backend. \param w The weights. \param dys `n · rows`.
/// \param dxs Added into, `n · cols`. \throws std::invalid_argument, cuda_exception.
template <ml::quant_block Block>
void linear_sequence_backward_input(const cuda_training_backend& b, const ml::quantized_weights<Block>& w, std::span<const float> dys, std::span<float> dxs)
{
    detail::cuda_quant::kernels(b).backward_input(b.math, detail::quant::traits<ml::quantized_weights<Block>>::of(w), dys, dxs);
}

/// \brief `dX += dY W` over dense minifloat weights. \param b The backend. \param w The weights. \param dys `n · rows`.
/// \param dxs Added into, `n · cols`. \throws std::invalid_argument, cuda_exception.
template <typename S>
    requires detail::quant::weights<ml::dense_weights<S>>
void linear_sequence_backward_input(const cuda_training_backend& b, const ml::dense_weights<S>& w, std::span<const float> dys, std::span<float> dxs)
{
    detail::cuda_quant::kernels(b).backward_input(b.math, detail::quant::traits<ml::dense_weights<S>>::of(w), dys, dxs);
}

/// \brief `dX += dY W` over NF4 weights: QLoRA's gradient through the frozen base. \param b The backend. \param w The
/// weights. \param dys `n · rows`. \param dxs Added into, `n · cols`. \throws std::invalid_argument, cuda_exception.
inline void linear_sequence_backward_input(const cuda_training_backend& b, const ml::nf4_weights& w, std::span<const float> dys, std::span<float> dxs)
{
    detail::cuda_quant::kernels(b).backward_input(b.math, detail::quant::traits<ml::nf4_weights>::of(w), dys, dxs);
}

/// \brief `Y = X Wᵀ + b` with Q8_0 activations over ggml blocks. \param b The backend. \param w The weights.
/// \param xs `n · cols`. \param ys Written, `n · rows`. \param bias `rows`, or empty. \throws std::invalid_argument, cuda_exception.
template <ml::quant_block Block>
void linear_sequence_q8(const cuda_training_backend& b, const ml::quantized_weights<Block>& w, std::span<const float> xs, std::span<float> ys,
                        std::span<const float> bias = {})
{
    detail::cuda_quant::kernels(b).forward_q8(detail::quant::traits<ml::quantized_weights<Block>>::of(w), xs, ys, bias);
}

/// @}

} // namespace snicholls::gpu

#endif /* snicholls_gpu_cuda_quant_hpp */
