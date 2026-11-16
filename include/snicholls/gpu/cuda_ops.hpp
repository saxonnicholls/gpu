//
//  cuda_ops.hpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::gpu::cuda_training_backend`: snicholls-core's batched training passes on an NVIDIA GPU with every step on the device, so a training step's data stays there.
/// \ingroup gpu
///
/// **Overview.** Core's batched passes (`ml::block_forward_sequence`, `ml::block_backward` and the attention, MLP and
/// linear passes beneath them) make every step a call with their backend (core's TRAINING.md T9): the products, and
/// `fill_rows`, `copy_rows`, `add_rows`, `add_bias_rows`, `sum_rows`, `mlp_activate_rows` and its backward,
/// `block_norm_rows` and its backward, and the four attention steps. This header overloads each for
/// `cuda_training_backend`, so with the weights, activations and gradients in `device_buffer`s the whole step runs on
/// the GPU: the products by cuBLAS, the other steps by kernels compiled at run time (NVRTC, to the device's machine code) when
/// `cuda_kernels` is created.
///
/// - `cuda_kernels::create(context)`: compiles the kernels for the context's device; `result` with a `cuda_failure`.
/// - `cuda_training_backend{{&context}, &kernels}`: the backend, two pointers.
///
/// \snippet cuda_ops_demo.cpp cuda_ops
///
/// **Relationships.** `cuda.hpp` (the context, the products, `device_buffer`), core's `ml/transformer_block.hpp`,
/// `ml/attention_layer.hpp`, `ml/mlp.hpp`, `ml/linear.hpp` (the operations it overloads and the reference it is checked
/// against). Group: \ref gpu.
///
/// **Implementations.** The elementwise steps one thread per element; the norms one thread per row (each row's sums in
/// order, as core's), the weights' and biases' gradients one thread per column summing the rows in order, so the
/// result does not depend on scheduling; the activations with core's formulas in float. The attention steps are staged
/// in this version: copied to the host, run by core's own code, copied back (the attention kernels are next).
///
/// **Guarantees.** Every step is enqueued on the context's stream, in the order the pass makes them, after the
/// previous product; the results agree with core's reference to rounding (the device's `expf`, `tanhf`, `sqrtf`
/// differ from the host's in the last bits). One training step at a time per context.
///
/// **Errors.** `cuda_kernels::create` returns a `cuda_failure` (with the compiler's log on stderr) if the kernels do not
/// compile. A launch or copy that fails throws `cuda_exception`.
///
/// **Performance.** `benchmarks/cuda_ops_benchmark.cpp`: a GPT-2-small layer's training step with its data on the
/// device, against the same step staged; see `BENCHMARKS.md`.
///
/// **Tests and demos.** `tests/cuda_ops_test.cpp` (each step against core's, and whole training steps of GPT-2-,
/// Llama- and nanochat-shaped blocks against the reference), `demos/cuda_ops_demo.cpp`.
///
/// **Dependencies.** `cuda.hpp`, `<nvrtc.h>`, core's `ml/transformer_block.hpp`. Link `CUDA::nvrtc` too.
///
/// **Known issues.** Float only. The norms' kernels use one thread per row: correct, not yet fast for long rows.
///
/// **Status.** Experimental, new in 0.2.0.

#ifndef snicholls_gpu_cuda_ops_hpp
#define snicholls_gpu_cuda_ops_hpp

#include <snicholls/gpu/cuda.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/transformer_block.hpp>

#include <nvrtc.h>

#include <cstddef>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls::gpu {

namespace detail::cuda_ops {

    inline constexpr const char* source = R"CU(
typedef unsigned long long u64;
__device__ float sigmoid_(float v) { if (v >= 0.0f) return 1.0f / (1.0f + expf(-v)); const float e = expf(v); return e / (1.0f + e); }
__device__ float gelu_tanh_2u_(float v) { const float k = 2.0f / 2.5066282746310002f; return 2.0f * k * (v + 0.044715f * v * v * v); }
__device__ float act_(float v, int kind)
{
    if (kind == 0) return v * (0.5f * erfcf(-v * 0.70710678118654752f));
    if (kind == 1) return v * sigmoid_(gelu_tanh_2u_(v));
    const float r = v > 0.0f ? v : 0.0f; return r * r;
}
__device__ float act_derivative_(float v, int kind)
{
    if (kind == 0) { const float cdf = 0.5f * erfcf(-v * 0.70710678118654752f); const float pdf = expf(-0.5f * v * v) / 2.5066282746310002f; return cdf + v * pdf; }
    if (kind == 1) { const float s = sigmoid_(gelu_tanh_2u_(v)); const float du = 2.0f / 2.5066282746310002f * (1.0f + 3.0f * 0.044715f * v * v); return s + 2.0f * v * s * (1.0f - s) * du; }
    return 2.0f * (v > 0.0f ? v : 0.0f);
}
extern "C" {
__global__ void fill(float* x, u64 n, float v) { const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i < n) x[i] = v; }
__global__ void copy(const float* a, float* x, u64 n) { const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i < n) x[i] = a[i]; }
__global__ void add(const float* a, float* x, u64 n) { const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i < n) x[i] = a[i] + x[i]; }
__global__ void add_bias_rows(float* y, const float* b, u64 rows, u64 r)
{ const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i < rows * r) y[i] += b[i % r]; }
__global__ void sum_rows(const float* y, float* into, u64 rows, u64 r)
{ const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= r) return; float s = into[i]; for (u64 t = 0; t < rows; ++t) s += y[t * r + i]; into[i] = s; }
__global__ void activate(const float* up, const float* gate, float* hidden, u64 m, int kind)
{
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= m) return;
    if (kind == 3) { const float g = gate[i]; hidden[i] = g * sigmoid_(g) * up[i]; } else hidden[i] = act_(up[i], kind);
}
__global__ void activate_backward(const float* up, const float* gate, float* dh, float* dgate, u64 m, int kind)
{
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= m) return;
    if (kind == 3) {
        const float g = gate[i], s = sigmoid_(g), silu = g * s, dsilu = s * (1.0f + g * (1.0f - s));
        dgate[i] = dh[i] * up[i] * dsilu; dh[i] = dh[i] * silu;
    } else dh[i] = dh[i] * act_derivative_(up[i], kind);
}
// One thread per row: core's layer norm (layer != 0) or RMS norm, each sum in index order; stats[2t], stats[2t+1] = mean, rstd
__global__ void norm_rows(const float* x, const float* w, const float* b, float* out, float* stats, u64 rows, u64 d, int layer, float eps)
{
    const u64 t = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (t >= rows) return;
    const float* in = x + t * d; float* y = out + t * d;
    float mean = 0.0f, rstd;
    if (layer) {
        for (u64 i = 0; i < d; ++i) mean += in[i];
        mean = mean / (float)d;
        float var = 0.0f; for (u64 i = 0; i < d; ++i) var += (in[i] - mean) * (in[i] - mean);
        var = var / (float)d; rstd = 1.0f / sqrtf(var + eps);
        for (u64 i = 0; i < d; ++i) y[i] = (in[i] - mean) * rstd * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
    } else {
        float ms = 0.0f; for (u64 i = 0; i < d; ++i) ms += in[i] * in[i];
        ms = ms / (float)d; rstd = 1.0f / sqrtf(ms + eps);
        for (u64 i = 0; i < d; ++i) y[i] = in[i] * rstd * (w ? w[i] : 1.0f);
    }
    if (stats) { stats[2 * t] = mean; stats[2 * t + 1] = rstd; }
}
// One thread per row: dxs_t += norm'(x_t)^T dh_t, from the statistics norm_rows left
__global__ void norm_rows_backward_add(const float* dh, const float* x, const float* w, const float* stats, float* dxs, u64 rows, u64 d, int layer)
{
    const u64 t = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (t >= rows) return;
    const float* g_ = dh + t * d; const float* in = x + t * d; float* gx = dxs + t * d;
    const float mean = stats[2 * t], rstd = stats[2 * t + 1];
    if (layer) {
        float mean_g = 0.0f, mean_gn = 0.0f;
        for (u64 i = 0; i < d; ++i) { const float norm = (in[i] - mean) * rstd; const float g = (w ? w[i] : 1.0f) * g_[i]; mean_g += g; mean_gn += g * norm; }
        mean_g /= (float)d; mean_gn /= (float)d;
        for (u64 i = 0; i < d; ++i) { const float norm = (in[i] - mean) * rstd; const float g = (w ? w[i] : 1.0f) * g_[i]; gx[i] += rstd * (g - mean_g - norm * mean_gn); }
    } else {
        float mean_gn = 0.0f;
        for (u64 i = 0; i < d; ++i) mean_gn += (w ? w[i] : 1.0f) * g_[i] * in[i] * rstd;
        mean_gn /= (float)d;
        for (u64 i = 0; i < d; ++i) { const float norm = in[i] * rstd; const float g = (w ? w[i] : 1.0f) * g_[i]; gx[i] += rstd * (g - norm * mean_gn); }
    }
}
// One thread per column: the weight's and bias's gradients, the rows summed in order
__global__ void norm_param_grads(const float* dh, const float* x, const float* stats, float* dw, float* db, u64 rows, u64 d, int layer)
{
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= d) return;
    float sw = dw ? dw[i] : 0.0f, sb = db ? db[i] : 0.0f;
    for (u64 t = 0; t < rows; ++t) {
        const float mean = layer ? stats[2 * t] : 0.0f, rstd = stats[2 * t + 1];
        const float norm = (x[t * d + i] - mean) * rstd, g = dh[t * d + i];
        sw += g * norm; sb += g;
    }
    if (dw) dw[i] = sw;
    if (db) db[i] = sb;
}
}
)CU";

    inline void check(nvrtcResult r, const char* what)
    {
        if (r != NVRTC_SUCCESS)
            throw cuda_exception(what, static_cast<int>(r));
    }

    // A device copy of a span, made on the host for a staged step and written back if asked
    template <typename T>
    struct staged {
        std::span<T> device;
        std::vector<std::remove_const_t<T>> host;
        explicit staged(std::span<T> d) : device(d), host(d.size())
        {
            if (!d.empty())
                detail::cuda::check(cudaMemcpy(host.data(), d.data(), d.size() * sizeof(T), cudaMemcpyDefault), "cudaMemcpy");
        }
        std::span<std::remove_const_t<T>> span() { return host; }
        void write_back()
        {
            if (!host.empty())
                detail::cuda::check(cudaMemcpy(const_cast<std::remove_const_t<T>*>(device.data()), host.data(), host.size() * sizeof(T),
                                               cudaMemcpyDefault),
                                    "cudaMemcpy");
        }
    };

} // namespace detail::cuda_ops

/// \brief The kernels of the training steps, compiled for one context's device.
class cuda_kernels {
public:
    /// \brief Compiles the kernels (NVRTC) for the context's device and loads them.
    /// \param context The device; must outlive the kernels.
    /// \return The kernels, or why they did not compile (the log on stderr).
    [[nodiscard]] static result<std::unique_ptr<cuda_kernels>, cuda_failure> create(const cuda_context& context)
    {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, context.device());
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, context.device());
        nvrtcProgram prog = nullptr;
        if (const nvrtcResult r = nvrtcCreateProgram(&prog, detail::cuda_ops::source, "snicholls_gpu_ops.cu", 0, nullptr, nullptr);
            r != NVRTC_SUCCESS) {
            std::fprintf(stderr, "snicholls::gpu::cuda_kernels: nvrtcCreateProgram: %s\n", nvrtcGetErrorString(r));
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(r)});
        }
        // Machine code for this device (a CUBIN, sm_XY), not PTX: nothing is left for the driver to compile, so a toolkit
        // newer than the driver works
        const std::string arch = "--gpu-architecture=sm_" + std::to_string(major * 10 + minor);
        const char* options[] = {arch.c_str(), "--fmad=false"};
        const nvrtcResult compiled = nvrtcCompileProgram(prog, 2, options);
        if (compiled != NVRTC_SUCCESS) {
            std::size_t size = 0;
            nvrtcGetProgramLogSize(prog, &size);
            std::string log(size, '\0');
            nvrtcGetProgramLog(prog, log.data());
            std::fprintf(stderr, "snicholls::gpu::cuda_kernels: %s\n", log.c_str());
            nvrtcDestroyProgram(&prog);
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(compiled)});
        }
        std::size_t size = 0;
        nvrtcGetCUBINSize(prog, &size);
        std::string cubin(size, '\0');
        nvrtcGetCUBIN(prog, cubin.data());
        nvrtcDestroyProgram(&prog);
        std::unique_ptr<cuda_kernels> k(new cuda_kernels(context));
        if (const cudaError_t e = cudaLibraryLoadData(&k->library_, cubin.data(), nullptr, nullptr, 0, nullptr, nullptr, 0); e != cudaSuccess) {
            std::fprintf(stderr, "snicholls::gpu::cuda_kernels: cudaLibraryLoadData: %s\n", cudaGetErrorString(e));
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(e)});
        }
        const std::pair<const char*, cudaKernel_t*> kernels[] = {
            {"fill", &k->fill_}, {"copy", &k->copy_}, {"add", &k->add_}, {"add_bias_rows", &k->add_bias_rows_}, {"sum_rows", &k->sum_rows_},
            {"activate", &k->activate_}, {"activate_backward", &k->activate_backward_}, {"norm_rows", &k->norm_rows_},
            {"norm_rows_backward_add", &k->norm_rows_backward_add_}, {"norm_param_grads", &k->norm_param_grads_}};
        for (const auto& [name, slot] : kernels)
            if (const cudaError_t e = cudaLibraryGetKernel(slot, k->library_, name); e != cudaSuccess) {
                std::fprintf(stderr, "snicholls::gpu::cuda_kernels: cudaLibraryGetKernel(%s): %s\n", name, cudaGetErrorString(e));
                return err(cuda_failure{cuda_error::call_failed, static_cast<int>(e)});
            }
        return ok(std::move(k));
    }

    /// \brief Not copyable.
    cuda_kernels(const cuda_kernels&) = delete;
    /// \brief Not assignable. \return Nothing; deleted.
    cuda_kernels& operator=(const cuda_kernels&) = delete;
    /// \brief Unloads the kernels.
    ~cuda_kernels()
    {
        if (library_ != nullptr)
            (void)cudaLibraryUnload(library_);
    }

    /// \brief Launches a kernel on the context's stream over `n` threads. \param k The kernel. \param n The threads.
    /// \param args The arguments, as `cudaLaunchKernel` takes them. \throws cuda_exception if the launch fails.
    void launch(cudaKernel_t k, std::size_t n, void** args) const
    {
        if (n == 0)
            return;
        constexpr unsigned block = 256;
        const auto grid = static_cast<unsigned>((n + block - 1) / block);
        if (const cudaError_t e = cudaLaunchKernel(reinterpret_cast<const void*>(k), dim3(grid), dim3(block), args, 0, context_->stream());
            e != cudaSuccess)
            throw cuda_exception(std::string("cudaLaunchKernel(") + name_of(k) + ") [" + cudaGetErrorString(e) + "]", static_cast<int>(e));
    }

    /// \brief The context. \return It.
    [[nodiscard]] const cuda_context& context() const noexcept { return *context_; }

    /// \brief A kernel's name, for messages. \param k The kernel. \return Its name.
    [[nodiscard]] const char* name_of(cudaKernel_t k) const noexcept
    {
        const std::pair<cudaKernel_t, const char*> names[] = {
            {fill_, "fill"}, {copy_, "copy"}, {add_, "add"}, {add_bias_rows_, "add_bias_rows"}, {sum_rows_, "sum_rows"}, {activate_, "activate"},
            {activate_backward_, "activate_backward"}, {norm_rows_, "norm_rows"}, {norm_rows_backward_add_, "norm_rows_backward_add"},
            {norm_param_grads_, "norm_param_grads"}};
        for (const auto& [kernel, name] : names)
            if (kernel == k)
                return name;
        return "?";
    }

    cudaKernel_t fill_ = nullptr, copy_ = nullptr, add_ = nullptr, add_bias_rows_ = nullptr, sum_rows_ = nullptr, activate_ = nullptr,
                 activate_backward_ = nullptr, norm_rows_ = nullptr, norm_rows_backward_add_ = nullptr,
                 norm_param_grads_ = nullptr;   ///< The kernels, for the overloads below.

private:
    explicit cuda_kernels(const cuda_context& c) : context_(&c) {}
    const cuda_context* context_;
    cudaLibrary_t library_ = nullptr;
};

/// \brief The CUDA backend with the training steps' kernels: core's batched passes run entirely on the device over
/// `device_buffer` memory. Two pointers, trivially copyable.
struct cuda_training_backend : cuda_backend {
    const cuda_kernels* kernels = nullptr;   ///< The compiled kernels; must outlive the backend.
};

static_assert(linalg::blas_backend<cuda_training_backend, float> && std::is_trivially_copyable_v<cuda_training_backend>);

namespace detail::cuda_ops {
    using u64 = unsigned long long;
    inline void sync(const cuda_training_backend& b) { cuda::check(cudaStreamSynchronize(b.context->stream()), "cudaStreamSynchronize"); }
    template <typename T>
    void* arg(const T& v) { return const_cast<void*>(static_cast<const void*>(&v)); }
    template <ml::mlp_kind K>
    constexpr int kind = K == ml::mlp_kind::gelu ? 0 : K == ml::mlp_kind::gelu_tanh ? 1 : K == ml::mlp_kind::relu_squared ? 2 : 3;
} // namespace detail::cuda_ops

/// \name core's training steps on the device, for `cuda_training_backend` (found by argument-dependent lookup)
/// @{

/// \brief `x = value` on the device. \param b The backend. \param x Device memory. \param value The value.
inline void fill_rows(const cuda_training_backend& b, std::span<float> x, std::type_identity_t<float> value)
{
    float* p = x.data();
    const detail::cuda_ops::u64 n = x.size();
    void* args[] = {&p, detail::cuda_ops::arg(n), &value};
    b.kernels->launch(b.kernels->fill_, x.size(), args);
}

/// \brief `to = from` on the device. \param b The backend. \param from Device memory. \param to Device memory.
inline void copy_rows(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> from, std::span<float> to)
{
    const float* f = from.data();
    float* t = to.data();
    const detail::cuda_ops::u64 n = to.size();
    void* args[] = {&f, &t, detail::cuda_ops::arg(n)};
    b.kernels->launch(b.kernels->copy_, to.size(), args);
}

/// \brief `x = a + x` on the device. \param b The backend. \param a Device memory. \param x Device memory, updated.
inline void add_rows(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> a, std::span<float> x)
{
    const float* ap = a.data();
    float* xp = x.data();
    const detail::cuda_ops::u64 n = x.size();
    void* args[] = {&ap, &xp, detail::cuda_ops::arg(n)};
    b.kernels->launch(b.kernels->add_, x.size(), args);
}

/// \brief `y_t += bias` on the device. \param b The backend. \param ys Device memory. \param bias Device memory.
inline void add_bias_rows(const cuda_training_backend& b, std::span<float> ys, std::type_identity_t<std::span<const float>> bias)
{
    float* y = ys.data();
    const float* bp = bias.data();
    const detail::cuda_ops::u64 r = bias.size(), rows = ys.size() / bias.size();
    void* args[] = {&y, &bp, detail::cuda_ops::arg(rows), detail::cuda_ops::arg(r)};
    b.kernels->launch(b.kernels->add_bias_rows_, ys.size(), args);
}

/// \brief `into += Σ_t ys_t` on the device, the rows in order. \param b The backend. \param ys Device memory.
/// \param into Device memory, added into.
inline void sum_rows(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> ys, std::span<float> into)
{
    const float* y = ys.data();
    float* ip = into.data();
    const detail::cuda_ops::u64 r = into.size(), rows = ys.size() / into.size();
    void* args[] = {&y, &ip, detail::cuda_ops::arg(rows), detail::cuda_ops::arg(r)};
    b.kernels->launch(b.kernels->sum_rows_, into.size(), args);
}

/// \brief The MLP's activation on the device. \tparam C The configuration. \param b The backend.
/// \param saved `up` (and `gate`) read, `hidden` written, all device memory.
template <ml::transformer_config C>
void mlp_activate_rows(const cuda_training_backend& b, const ml::mlp_activations<float>& saved)
{
    const float* up = saved.up.data();
    const float* gate = saved.gate.empty() ? nullptr : saved.gate.data();
    float* hidden = saved.hidden.data();
    const detail::cuda_ops::u64 m = saved.up.size();
    int kind = detail::cuda_ops::kind<C.mlp>;
    void* args[] = {&up, &gate, &hidden, detail::cuda_ops::arg(m), &kind};
    b.kernels->launch(b.kernels->activate_, saved.up.size(), args);
}

/// \brief The backward pass of the MLP's activation on the device. \tparam C The configuration. \param b The backend.
/// \param saved What the forward saved. \param dh Replaced by the gradient in `up`. \param dgate SwiGLU's gate gradient.
template <ml::transformer_config C>
void mlp_activate_rows_backward(const cuda_training_backend& b, const ml::mlp_activations<float>& saved, std::span<float> dh,
                                std::span<float> dgate)
{
    const float* up = saved.up.data();
    const float* gate = saved.gate.empty() ? nullptr : saved.gate.data();
    float* dhp = dh.data();
    float* dg = dgate.empty() ? nullptr : dgate.data();
    const detail::cuda_ops::u64 m = dh.size();
    int kind = detail::cuda_ops::kind<C.mlp>;
    void* args[] = {&up, &gate, &dhp, &dg, detail::cuda_ops::arg(m), &kind};
    b.kernels->launch(b.kernels->activate_backward_, dh.size(), args);
}

/// \brief `block_norm` of every row on the device. \tparam C The configuration. \param b The backend.
/// \param x Device memory. \param weight Device memory, or empty. \param bias Device memory, or empty.
/// \param out Device memory.
template <ml::transformer_config C>
void block_norm_rows(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> x,
                     std::type_identity_t<std::span<const float>> weight, std::type_identity_t<std::span<const float>> bias, std::span<float> out)
{
    const float* xp = x.data();
    const float* wp = weight.empty() ? nullptr : weight.data();
    const float* bp = bias.empty() ? nullptr : bias.data();
    float* op = out.data();
    float* stats = nullptr;
    const detail::cuda_ops::u64 d = C.d_model, rows = x.size() / C.d_model;
    int layer = C.norm == ml::norm_kind::layer ? 1 : 0;
    float eps = ml::norm_eps_v<C, float>;
    void* args[] = {&xp, &wp, &bp, &op, &stats, detail::cuda_ops::arg(rows), detail::cuda_ops::arg(d), &layer, &eps};
    b.kernels->launch(b.kernels->norm_rows_, rows, args);
}

/// \brief The backward pass of `block_norm_rows` on the device, added into `dxs`; the weight's and bias's gradients
/// added into, the rows in order. \tparam C The configuration. \param b The backend. \param dh The gradient in the
/// norm's output. \param x Its input. \param weight The scale. \param bias The shift. \param dxs Added into.
/// \param dweight Added into, or empty. \param dbias Added into, or empty.
template <ml::transformer_config C>
void block_norm_rows_backward_add(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> dh,
                                  std::type_identity_t<std::span<const float>> x, std::type_identity_t<std::span<const float>> weight,
                                  std::type_identity_t<std::span<const float>> bias, std::span<float> dxs,
                                  std::type_identity_t<std::span<float>> dweight, std::type_identity_t<std::span<float>> dbias)
{
    namespace dc = detail::cuda_ops;
    const dc::u64 d = C.d_model, rows = x.size() / C.d_model;
    device_buffer<float> stats(*b.context, 2 * rows), scratch(*b.context, x.size());   // the forward's statistics, recomputed
    const float* xp = x.data();
    const float* wp = weight.empty() ? nullptr : weight.data();
    const float* bp = bias.empty() ? nullptr : bias.data();
    float* sp = stats.span().data();
    float* tmp = scratch.span().data();
    int layer = C.norm == ml::norm_kind::layer ? 1 : 0;
    float eps = ml::norm_eps_v<C, float>;
    void* a1[] = {&xp, &wp, &bp, &tmp, &sp, dc::arg(rows), dc::arg(d), &layer, &eps};
    b.kernels->launch(b.kernels->norm_rows_, rows, a1);
    const float* g = dh.data();
    float* gx = dxs.data();
    void* a2[] = {&g, &xp, &wp, &sp, &gx, dc::arg(rows), dc::arg(d), &layer};
    b.kernels->launch(b.kernels->norm_rows_backward_add_, rows, a2);
    float* gw = dweight.empty() ? nullptr : dweight.data();
    float* gb = (bias.empty() || dbias.empty()) ? nullptr : dbias.data();
    void* a3[] = {&g, &xp, &sp, &gw, &gb, dc::arg(rows), dc::arg(d), &layer};
    if (gw != nullptr || gb != nullptr)
        b.kernels->launch(b.kernels->norm_param_grads_, d, a3);
    dc::sync(b);   // the scratch buffers are freed on return
}

/// \brief QK norm and RoPE on the device: staged in this version (core's code on host copies).
/// \tparam C The configuration. \param b The backend. \param w The weights. \param act The activations.
/// \param n The tokens. \param rope_src The rope source (host memory).
template <ml::transformer_config C, typename W, typename R>
void attention_position_rows(const cuda_training_backend& b, const ml::attention_weights<W, float>& w,
                             const ml::attention_activations<float>& act, std::size_t n, const R& rope_src)
{
    if constexpr (C.qk_norm == ml::qk_norm_kind::none && C.rope == ml::rope_kind::none) {
        (void)b, (void)w, (void)act, (void)n, (void)rope_src;   // nothing to do
    } else {
        detail::cuda_ops::sync(b);
        using S = detail::cuda_ops::staged<float>;
        using SC = detail::cuda_ops::staged<const float>;
        S q(act.q), k(act.k), qr(act.q_raw), kr(act.k_raw);
        SC qn(w.q_norm), kn(w.k_norm);
        auto wh = w;
        wh.q_norm = qn.span();
        wh.k_norm = kn.span();
        ml::attention_activations<float> a = act;
        a.q = q.span(), a.k = k.span(), a.q_raw = qr.span(), a.k_raw = kr.span();
        ml::attention_position_rows<C>(linalg::reference, wh, a, n, rope_src);
        q.write_back(), k.write_back(), qr.write_back(), kr.write_back();
    }
}

/// \brief Every token's heads on the device: staged in this version. \tparam C The configuration.
/// \param b The backend. \param act The activations. \param n The tokens.
template <ml::transformer_config C>
void attention_attend_rows(const cuda_training_backend& b, const ml::attention_activations<float>& act, std::size_t n)
{
    detail::cuda_ops::sync(b);
    using S = detail::cuda_ops::staged<float>;
    S q(act.q), k(act.k), v(act.v), y(act.y), lse(act.lse);
    ml::attention_activations<float> a = act;
    a.q = q.span(), a.k = k.span(), a.v = v.span(), a.y = y.span(), a.lse = lse.span();
    ml::attention_attend_rows<C>(linalg::reference, a, n);
    y.write_back(), lse.write_back();
}

/// \brief The backward pass of the heads on the device: staged in this version. \tparam C The configuration.
/// \param b The backend. \param act The activations. \param dy The heads' output gradients. \param grads The
/// gradients. \param n The tokens.
template <ml::transformer_config C>
void attention_attend_rows_backward(const cuda_training_backend& b, const ml::attention_activations<float>& act,
                                    std::type_identity_t<std::span<const float>> dy, const ml::attention_gradients<float>& grads, std::size_t n)
{
    detail::cuda_ops::sync(b);
    using S = detail::cuda_ops::staged<float>;
    using SC = detail::cuda_ops::staged<const float>;
    S q(act.q), k(act.k), v(act.v), y(act.y), lse(act.lse), gq(grads.q), gk(grads.k), gv(grads.v);
    SC dys(dy);
    ml::attention_activations<float> a = act;
    a.q = q.span(), a.k = k.span(), a.v = v.span(), a.y = y.span(), a.lse = lse.span();
    ml::attention_gradients<float> g = grads;
    g.q = gq.span(), g.k = gk.span(), g.v = gv.span();
    ml::attention_attend_rows_backward<C>(linalg::reference, a, std::span<const float>(dys.span()), g, n);
    gq.write_back(), gk.write_back(), gv.write_back();
}

/// \brief The backward pass of QK norm and RoPE on the device: staged in this version. \tparam C The configuration.
/// \param b The backend. \param w The weights. \param act The activations. \param grads The gradients.
/// \param n The tokens. \param rope_src The rope source (host memory).
template <ml::transformer_config C, typename W, typename R>
void attention_position_rows_backward(const cuda_training_backend& b, const ml::attention_weights<W, float>& w,
                                      const ml::attention_activations<float>& act, const ml::attention_gradients<float>& grads,
                                      std::size_t n, const R& rope_src)
{
    if constexpr (C.qk_norm == ml::qk_norm_kind::none && C.rope == ml::rope_kind::none) {
        (void)b, (void)w, (void)act, (void)grads, (void)n, (void)rope_src;
    } else {
        detail::cuda_ops::sync(b);
        using S = detail::cuda_ops::staged<float>;
        using SC = detail::cuda_ops::staged<const float>;
        S gq(grads.q), gk(grads.k), gqn(grads.q_norm), gkn(grads.k_norm), qr(act.q_raw), kr(act.k_raw);
        SC qn(w.q_norm), kn(w.k_norm);
        auto wh = w;
        wh.q_norm = qn.span();
        wh.k_norm = kn.span();
        ml::attention_activations<float> a = act;
        a.q_raw = qr.span(), a.k_raw = kr.span();
        ml::attention_gradients<float> g = grads;
        g.q = gq.span(), g.k = gk.span(), g.q_norm = gqn.span(), g.k_norm = gkn.span();
        ml::attention_position_rows_backward<C>(linalg::reference, wh, a, g, n, rope_src);
        gq.write_back(), gk.write_back(), gqn.write_back(), gkn.write_back();
    }
}

/// @}

} // namespace snicholls::gpu

#endif /* snicholls_gpu_cuda_ops_hpp */
