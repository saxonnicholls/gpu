//
//  cuda_rows.hpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::gpu::cuda_rows_kernels`: the key-value cache's operations on CUDA, from the source
/// `opencl_device.hpp` builds for OpenCL (`detail/rows_source.hpp`), so a forward pass over new tokens is written once
/// against core's backend concept and runs on both.
/// \ingroup gpu
///
/// **Overview.** `cuda_ops.hpp` already overloads core's row operations for `cuda_training_backend` (norms, RoPE,
/// attention within a sequence, the MLP's activations); a pass over new tokens against a cache needs two more,
/// `kv_store_rows` and `attention_attend_cached_rows` (their host reference is in the consumer's model until core has
/// them). This header compiles those kernels from the shared source with NVRTC and overloads the two operations for
/// `cuda_training_backend`, found by argument-dependent lookup as `opencl_device.hpp`'s are for
/// `opencl_device_backend`. CUDA's device pointers are real, so every buffer offset is 0.
///
/// \snippet cuda_rows_demo.cpp rows
///
/// **Relationships.** `opencl_device.hpp` (the same operations on OpenCL), `cuda_ops.hpp` (the backend, its other
/// operations), `detail/rows_source.hpp`. Group: \ref gpu.
///
/// **Implementations.** One kernel source for both APIs.
///
/// **Guarantees.** As `opencl_device.hpp`'s: the cache's writes exact; attention in core's order (keys in tiles of 64,
/// each scored by one thread with a sequential dot product, folded in key order), differing from the host only by the
/// device's `expf` and `logf`. Deterministic.
///
/// **Errors.** `create` returns a `cuda_failure` when NVRTC or the load fails (the log on stderr); a launch that fails
/// throws `cuda_exception`; a cache write or read past its capacity throws `std::invalid_argument`; a backend without
/// `rows` throws `std::logic_error`.
///
/// **Performance.** Not yet run on an NVIDIA GPU: compiled without the toolkit (clang's NVPTX target, and the host code
/// against the CUDA 12.9 headers); measured against OpenCL when the NVIDIA machine returns.
///
/// **Tests and demos.** `tests/cuda_rows_test.cpp`, `demos/cuda_rows_demo.cpp` (where CUDA is).
///
/// **Dependencies.** The CUDA runtime and NVRTC, `cuda_ops.hpp`, `detail/quant_source.hpp` (the CUDA prelude),
/// `detail/rows_source.hpp`.
///
/// **Known issues.** Heads of at most 128 channels.
///
/// **Status.** Experimental. CUDA: not yet run on an NVIDIA GPU.

#ifndef snicholls_gpu_cuda_rows_hpp
#define snicholls_gpu_cuda_rows_hpp

#include <snicholls/gpu/cuda_ops.hpp>
#include <snicholls/gpu/detail/quant_source.hpp>
#include <snicholls/gpu/detail/rows_source.hpp>

#include <snicholls/core/ml/transformer_config.hpp>

#include <nvrtc.h>

#include <cstddef>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace snicholls::gpu {

/// \brief The shared row kernels, compiled by NVRTC for one CUDA context.
class cuda_rows_kernels {
public:
    /// \brief Compiles and loads the kernels. \param context The device; must outlive the kernels.
    /// \return The kernels, or why they did not compile (the log on stderr).
    [[nodiscard]] static result<std::unique_ptr<cuda_rows_kernels>, cuda_failure> create(const cuda_context& context)
    {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, context.device());
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, context.device());
        const std::string source = std::string(detail::quant::cuda_prelude) + detail::rows::math_cuda + detail::rows::source;
        nvrtcProgram prog = nullptr;
        if (const nvrtcResult r = nvrtcCreateProgram(&prog, source.c_str(), "snicholls_gpu_rows.cu", 0, nullptr, nullptr); r != NVRTC_SUCCESS) {
            std::fprintf(stderr, "snicholls::gpu::cuda_rows_kernels: nvrtcCreateProgram: %s\n", nvrtcGetErrorString(r));
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(r)});
        }
        const std::string arch = "--gpu-architecture=sm_" + std::to_string(major * 10 + minor);
        const char* options[] = {arch.c_str(), "--fmad=false", "--prec-div=true", "--ftz=false", "--std=c++17"};
        if (const nvrtcResult compiled = nvrtcCompileProgram(prog, 5, options); compiled != NVRTC_SUCCESS) {
            std::size_t size = 0;
            nvrtcGetProgramLogSize(prog, &size);
            std::string log(size, '\0');
            nvrtcGetProgramLog(prog, log.data());
            std::fprintf(stderr, "snicholls::gpu::cuda_rows_kernels: %s\n", log.c_str());
            nvrtcDestroyProgram(&prog);
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(compiled)});
        }
        std::size_t size = 0;
        nvrtcGetCUBINSize(prog, &size);
        std::string cubin(size, '\0');
        nvrtcGetCUBIN(prog, cubin.data());
        nvrtcDestroyProgram(&prog);
        std::unique_ptr<cuda_rows_kernels> k(new cuda_rows_kernels(context));
        if (const cudaError_t e = cudaLibraryLoadData(&k->library_, cubin.data(), nullptr, nullptr, 0, nullptr, nullptr, 0); e != cudaSuccess) {
            std::fprintf(stderr, "snicholls::gpu::cuda_rows_kernels: cudaLibraryLoadData: %s\n", cudaGetErrorString(e));
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(e)});
        }
        for (const auto& [name, slot] : {std::pair<const char*, cudaKernel_t*>{"rows_attend", &k->attend_}, {"rows_store", &k->store_}})
            if (const cudaError_t e = cudaLibraryGetKernel(slot, k->library_, name); e != cudaSuccess) {
                std::fprintf(stderr, "snicholls::gpu::cuda_rows_kernels: cudaLibraryGetKernel(%s): %s\n", name, cudaGetErrorString(e));
                return err(cuda_failure{cuda_error::call_failed, static_cast<int>(e)});
            }
        return ok(std::move(k));
    }

    /// \brief Not copyable.
    cuda_rows_kernels(const cuda_rows_kernels&) = delete;
    /// \brief Not assignable. \return Nothing; deleted.
    cuda_rows_kernels& operator=(const cuda_rows_kernels&) = delete;
    /// \brief Unloads the kernels.
    ~cuda_rows_kernels()
    {
        if (library_ != nullptr)
            (void)cudaLibraryUnload(library_);
    }

    /// \brief Launches `groups` blocks of `threads`. \param attend The attention kernel, else the cache's writes.
    /// \param groups The blocks. \param threads Their threads. \param args The arguments. \throws cuda_exception.
    void launch(bool attend, std::size_t groups, unsigned threads, void** args) const
    {
        if (groups == 0)
            return;
        const cudaKernel_t k = attend ? attend_ : store_;
        if (const cudaError_t e = cudaLaunchKernel(reinterpret_cast<const void*>(k), dim3(unsigned(groups)), dim3(threads), args, 0, context_->stream());
            e != cudaSuccess)
            throw cuda_exception(std::string("cudaLaunchKernel(rows) [") + cudaGetErrorString(e) + "]", static_cast<int>(e));
    }

private:
    explicit cuda_rows_kernels(const cuda_context& c) : context_(&c) {}
    const cuda_context* context_;
    cudaLibrary_t library_ = nullptr;
    cudaKernel_t attend_ = nullptr, store_ = nullptr;
};

namespace detail::cuda_rows {
    inline const cuda_rows_kernels& kernels(const cuda_training_backend& b)
    {
        if (b.rows == nullptr)
            throw std::logic_error("snicholls::gpu: the cache's operations need cuda_training_backend::rows (cuda_rows_kernels)");
        return *b.rows;
    }
    using u64 = unsigned long long;
} // namespace detail::cuda_rows

/// \brief Attention of `n` new queries at positions `first ..` over a cache on the device (`[kv_head][capacity][head_dim]`),
/// causal and windowed as the configuration says: `opencl_device.hpp`'s operation on CUDA. \tparam C The configuration.
/// \param b The backend. \param q `n` rows of `q_size`. \param keys The keys. \param values The values. \param first The
/// first query's position. \param n The queries. \param capacity The rows a head. \param y `n` rows. \param lse `n · n_heads`.
template <ml::transformer_config C>
void attention_attend_cached_rows(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> q,
                                  std::type_identity_t<std::span<const float>> keys, std::type_identity_t<std::span<const float>> values,
                                  std::size_t first, std::size_t n, std::size_t capacity, std::span<float> y, std::span<float> lse)
{
    static_assert(C.head_dim <= 128, "snicholls::gpu: the cached attention takes heads of at most 128 channels");
    static_assert(C.attention_softcap == 0, "snicholls::gpu: the cached attention has no soft cap");
    namespace d = detail::cuda_rows;
    if (n == 0)
        return;
    if (first + n > capacity)
        throw std::invalid_argument("snicholls::gpu: attention past the cache's capacity");
    const float* pq = q.data();
    const float* pk = keys.data();
    const float* pv = values.data();
    float* py = y.data();
    float* pl = lse.data();
    d::u64 zero = 0, f = first, nn = n, cap = capacity, window = C.window, dh = C.head_dim;
    int H = int(C.n_heads), KVH = int(C.n_kv_heads);
    float scale = 1.0f / snicholls::math::sqrt(float(C.head_dim));
    void* args[] = {&pq, &zero, &pk, &zero, &pv, &zero, &f, &nn, &cap, &window, &H, &KVH, &dh, &scale, &py, &zero, &pl, &zero};
    d::kernels(b).launch(true, n * C.n_heads, 64, args);
}

/// \brief The keys and values of `n` new tokens (`[kv_head][n][head_dim]`) into a cache's rows `first ..` on the device.
/// \tparam C The configuration. \param b The backend. \param k The keys. \param v The values. \param keys One layer's
/// cache keys. \param values Its values. \param first The first position. \param n The tokens. \param capacity The rows a head.
template <ml::transformer_config C>
void kv_store_rows(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> k, std::type_identity_t<std::span<const float>> v,
                   std::span<float> keys, std::span<float> values, std::size_t first, std::size_t n, std::size_t capacity)
{
    namespace d = detail::cuda_rows;
    if (n == 0)
        return;
    if (first + n > capacity)
        throw std::invalid_argument("snicholls::gpu: a cache write past its capacity");
    const float* pk = k.data();
    const float* pv = v.data();
    float* pks = keys.data();
    float* pvs = values.data();
    d::u64 zero = 0, f = first, nn = n, cap = capacity, dh = C.head_dim;
    int KVH = int(C.n_kv_heads);
    void* args[] = {&pk, &zero, &pv, &zero, &pks, &zero, &pvs, &zero, &f, &nn, &cap, &KVH, &dh};
    d::kernels(b).launch(false, (C.n_kv_heads * n * C.head_dim + 255) / 256, 256, args);
}

} // namespace snicholls::gpu

#endif // snicholls_gpu_cuda_rows_hpp
