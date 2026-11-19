//
//  cuda_moe.hpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::gpu::cuda_moe_kernels`: snicholls-core's mixture-of-experts steps (`ml/moe.hpp`) on an NVIDIA GPU, so an MoE transformer trains with its routing, permutation, grouped products and balance losses on the device.
/// \ingroup gpu
///
/// **Overview.** Core's mixture of experts (TRAINING.md T10) makes every step a free function taking the backend first:
/// routing, the histogram and its prefix sum, the stable permutation, gather and combine, the grouped products and the
/// balance terms. This header overloads each for `cuda_training_backend` (found by argument-dependent lookup), so with
/// the router logits, the experts' weights and every activation in `device_buffer`s, core's `ml::moe_forward` and
/// `ml::moe_backward` run on the GPU; the experts' activations are `cuda_ops.hpp`'s fused kernels over the permuted
/// rows, the router and the shared experts its dense products.
///
/// - `cuda_moe_kernels::create(context)`: compiles the kernels for the context's device (NVRTC); `result` with a
///   `cuda_failure`.
/// - `backend.moe = moe_kernels.get()`: the training backend carries a pointer to them; an overload here throws
///   `std::logic_error` on a backend without one.
///
/// \snippet cuda_moe_demo.cpp moe
///
/// **The steps on the device.**
///
/// | step | kernel | against core's reference |
/// |---|---|---|
/// | `moe_route_rows` | a thread per row: the affinities, core's insertion top-K over the keys, the weights | the experts to the bit away from ties within rounding of the affinities; the weights within rounding (the device's `expf`) |
/// | `moe_route_rows_backward` | a thread per row, the weights recomputed | within rounding |
/// | `moe_count_rows`, `moe_offsets` | a shared-memory histogram per block (integer atomics: exact, order-free), the prefix sum | to the bit |
/// | `moe_permutation` | a counting sort: each block's histogram of its chunk, each expert's start in each chunk, then each block ranks its pairs tile by tile (warp `match_any`, the warps in order) | to the bit (the order is unique) |
/// | `moe_gather_rows`, its backward | a thread per value (float4 where aligned); the backward adds the `K` slots in slot order | to the bit |
/// | `moe_combine_rows`, its backward | a thread per value in slot order; `dweights` a thread per pair summing its row in element order | to the bit |
/// | `grouped_linear`, its backward | one launch per product over every expert (below) | within the product's rounding |
/// | `moe_update_bias` | a thread per expert, the sign in 64-bit integers | to the bit |
/// | `moe_balance_loss`, `moe_z_loss` | the probabilities a thread per row, the sums per (group, expert) in row order, the loss in core's order | within rounding (`expf`, `logf`) |
/// | `moe_send_offsets`, `moe_receive_offsets`, `moe_receive_layout` | expert parallelism's layouts from every rank's counts in device memory | to the bit |
/// | `moe_all_to_all`, `moe_permute_rows`, `moe_unpermute_rows` | in-process ranks on one device, a thread per element, any 1-, 2-, 4- or 8-byte payload | to the bit |
/// | `to_bfloat16_rows`, `to_float_rows` | a thread per value, core's rounding (a NaN kept quiet with its sign) | to the bit |
///
/// The experts' activations are `cuda_ops.hpp`'s `mlp_activate_rows` and its backward over the permuted rows, the
/// router and the shared experts its dense products; so core's `ml::moe_forward`, `ml::moe_backward` and the block of
/// `ml/moe_block.hpp` (`block_forward_sequence`, `block_backward`) run on the device unchanged.
///
/// **The grouped products.** The segments' sizes come from the routing, so they are known only on the device.
/// `grouped_linear` and its backward run as one kernel each (`grouped_path::kernel`, the default): the segments' bounds
/// are read from the offsets in device memory, the grid sized from the rows (an upper bound of `rows / 128 + E` row
/// tiles; a block past the last segment returns), 128 x 128 tiles on the tensor cores with float accumulators: `mma.sync
/// m16n8k8` in TF32 under `cuda_math::tf32`, `m16n8k16` in bfloat16 under `cuda_math::bf16` (the operands rounded as
/// they are loaded, no conversion pass), and under `cuda_math::float32` three TF32 products per tile (the operands
/// split into a TF32 part and the remainder, the small-times-small term dropped), which is float's accuracy. The weight
/// gradient's reduction runs over a segment's rows in order inside one block, so nothing is atomic and the result is
/// the same every run. Nothing waits for the host, so a whole MoE forward and backward pass can be captured as a
/// `cuda_graph`. `grouped_path::cublas` instead copies the offsets to the host (one wait per call) and makes each
/// expert's product through cuBLAS, as core's reference loops them; it cannot be captured.
///
/// **Losses in a graph.** `moe_balance_loss` and `moe_z_loss` return a number. Outside a capture they wait for it;
/// during a capture (or with `cuda_moe_kernels::losses_on_device`) they write it to `cuda_moe_kernels::losses()` (the
/// balance loss at 0, the z-loss at 1), return a quiet NaN, and the gradient is unaffected.
///
/// **Relationships.** `cuda_ops.hpp` (the backend, the activations, the dense products), `cuda.hpp` (the context,
/// `device_buffer`, `cuda_graph`), core's `ml/moe.hpp` (the operations it overloads and the reference it is checked
/// against). Group: \ref gpu.
///
/// **Guarantees.** Every step is queued on the context's stream; none waits, except the two losses outside a capture
/// and `grouped_path::cublas`. Every kernel is deterministic: two runs give the same bits. The kernels' working space
/// is kept between calls; run a step once uncaptured before capturing it, so the space has its size.
///
/// **Errors.** `cuda_moe_kernels::create` returns a `cuda_failure` if the kernels do not compile. An overload throws
/// `std::logic_error` if the backend has no `moe` kernels, `std::invalid_argument` past the device's limits (more than
/// 1024 experts), `cuda_exception` if a launch fails.
///
/// **Performance.** `benchmarks/cuda_moe_benchmark.cpp`, `docs/performance.md`. On an RTX PRO 4000 Blackwell (CUDA 13.3,
/// Release, 2026-11-19) while another training job kept the GPU fully busy, so upper bounds and noisy: the grouped up
/// projection of 4096 tokens of 768 (eight experts of 2048, two per token) forward in 3.8 ms in TF32 against 3.5 ms for
/// cuBLAS expert by expert and 3.4 ms for `cublasGemmGroupedBatchedEx`; in bfloat16 3.3 ms against 2.9 ms; in float 7.5
/// ms against 7.7 ms. With 64 experts of 256 (eight per token, 32768 rows) the kernel matches or beats cuBLAS's loop in
/// every math (float 4.0 ms against 6.9). cuBLASLt's grouped matrix layouts are not supported on this device (sm_120).
///
/// **Tests and demos.** `tests/cuda_moe_test.cpp` (each step against core's reference on the host, the edge cases,
/// two runs bit for bit, whole MoE layers and an MoE block forward and backward, a captured graph, expert parallelism's
/// layouts and payloads), `demos/cuda_moe_demo.cpp`:
///
/// \snippet cuda_moe_demo.cpp bias
///
/// \snippet cuda_moe_demo.cpp graph
///
/// **Dependencies.** `cuda_ops.hpp`, `<nvrtc.h>`, core's `ml/moe.hpp`, `ml/moe_block.hpp`, `ml/expert_parallel.hpp`,
/// `ml/minifloat.hpp`.
///
/// **Known issues.** Float only (the payload rows also in bfloat16). At most 1024 experts and 64 ranks. The selection is decided on affinities the device computes with
/// its own `expf`, so two keys within an ulp of each other may be ordered differently from the host's.
///
/// **Status.** Experimental, new in 0.8.0.

#ifndef snicholls_gpu_cuda_moe_hpp
#define snicholls_gpu_cuda_moe_hpp

#include <snicholls/gpu/cuda_ops.hpp>

#include <snicholls/core/ml/expert_parallel.hpp>
#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/moe.hpp>
#include <snicholls/core/ml/moe_block.hpp>

#include <nvrtc.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls::gpu {

namespace detail::cuda_moe {

    inline constexpr const char* source = R"CU(
typedef unsigned long long u64;
#define INFINITY __int_as_float(0x7f800000)
#define NAN __int_as_float(0x7fffffff)
// core's sigmoid and sigmoid_derivative (ml/activations.hpp)
__device__ float sigmoid_(float v) { if (v >= 0.0f) return 1.0f / (1.0f + expf(-v)); const float e = expf(v); return e / (1.0f + e); }
__device__ float sigmoid_derivative_(float v)
{ const float e = expf(-fabsf(v)); const float r = 1.0f / (1.0f + e); return v >= 0.0f ? r * (e * r) : (e * r) * r; }
// core's max_of: the largest value, or the first NaN
__device__ float max_of_(const float* z, u64 E) { float m = -INFINITY; for (u64 e = 0; e < E; ++e) { const float x = z[e]; if (x != x) return x; if (x > m) m = x; } return m; }
// core's softmax of a row: the maximum, then e^(x - max) summed in order, then each times 1 / sum
struct softmax_ { float m, inv; };
__device__ softmax_ softmax_of_(const float* z, u64 E)
{ const float m = max_of_(z, E); float s = 0.0f; for (u64 e = 0; e < E; ++e) s += expf(z[e] - m); return {m, 1.0f / s}; }
__device__ float prob_(const float* z, u64 e, softmax_ q) { return expf(z[e] - q.m) * q.inv; }
// The affinity of expert e: z, softmax(z) or sigmoid(z) (scoring 0, 1, 2)
__device__ float affinity_(const float* z, u64 e, int scoring, softmax_ q) { return scoring == 0 ? z[e] : scoring == 1 ? prob_(z, e, q) : sigmoid_(z[e]); }
// The chosen experts' softmax_top_k weights before the scale, into q
__device__ void top_softmax_(const float* z, const unsigned* ids, int K, float* q)
{
    float m = z[ids[0]];
    for (int i = 1; i < K; ++i) m = z[ids[i]] > m ? z[ids[i]] : m;
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) { q[i] = expf(z[ids[i]] - m); sum += q[i]; }
    for (int i = 0; i < K; ++i) q[i] /= sum;
}
extern "C" {
// One thread per row: core's moe_route_rows. The top-K is core's insertion (ml/sampling.hpp top_k), its values kept in
// the row's weights until they are written
__global__ void moe_route(const float* logits, const float* bias, unsigned* experts, float* weights, u64 rows, u64 E, int K, int scoring, float scale)
{
    const u64 t = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (t >= rows) return;
    const float* z = logits + t * E; unsigned* ids = experts + t * K; float* w = weights + t * K;
    const softmax_ q = scoring == 1 ? softmax_of_(z, E) : softmax_{0.0f, 0.0f};
    int count = 0;
    for (u64 e = 0; e < E; ++e) {
        const float s = affinity_(z, e, scoring, q), x = bias ? s + bias[e] : s;
        if (count == K && !(x > w[K - 1])) continue;
        int at = count < K ? count++ : K - 1;
        for (; at > 0 && x > w[at - 1]; --at) { w[at] = w[at - 1]; ids[at] = ids[at - 1]; }
        w[at] = x; ids[at] = (unsigned)e;
    }
    if (scoring == 0) {
        float m = z[ids[0]];
        for (int i = 1; i < K; ++i) m = z[ids[i]] > m ? z[ids[i]] : m;
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) { w[i] = expf(z[ids[i]] - m); sum += w[i]; }
        for (int i = 0; i < K; ++i) w[i] = (w[i] / sum) * scale;
    } else if (scoring == 1) {
        for (int i = 0; i < K; ++i) w[i] = prob_(z, ids[i], q) * scale;
    } else {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += sigmoid_(z[ids[i]]);
        for (int i = 0; i < K; ++i) w[i] = sigmoid_(z[ids[i]]) / sum * scale;
    }
}
// One thread per row: core's moe_route_rows_backward, dlogits written
__global__ void moe_route_backward(const float* logits, const unsigned* experts, const float* dweights, float* dlogits, u64 rows, u64 E, int K,
                                   int scoring, float scale)
{
    const u64 t = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (t >= rows) return;
    const float* z = logits + t * E; const unsigned* ids = experts + t * K; const float* dw = dweights + t * K; float* g = dlogits + t * E;
    if (scoring == 1) {   // softmax_backward of the whole row, dp the scaled weight gradient at the chosen experts
        const softmax_ q = softmax_of_(z, E);
        float dot = 0.0f;
        for (u64 e = 0; e < E; ++e) {
            float dp = 0.0f; for (int i = 0; i < K; ++i) if (ids[i] == e) dp = scale * dw[i];
            dot += prob_(z, e, q) * dp;
        }
        for (u64 e = 0; e < E; ++e) {
            float dp = 0.0f; for (int i = 0; i < K; ++i) if (ids[i] == e) dp = scale * dw[i];
            g[e] = prob_(z, e, q) * (dp - dot);
        }
        return;
    }
    // The chosen experts' normalised weights q (in g's chosen places until the end: recomputed per slot instead)
    float dot = 0.0f, sum = 0.0f, m = 0.0f;
    if (scoring == 0) {
        m = z[ids[0]];
        for (int i = 1; i < K; ++i) m = z[ids[i]] > m ? z[ids[i]] : m;
        for (int i = 0; i < K; ++i) sum += expf(z[ids[i]] - m);
        for (int i = 0; i < K; ++i) dot += (expf(z[ids[i]] - m) / sum) * (scale * dw[i]);
    } else {
        for (int i = 0; i < K; ++i) sum += sigmoid_(z[ids[i]]);
        for (int i = 0; i < K; ++i) dot += sigmoid_(z[ids[i]]) / sum * (scale * dw[i]);
    }
    for (u64 e = 0; e < E; ++e) g[e] = 0.0f;
    for (int i = 0; i < K; ++i)
        g[ids[i]] = scoring == 0 ? (expf(z[ids[i]] - m) / sum) * (scale * dw[i] - dot) : (scale * dw[i] - dot) / sum * sigmoid_derivative_(z[ids[i]]);
}
// The histogram: each block counts its part in shared memory, then adds its counts (integers, so in any order)
__global__ void moe_count(const unsigned* experts, u64 n, unsigned E, unsigned* counts)
{
    extern __shared__ unsigned h[];
    for (unsigned e = threadIdx.x; e < E; e += blockDim.x) h[e] = 0;
    __syncthreads();
    for (u64 j = blockIdx.x * (u64)blockDim.x + threadIdx.x; j < n; j += (u64)gridDim.x * blockDim.x) atomicAdd(&h[experts[j]], 1u);
    __syncthreads();
    for (unsigned e = threadIdx.x; e < E; e += blockDim.x) if (h[e]) atomicAdd(&counts[e], h[e]);
}
__global__ void moe_offsets(const unsigned* counts, unsigned* offsets, unsigned E)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    offsets[0] = 0; for (unsigned e = 0; e < E; ++e) offsets[e + 1] = offsets[e] + counts[e];
}
// The permutation, a counting sort in three kernels over chunks of CHUNK pairs, a block each.
#define CHUNK 1024
// 1. each chunk's histogram: bc[b E + e]
__global__ void moe_chunk_counts(const unsigned* experts, u64 n, unsigned E, unsigned* bc)
{
    extern __shared__ unsigned h[];
    for (unsigned e = threadIdx.x; e < E; e += blockDim.x) h[e] = 0;
    __syncthreads();
    const u64 first = (u64)blockIdx.x * CHUNK;
    for (u64 j = first + threadIdx.x; j < first + CHUNK && j < n; j += blockDim.x) atomicAdd(&h[experts[j]], 1u);
    __syncthreads();
    for (unsigned e = threadIdx.x; e < E; e += blockDim.x) bc[(u64)blockIdx.x * E + e] = h[e];
}
// 2. a thread per expert: where its pairs of each chunk start, the chunks in order from offsets[e]
__global__ void moe_chunk_starts(const unsigned* bc, const unsigned* offsets, unsigned chunks, unsigned E, unsigned* starts)
{
    const unsigned e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= E) return;
    unsigned at = offsets[e];
    for (unsigned b = 0; b < chunks; ++b) { starts[(u64)b * E + e] = at; at += bc[(u64)b * E + e]; }
}
// 3. a block per chunk, 256 threads, its pairs in tiles of 256 in order: a pair's position is its expert's start in
// the chunk, plus the pairs of that expert in the chunk's earlier tiles, in the tile's earlier warps, and in its warp
// before it (match_any). Every count is an integer, so the positions are exact and unique.
__global__ void moe_rank(const unsigned* experts, u64 n, unsigned E, const unsigned* starts, unsigned* order, unsigned* inverse)
{
    extern __shared__ unsigned sh[];
    unsigned* running = sh;        // E
    unsigned* wc = sh + E;         // 8 E: each warp's count of each expert in the current tile
    for (unsigned e = threadIdx.x; e < E; e += blockDim.x) { running[e] = starts[(u64)blockIdx.x * E + e]; for (int w = 0; w < 8; ++w) wc[w * E + e] = 0; }
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const u64 first = (u64)blockIdx.x * CHUNK;
    for (u64 tile = first; tile < first + CHUNK && tile < n; tile += 256) {
        const u64 j = tile + threadIdx.x;
        const bool live = j < n && j < first + CHUNK;
        const unsigned e = live ? experts[j] : 0xffffffffu;
        const unsigned peers = __match_any_sync(0xffffffffu, e);
        const unsigned below = __popc(peers & ((1u << lane) - 1u));
        const bool leader = below == 0;
        if (live && leader) wc[warp * E + e] = __popc(peers);
        __syncthreads();
        if (live) {
            unsigned p = running[e] + below;
            for (int w = 0; w < warp; ++w) p += wc[w * E + e];
            order[p] = (unsigned)j; inverse[j] = p;
        }
        __syncthreads();
        if (live && leader) { atomicAdd(&running[e], __popc(peers)); wc[warp * E + e] = 0; }
        __syncthreads();
    }
}
// Gather: xp[p] = xs[order[p] / K], a thread per value or per float4
__global__ void moe_gather(const float* xs, const unsigned* order, float* xp, u64 n, u64 d, unsigned K, int vec)
{
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (vec) { const u64 q = d / 4; if (i >= n * q) return; const u64 p = i / q, c = i % q;
               reinterpret_cast<float4*>(xp)[i] = reinterpret_cast<const float4*>(xs)[(u64)(order[p] / K) * q + c]; return; }
    if (i >= n * d) return;
    const u64 p = i / d, c = i % d; xp[i] = xs[(u64)(order[p] / K) * d + c];
}
// The gather's backward: dxs_r = dxs_r + dxp[inverse[r K + s]] for s = 0 .. K-1, a thread per value
__global__ void moe_gather_backward(const float* dxp, const unsigned* inverse, float* dxs, u64 rows, u64 d, unsigned K)
{
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= rows * d) return;
    const u64 r = i / d, c = i % d; float a = dxs[i];
    for (unsigned s = 0; s < K; ++s) a = a + dxp[(u64)inverse[r * K + s] * d + c];
    dxs[i] = a;
}
// The combine: y = w_0 yp_0, then y = y + w_s yp_s in slot order, a thread per value
__global__ void moe_combine(const float* yp, const float* w, const unsigned* inverse, float* ys, u64 rows, u64 d, unsigned K)
{
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= rows * d) return;
    const u64 r = i / d, c = i % d;
    float y = w[r * K] * yp[(u64)inverse[r * K] * d + c];
    for (unsigned s = 1; s < K; ++s) { const float v = w[r * K + s] * yp[(u64)inverse[r * K + s] * d + c]; y = y + v; }
    ys[i] = y;
}
// The combine's backward: dyp[inverse[r K + s]] = w_rs dys_r, a thread per value
__global__ void moe_combine_backward_values(const float* dys, const float* w, const unsigned* inverse, float* dyp, u64 rows, u64 d, unsigned K)
{
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= rows * K * d) return;
    const u64 j = i / d, c = i % d, r = j / K;
    dyp[(u64)inverse[j] * d + c] = w[j] * dys[r * d + c];
}
// ... and dweights[j] = sum_c yp[inverse[j]][c] dys_r[c] from 0 in c order, a thread per pair
__global__ void moe_combine_backward_weights(const float* dys, const float* yp, const unsigned* inverse, float* dweights, u64 rows, u64 d, unsigned K)
{
    const u64 j = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (j >= rows * K) return;
    const float* a = yp + (u64)inverse[j] * d; const float* b = dys + (j / K) * d;
    float dot = 0.0f;
    for (u64 c = 0; c < d; ++c) { const float v = a[c] * b[c]; dot = dot + v; }
    dweights[j] = dot;
}
// The bias's gradient per segment, core's sum_rows: db[e r + i] += dys[t][i] for t in expert e's segment, in order;
// a thread per (expert, column)
__global__ void moe_segment_sums(const float* dys, const unsigned* off, float* db, unsigned E, u64 r)
{
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= E * r) return;
    const u64 e = i / r, c = i % r, base = off[0];
    float a = db[i];
    for (u64 t = off[e] - base; t < off[e + 1] - base; ++t) a += dys[t * r + c];
    db[i] = a;
}
// DeepSeek-V3's bias rule, a thread per expert: bias_e += gamma sign(total - E counts_e), in 64-bit integers
__global__ void moe_update_bias(const unsigned* counts, float* bias, unsigned E, float gamma)
{
    const unsigned e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= E) return;
    u64 total = 0; for (unsigned k = 0; k < E; ++k) total += counts[k];
    const u64 scaled = (u64)E * counts[e];
    if (scaled < total) bias[e] = bias[e] + gamma; else if (scaled > total) bias[e] = bias[e] - gamma;
}
// The balance loss's router probabilities, a thread per row: softmax(z), or for sigmoid scoring the sigmoids over
// their sum (divided, as core)
__global__ void moe_probabilities(const float* logits, float* P, u64 rows, u64 E, int sig)
{
    const u64 t = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (t >= rows) return;
    const float* z = logits + t * E; float* p = P + t * E;
    if (!sig) { const softmax_ q = softmax_of_(z, E); for (u64 e = 0; e < E; ++e) p[e] = prob_(z, e, q); return; }
    float sum = 0.0f; for (u64 e = 0; e < E; ++e) { p[e] = sigmoid_(z[e]); sum += p[e]; }
    for (u64 e = 0; e < E; ++e) p[e] = p[e] / sum;
}
// A thread per (group, expert): the group's probabilities summed in row order, and its count of the expert
__global__ void moe_group_sums(const float* P, const unsigned* experts, float* sum_p, unsigned* cnt, u64 groups, u64 n, u64 E, unsigned K)
{
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= groups * E) return;
    const u64 s = i / E, e = i % E;
    float a = 0.0f; for (u64 t = s * n; t < (s + 1) * n; ++t) a += P[t * E + e];
    unsigned c = 0; for (u64 j = s * n * K; j < (s + 1) * n * K; ++j) c += experts[j] == e ? 1u : 0u;
    sum_p[i] = a; cnt[i] = c;
}
// One thread: core's loss in its order (each group's dot over the experts, the groups in order, over the
// denominator), and each group's gradient in the probabilities, g[s E + e] = ((alpha E) f_e) / (n D)
__global__ void moe_balance_finish(const float* sum_p, const unsigned* cnt, float* g, float* loss, u64 groups, u64 E, float ae, float nk, float nf,
                                   float nden, float den)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    float l = 0.0f;
    for (u64 s = 0; s < groups; ++s) {
        float dot = 0.0f;
        for (u64 e = 0; e < E; ++e) { const float f = (float)cnt[s * E + e] / nk; dot += f * (sum_p[s * E + e] / nf); if (g) g[s * E + e] = ae * f / nden; }
        l += ae * dot;
    }
    *loss = l / den;
}
// A thread per row: the balance gradient taken through the probabilities, added into dz (core's probabilities_backward)
__global__ void moe_balance_backward(const float* logits, const float* P, const float* g, float* dlogits, u64 rows, u64 n, u64 E, int sig)
{
    const u64 t = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (t >= rows) return;
    const float* z = logits + t * E; const float* p = P + t * E; const float* gs = g + (t / n) * E; float* dz = dlogits + t * E;
    float dot = 0.0f; for (u64 e = 0; e < E; ++e) dot += p[e] * gs[e];
    if (!sig) { for (u64 e = 0; e < E; ++e) dz[e] += p[e] * (gs[e] - dot); return; }
    float sum = 0.0f; for (u64 e = 0; e < E; ++e) sum += sigmoid_(z[e]);
    for (u64 e = 0; e < E; ++e) dz[e] += (gs[e] - dot) / sum * sigmoid_derivative_(z[e]);
}
// The z-loss, a thread per row: lse (core's logsumexp: max + log(sum e^(z - max)), the max itself when it is -inf or
// NaN), its square kept for the sum, and the gradient ((2 beta) lse / D) softmax(z) added into dz
__global__ void moe_z_rows(const float* logits, float* dlogits, float* sq, u64 rows, u64 E, float twobeta, float den)
{
    const u64 t = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (t >= rows) return;
    const float* z = logits + t * E;
    const float m = max_of_(z, E);
    float s = 1.0f;
    if (!(m == -INFINITY || m != m)) { s = 0.0f; for (u64 e = 0; e < E; ++e) s += expf(z[e] - m); }
    const float lse = m + logf(s);
    sq[t] = lse * lse;
    if (!dlogits) return;
    const softmax_ q = softmax_of_(z, E);
    const float c = twobeta * lse / den;
    for (u64 e = 0; e < E; ++e) dlogits[t * E + e] += c * prob_(z, e, q);
}
__global__ void moe_z_finish(const float* sq, u64 rows, float beta, float den, float* loss)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    float s = 0.0f; for (u64 t = 0; t < rows; ++t) s += sq[t];
    *loss = beta * s / den;
}
}
// Expert parallelism (ml/expert_parallel.hpp): tables of up to 64 ranks' pointers, by value
#define MAX_RANKS 64
struct ptrs_ { const unsigned* p[MAX_RANKS]; };
struct bptrs_ { const void* p[MAX_RANKS]; };
struct wptrs_ { void* p[MAX_RANKS]; u64 size[MAX_RANKS]; };
extern "C" {
__global__ void moe_send_offsets(const unsigned* offsets, unsigned* send, unsigned ranks, unsigned per)
{ const unsigned r = blockIdx.x * blockDim.x + threadIdx.x; if (r <= ranks) send[r] = offsets[r * per]; }
__global__ void moe_receive_offsets(ptrs_ counts, unsigned ranks, unsigned first, unsigned local, unsigned* out)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    out[0] = 0;
    for (unsigned s = 0; s < ranks; ++s) { unsigned n = 0; for (unsigned j = 0; j < local; ++j) n += counts.p[s][first + j]; out[s + 1] = out[s] + n; }
}
// A block per (local expert j, source s): its run of the order, (expert, source, row, slot), in core's arithmetic
__global__ void moe_receive_layout(ptrs_ counts, unsigned ranks, unsigned first, unsigned local, unsigned* expert_offsets, unsigned* order)
{
    const unsigned j = blockIdx.x / ranks, s = blockIdx.x % ranks;
    __shared__ unsigned p0, base, cnt;
    if (threadIdx.x == 0) {
        unsigned p = 0;
        for (unsigned jj = 0; jj < j; ++jj) for (unsigned ss = 0; ss < ranks; ++ss) p += counts.p[ss][first + jj];
        for (unsigned ss = 0; ss < s; ++ss) p += counts.p[ss][first + j];
        unsigned source_start = 0;
        for (unsigned ss = 0; ss < s; ++ss) for (unsigned i = 0; i < local; ++i) source_start += counts.p[ss][first + i];
        unsigned within = 0; for (unsigned i = 0; i < j; ++i) within += counts.p[s][first + i];
        p0 = p; base = source_start + within; cnt = counts.p[s][first + j];
        if (s == 0) expert_offsets[j] = p;
        if (j + 1 == local && s + 1 == ranks) expert_offsets[local] = p + cnt;
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < cnt; i += blockDim.x) order[p0 + i] = base + i;
}
__global__ void moe_to_bf16(const float* in, unsigned short* out, u64 n)
{
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= n) return;
    const unsigned u = __float_as_uint(in[i]);   // core's to_bfloat16: a NaN keeps its sign and is made quiet
    out[i] = (u & 0x7fffffffu) > 0x7f800000u ? (unsigned short)((u >> 16) | 0x40u) : (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__global__ void moe_to_float(const unsigned short* in, float* out, u64 n)
{ const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i < n) out[i] = __uint_as_float((unsigned)in[i] << 16); }
}
// Destination r (grid y) receives, in source order, each source's rows [so[s][r], so[s][r + 1]): a thread per element
template <typename U>
__device__ void a2a_(bptrs_ send, ptrs_ so, unsigned ranks, u64 width, wptrs_ recv)
{
    const unsigned r = blockIdx.y; const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i >= recv.size[r]) return;
    const u64 q = i / width, c = i % width;
    u64 at = 0;
    for (unsigned s = 0; s < ranks; ++s) {
        const u64 len = so.p[s][r + 1] - so.p[s][r];
        if (q < at + len) { reinterpret_cast<U*>(recv.p[r])[i] = reinterpret_cast<const U*>(send.p[s])[(so.p[s][r] + q - at) * width + c]; return; }
        at += len;
    }
}
template <typename U>
__device__ void permute_(const U* from, const unsigned* order, U* to, u64 rows, u64 width, int inverse)
{
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= rows * width) return;
    const u64 p = i / width, c = i % width;
    if (inverse) to[(u64)order[p] * width + c] = from[i]; else to[i] = from[(u64)order[p] * width + c];
}
#define SNICHOLLS_UNIT(N, U)                                                                                                         \
    extern "C" __global__ void moe_all_to_all_##N(bptrs_ send, ptrs_ so, unsigned ranks, u64 width, wptrs_ recv) { a2a_<U>(send, so, ranks, width, recv); } \
    extern "C" __global__ void moe_permute_##N(const void* from, const unsigned* order, void* to, u64 rows, u64 width, int inverse)  \
    { permute_<U>(reinterpret_cast<const U*>(from), order, reinterpret_cast<U*>(to), rows, width, inverse); }
SNICHOLLS_UNIT(1, unsigned char)
SNICHOLLS_UNIT(2, unsigned short)
SNICHOLLS_UNIT(4, unsigned)
SNICHOLLS_UNIT(8, u64)
// The grouped product over expert segments whose bounds are in device memory. Operands are row-major logical
// matrices given by (pointer, leading dimension, transposed); C (m x n) = alpha A (m x kk) B (kk x n) + beta C, plus a
// bias row. A block makes a 128 x 128 tile of C with 8 warps (2 x 4, 64 x 32 each) on the tensor cores: mode 0 TF32,
// 1 bfloat16, 2 three TF32 products (float's accuracy). Tiles of k are fetched into registers (float4 along the
// contiguous dimension) while the previous is multiplied, then stored to shared memory as 32-bit words [k][row]: a
// TF32 value, or two bfloat16 adjacent in k, so the fragments of m16n8k8 and m16n8k16 are read the same way.
#define SG_BM 128
#define SG_BN 128
#define SG_BKW 16
__device__ __forceinline__ unsigned sg_tf32(float x) { unsigned r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
__device__ __forceinline__ unsigned sg_bf2(float lo, float hi) { unsigned r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ void sg_mma_tf32(float* c, const unsigned* a, unsigned b0, unsigned b1)
{
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void sg_mma_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <int MODE>
struct sg_frag {
    static constexpr int KV = MODE == 1 ? 2 * SG_BKW : SG_BKW;   // k values per tile
    static constexpr int N4 = SG_BM * KV / 4 / 256;              // float4s per thread
    float4 v[N4];
};
// A tile of a logical (rows x kk) operand into registers, zero outside (rows, kk); VEC: float4 loads (the contiguous
// dimension a multiple of 4, the pointer 16-byte aligned)
template <int MODE, int VEC>
__device__ __forceinline__ void sg_fetch(sg_frag<MODE>& f, const float* p, u64 ld, int tr, u64 r0, u64 rows, u64 k0, u64 kk)
{
    constexpr int KV = sg_frag<MODE>::KV;
#pragma unroll
    for (int x = 0; x < sg_frag<MODE>::N4; ++x) {
        const int i = threadIdx.x + 256 * x;
        float* o = &f.v[x].x;
        if (!tr) {   // k contiguous: four k of one row
            const int r = i / (KV / 4), k4 = i % (KV / 4); const u64 gr = r0 + r, gk = k0 + 4 * k4;
            if (VEC) f.v[x] = gr < rows && gk < kk ? *reinterpret_cast<const float4*>(p + gr * ld + gk) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            else
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = gr < rows && gk + j < kk ? p[gr * ld + gk + j] : 0.0f;
        } else {     // rows contiguous: four rows of one k
            const int r4 = i % (SG_BM / 4), k = i / (SG_BM / 4); const u64 gr = r0 + 4 * r4, gk = k0 + k;
            if (VEC) f.v[x] = gr < rows && gk < kk ? *reinterpret_cast<const float4*>(p + gk * ld + gr) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            else
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = gr + j < rows && gk < kk ? p[gk * ld + gr + j] : 0.0f;
        }
    }
}
// Shared memory holds each operand's tile in the order its fragments are read, so a lane reads its fragments with
// 128-bit loads: word (row r, k word k) of A at sg_a(r, k) (a 16 x 8 block's four words of one lane together: a0..a3
// of m16n8k8, or of m16n8k16 in k pairs), of B at sg_b(c, k) (two 8-column blocks' b0, b1 together)
__device__ __forceinline__ int sg_a(int r, int k)
{ const int kk = k & 7, rr = r & 15; return ((((k >> 3) * 8 + (r >> 4)) * 32 + (rr & 7) * 4 + (kk & 3)) << 2) + (rr >> 3) + 2 * (kk >> 2); }
__device__ __forceinline__ int sg_b(int c, int k)
{ const int kk = k & 7; return ((((k >> 3) * 8 + (c >> 4)) * 32 + (c & 7) * 4 + (kk & 3)) << 2) + ((c >> 3) & 1) * 2 + (kk >> 2); }
template <int MODE, int IS_A>
__device__ __forceinline__ void sg_store(const sg_frag<MODE>& f, unsigned* w, unsigned* lo, int tr)
{
    constexpr int KV = sg_frag<MODE>::KV;
#pragma unroll
    for (int x = 0; x < sg_frag<MODE>::N4; ++x) {
        const int i = threadIdx.x + 256 * x;
        const float* o = &f.v[x].x;
        if (!tr) {
            const int r = i / (KV / 4), k4 = i % (KV / 4);
            if (MODE == 1) { w[IS_A ? sg_a(r, 2 * k4) : sg_b(r, 2 * k4)] = sg_bf2(o[0], o[1]); w[IS_A ? sg_a(r, 2 * k4 + 1) : sg_b(r, 2 * k4 + 1)] = sg_bf2(o[2], o[3]); }
            else
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int at = IS_A ? sg_a(r, 4 * k4 + j) : sg_b(r, 4 * k4 + j);
                    const unsigned big = sg_tf32(o[j]); w[at] = big; if (MODE == 2) lo[at] = sg_tf32(o[j] - __uint_as_float(big));
                }
        } else {
            const int r4 = i % (SG_BM / 4), k = i / (SG_BM / 4);
            if (MODE == 1) {   // k even into a word's low half, odd into its high half
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    reinterpret_cast<unsigned short*>(w)[2 * (IS_A ? sg_a(4 * r4 + j, k / 2) : sg_b(4 * r4 + j, k / 2)) + (k & 1)] = (unsigned short)(sg_bf2(o[j], 0.0f) & 0xffffu);
            } else
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int at = IS_A ? sg_a(4 * r4 + j, k) : sg_b(4 * r4 + j, k);
                    const unsigned big = sg_tf32(o[j]); w[at] = big; if (MODE == 2) lo[at] = sg_tf32(o[j] - __uint_as_float(big));
                }
        }
    }
}
// The block's tile of C: rows [m0, m0 + 128) of A (m x kk), columns [n0, n0 + 128) of B (kk x n); bias (n) or null
template <int MODE, int VA, int VB>
__device__ void sg_tile(const float* A, u64 lda, int ta, const float* B, u64 ldb, int tb, float* C, u64 ldc, const float* bias, u64 m0, u64 m, u64 n0,
                        u64 n, u64 kk, float beta)
{
    __shared__ __align__(16) unsigned as[SG_BKW * SG_BM], bs[SG_BKW * SG_BN];
    __shared__ __align__(16) unsigned al[MODE == 2 ? SG_BKW * SG_BM : 4], bl[MODE == 2 ? SG_BKW * SG_BN : 4];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, q4 = lane & 3;
    const int wm = (warp / 4) * 64, wn = (warp % 4) * 32;
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
    constexpr int KV = sg_frag<MODE>::KV;
    sg_frag<MODE> fa, fb;
    // A as (m x kk); B as (n x kk), that is Bᵀ
    if (kk > 0) { sg_fetch<MODE, VA>(fa, A, lda, ta, m0, m, 0, kk); sg_fetch<MODE, VB>(fb, B, ldb, !tb, n0, n, 0, kk); }
    for (u64 k0 = 0; k0 < kk; k0 += KV) {
        __syncthreads();
        sg_store<MODE, 1>(fa, as, al, ta);
        sg_store<MODE, 0>(fb, bs, bl, !tb);
        __syncthreads();
        if (k0 + KV < kk) { sg_fetch<MODE, VA>(fa, A, lda, ta, m0, m, k0 + KV, kk); sg_fetch<MODE, VB>(fb, B, ldb, !tb, n0, n, k0 + KV, kk); }
        // Under float (mode 2) a tile's products are accumulated apart and then added in float: the tensor cores'
        // own accumulation does not round to nearest, which over a long reduction (a weight gradient's segment) costs digits
        float part[MODE == 2 ? 4 : 1][4][4];
        if (MODE == 2)
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) part[i][j][0] = part[i][j][1] = part[i][j][2] = part[i][j][3] = 0.0f;
#pragma unroll
        for (int s = 0; s < SG_BKW / 8; ++s) {
            unsigned a[4][4], b[4][2];
#pragma unroll
            for (int i = 0; i < 4; ++i) { const uint4 v = reinterpret_cast<const uint4*>(as)[(s * 8 + wm / 16 + i) * 32 + lane]; a[i][0] = v.x; a[i][1] = v.y; a[i][2] = v.z; a[i][3] = v.w; }
#pragma unroll
            for (int j = 0; j < 2; ++j) { const uint4 v = reinterpret_cast<const uint4*>(bs)[(s * 8 + wn / 16 + j) * 32 + lane]; b[2 * j][0] = v.x; b[2 * j][1] = v.y; b[2 * j + 1][0] = v.z; b[2 * j + 1][1] = v.w; }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (MODE == 1) sg_mma_bf16(acc[i][j], a[i], b[j][0], b[j][1]);
                    else if (MODE == 0) sg_mma_tf32(acc[i][j], a[i], b[j][0], b[j][1]);
                    else sg_mma_tf32(part[i][j], a[i], b[j][0], b[j][1]);
                }
            if (MODE == 2) {   // the small parts: a_lo b_hi + a_hi b_lo
                unsigned x[4][4], y[4][2];
#pragma unroll
                for (int i = 0; i < 4; ++i) { const uint4 v = reinterpret_cast<const uint4*>(al)[(s * 8 + wm / 16 + i) * 32 + lane]; x[i][0] = v.x; x[i][1] = v.y; x[i][2] = v.z; x[i][3] = v.w; }
#pragma unroll
                for (int j = 0; j < 2; ++j) { const uint4 v = reinterpret_cast<const uint4*>(bl)[(s * 8 + wn / 16 + j) * 32 + lane]; y[2 * j][0] = v.x; y[2 * j][1] = v.y; y[2 * j + 1][0] = v.z; y[2 * j + 1][1] = v.w; }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) { sg_mma_tf32(part[i][j], x[i], b[j][0], b[j][1]); sg_mma_tf32(part[i][j], a[i], y[j][0], y[j][1]); }
            }
        }
        if (MODE == 2)
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[i][j][e] += part[MODE == 2 ? i : 0][j][e];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const u64 r = m0 + wm + 16 * i + g + 8 * (e / 2), c = n0 + wn + 8 * j + 2 * q4 + (e % 2);
                if (r < m && c < n) {
                    float* o = C + r * ldc + c;
                    float v = acc[i][j][e];
                    if (beta != 0.0f) v = v + beta * *o;   // beta is 0 or 1: C += A B
                    if (bias) v = v + bias[c];
                    *o = v;
                }
            }
}
// Rows varying: rows [off[e] - off[0], off[e + 1] - off[0]) of A and C are expert e's; B is expert e's matrix
// (B + e b_stride), bias expert e's row (bias + e n). Grid (n tiles, an upper bound of the row tiles): block y finds its
// segment from the offsets; past the last it returns
template <int MODE, int V>
__device__ void sg_rows(const float* A, u64 lda, const float* B, u64 ldb, int tb, u64 b_stride, float* C, u64 ldc, const float* bias, const unsigned* off,
                        unsigned E, u64 n, u64 kk, float beta)
{
    __shared__ int seg, first;
    if (threadIdx.x == 0) {
        int t = blockIdx.y; seg = -1;
        for (unsigned e = 0; e < E; ++e) { const int tiles = (int)((off[e + 1] - off[e] + SG_BM - 1) / SG_BM); if (t < tiles) { seg = (int)e; first = t; break; } t -= tiles; }
    }
    __syncthreads();
    if (seg < 0) return;
    const u64 r0 = off[seg] - off[0];
    sg_tile<MODE, V, V>(A + r0 * lda, lda, 0, B + seg * b_stride, ldb, tb, C + r0 * ldc, ldc, bias ? bias + seg * n : 0, (u64)first * SG_BM,
                        off[seg + 1] - off[seg], (u64)blockIdx.x * SG_BN, n, kk, beta);
}
// The reduction varying: C_e (m x n) = A_eᵀ B_e + beta C_e over expert e's rows, A_e (rows x m) and B_e (rows x n);
// grid (n tiles, m tiles, E)
template <int MODE, int V>
__device__ void sg_reduce(const float* A, u64 lda, const float* B, u64 ldb, float* C, u64 ldc, u64 c_stride, const unsigned* off, u64 m, u64 n, float beta)
{
    const unsigned e = blockIdx.z; const u64 r0 = off[e] - off[0], rows = off[e + 1] - off[e];
    if (rows == 0) return;   // an empty segment leaves its gradient as it is, as core skips it
    sg_tile<MODE, V, V>(A + r0 * lda, lda, 1, B + r0 * ldb, ldb, 0, C + e * c_stride, ldc, 0, (u64)blockIdx.y * SG_BM, m, (u64)blockIdx.x * SG_BN, n,
                        rows, beta);
}
#define SNICHOLLS_SEG(MODE, V)                                                                                                        \
    extern "C" __global__ void __launch_bounds__(256) seg_rows_##MODE##_##V(const float* A, u64 lda, const float* B, u64 ldb, int tb, u64 b_stride, \
        float* C, u64 ldc, const float* bias, const unsigned* off, unsigned E, u64 n, u64 kk, float beta)                             \
    { sg_rows<MODE, V>(A, lda, B, ldb, tb, b_stride, C, ldc, bias, off, E, n, kk, beta); }                                           \
    extern "C" __global__ void __launch_bounds__(256) seg_reduce_##MODE##_##V(const float* A, u64 lda, const float* B, u64 ldb, float* C, u64 ldc, \
        u64 c_stride, const unsigned* off, u64 m, u64 n, float beta)                                                                  \
    { sg_reduce<MODE, V>(A, lda, B, ldb, C, ldc, c_stride, off, m, n, beta); }
SNICHOLLS_SEG(0, 0)
SNICHOLLS_SEG(0, 1)
SNICHOLLS_SEG(1, 0)
SNICHOLLS_SEG(1, 1)
SNICHOLLS_SEG(2, 0)
SNICHOLLS_SEG(2, 1)
)CU";

    inline constexpr unsigned chunk = 1024;        // pairs per block of the permutation: as CHUNK in the source
    inline constexpr unsigned tile = 128;          // rows of a grouped product's tile: as SG_BM and SG_BN
    inline constexpr std::size_t max_experts = 1024;
    inline constexpr std::size_t max_ranks = 64;   // as MAX_RANKS in the source
    struct ptrs { const std::uint32_t* p[max_ranks]; };
    struct bptrs { const void* p[max_ranks]; };
    struct wptrs { void* p[max_ranks]; unsigned long long size[max_ranks]; };

} // namespace detail::cuda_moe

/// \brief How `grouped_linear` and its backward run on the device.
enum class grouped_path {
    kernel,   ///< One grouped kernel per product, the segments read from device memory: nothing waits, a graph can capture it.
    cublas,   ///< The offsets copied to the host (one wait per call), then each expert's product through cuBLAS, as core's
              ///< reference loops them. Not capturable.
};

/// \brief The mixture-of-experts kernels, compiled for one context's device, and their working space.
class cuda_moe_kernels {
public:
    /// \brief Compiles the kernels (NVRTC) for the context's device and loads them.
    /// \param context The device; must outlive the kernels.
    /// \return The kernels, or why they did not compile (the log on stderr).
    [[nodiscard]] static result<std::unique_ptr<cuda_moe_kernels>, cuda_failure> create(const cuda_context& context)
    {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, context.device());
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, context.device());
        nvrtcProgram prog = nullptr;
        if (const nvrtcResult r = nvrtcCreateProgram(&prog, detail::cuda_moe::source, "snicholls_gpu_moe.cu", 0, nullptr, nullptr); r != NVRTC_SUCCESS) {
            std::fprintf(stderr, "snicholls::gpu::cuda_moe_kernels: nvrtcCreateProgram: %s\n", nvrtcGetErrorString(r));
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(r)});
        }
        const std::string arch = "--gpu-architecture=sm_" + std::to_string(major * 10 + minor);
        const char* options[] = {arch.c_str(), "--fmad=false", "--std=c++17"};
        const nvrtcResult compiled = nvrtcCompileProgram(prog, 3, options);
        if (compiled != NVRTC_SUCCESS) {
            std::size_t size = 0;
            nvrtcGetProgramLogSize(prog, &size);
            std::string log(size, '\0');
            nvrtcGetProgramLog(prog, log.data());
            std::fprintf(stderr, "snicholls::gpu::cuda_moe_kernels: %s\n", log.c_str());
            nvrtcDestroyProgram(&prog);
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(compiled)});
        }
        std::size_t size = 0;
        nvrtcGetCUBINSize(prog, &size);
        std::string cubin(size, '\0');
        nvrtcGetCUBIN(prog, cubin.data());
        nvrtcDestroyProgram(&prog);
        std::unique_ptr<cuda_moe_kernels> k(new cuda_moe_kernels(context));
        if (const cudaError_t e = cudaLibraryLoadData(&k->library_, cubin.data(), nullptr, nullptr, 0, nullptr, nullptr, 0); e != cudaSuccess) {
            std::fprintf(stderr, "snicholls::gpu::cuda_moe_kernels: cudaLibraryLoadData: %s\n", cudaGetErrorString(e));
            return err(cuda_failure{cuda_error::call_failed, static_cast<int>(e)});
        }
        std::vector<std::pair<std::string, cudaKernel_t*>> all = {
            {"moe_route", &k->route_}, {"moe_route_backward", &k->route_backward_}, {"moe_count", &k->count_}, {"moe_offsets", &k->offsets_},
            {"moe_chunk_counts", &k->chunk_counts_}, {"moe_chunk_starts", &k->chunk_starts_}, {"moe_rank", &k->rank_}, {"moe_gather", &k->gather_},
            {"moe_gather_backward", &k->gather_backward_}, {"moe_combine", &k->combine_}, {"moe_combine_backward_values", &k->combine_backward_values_},
            {"moe_combine_backward_weights", &k->combine_backward_weights_}, {"moe_segment_sums", &k->segment_sums_},
            {"moe_update_bias", &k->update_bias_}, {"moe_probabilities", &k->probabilities_}, {"moe_group_sums", &k->group_sums_},
            {"moe_balance_finish", &k->balance_finish_}, {"moe_balance_backward", &k->balance_backward_}, {"moe_z_rows", &k->z_rows_},
            {"moe_z_finish", &k->z_finish_}, {"moe_send_offsets", &k->send_offsets_}, {"moe_receive_offsets", &k->receive_offsets_},
            {"moe_receive_layout", &k->receive_layout_}, {"moe_to_bf16", &k->to_bf16_}, {"moe_to_float", &k->to_float_}};
        for (int u = 0; u < 4; ++u) {
            const std::string n = std::to_string(1 << u);
            all.emplace_back("moe_all_to_all_" + n, &k->all_to_all_[u]);
            all.emplace_back("moe_permute_" + n, &k->permute_[u]);
        }
        for (int mode = 0; mode < 3; ++mode)
            for (int v = 0; v < 2; ++v) {
                const std::string s = std::to_string(mode) + "_" + std::to_string(v);
                all.emplace_back("seg_rows_" + s, &k->seg_rows_[mode][v]);
                all.emplace_back("seg_reduce_" + s, &k->seg_reduce_[mode][v]);
            }
        for (const auto& [name, slot] : all)
            if (const cudaError_t e = cudaLibraryGetKernel(slot, k->library_, name.c_str()); e != cudaSuccess) {
                std::fprintf(stderr, "snicholls::gpu::cuda_moe_kernels: cudaLibraryGetKernel(%s): %s\n", name.c_str(), cudaGetErrorString(e));
                return err(cuda_failure{cuda_error::call_failed, static_cast<int>(e)});
            }
        k->losses_ = std::make_unique<device_buffer<float>>(context, 2);
        return ok(std::move(k));
    }

    /// \brief Not copyable.
    cuda_moe_kernels(const cuda_moe_kernels&) = delete;
    /// \brief Not assignable. \return Nothing; deleted.
    cuda_moe_kernels& operator=(const cuda_moe_kernels&) = delete;
    /// \brief Unloads the kernels.
    ~cuda_moe_kernels()
    {
        if (library_ != nullptr)
            (void)cudaLibraryUnload(library_);
    }

    grouped_path path = grouped_path::kernel;   ///< How the grouped products run.
    bool losses_on_device = false;              ///< The losses never waited for: written to `losses()`, NaN returned.

    /// \brief Where the balance loss (0) and the z-loss (1) are written when they are not returned (in a capture, or
    /// with `losses_on_device`). \return Two floats of device memory.
    [[nodiscard]] std::span<float> losses() const noexcept { return losses_->span(); }

    /// \brief Launches a kernel on the context's stream. \param k The kernel. \param grid The blocks. \param block The
    /// threads per block. \param shared Dynamic shared memory in bytes. \param args The arguments.
    /// \throws cuda_exception if the launch fails.
    void launch(cudaKernel_t k, dim3 grid, unsigned block, std::size_t shared, void** args) const
    {
        if (grid.x == 0 || grid.y == 0 || grid.z == 0)
            return;
        if (const cudaError_t e = cudaLaunchKernel(reinterpret_cast<const void*>(k), grid, dim3(block), args, shared, context_->stream()); e != cudaSuccess)
            throw cuda_exception(std::string("cudaLaunchKernel(moe) [") + cudaGetErrorString(e) + "]", static_cast<int>(e));
    }
    /// \brief Launches a kernel over `n` threads in blocks of 256. \param k The kernel. \param n The threads.
    /// \param args The arguments. \throws cuda_exception if the launch fails.
    void launch(cudaKernel_t k, std::size_t n, void** args) const { launch(k, dim3(unsigned((n + 255) / 256)), 256, 0, args); }

    /// \brief 32-bit working space of at least `n` values, kept between calls. \param n The size. \return The space.
    [[nodiscard]] std::span<std::uint32_t> scratch_u32(std::size_t n) const
    {
        if (!u32_ || u32_->size() < n)
            u32_ = std::make_unique<device_buffer<std::uint32_t>>(*context_, n);
        return u32_->span().first(n);
    }
    /// \brief Float working space of at least `n` values, kept between calls. \param n The size. \return The space.
    [[nodiscard]] std::span<float> scratch(std::size_t n) const
    {
        if (!f32_ || f32_->size() < n)
            f32_ = std::make_unique<device_buffer<float>>(*context_, n);
        return f32_->span().first(n);
    }

    /// \brief The context. \return It.
    [[nodiscard]] const cuda_context& context() const noexcept { return *context_; }

    /// \name The kernels, for the overloads below
    /// @{
    cudaKernel_t route_ = nullptr;                      ///< `moe_route`.
    cudaKernel_t route_backward_ = nullptr;             ///< `moe_route_backward`.
    cudaKernel_t count_ = nullptr;                      ///< `moe_count`.
    cudaKernel_t offsets_ = nullptr;                    ///< `moe_offsets`.
    cudaKernel_t chunk_counts_ = nullptr;               ///< `moe_chunk_counts`.
    cudaKernel_t chunk_starts_ = nullptr;               ///< `moe_chunk_starts`.
    cudaKernel_t rank_ = nullptr;                       ///< `moe_rank`.
    cudaKernel_t gather_ = nullptr;                     ///< `moe_gather`.
    cudaKernel_t gather_backward_ = nullptr;            ///< `moe_gather_backward`.
    cudaKernel_t combine_ = nullptr;                    ///< `moe_combine`.
    cudaKernel_t combine_backward_values_ = nullptr;    ///< `moe_combine_backward_values`.
    cudaKernel_t combine_backward_weights_ = nullptr;   ///< `moe_combine_backward_weights`.
    cudaKernel_t segment_sums_ = nullptr;               ///< `moe_segment_sums`.
    cudaKernel_t update_bias_ = nullptr;                ///< `moe_update_bias`.
    cudaKernel_t probabilities_ = nullptr;              ///< `moe_probabilities`.
    cudaKernel_t group_sums_ = nullptr;                 ///< `moe_group_sums`.
    cudaKernel_t balance_finish_ = nullptr;             ///< `moe_balance_finish`.
    cudaKernel_t balance_backward_ = nullptr;           ///< `moe_balance_backward`.
    cudaKernel_t z_rows_ = nullptr;                     ///< `moe_z_rows`.
    cudaKernel_t z_finish_ = nullptr;                   ///< `moe_z_finish`.
    cudaKernel_t send_offsets_ = nullptr;               ///< `moe_send_offsets`.
    cudaKernel_t receive_offsets_ = nullptr;            ///< `moe_receive_offsets`.
    cudaKernel_t receive_layout_ = nullptr;             ///< `moe_receive_layout`.
    cudaKernel_t to_bf16_ = nullptr;                    ///< `moe_to_bf16`.
    cudaKernel_t to_float_ = nullptr;                   ///< `moe_to_float`.
    cudaKernel_t all_to_all_[4] = {};                   ///< `moe_all_to_all_N`, elements of 1, 2, 4 and 8 bytes.
    cudaKernel_t permute_[4] = {};                      ///< `moe_permute_N`, elements of 1, 2, 4 and 8 bytes.
    cudaKernel_t seg_rows_[3][2] = {};                  ///< `seg_rows_M_V`: the grouped product, rows varying, per mode and vector loads.
    cudaKernel_t seg_reduce_[3][2] = {};                ///< `seg_reduce_M_V`: the weight gradient, the reduction varying.
    /// @}

private:
    explicit cuda_moe_kernels(const cuda_context& c) : context_(&c) {}
    const cuda_context* context_;
    cudaLibrary_t library_ = nullptr;
    std::unique_ptr<device_buffer<float>> losses_;
    mutable std::unique_ptr<device_buffer<std::uint32_t>> u32_;
    mutable std::unique_ptr<device_buffer<float>> f32_;
};

namespace detail::cuda_moe {
    using u64 = unsigned long long;
    using cuda_ops::arg;

    inline const cuda_moe_kernels& kernels(const cuda_training_backend& b)
    {
        if (b.moe == nullptr)
            throw std::logic_error("snicholls::gpu: a mixture-of-experts step needs cuda_training_backend::moe (cuda_moe_kernels)");
        return *b.moe;
    }
    template <ml::transformer_config C>
    constexpr int scoring = C.scoring == ml::moe_scoring::softmax_top_k ? 0 : C.scoring == ml::moe_scoring::softmax ? 1 : 2;
    template <ml::transformer_config C>
    void check_experts()
    {
        static_assert(C.experts > 0, "snicholls::gpu: a mixture-of-experts step needs a configuration with experts");
        if constexpr (C.experts > max_experts)
            throw std::invalid_argument("snicholls::gpu: more than 1024 experts");
    }
    inline bool capturing(const cuda_training_backend& b)
    {
        cudaStreamCaptureStatus s = cudaStreamCaptureStatusNone;
        cuda::check(cudaStreamIsCapturing(b.context->stream(), &s), "cudaStreamIsCapturing");
        return s != cudaStreamCaptureStatusNone;
    }
    // A loss written at slot i of losses(): returned (one wait) unless it stays on the device
    inline float loss_result(const cuda_training_backend& b, int i)
    {
        const auto& k = kernels(b);
        if (k.losses_on_device || capturing(b))
            return std::numeric_limits<float>::quiet_NaN();
        float v = 0.0f;
        cuda::check(cudaMemcpyAsync(&v, k.losses().data() + i, sizeof(float), cudaMemcpyDeviceToHost, b.context->stream()), "cudaMemcpyAsync");
        cuda::check(cudaStreamSynchronize(b.context->stream()), "cudaStreamSynchronize");
        return v;
    }
    // The grouped kernels' mode for the backend's math: TF32, bfloat16, or three TF32 products for float
    inline int mode(const cuda_training_backend& b) noexcept { return b.math == cuda_math::tf32 ? 0 : b.math == cuda_math::bf16 ? 1 : 2; }
    inline bool aligned16(const void* p) noexcept { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }
    // C = A B (+ C) (+ bias) over the segments, rows varying: A rows x kk (row-major, lda), B expert e's (kk x n) from
    // B + e b_stride, stored transposed when tb
    inline void seg_rows(const cuda_training_backend& b, const float* A, u64 lda, const float* B, u64 ldb, int tb, u64 b_stride, float* C, u64 ldc,
                         const float* bias, const std::uint32_t* off, unsigned E, std::size_t rows, u64 n, u64 kk, float beta)
    {
        const auto& k = kernels(b);
        const int v = lda % 4 == 0 && ldb % 4 == 0 && aligned16(A) && aligned16(B) && b_stride % 4 == 0 ? 1 : 0;
        const unsigned ytiles = unsigned((rows + tile - 1) / tile + E);
        void* args[] = {&A, &lda, &B, &ldb, &tb, &b_stride, &C, &ldc, &bias, &off, &E, &n, &kk, &beta};
        k.launch(k.seg_rows_[mode(b)][v], dim3(unsigned((n + tile - 1) / tile), ytiles), 256, 0, args);
    }
    // C_e = A_eᵀ B_e + beta C_e over the segments, the reduction varying
    inline void seg_reduce(const cuda_training_backend& b, const float* A, u64 lda, const float* B, u64 ldb, float* C, u64 ldc, u64 c_stride,
                           const std::uint32_t* off, unsigned E, u64 m, u64 n, float beta)
    {
        const auto& k = kernels(b);
        const int v = lda % 4 == 0 && ldb % 4 == 0 && aligned16(A) && aligned16(B) ? 1 : 0;
        void* args[] = {&A, &lda, &B, &ldb, &C, &ldc, &c_stride, &off, &m, &n, &beta};
        k.launch(k.seg_reduce_[mode(b)][v], dim3(unsigned((n + tile - 1) / tile), unsigned((m + tile - 1) / tile), E), 256, 0, args);
    }
    // The offsets on the host, for grouped_path::cublas (one wait)
    inline std::vector<std::uint32_t> host_offsets(const cuda_training_backend& b, std::span<const std::uint32_t> offsets)
    {
        if (capturing(b))
            throw std::logic_error("snicholls::gpu: grouped_path::cublas reads the offsets on the host, which a graph capture cannot");
        std::vector<std::uint32_t> h(offsets.size());
        cuda::check(cudaMemcpyAsync(h.data(), offsets.data(), h.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, b.context->stream()),
                    "cudaMemcpyAsync");
        cuda::check(cudaStreamSynchronize(b.context->stream()), "cudaStreamSynchronize");
        return h;
    }
} // namespace detail::cuda_moe

/// \name core's mixture-of-experts steps on the device, for `cuda_training_backend` (found by argument-dependent lookup)
/// @{

/// \brief `ml::moe_route_rows` on the device: a thread per row. \tparam C A mixture-of-experts configuration.
/// \param b The backend (with `moe`). \param logits `rows · E`. \param bias `E`, or empty. \param experts Receives `rows · K`.
/// \param weights Receives `rows · K`.
template <ml::transformer_config C>
void moe_route_rows(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> logits, std::type_identity_t<std::span<const float>> bias,
                    std::span<std::uint32_t> experts, std::span<float> weights)
{
    namespace dm = detail::cuda_moe;
    dm::check_experts<C>();
    const float* z = logits.data();
    const float* bp = bias.empty() ? nullptr : bias.data();
    std::uint32_t* ep = experts.data();
    float* wp = weights.data();
    const dm::u64 rows = logits.size() / C.experts, E = C.experts;
    int K = int(C.top_k), scoring = dm::scoring<C>;
    float scale = ml::route_scale_v<C, float>;
    void* args[] = {&z, &bp, &ep, &wp, dm::arg(rows), dm::arg(E), &K, &scoring, &scale};
    dm::kernels(b).launch(dm::kernels(b).route_, rows, args);
}

/// \brief `ml::moe_route_rows_backward` on the device: a thread per row, `dlogits` written. \tparam C The configuration.
/// \param b The backend. \param logits The router logits. \param experts The chosen experts. \param dweights The
/// gradient in the weights. \param dlogits Receives the gradient in the logits.
template <ml::transformer_config C>
void moe_route_rows_backward(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> logits, std::span<const std::uint32_t> experts,
                             std::type_identity_t<std::span<const float>> dweights, std::span<float> dlogits)
{
    namespace dm = detail::cuda_moe;
    dm::check_experts<C>();
    const float* z = logits.data();
    const std::uint32_t* ep = experts.data();
    const float* dw = dweights.data();
    float* g = dlogits.data();
    const dm::u64 rows = logits.size() / C.experts, E = C.experts;
    int K = int(C.top_k), scoring = dm::scoring<C>;
    float scale = ml::route_scale_v<C, float>;
    void* args[] = {&z, &ep, &dw, &g, dm::arg(rows), dm::arg(E), &K, &scoring, &scale};
    dm::kernels(b).launch(dm::kernels(b).route_backward_, rows, args);
}

/// \brief `ml::moe_count_rows` on the device: a shared-memory histogram per block, integer atomics (exact).
/// \tparam C The configuration. \param b The backend. \param experts `rows · K`. \param counts Receives `E`.
template <ml::transformer_config C>
void moe_count_rows(const cuda_training_backend& b, std::span<const std::uint32_t> experts, std::span<std::uint32_t> counts)
{
    namespace dm = detail::cuda_moe;
    dm::check_experts<C>();
    const auto& k = dm::kernels(b);
    detail::cuda::check(cudaMemsetAsync(counts.data(), 0, C.experts * sizeof(std::uint32_t), b.context->stream()), "cudaMemsetAsync");
    const std::uint32_t* ep = experts.data();
    std::uint32_t* cp = counts.data();
    const dm::u64 n = experts.size();
    unsigned E = unsigned(C.experts);
    void* args[] = {&ep, dm::arg(n), &E, &cp};
    const unsigned grid = unsigned(std::min<std::size_t>(256, (n + 255) / 256));
    k.launch(k.count_, dim3(grid), 256, E * sizeof(std::uint32_t), args);
}

/// \brief `ml::moe_offsets` on the device (one thread: `E` additions). \tparam C The configuration. \param b The
/// backend. \param counts `E`. \param offsets Receives `E + 1`.
template <ml::transformer_config C>
void moe_offsets(const cuda_training_backend& b, std::span<const std::uint32_t> counts, std::span<std::uint32_t> offsets)
{
    namespace dm = detail::cuda_moe;
    dm::check_experts<C>();
    const std::uint32_t* cp = counts.data();
    std::uint32_t* op = offsets.data();
    unsigned E = unsigned(C.experts);
    void* args[] = {&cp, &op, &E};
    dm::kernels(b).launch(dm::kernels(b).offsets_, 1, args);
}

/// \brief `ml::moe_permutation` on the device: a counting sort in three kernels, every position from integer counts,
/// so the unique stable order to the bit. \tparam C The configuration. \param b The backend. \param experts `rows · K`.
/// \param offsets `E + 1`. \param order Receives `rows · K`. \param inverse Receives `rows · K`.
template <ml::transformer_config C>
void moe_permutation(const cuda_training_backend& b, std::span<const std::uint32_t> experts, std::span<const std::uint32_t> offsets,
                     std::span<std::uint32_t> order, std::span<std::uint32_t> inverse)
{
    namespace dm = detail::cuda_moe;
    dm::check_experts<C>();
    const auto& k = dm::kernels(b);
    const dm::u64 n = experts.size();
    if (n == 0)
        return;
    unsigned E = unsigned(C.experts), chunks = unsigned((n + dm::chunk - 1) / dm::chunk);
    const auto space = k.scratch_u32(2 * std::size_t(chunks) * E);
    std::uint32_t* bc = space.data();
    std::uint32_t* starts = space.data() + std::size_t(chunks) * E;
    const std::uint32_t* ep = experts.data();
    const std::uint32_t* op = offsets.data();
    std::uint32_t* ord = order.data();
    std::uint32_t* inv = inverse.data();
    void* a1[] = {&ep, dm::arg(n), &E, &bc};
    k.launch(k.chunk_counts_, dim3(chunks), 256, E * sizeof(std::uint32_t), a1);
    const std::uint32_t* cbc = bc;
    void* a2[] = {&cbc, &op, &chunks, &E, &starts};
    k.launch(k.chunk_starts_, E, a2);
    const std::uint32_t* cst = starts;
    void* a3[] = {&ep, dm::arg(n), &E, &cst, &ord, &inv};
    k.launch(k.rank_, dim3(chunks), 256, 9 * E * sizeof(std::uint32_t), a3);
}

/// \brief `ml::moe_gather_rows` on the device, a thread per value (float4 where aligned). \tparam C The configuration.
/// \param b The backend. \param xs `rows · d_model`. \param order `rows · K`. \param xp Receives `rows · K · d_model`.
template <ml::transformer_config C>
void moe_gather_rows(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> xs, std::span<const std::uint32_t> order,
                     std::span<float> xp)
{
    namespace dm = detail::cuda_moe;
    dm::check_experts<C>();
    const float* x = xs.data();
    const std::uint32_t* o = order.data();
    float* y = xp.data();
    const dm::u64 n = order.size(), d = C.d_model;
    unsigned K = unsigned(C.top_k);
    int vec = d % 4 == 0 && dm::aligned16(x) && dm::aligned16(y) ? 1 : 0;
    void* args[] = {&x, &o, &y, dm::arg(n), dm::arg(d), &K, &vec};
    dm::kernels(b).launch(dm::kernels(b).gather_, vec ? n * d / 4 : n * d, args);
}

/// \brief `ml::moe_gather_rows_backward` on the device: `dxs` added into, the slots in order, a thread per value.
/// \tparam C The configuration. \param b The backend. \param dxp `rows · K · d_model`. \param inverse `rows · K`.
/// \param dxs Added into, `rows · d_model`.
template <ml::transformer_config C>
void moe_gather_rows_backward(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> dxp, std::span<const std::uint32_t> inverse,
                              std::span<float> dxs)
{
    namespace dm = detail::cuda_moe;
    dm::check_experts<C>();
    const float* g = dxp.data();
    const std::uint32_t* inv = inverse.data();
    float* x = dxs.data();
    const dm::u64 rows = inverse.size() / C.top_k, d = C.d_model;
    unsigned K = unsigned(C.top_k);
    void* args[] = {&g, &inv, &x, dm::arg(rows), dm::arg(d), &K};
    dm::kernels(b).launch(dm::kernels(b).gather_backward_, rows * d, args);
}

/// \brief `ml::moe_combine_rows` on the device, a thread per value in slot order. \tparam C The configuration.
/// \param b The backend. \param yp `rows · K · d_model`. \param weights `rows · K`. \param inverse `rows · K`.
/// \param ys Receives `rows · d_model`.
template <ml::transformer_config C>
void moe_combine_rows(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> yp, std::type_identity_t<std::span<const float>> weights,
                      std::span<const std::uint32_t> inverse, std::span<float> ys)
{
    namespace dm = detail::cuda_moe;
    dm::check_experts<C>();
    const float* y = yp.data();
    const float* w = weights.data();
    const std::uint32_t* inv = inverse.data();
    float* out = ys.data();
    const dm::u64 rows = inverse.size() / C.top_k, d = C.d_model;
    unsigned K = unsigned(C.top_k);
    void* args[] = {&y, &w, &inv, &out, dm::arg(rows), dm::arg(d), &K};
    dm::kernels(b).launch(dm::kernels(b).combine_, rows * d, args);
}

/// \brief `ml::moe_combine_rows_backward` on the device: `dyp` a thread per value, `dweights` a thread per pair
/// summing its row in order. \tparam C The configuration. \param b The backend. \param dys `rows · d_model`.
/// \param yp `rows · K · d_model`. \param weights `rows · K`. \param inverse `rows · K`. \param dyp Receives
/// `rows · K · d_model`. \param dweights Receives `rows · K`.
template <ml::transformer_config C>
void moe_combine_rows_backward(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> dys,
                               std::type_identity_t<std::span<const float>> yp, std::type_identity_t<std::span<const float>> weights,
                               std::span<const std::uint32_t> inverse, std::type_identity_t<std::span<float>> dyp, std::span<float> dweights)
{
    namespace dm = detail::cuda_moe;
    dm::check_experts<C>();
    const auto& k = dm::kernels(b);
    const float* g = dys.data();
    const float* y = yp.data();
    const float* w = weights.data();
    const std::uint32_t* inv = inverse.data();
    float* gy = dyp.data();
    float* gw = dweights.data();
    const dm::u64 rows = inverse.size() / C.top_k, d = C.d_model;
    unsigned K = unsigned(C.top_k);
    void* a1[] = {&g, &w, &inv, &gy, dm::arg(rows), dm::arg(d), &K};
    k.launch(k.combine_backward_values_, rows * K * d, a1);
    void* a2[] = {&g, &y, &inv, &gw, dm::arg(rows), dm::arg(d), &K};
    k.launch(k.combine_backward_weights_, rows * K, a2);
}

/// \brief `ml::grouped_linear` on the device: `Y_e = X_e W_eᵀ + b_e` for every expert in one launch, the segments
/// read from device memory (`grouped_path::kernel`), or each expert through cuBLAS (`grouped_path::cublas`).
/// \param b The backend. \param w The experts' matrices (device memory). \param offsets `experts + 1` (device memory).
/// \param xs The rows. \param ys Receives the outputs. \param bias `experts · rows`, or empty.
inline void grouped_linear(const cuda_training_backend& b, const ml::grouped_weights<float>& w, std::span<const std::uint32_t> offsets,
                           std::type_identity_t<std::span<const float>> xs, std::type_identity_t<std::span<float>> ys,
                           std::type_identity_t<std::span<const float>> bias = {})
{
    namespace dm = detail::cuda_moe;
    if (dm::kernels(b).path == grouped_path::cublas) {
        const auto h = dm::host_offsets(b, offsets);
        ml::grouped_linear(b, w, std::span<const std::uint32_t>(h), xs, ys, bias);
        return;
    }
    if (w.cols == 0 || w.rows == 0 || w.experts == 0)
        return;
    const std::size_t rows = xs.size() / w.cols;
    if (rows == 0)
        return;
    dm::seg_rows(b, xs.data(), w.cols, w.data.data(), w.cols, 1, w.rows * w.cols, ys.data(), w.rows, bias.empty() ? nullptr : bias.data(), offsets.data(),
                 unsigned(w.experts), rows, w.rows, w.cols, 0.0f);
}

/// \brief `ml::grouped_linear_backward` on the device: `dX_e += dY_e W_e` and `dW_e += dY_eᵀ X_e` one launch each over
/// every expert, `db_e += Σ dY_e` a thread per (expert, column) in row order. \param b The backend. \param w The
/// matrices. \param offsets `experts + 1`. \param xs The forward's rows. \param dys The gradients in its outputs.
/// \param dxs Added into, or empty. \param dw Added into. \param db Added into, or empty.
inline void grouped_linear_backward(const cuda_training_backend& b, const ml::grouped_weights<float>& w, std::span<const std::uint32_t> offsets,
                                    std::type_identity_t<std::span<const float>> xs, std::type_identity_t<std::span<const float>> dys,
                                    std::type_identity_t<std::span<float>> dxs, std::type_identity_t<std::span<float>> dw,
                                    std::type_identity_t<std::span<float>> db = {})
{
    namespace dm = detail::cuda_moe;
    const auto& k = dm::kernels(b);
    if (k.path == grouped_path::cublas) {
        const auto h = dm::host_offsets(b, offsets);
        ml::grouped_linear_backward(b, w, std::span<const std::uint32_t>(h), xs, dys, dxs, dw, db);
        return;
    }
    if (w.cols == 0 || w.rows == 0 || w.experts == 0)
        return;
    const std::size_t rows = xs.size() / w.cols;
    if (rows == 0)
        return;
    const unsigned E = unsigned(w.experts);
    if (!dxs.empty())   // dX (rows x cols) += dY (rows x out) W_e (out x cols)
        dm::seg_rows(b, dys.data(), w.rows, w.data.data(), w.cols, 0, w.rows * w.cols, dxs.data(), w.cols, nullptr, offsets.data(), E, rows, w.cols, w.rows,
                     1.0f);
    // dW_e (out x cols) += dY_eᵀ X_e
    dm::seg_reduce(b, dys.data(), w.rows, xs.data(), w.cols, dw.data(), w.cols, w.rows * w.cols, offsets.data(), E, w.rows, w.cols, 1.0f);
    if (!db.empty()) {
        const float* g = dys.data();
        const std::uint32_t* off = offsets.data();
        float* bp = db.data();
        unsigned e = E;
        const dm::u64 r = w.rows;
        void* args[] = {&g, &off, &bp, &e, dm::arg(r)};
        k.launch(k.segment_sums_, std::size_t(E) * w.rows, args);
    }
}

/// \brief `ml::moe_update_bias` on the device, a thread per expert, exact. \tparam C The configuration. \param b The
/// backend. \param counts `E`. \param bias `E`, updated.
template <ml::transformer_config C>
void moe_update_bias(const cuda_training_backend& b, std::span<const std::uint32_t> counts, std::span<float> bias)
{
    namespace dm = detail::cuda_moe;
    dm::check_experts<C>();
    const std::uint32_t* cp = counts.data();
    float* bp = bias.data();
    unsigned E = unsigned(C.experts);
    float gamma = ml::bias_update_rate_v<C, float>;
    void* args[] = {&cp, &bp, &E, &gamma};
    dm::kernels(b).launch(dm::kernels(b).update_bias_, E, args);
}

/// \brief `ml::moe_balance_loss` on the device: the gradient added into `dlogits` (if not empty) and the loss, which
/// is returned outside a capture (one wait) and otherwise written to `cuda_moe_kernels::losses()[0]` with NaN returned.
/// \tparam C The configuration. \param b The backend. \param logits `rows · E`. \param experts `rows · K`.
/// \param dlogits Added into, or empty. \param groups The groups. \param denominator The denominator, 0 for `groups`.
/// \return The loss, or NaN.
template <ml::transformer_config C>
float moe_balance_loss(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> logits, std::span<const std::uint32_t> experts,
                       std::span<float> dlogits, std::size_t groups = 1, std::size_t denominator = 0)
{
    namespace dm = detail::cuda_moe;
    dm::check_experts<C>();
    const auto& k = dm::kernels(b);
    constexpr std::size_t E = C.experts, K = C.top_k;
    const std::size_t rows = logits.size() / E, n = groups == 0 ? 0 : rows / groups, den = denominator == 0 ? groups : denominator;
    float* loss = k.losses().data();
    if (n == 0) {
        detail::cuda::check(cudaMemsetAsync(loss, 0, sizeof(float), b.context->stream()), "cudaMemsetAsync");
        return dm::loss_result(b, 0);
    }
    const auto space = k.scratch(rows * E + 2 * groups * E);
    float* P = space.data();
    float* sum_p = P + rows * E;
    float* g = sum_p + groups * E;
    const auto cs = k.scratch_u32(groups * E);
    std::uint32_t* cnt = cs.data();
    const float* z = logits.data();
    const std::uint32_t* ep = experts.data();
    int sig = C.scoring == ml::moe_scoring::sigmoid ? 1 : 0;
    void* a1[] = {&z, &P, dm::arg(rows), dm::arg(E), &sig};
    k.launch(k.probabilities_, rows, a1);
    unsigned kk = unsigned(K);
    void* a2[] = {&P, &ep, &sum_p, &cnt, dm::arg(groups), dm::arg(n), dm::arg(E), &kk};
    k.launch(k.group_sums_, groups * E, a2);
    float ae = ml::balance_loss_weight_v<C, float> * float(E), nk = float(n * K), nf = float(n), nden = float(n * den), fden = float(den);
    float* gp = dlogits.empty() ? nullptr : g;
    const float* csum = sum_p;
    const std::uint32_t* ccnt = cnt;
    void* a3[] = {&csum, &ccnt, &gp, &loss, dm::arg(groups), dm::arg(E), &ae, &nk, &nf, &nden, &fden};
    k.launch(k.balance_finish_, 1, a3);
    if (!dlogits.empty()) {
        const float* cP = P;
        const float* cg = g;
        float* dz = dlogits.data();
        void* a4[] = {&z, &cP, &cg, &dz, dm::arg(rows), dm::arg(n), dm::arg(E), &sig};
        k.launch(k.balance_backward_, rows, a4);
    }
    return dm::loss_result(b, 0);
}

/// \brief `ml::moe_z_loss` on the device: the gradient added into `dlogits` (if not empty) and the loss, returned
/// outside a capture and otherwise written to `cuda_moe_kernels::losses()[1]` with NaN returned. \tparam C The
/// configuration. \param b The backend. \param logits `rows · E`. \param dlogits Added into, or empty. \param denominator
/// The denominator, 0 for the rows. \return The loss, or NaN.
template <ml::transformer_config C>
float moe_z_loss(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> logits, std::span<float> dlogits, std::size_t denominator = 0)
{
    namespace dm = detail::cuda_moe;
    dm::check_experts<C>();
    const auto& k = dm::kernels(b);
    constexpr std::size_t E = C.experts;
    const std::size_t rows = logits.size() / E, den = denominator == 0 ? rows : denominator;
    float* loss = k.losses().data() + 1;
    if (rows == 0) {
        detail::cuda::check(cudaMemsetAsync(loss, 0, sizeof(float), b.context->stream()), "cudaMemsetAsync");
        return dm::loss_result(b, 1);
    }
    float* sq = k.scratch(rows).data();
    const float* z = logits.data();
    float* dz = dlogits.empty() ? nullptr : dlogits.data();
    float beta = ml::z_loss_weight_v<C, float>, twobeta = 2.0f * beta, fden = float(den);
    void* a1[] = {&z, &dz, &sq, dm::arg(rows), dm::arg(E), &twobeta, &fden};
    k.launch(k.z_rows_, rows, a1);
    const float* csq = sq;
    void* a2[] = {&csq, dm::arg(rows), &beta, &fden, &loss};
    k.launch(k.z_finish_, 1, a2);
    return dm::loss_result(b, 1);
}

/// \brief `ml::moe_send_offsets` on the device: `send_offsets[r] = offsets[r · E / P]`. \tparam C The configuration.
/// \param b The backend. \param offsets `E + 1`. \param ranks `P`, dividing `E`. \param send_offsets Receives `P + 1`.
template <ml::transformer_config C>
void moe_send_offsets(const cuda_training_backend& b, std::span<const std::uint32_t> offsets, std::size_t ranks, std::span<std::uint32_t> send_offsets)
{
    namespace dm = detail::cuda_moe;
    dm::check_experts<C>();
    const std::uint32_t* op = offsets.data();
    std::uint32_t* sp = send_offsets.data();
    unsigned P = unsigned(ranks), per = unsigned(C.experts / ranks);
    void* args[] = {&op, &sp, &P, &per};
    dm::kernels(b).launch(dm::kernels(b).send_offsets_, ranks + 1, args);
}

namespace detail::cuda_moe {
    inline ptrs table(std::span<const std::span<const std::uint32_t>> counts)
    {
        if (counts.size() > max_ranks)
            throw std::invalid_argument("snicholls::gpu: expert parallelism over more than 64 ranks");
        ptrs t{};
        for (std::size_t s = 0; s < counts.size(); ++s)
            t.p[s] = counts[s].data();
        return t;
    }
} // namespace detail::cuda_moe

/// \brief `ml::moe_receive_offsets` on the device (one thread; every source's counts in device memory). \tparam C The
/// configuration. \param b The backend. \param counts `P` spans of `E`. \param rank This rank. \param receive_offsets Receives `P + 1`.
template <ml::transformer_config C>
void moe_receive_offsets(const cuda_training_backend& b, std::span<const std::span<const std::uint32_t>> counts, std::size_t rank,
                         std::span<std::uint32_t> receive_offsets)
{
    namespace dm = detail::cuda_moe;
    dm::check_experts<C>();
    dm::ptrs t = dm::table(counts);
    unsigned P = unsigned(counts.size()), local = unsigned(C.experts / counts.size()), first = unsigned(rank) * local;
    std::uint32_t* out = receive_offsets.data();
    void* args[] = {&t, &P, &first, &local, &out};
    dm::kernels(b).launch(dm::kernels(b).receive_offsets_, 1, args);
}

/// \brief `ml::moe_receive_layout` on the device: a block per (local expert, source) writing its run of the order.
/// \tparam C The configuration. \param b The backend. \param counts `P` spans of `E`. \param rank This rank.
/// \param expert_offsets Receives `E / P + 1`. \param order Receives one index per received row.
template <ml::transformer_config C>
void moe_receive_layout(const cuda_training_backend& b, std::span<const std::span<const std::uint32_t>> counts, std::size_t rank,
                        std::span<std::uint32_t> expert_offsets, std::span<std::uint32_t> order)
{
    namespace dm = detail::cuda_moe;
    dm::check_experts<C>();
    dm::ptrs t = dm::table(counts);
    unsigned P = unsigned(counts.size()), local = unsigned(C.experts / counts.size()), first = unsigned(rank) * local;
    std::uint32_t* eo = expert_offsets.data();
    std::uint32_t* op = order.data();
    void* args[] = {&t, &P, &first, &local, &eo, &op};
    dm::kernels(b).launch(dm::kernels(b).receive_layout_, dim3(local * P), 256, 0, args);
}

namespace detail::cuda_moe {
    template <typename T>
    constexpr int unit = sizeof(T) == 1 ? 0 : sizeof(T) == 2 ? 1 : sizeof(T) == 4 ? 2 : 3;
    template <typename T>
    concept payload = std::is_trivially_copyable_v<T> && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
} // namespace detail::cuda_moe

/// \brief `ml::moe_all_to_all` over in-process ranks whose buffers are all on this device: one launch, a thread per
/// received element, the sources in order; the bits copied. \tparam T The payload (1, 2, 4 or 8 bytes). \param b The
/// backend. \param send `P` spans. \param send_offsets `P` spans of `P + 1` (device memory). \param width The elements
/// of a row. \param receive `P` spans, each exactly the rows it receives.
template <typename T>
    requires detail::cuda_moe::payload<T>
void moe_all_to_all(const cuda_training_backend& b, std::span<const std::span<const T>> send, std::span<const std::span<const std::uint32_t>> send_offsets,
                    std::size_t width, std::span<const std::span<T>> receive)
{
    namespace dm = detail::cuda_moe;
    const std::size_t P = send.size();
    if (P > dm::max_ranks)
        throw std::invalid_argument("snicholls::gpu: expert parallelism over more than 64 ranks");
    dm::bptrs s{};
    dm::wptrs r{};
    std::size_t most = 0;
    for (std::size_t i = 0; i < P; ++i) {
        s.p[i] = send[i].data(), r.p[i] = receive[i].data(), r.size[i] = receive[i].size();
        most = std::max(most, receive[i].size());
    }
    dm::ptrs so = dm::table(send_offsets);
    unsigned ranks = unsigned(P);
    dm::u64 w = width;
    void* args[] = {&s, &so, &ranks, &w, &r};
    const auto& k = dm::kernels(b);
    k.launch(k.all_to_all_[dm::unit<T>], dim3(unsigned((most + 255) / 256), ranks), 256, 0, args);
}

/// \brief `ml::moe_permute_rows` on the device: `to[p] = from[order[p]]`, a thread per element. \tparam T The
/// payload. \param b The backend. \param from The rows. \param order The source row of each. \param to Receives
/// `order.size()` rows. \param width The elements of a row.
template <typename T>
    requires detail::cuda_moe::payload<T>
void moe_permute_rows(const cuda_training_backend& b, std::type_identity_t<std::span<const T>> from, std::span<const std::uint32_t> order, std::span<T> to,
                      std::size_t width)
{
    namespace dm = detail::cuda_moe;
    const void* f = from.data();
    const std::uint32_t* o = order.data();
    void* t = to.data();
    const dm::u64 rows = order.size(), w = width;
    int inverse = 0;
    void* args[] = {&f, &o, &t, dm::arg(rows), dm::arg(w), &inverse};
    dm::kernels(b).launch(dm::kernels(b).permute_[dm::unit<T>], rows * width, args);
}

/// \brief `ml::moe_unpermute_rows` on the device: `to[order[p]] = from[p]`, a thread per element. \tparam T The
/// payload. \param b The backend. \param from The permuted rows. \param order The order. \param to Receives the rows.
/// \param width The elements of a row.
template <typename T>
    requires detail::cuda_moe::payload<T>
void moe_unpermute_rows(const cuda_training_backend& b, std::type_identity_t<std::span<const T>> from, std::span<const std::uint32_t> order,
                        std::span<T> to, std::size_t width)
{
    namespace dm = detail::cuda_moe;
    const void* f = from.data();
    const std::uint32_t* o = order.data();
    void* t = to.data();
    const dm::u64 rows = order.size(), w = width;
    int inverse = 1;
    void* args[] = {&f, &o, &t, dm::arg(rows), dm::arg(w), &inverse};
    dm::kernels(b).launch(dm::kernels(b).permute_[dm::unit<T>], rows * width, args);
}

/// \brief `ml::to_bfloat16_rows` on the device, core's rounding to the bit. \param b The backend. \param from The
/// values. \param to Receives them (the shorter of the two sizes).
inline void to_bfloat16_rows(const cuda_training_backend& b, std::span<const float> from, std::span<ml::bfloat16> to)
{
    namespace dm = detail::cuda_moe;
    const float* f = from.data();
    void* t = to.data();
    const dm::u64 n = std::min(from.size(), to.size());
    void* args[] = {&f, &t, dm::arg(n)};
    dm::kernels(b).launch(dm::kernels(b).to_bf16_, n, args);
}

/// \brief `ml::to_float_rows` on the device, exact. \param b The backend. \param from The values. \param to Receives
/// them (the shorter of the two sizes).
inline void to_float_rows(const cuda_training_backend& b, std::span<const ml::bfloat16> from, std::span<float> to)
{
    namespace dm = detail::cuda_moe;
    const void* f = from.data();
    float* t = to.data();
    const dm::u64 n = std::min(from.size(), to.size());
    void* args[] = {&f, &t, dm::arg(n)};
    dm::kernels(b).launch(dm::kernels(b).to_float_, n, args);
}

/// @}

} // namespace snicholls::gpu

#endif /* snicholls_gpu_cuda_moe_hpp */
