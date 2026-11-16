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
/// **Implementations.** The elementwise steps one thread per element; the norms a warp per row (each lane's part of a
/// row's sums in order, then the lanes in a fixed order), the weights' and biases' gradients one thread per column summing the rows in order, so the
/// result does not depend on scheduling; the activations with core's formulas in float. Attention, up to
/// `cuda_training_backend::gemm_attention_max` tokens (4096), is matrix products head by head, as llm.c's: the scores
/// `S = Q Kᵀ` through cuBLAS (the tensor cores under TF32), a softmax kernel writing the log-sum-exp, `Y = P V`; the
/// backward recomputes `P` from the log-sum-exp and makes the four gradient products, a group's heads adding into
/// its keys' and values' gradients in order. Longer sequences take tiled kernels for heads of 16, 32, 64 or 128
/// channels: a thread per query or key with its row in registers, the other side through shared memory a tile at a
/// time, the keys' gradients split over the queries into partials added in a fixed order; other head sizes, up to
/// 256, a warp per (token, head). Every path is deterministic. QK norm and RoPE one thread per
/// (token, head), the norms' weight gradients summed in core's order. The kernels' working space
/// (`cuda_kernels::scratch`) is kept between calls, so a step allocates nothing.
///
/// **Guarantees.** Every step is enqueued on the context's stream, in the order the pass makes them, after the
/// previous product; the results agree with core's reference to rounding (the device's `expf`, `tanhf`, `sqrtf`
/// differ from the host's in the last bits). One training step at a time per context.
///
/// **Errors.** `cuda_kernels::create` returns a `cuda_failure` (with the compiler's log on stderr) if the kernels do not
/// compile. A launch or copy that fails throws `cuda_exception`.
///
/// **Performance.** `benchmarks/cuda_ops_benchmark.cpp`: a GPT-2-small layer's training step with its data on the
/// device, against the same step staged; see `BENCHMARKS.md`. A whole GPT-2 small step (124M parameters, one sequence
/// of 1024 tokens, forward, backward and clipping) on an RTX PRO 4000 Blackwell, 2026-11-17: 126 ms in float, 78.3 ms
/// with `cuda_math::tf32`, against llm.c's 64.4 ms in bfloat16 (288 ms with attention a warp per (token, head), 98.5 ms
/// with the tiled kernels under TF32).
///
/// **Tests and demos.** `tests/cuda_ops_test.cpp` (each step against core's, and whole training steps of GPT-2-,
/// Llama- and nanochat-shaped blocks against the reference), `demos/cuda_ops_demo.cpp`.
///
/// **Dependencies.** `cuda.hpp`, `<nvrtc.h>`, core's `ml/transformer_block.hpp`. Link `CUDA::nvrtc` too.
///
/// **Known issues.** Float only (the products can be TF32). The attention's products are one call per head (the host's
/// share of a step is about a seventh); its scores take `3 n²` floats of scratch. The tiled kernels for long sequences
/// run on the CUDA cores, not the tensor cores.
///
/// **Status.** Experimental, new in 0.2.0.

#ifndef snicholls_gpu_cuda_ops_hpp
#define snicholls_gpu_cuda_ops_hpp

#include <snicholls/gpu/cuda.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/embedding.hpp>
#include <snicholls/core/ml/optim.hpp>
#include <snicholls/core/ml/softmax.hpp>
#include <snicholls/core/ml/transformer_block.hpp>

#include <nvrtc.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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
// A sum over a warp, for the norms (defined before the attention kernels' copy, which uses LANES)
__device__ float norm_warp_sum_(float s)
{
    for (int o = 16; o > 0; o /= 2) s += __shfl_down_sync(0xffffffffu, s, o);
    return __shfl_sync(0xffffffffu, s, 0);
}
// One warp per row: core's layer norm (layer != 0) or RMS norm; stats[2t], stats[2t+1] = mean, rstd
__global__ void norm_rows(const float* x, const float* w, const float* b, float* out, float* stats, u64 rows, u64 d, int layer, float eps)
{
    const u64 t = (blockIdx.x * (u64)blockDim.x + threadIdx.x) / 32; const int lane = threadIdx.x % 32;
    if (t >= rows) return;
    const float* in = x + t * d; float* y = out + t * d;
    float mean = 0.0f, rstd;
    if (layer) {
        float s = 0.0f; for (u64 i = lane; i < d; i += 32) s += in[i];
        mean = norm_warp_sum_(s) / (float)d;
        float v = 0.0f; for (u64 i = lane; i < d; i += 32) v += (in[i] - mean) * (in[i] - mean);
        rstd = 1.0f / sqrtf(norm_warp_sum_(v) / (float)d + eps);
        for (u64 i = lane; i < d; i += 32) y[i] = (in[i] - mean) * rstd * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
    } else {
        float s = 0.0f; for (u64 i = lane; i < d; i += 32) s += in[i] * in[i];
        rstd = 1.0f / sqrtf(norm_warp_sum_(s) / (float)d + eps);
        for (u64 i = lane; i < d; i += 32) y[i] = in[i] * rstd * (w ? w[i] : 1.0f);
    }
    if (stats && lane == 0) { stats[2 * t] = mean; stats[2 * t + 1] = rstd; }
}
// One warp per row: dxs_t += norm'(x_t)^T dh_t, from the statistics norm_rows left
__global__ void norm_rows_backward_add(const float* dh, const float* x, const float* w, const float* stats, float* dxs, u64 rows, u64 d, int layer)
{
    const u64 t = (blockIdx.x * (u64)blockDim.x + threadIdx.x) / 32; const int lane = threadIdx.x % 32;
    if (t >= rows) return;
    const float* g_ = dh + t * d; const float* in = x + t * d; float* gx = dxs + t * d;
    const float mean = stats[2 * t], rstd = stats[2 * t + 1];
    if (layer) {
        float sg = 0.0f, sgn = 0.0f;
        for (u64 i = lane; i < d; i += 32) { const float norm = (in[i] - mean) * rstd; const float g = (w ? w[i] : 1.0f) * g_[i]; sg += g; sgn += g * norm; }
        const float mean_g = norm_warp_sum_(sg) / (float)d, mean_gn = norm_warp_sum_(sgn) / (float)d;
        for (u64 i = lane; i < d; i += 32) { const float norm = (in[i] - mean) * rstd; const float g = (w ? w[i] : 1.0f) * g_[i]; gx[i] += rstd * (g - mean_g - norm * mean_gn); }
    } else {
        float sgn = 0.0f; for (u64 i = lane; i < d; i += 32) sgn += (w ? w[i] : 1.0f) * g_[i] * in[i] * rstd;
        const float mean_gn = norm_warp_sum_(sgn) / (float)d;
        for (u64 i = lane; i < d; i += 32) { const float norm = in[i] * rstd; const float g = (w ? w[i] : 1.0f) * g_[i]; gx[i] += rstd * (g - norm * mean_gn); }
    }
}
#define MAXD 256
#define INFINITY __int_as_float(0x7f800000)   // NVRTC has no <math.h>
__device__ float dot_(const float* a, const float* b, int d) { float s = 0.0f; for (int i = 0; i < d; ++i) s += a[i] * b[i]; return s; }
// core's rms_norm of one head, in place: y = x rstd w (w may be null); returns rstd
__device__ float rms_(float* x, const float* w, int d, float eps)
{ float ms = 0.0f; for (int i = 0; i < d; ++i) ms += x[i] * x[i]; ms = ms / (float)d; const float rstd = 1.0f / sqrtf(ms + eps);
  for (int i = 0; i < d; ++i) x[i] = x[i] * rstd * (w ? w[i] : 1.0f); return rstd; }
__device__ float rstd_(const float* x, int d, float eps)
{ float ms = 0.0f; for (int i = 0; i < d; ++i) ms += x[i] * x[i]; ms = ms / (float)d; return 1.0f / sqrtf(ms + eps); }
// core's rope rotation of one head by sign * theta
__device__ void rotate_(float* x, const float* c, const float* s, int d, int half, float sign)
{
    const int h = d / 2;
    for (int i = 0; i < h; ++i) {
        float* a = half ? x + i : x + 2 * i; float* b = half ? x + i + h : x + 2 * i + 1;
        const float x1 = *a, x2 = *b, sn = sign * s[i];
        *a = x1 * c[i] - x2 * sn; *b = x1 * sn + x2 * c[i];
    }
}
// core's rms_norm_backward in place on g (the gradient in the norm's output becomes its input's); contrib = g * norm
__device__ void rms_backward_(float* g, const float* x, const float* w, float rstd, float* contrib, int d)
{
    float mean_gn = 0.0f; for (int i = 0; i < d; ++i) mean_gn += (w ? w[i] : 1.0f) * g[i] * x[i] * rstd;
    mean_gn /= (float)d;
    for (int i = 0; i < d; ++i) { const float norm = x[i] * rstd; const float gi = (w ? w[i] : 1.0f) * g[i];
        if (contrib) contrib[i] = g[i] * norm; g[i] = rstd * (gi - norm * mean_gn); }
}
// One thread per (token, head) of the queries then the keys: the raw copies, QK norm before or after RoPE, RoPE
__global__ void position(float* q, float* k, float* q_raw, float* k_raw, const float* qnw, const float* knw, const float* cs, const float* sn,
                         u64 n, int H, int KVH, int d, int qk, int rope, int half, float eps)
{
    const u64 idx = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (idx >= n * (u64)(H + KVH)) return;
    const u64 t = idx / (H + KVH); const int h = (int)(idx % (H + KVH));
    float* head; float* raw; const float* w;
    if (h < H) { head = q + t * (u64)(H * d) + (u64)h * d; raw = q_raw ? q_raw + t * (u64)(H * d) + (u64)h * d : 0; w = qnw; }
    else { const int hk = h - H; head = k + ((u64)hk * n + t) * d; raw = k_raw ? k_raw + t * (u64)(KVH * d) + (u64)hk * d : 0; w = knw; }
    if (qk) for (int i = 0; i < d; ++i) raw[i] = head[i];
    if (qk == 1) rms_(head, w, d, eps);
    if (rope) rotate_(head, cs + t * (u64)(d / 2), sn + t * (u64)(d / 2), d, half, 1.0f);
    if (qk == 2) rms_(head, w, d, eps);
}
// The backward of position, in place on the gradients; each head's norm-weight contributions written for an in-order sum
__global__ void position_backward(float* gq, float* gk, const float* q_raw, const float* k_raw, const float* qnw, const float* knw,
                                  const float* cs, const float* sn, float* cq, float* ck, u64 n, int H, int KVH, int d, int qk, int rope, int half, float eps)
{
    const u64 idx = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (idx >= n * (u64)(H + KVH)) return;
    const u64 t = idx / (H + KVH); const int h = (int)(idx % (H + KVH));
    float* g; const float* raw; const float* w; float* contrib;
    if (h < H) { g = gq + t * (u64)(H * d) + (u64)h * d; raw = q_raw ? q_raw + t * (u64)(H * d) + (u64)h * d : 0; w = qnw;
                 contrib = cq ? cq + (t * H + h) * (u64)d : 0; }
    else { const int hk = h - H; g = gk + ((u64)hk * n + t) * d; raw = k_raw ? k_raw + t * (u64)(KVH * d) + (u64)hk * d : 0; w = knw;
           contrib = ck ? ck + (t * KVH + hk) * (u64)d : 0; }
    const float* c = cs + t * (u64)(d / 2); const float* s = sn + t * (u64)(d / 2);
    float x[MAXD];
    if (qk == 2) { for (int i = 0; i < d; ++i) x[i] = raw[i]; if (rope) rotate_(x, c, s, d, half, 1.0f); rms_backward_(g, x, w, rstd_(x, d, eps), contrib, d); }
    if (rope) rotate_(g, c, s, d, half, -1.0f);
    if (qk == 1) { for (int i = 0; i < d; ++i) x[i] = raw[i]; rms_backward_(g, x, w, rstd_(x, d, eps), contrib, d); }
}
__device__ u64 first_(u64 t, u64 window) { return (window == 0 || window > t) ? 0 : t + 1 - window; }
// One warp per (token, head): lane l holds channels l, l + 32, ...; a dot product sums each lane's channels in order,
// then the lanes down to lane 0, then broadcasts lane 0's sum, so every lane has the same value
#define LANES 32
#define PER_LANE 8   // MAXD / LANES
__device__ float warp_sum_(float s)
{
    for (int o = LANES / 2; o > 0; o /= 2) s += __shfl_down_sync(0xffffffffu, s, o);
    return __shfl_sync(0xffffffffu, s, 0);
}
__device__ float warp_dot_(const float* a, const float* b, int d, int lane)
{
    float s = 0.0f; for (int c = lane; c < d; c += LANES) s += a[c] * b[c];
    return warp_sum_(s);
}
// core's one-pass softmax over the visible keys of the head's key-value head
__global__ void attend(const float* q, const float* k, const float* v, float* y, float* lse, u64 n, int H, int KVH, int d, u64 window, float scale, float cap)
{
    const u64 w = (blockIdx.x * (u64)blockDim.x + threadIdx.x) / LANES; const int lane = threadIdx.x % LANES;
    if (w >= n * (u64)H) return;
    const u64 t = w / H; const int h = (int)(w % H), kvh = h / (H / KVH);
    const float* qh = q + t * (u64)(H * d) + (u64)h * d;
    float acc[PER_LANE]; for (int r = 0; r < PER_LANE; ++r) acc[r] = 0.0f;
    float mx = -INFINITY, sum = 0.0f;
    for (u64 j = first_(t, window); j <= t; ++j) {
        const float* kj = k + ((u64)kvh * n + j) * d; const float* vj = v + ((u64)kvh * n + j) * d;
        float sc = scale * warp_dot_(qh, kj, d, lane);
        if (cap > 0.0f) sc = cap * tanhf(sc / cap);
        if (sc == -INFINITY) continue;
        if (sc > mx) { const float r = expf(mx - sc); for (int c = lane, x = 0; c < d; c += LANES, ++x) acc[x] = acc[x] * r + vj[c]; sum = sum * r + 1.0f; mx = sc; }
        else { const float e = expf(sc - mx); for (int c = lane, x = 0; c < d; c += LANES, ++x) acc[x] += e * vj[c]; sum += e; }
    }
    float* yh = y + t * (u64)(H * d) + (u64)h * d;
    for (int c = lane, x = 0; c < d; c += LANES, ++x) yh[c] = sum == 0.0f ? 0.0f : acc[x] / sum;
    if (lane == 0) lse[t * H + h] = sum == 0.0f ? -INFINITY : mx + logf(sum);
}
// The backward, queries: one warp per (token, query head) writes its query's gradient and D = dy . y
__global__ void attend_backward_queries(const float* q, const float* k, const float* v, const float* y, const float* lse, const float* dy,
                                        float* gq, float* D, u64 n, int H, int KVH, int d, u64 window, float scale, float cap)
{
    const u64 w = (blockIdx.x * (u64)blockDim.x + threadIdx.x) / LANES; const int lane = threadIdx.x % LANES;
    if (w >= n * (u64)H) return;
    const u64 t = w / H; const int h = (int)(w % H), kvh = h / (H / KVH);
    const u64 at = t * (u64)(H * d) + (u64)h * d;
    const float* qh = q + at; const float* g = dy + at; float* gqh = gq + at;
    float acc[PER_LANE]; for (int r = 0; r < PER_LANE; ++r) acc[r] = 0.0f;
    const float l = lse[t * H + h];
    if (l != -INFINITY) {
        const float bigd = warp_dot_(g, y + at, d, lane);
        if (lane == 0) D[t * H + h] = bigd;
        for (u64 j = first_(t, window); j <= t; ++j) {
            const float* kj = k + ((u64)kvh * n + j) * d; const float* vj = v + ((u64)kvh * n + j) * d;
            const float r = scale * warp_dot_(qh, kj, d, lane), gv = warp_dot_(g, vj, d, lane);
            float ds;
            if (cap > 0.0f) { const float th = tanhf(r / cap); ds = expf(cap * th - l) * (gv - bigd) * (1.0f - th * th); }
            else ds = expf(r - l) * (gv - bigd);
            for (int c = lane, x = 0; c < d; c += LANES, ++x) acc[x] += scale * ds * kj[c];
        }
    } else if (lane == 0) D[t * H + h] = 0.0f;
    for (int c = lane, x = 0; c < d; c += LANES, ++x) gqh[c] = acc[x];
}
// The backward, keys and values: one warp per (key-value head, key) adds every query's contribution in core's order
// (query token ascending, then query head), so the sums do not depend on scheduling
__global__ void attend_backward_keys(const float* q, const float* k, const float* v, const float* lse, const float* dy, const float* D,
                                     float* gk, float* gv, u64 n, int H, int KVH, int d, u64 window, float scale, float cap)
{
    const u64 w = (blockIdx.x * (u64)blockDim.x + threadIdx.x) / LANES; const int lane = threadIdx.x % LANES;
    if (w >= n * (u64)KVH) return;
    const int kvh = (int)(w / n); const u64 j = w % n; const int group = H / KVH;
    const u64 at = ((u64)kvh * n + j) * d;
    const float* kj = k + at; const float* vj = v + at;
    float ak[PER_LANE], av[PER_LANE];
    for (int c = lane, x = 0; c < d; c += LANES, ++x) { ak[x] = gk[at + c]; av[x] = gv[at + c]; }
    const u64 last = (window == 0 || j + window - 1 >= n - 1) ? n - 1 : j + window - 1;
    for (u64 t = j; t <= last; ++t)
        for (int h = kvh * group; h < (kvh + 1) * group; ++h) {
            const float l = lse[t * H + h]; if (l == -INFINITY) continue;
            const u64 qa = t * (u64)(H * d) + (u64)h * d;
            const float* qh = q + qa; const float* g = dy + qa;
            const float r = scale * warp_dot_(qh, kj, d, lane), gvd = warp_dot_(g, vj, d, lane);
            float p, ds;
            if (cap > 0.0f) { const float th = tanhf(r / cap); p = expf(cap * th - l); ds = p * (gvd - D[t * H + h]) * (1.0f - th * th); }
            else { p = expf(r - l); ds = p * (gvd - D[t * H + h]); }
            for (int c = lane, x = 0; c < d; c += LANES, ++x) { av[x] += p * g[c]; ak[x] += scale * ds * qh[c]; }
        }
    for (int c = lane, x = 0; c < d; c += LANES, ++x) { gk[at + c] = ak[x]; gv[at + c] = av[x]; }
}
}
// Tiled attention for a head of D channels (16, 32, 64 or 128): a thread per query (forward, query gradients) or per
// key (key and value gradients) with its row in registers, the other side staged through shared memory a tile at a
// time and read by every thread of the block. Each thread's sums run in a fixed order: deterministic.
#define TK 32   // keys per tile
// A dot product of D channels as 8 interleaved partial sums, added in a fixed tree: eight independent chains instead
// of one, the order still fixed
template <int D>
__device__ __forceinline__ float dot8_(const float* a, const float* b)
{
    float p[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int c = 0; c < D; ++c) p[c % 8] += a[c] * b[c];
    return ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]));
}
template <int D>
__device__ void attend_tiled_(const float* q, const float* k, const float* v, float* y, float* lse, u64 n, int H, int KVH, u64 window, float scale, float cap)
{
    __shared__ float ks[TK * D], vs[TK * D];
    const int h = blockIdx.y, kvh = h / (H / KVH);
    const u64 q0 = (u64)blockIdx.x * blockDim.x, t = q0 + threadIdx.x, qlast = min(q0 + blockDim.x, n) - 1;
    const bool live = t < n;
    float qr[D], acc[D];
    for (int c = 0; c < D; ++c) { qr[c] = live ? q[t * (u64)(H * D) + (u64)h * D + c] : 0.0f; acc[c] = 0.0f; }
    float mx = -INFINITY, sum = 0.0f;
    const u64 mine = live ? first_(t, window) : 0;
    for (u64 jt = first_(q0, window); jt <= qlast; jt += TK) {
        __syncthreads();
        const float* kt = k + ((u64)kvh * n + jt) * D; const float* vt = v + ((u64)kvh * n + jt) * D;
        for (int i = threadIdx.x; i < TK * D; i += blockDim.x) { const bool in = jt + i / D < n; ks[i] = in ? kt[i] : 0.0f; vs[i] = in ? vt[i] : 0.0f; }
        __syncthreads();
        if (!live) continue;
        for (int x = 0; x < TK; ++x) {
            const u64 j = jt + x; if (j < mine || j > t) continue;
            float sc = scale * dot8_<D>(qr, ks + x * D);
            if (cap > 0.0f) sc = cap * tanhf(sc / cap);
            if (sc == -INFINITY) continue;
            if (sc > mx) { const float r = expf(mx - sc); for (int c = 0; c < D; ++c) acc[c] = acc[c] * r + vs[x * D + c]; sum = sum * r + 1.0f; mx = sc; }
            else { const float e = expf(sc - mx); for (int c = 0; c < D; ++c) acc[c] += e * vs[x * D + c]; sum += e; }
        }
    }
    if (!live) return;
    float* yh = y + t * (u64)(H * D) + (u64)h * D;
    for (int c = 0; c < D; ++c) yh[c] = sum == 0.0f ? 0.0f : acc[c] / sum;
    lse[t * H + h] = sum == 0.0f ? -INFINITY : mx + logf(sum);
}
template <int D>
__device__ void attend_backward_queries_tiled_(const float* q, const float* k, const float* v, const float* y, const float* lse, const float* dy,
                                               float* gq, float* Dd, u64 n, int H, int KVH, u64 window, float scale, float cap)
{
    __shared__ float ks[TK * D], vs[TK * D];
    const int h = blockIdx.y, kvh = h / (H / KVH);
    const u64 q0 = (u64)blockIdx.x * blockDim.x, t = q0 + threadIdx.x, qlast = min(q0 + blockDim.x, n) - 1;
    const u64 at = t * (u64)(H * D) + (u64)h * D;
    const float l = t < n ? lse[t * H + h] : -INFINITY;
    const bool live = l != -INFINITY;
    float qr[D], g[D], acc[D], bigd = 0.0f;
    for (int c = 0; c < D; ++c) {
        qr[c] = live ? q[at + c] : 0.0f; g[c] = live ? dy[at + c] : 0.0f; acc[c] = 0.0f;
        bigd += live ? g[c] * y[at + c] : 0.0f;
    }
    const u64 mine = t < n ? first_(t, window) : 0;
    for (u64 jt = first_(q0, window); jt <= qlast; jt += TK) {
        __syncthreads();
        const float* kt = k + ((u64)kvh * n + jt) * D; const float* vt = v + ((u64)kvh * n + jt) * D;
        for (int i = threadIdx.x; i < TK * D; i += blockDim.x) { const bool in = jt + i / D < n; ks[i] = in ? kt[i] : 0.0f; vs[i] = in ? vt[i] : 0.0f; }
        __syncthreads();
        if (!live) continue;
        for (int x = 0; x < TK; ++x) {
            const u64 j = jt + x; if (j < mine || j > t) continue;
            const float r = scale * dot8_<D>(qr, ks + x * D), gv = dot8_<D>(g, vs + x * D);
            float ds;
            if (cap > 0.0f) { const float th = tanhf(r / cap); ds = expf(cap * th - l) * (gv - bigd) * (1.0f - th * th); }
            else ds = expf(r - l) * (gv - bigd);
            for (int c = 0; c < D; ++c) acc[c] += scale * ds * ks[x * D + c];
        }
    }
    if (t >= n) return;
    Dd[t * H + h] = bigd;
    for (int c = 0; c < D; ++c) gq[at + c] = acc[c];
}
// Keys and values: a thread per key, its key and value rows in shared memory (padded against bank conflicts), the
// queries and their output gradients staged TQ tokens at a time, one query head after another
template <int D>
__device__ void attend_backward_keys_tiled_(const float* q, const float* k, const float* v, const float* lse, const float* dy, const float* Dd,
                                            float* pk, float* pv, u64 n, int H, int KVH, u64 window, float scale, float cap)
{
    constexpr int TQ = 1024 / D, KB = 32;   // query tokens per tile; keys per block
    __shared__ float qs[TQ * D], gs[TQ * D], ls[TQ], ds_[TQ], kj[KB * (D + 1)], vj[KB * (D + 1)];
    const int kvh = blockIdx.y, group = H / KVH;
    const u64 j0 = (u64)blockIdx.x * KB, j = j0 + threadIdx.x, jlast = min(j0 + KB, n) - 1;
    const bool live = j < n;
    const u64 at = ((u64)kvh * n + j) * D;
    float ak[D], av[D];
    for (int c = 0; c < D; ++c) {
        kj[threadIdx.x * (D + 1) + c] = live ? k[at + c] : 0.0f; vj[threadIdx.x * (D + 1) + c] = live ? v[at + c] : 0.0f;
        ak[c] = 0.0f; av[c] = 0.0f;
    }
    const float* kr = kj + threadIdx.x * (D + 1); const float* vr = vj + threadIdx.x * (D + 1);
    const u64 last = (window == 0 || j + window - 1 >= n - 1) ? n - 1 : j + window - 1;
    const u64 tend = (window == 0 || jlast + window - 1 >= n - 1) ? n - 1 : jlast + window - 1;
    // This block's share of the query tiles: split z of gridDim.z, contiguous, so the partials add up in a fixed order
    const u64 tiles = (tend - j0) / TQ + 1, from = tiles * blockIdx.z / gridDim.z, to = tiles * (blockIdx.z + 1) / gridDim.z;
    for (u64 tt = j0 + from * TQ; tt < j0 + to * TQ; tt += TQ)
        for (int h = kvh * group; h < (kvh + 1) * group; ++h) {
            __syncthreads();
            for (int i = threadIdx.x; i < TQ * D; i += blockDim.x) {
                const u64 t = tt + i / D; const bool in = t < n; const u64 qa = t * (u64)(H * D) + (u64)h * D + i % D;
                qs[i] = in ? q[qa] : 0.0f; gs[i] = in ? dy[qa] : 0.0f;
            }
            for (int i = threadIdx.x; i < TQ; i += blockDim.x) { const bool in = tt + i < n; ls[i] = in ? lse[(tt + i) * H + h] : -INFINITY; ds_[i] = in ? Dd[(tt + i) * H + h] : 0.0f; }
            __syncthreads();
            if (!live) continue;
            for (int x = 0; x < TQ; ++x) {
                const u64 t = tt + x; if (t < j || t > last) continue;
                const float l = ls[x]; if (l == -INFINITY) continue;
                const float r = scale * dot8_<D>(qs + x * D, kr), gvd = dot8_<D>(gs + x * D, vr);
                float p, ds;
                if (cap > 0.0f) { const float th = tanhf(r / cap); p = expf(cap * th - l); ds = p * (gvd - ds_[x]) * (1.0f - th * th); }
                else { p = expf(r - l); ds = p * (gvd - ds_[x]); }
                for (int c = 0; c < D; ++c) { av[c] += p * gs[x * D + c]; ak[c] += scale * ds * qs[x * D + c]; }
            }
        }
    if (!live) return;
    const u64 part = (u64)blockIdx.z * KVH * n * D + at;
    for (int c = 0; c < D; ++c) { pk[part + c] = ak[c]; pv[part + c] = av[c]; }
}
// The keys' and values' gradients: each split's partial added in, split by split
extern "C" __global__ void attend_keys_reduce(float* gk, float* gv, const float* pk, const float* pv, u64 m, int splits)
{
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= m) return;
    float a = gk[i], b = gv[i];
    for (int z = 0; z < splits; ++z) { a += pk[z * m + i]; b += pv[z * m + i]; }
    gk[i] = a; gv[i] = b;
}
// Attention as matrix products, one head at a time: the scores S (n x n, row t the query) come from cuBLAS; these
// kernels make them probabilities and back. A warp per row, its sums lane by lane in order and then a fixed tree.
// The softmax: S becomes P in place, masked to the visible keys and capped; lse[t H + h] written
extern "C" __global__ void attn_softmax(float* S, float* lse, u64 n, int H, int h, u64 window, float cap)
{
    const u64 t = (blockIdx.x * (u64)blockDim.x + threadIdx.x) / LANES; const int lane = threadIdx.x % LANES;
    if (t >= n) return;
    float* row = S + t * n; const u64 lo = first_(t, window);
    float mx = -INFINITY;
    for (u64 j = lo + lane; j <= t; j += LANES) { float sc = row[j]; if (cap > 0.0f) sc = cap * tanhf(sc / cap); row[j] = sc; mx = fmaxf(mx, sc); }
    for (int o = LANES / 2; o > 0; o /= 2) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, o));
    mx = __shfl_sync(0xffffffffu, mx, 0);
    float sum = 0.0f;
    if (mx != -INFINITY) for (u64 j = lo + lane; j <= t; j += LANES) sum += expf(row[j] - mx);
    sum = warp_sum_(sum);
    const float l = sum == 0.0f ? -INFINITY : mx + logf(sum);
    for (u64 j = lane; j < n; j += LANES) row[j] = (j >= lo && j <= t && l != -INFINITY) ? expf(row[j] - l) : 0.0f;
    if (lane == 0) lse[t * H + h] = l;
}
// The probabilities again, from the raw scores and the saved lse: P written, S kept
extern "C" __global__ void attn_probs(const float* S, float* P, const float* lse, u64 n, int H, int h, u64 window, float cap)
{
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= n * n) return;
    const u64 t = i / n, j = i % n; const float l = lse[t * H + h];
    float sc = S[i]; if (cap > 0.0f) sc = cap * tanhf(sc / cap);
    P[i] = (j >= first_(t, window) && j <= t && l != -INFINITY) ? expf(sc - l) : 0.0f;
}
// The scores' gradient, in place over dP: dS = P (dP - D_t), times 1 - tanh^2 under a cap, D_t = dy_t . y_t
extern "C" __global__ void attn_dscores(float* dP, const float* P, const float* S, const float* dy, const float* y, u64 n, int H, int h, int d, float cap)
{
    const u64 t = (blockIdx.x * (u64)blockDim.x + threadIdx.x) / LANES; const int lane = threadIdx.x % LANES;
    if (t >= n) return;
    const u64 at = t * (u64)(H * d) + (u64)h * d;
    const float bigd = warp_dot_(dy + at, y + at, d, lane);
    for (u64 j = lane; j < n; j += LANES) {
        const u64 i = t * n + j;
        float v = P[i] * (dP[i] - bigd);
        if (cap > 0.0f) { const float th = tanhf(S[i] / cap); v *= 1.0f - th * th; }
        dP[i] = v;
    }
}
#define SNICHOLLS_ATTEND_TILED(D)                                                                                                        \
    extern "C" __global__ void attend_tiled_##D(const float* q, const float* k, const float* v, float* y, float* lse, u64 n, int H, int KVH, \
                                                u64 window, float scale, float cap)                                                   \
    { attend_tiled_<D>(q, k, v, y, lse, n, H, KVH, window, scale, cap); }                                                              \
    extern "C" __global__ void attend_backward_queries_tiled_##D(const float* q, const float* k, const float* v, const float* y,      \
        const float* lse, const float* dy, float* gq, float* Dd, u64 n, int H, int KVH, u64 window, float scale, float cap)           \
    { attend_backward_queries_tiled_<D>(q, k, v, y, lse, dy, gq, Dd, n, H, KVH, window, scale, cap); }                                 \
    extern "C" __global__ void attend_backward_keys_tiled_##D(const float* q, const float* k, const float* v, const float* lse,       \
        const float* dy, const float* Dd, float* pk, float* pv, u64 n, int H, int KVH, u64 window, float scale, float cap)            \
    { attend_backward_keys_tiled_<D>(q, k, v, lse, dy, Dd, pk, pv, n, H, KVH, window, scale, cap); }
SNICHOLLS_ATTEND_TILED(16)
SNICHOLLS_ATTEND_TILED(32)
SNICHOLLS_ATTEND_TILED(64)
SNICHOLLS_ATTEND_TILED(128)
extern "C" {
// The model's other steps: embeddings, the fused classifier, AdamW, and the reductions
__global__ void embed(const float* table, const float* pos, const unsigned* ids, float* out, u64 n, u64 d)
{ const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= n * d) return; const u64 t = i / d, c = i % d;
  const float e = table[(u64)ids[t] * d + c]; out[i] = pos ? e + pos[i] : e; }
// One thread per channel: each token's row added into its table row in token order (as core's), and into its position's
__global__ void embed_backward(const float* dout, const unsigned* ids, float* dtable, float* dpos, u64 n, u64 d)
{ const u64 c = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (c >= d) return;
  for (u64 t = 0; t < n; ++t) { dtable[(u64)ids[t] * d + c] += dout[t * d + c]; if (dpos) dpos[t * d + c] += dout[t * d + c]; } }
// One warp per row: core's cross_entropy_with_gradient in place (the loss, then the row replaced by scale (softmax - onehot))
__global__ void cross_entropy_rows(float* logits, const unsigned* targets, float* losses, u64 n, u64 V, float scale)
{
    const u64 t = (blockIdx.x * (u64)blockDim.x + threadIdx.x) / 32; const int lane = threadIdx.x % 32;
    if (t >= n) return;
    float* x = logits + t * V; const unsigned target = targets[t];
    float mx = -INFINITY; for (u64 i = lane; i < V; i += 32) mx = fmaxf(mx, x[i]);
    for (int o = 16; o > 0; o /= 2) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, o));
    mx = __shfl_sync(0xffffffffu, mx, 0);
    float s = 0.0f; for (u64 i = lane; i < V; i += 32) s += expf(x[i] - mx);
    for (int o = 16; o > 0; o /= 2) s += __shfl_down_sync(0xffffffffu, s, o);
    s = __shfl_sync(0xffffffffu, s, 0);
    const float lse = mx + logf(s), xt = x[target];
    __syncwarp();   // every lane has read x[target] before any lane overwrites the row
    for (u64 i = lane; i < V; i += 32) x[i] = scale * (expf(x[i] - lse) - (i == target ? 1.0f : 0.0f));
    if (lane == 0) losses[t] = lse - xt;
}
__global__ void adamw(float* p, const float* g, float* m, float* v, u64 n, float lr, float b1, float b2, float eps, float wd, float c1, float c2)
{
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= n) return;
    const float gi = g[i];
    const float mi = b1 * m[i] + (1.0f - b1) * gi, vi = b2 * v[i] + (1.0f - b2) * gi * gi;
    m[i] = mi; v[i] = vi;
    const float m_hat = mi / c1, v_hat = vi / c2;
    p[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + wd * p[i]);
}
__global__ void scale(float* x, u64 n, float alpha) { const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i < n) x[i] *= alpha; }
// A fixed grid of 256-thread blocks: each thread's strided part, then the block's tree in shared memory; the partials are
// summed on the host in order, so a sum does not depend on scheduling
__global__ void partial_sums(const float* x, u64 n, int squares, float* partials)
{
    __shared__ float sh[256];
    float s = 0.0f;
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) s += squares ? x[i] * x[i] : x[i];
    sh[threadIdx.x] = s; __syncthreads();
    for (int o = 128; o > 0; o /= 2) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) partials[blockIdx.x] = sh[0];
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

} // namespace detail::cuda_ops

/// \brief The kernels of the training steps, compiled for one context's device, and their working space.
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
            {"norm_rows_backward_add", &k->norm_rows_backward_add_}, {"norm_param_grads", &k->norm_param_grads_}, {"position", &k->position_},
            {"position_backward", &k->position_backward_}, {"attend", &k->attend_}, {"attend_backward_queries", &k->attend_backward_queries_},
            {"attend_backward_keys", &k->attend_backward_keys_}, {"embed", &k->embed_}, {"embed_backward", &k->embed_backward_},
            {"cross_entropy_rows", &k->cross_entropy_rows_}, {"adamw", &k->adamw_}, {"scale", &k->scale_}, {"partial_sums", &k->partial_sums_},
            {"attend_keys_reduce", &k->attend_keys_reduce_}, {"attn_softmax", &k->attn_softmax_}, {"attn_probs", &k->attn_probs_},
            {"attn_dscores", &k->attn_dscores_}};
        std::vector<std::pair<std::string, cudaKernel_t*>> all;
        for (const auto& [name, slot] : kernels)
            all.emplace_back(name, slot);
        for (std::size_t i = 0; i < 4; ++i) {
            const std::string d = std::to_string(16 << i);
            all.emplace_back("attend_tiled_" + d, &k->attend_tiled_[i]);
            all.emplace_back("attend_backward_queries_tiled_" + d, &k->attend_backward_queries_tiled_[i]);
            all.emplace_back("attend_backward_keys_tiled_" + d, &k->attend_backward_keys_tiled_[i]);
        }
        for (const auto& [name, slot] : all)
            if (const cudaError_t e = cudaLibraryGetKernel(slot, k->library_, name.c_str()); e != cudaSuccess) {
                std::fprintf(stderr, "snicholls::gpu::cuda_kernels: cudaLibraryGetKernel(%s): %s\n", name.c_str(), cudaGetErrorString(e));
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

    /// \brief Launches a kernel with an explicit grid of `block`-thread blocks. \param k The kernel. \param grid The blocks.
    /// \param block The threads per block. \param args The arguments. \throws cuda_exception if the launch fails.
    void launch_grid(cudaKernel_t k, unsigned grid, unsigned block, void** args) const
    {
        if (const cudaError_t e = cudaLaunchKernel(reinterpret_cast<const void*>(k), dim3(grid), dim3(block), args, 0, context_->stream());
            e != cudaSuccess)
            throw cuda_exception(std::string("cudaLaunchKernel(") + name_of(k) + ") [" + cudaGetErrorString(e) + "]", static_cast<int>(e));
    }

    /// \brief Launches a kernel over a two-dimensional grid. \param k The kernel. \param x The blocks along x.
    /// \param y The blocks along y. \param block The threads per block. \param args The arguments.
    /// \throws cuda_exception if the launch fails.
    void launch_grid2(cudaKernel_t k, unsigned x, std::size_t y, unsigned block, void** args) const { launch_grid3(k, x, y, 1, block, args); }

    /// \brief Launches a kernel over a three-dimensional grid. \param k The kernel. \param x The blocks along x.
    /// \param y The blocks along y. \param z The blocks along z. \param block The threads per block.
    /// \param args The arguments. \throws cuda_exception if the launch fails.
    void launch_grid3(cudaKernel_t k, unsigned x, std::size_t y, std::size_t z, unsigned block, void** args) const
    {
        if (x == 0 || y == 0 || z == 0)
            return;
        if (const cudaError_t e = cudaLaunchKernel(reinterpret_cast<const void*>(k), dim3(x, unsigned(y), unsigned(z)), dim3(block), args, 0,
                                                   context_->stream());
            e != cudaSuccess)
            throw cuda_exception(std::string("cudaLaunchKernel(") + name_of(k) + ") [" + cudaGetErrorString(e) + "]", static_cast<int>(e));
    }

    /// \brief Working space in device memory of at least `floats` floats, kept between calls and grown when asked for
    /// more: the kernels' scratch, so a step allocates nothing. Its contents are the caller's until the next call; the
    /// kernels are for one host thread at a time, as their context's one stream is.
    /// \param floats The size. \return The space.
    [[nodiscard]] std::span<float> scratch(std::size_t floats) const
    {
        if (!scratch_ || scratch_->span().size() < floats)
            scratch_ = std::make_unique<device_buffer<float>>(*context_, floats);
        return scratch_->span().first(floats);
    }

    /// \brief The context. \return It.
    [[nodiscard]] const cuda_context& context() const noexcept { return *context_; }

    /// \brief A kernel's name, for messages. \param k The kernel. \return Its name.
    [[nodiscard]] const char* name_of(cudaKernel_t k) const noexcept
    {
        const std::pair<cudaKernel_t, const char*> names[] = {
            {fill_, "fill"}, {copy_, "copy"}, {add_, "add"}, {add_bias_rows_, "add_bias_rows"}, {sum_rows_, "sum_rows"}, {activate_, "activate"},
            {activate_backward_, "activate_backward"}, {norm_rows_, "norm_rows"}, {norm_rows_backward_add_, "norm_rows_backward_add"},
            {norm_param_grads_, "norm_param_grads"}, {position_, "position"}, {position_backward_, "position_backward"}, {attend_, "attend"},
            {attend_backward_queries_, "attend_backward_queries"}, {attend_backward_keys_, "attend_backward_keys"}, {embed_, "embed"},
            {embed_backward_, "embed_backward"}, {cross_entropy_rows_, "cross_entropy_rows"}, {adamw_, "adamw"}, {scale_, "scale"},
            {partial_sums_, "partial_sums"}, {attend_keys_reduce_, "attend_keys_reduce"}, {attn_softmax_, "attn_softmax"},
            {attn_probs_, "attn_probs"}, {attn_dscores_, "attn_dscores"}};
        for (const auto& [kernel, name] : names)
            if (kernel == k)
                return name;
        for (std::size_t i = 0; i < 4; ++i)
            if (k == attend_tiled_[i] || k == attend_backward_queries_tiled_[i] || k == attend_backward_keys_tiled_[i])
                return k == attend_tiled_[i] ? "attend_tiled" : k == attend_backward_queries_tiled_[i] ? "attend_backward_queries_tiled" : "attend_backward_keys_tiled";
        return "?";
    }

    /// \name The kernels, for the overloads below
    /// @{
    cudaKernel_t fill_ = nullptr;   ///< `fill`.
    cudaKernel_t copy_ = nullptr;   ///< `copy`.
    cudaKernel_t add_ = nullptr;   ///< `add`.
    cudaKernel_t add_bias_rows_ = nullptr;   ///< `add_bias_rows`.
    cudaKernel_t sum_rows_ = nullptr;   ///< `sum_rows`.
    cudaKernel_t activate_ = nullptr;   ///< `activate`.
    cudaKernel_t activate_backward_ = nullptr;   ///< `activate_backward`.
    cudaKernel_t norm_rows_ = nullptr;   ///< `norm_rows`.
    cudaKernel_t norm_rows_backward_add_ = nullptr;   ///< `norm_rows_backward_add`.
    cudaKernel_t norm_param_grads_ = nullptr;   ///< `norm_param_grads`.
    cudaKernel_t position_ = nullptr;   ///< `position`.
    cudaKernel_t position_backward_ = nullptr;   ///< `position_backward`.
    cudaKernel_t attend_ = nullptr;   ///< `attend`.
    cudaKernel_t attend_backward_queries_ = nullptr;   ///< `attend_backward_queries`.
    cudaKernel_t attend_backward_keys_ = nullptr;   ///< `attend_backward_keys`.
    cudaKernel_t embed_ = nullptr;   ///< `embed`.
    cudaKernel_t embed_backward_ = nullptr;   ///< `embed_backward`.
    cudaKernel_t cross_entropy_rows_ = nullptr;   ///< `cross_entropy_rows`.
    cudaKernel_t adamw_ = nullptr;   ///< `adamw`.
    cudaKernel_t scale_ = nullptr;   ///< `scale`.
    cudaKernel_t partial_sums_ = nullptr;   ///< `partial_sums`.
    cudaKernel_t attend_keys_reduce_ = nullptr;   ///< `attend_keys_reduce`.
    cudaKernel_t attn_softmax_ = nullptr;         ///< `attn_softmax`.
    cudaKernel_t attn_probs_ = nullptr;           ///< `attn_probs`.
    cudaKernel_t attn_dscores_ = nullptr;         ///< `attn_dscores`.
    std::array<cudaKernel_t, 4> attend_tiled_{};                    ///< `attend_tiled_D`, for heads of 16, 32, 64 and 128 channels.
    std::array<cudaKernel_t, 4> attend_backward_queries_tiled_{};   ///< `attend_backward_queries_tiled_D`.
    std::array<cudaKernel_t, 4> attend_backward_keys_tiled_{};      ///< `attend_backward_keys_tiled_D`.
    /// @}

private:
    explicit cuda_kernels(const cuda_context& c) : context_(&c) {}
    const cuda_context* context_;
    cudaLibrary_t library_ = nullptr;
    mutable std::unique_ptr<device_buffer<float>> scratch_;
};

/// \brief The CUDA backend with the training steps' kernels: core's batched passes run entirely on the device over
/// `device_buffer` memory. Trivially copyable.
struct cuda_training_backend : cuda_backend {
    const cuda_kernels* kernels = nullptr;   ///< The compiled kernels; must outlive the backend.
    std::size_t gemm_attention_max = 4096;   ///< The longest sequence whose attention runs as matrix products (cuBLAS,
                                             ///< so on the tensor cores under `cuda_math::tf32`), its `n x n` scores
                                             ///< three times in the kernels' scratch; longer ones take the tiled kernels.
};

static_assert(linalg::blas_backend<cuda_training_backend, float> && std::is_trivially_copyable_v<cuda_training_backend>);

namespace detail::cuda_ops {
    using u64 = unsigned long long;
    inline void sync(const cuda_training_backend& b) { cuda::check(cudaStreamSynchronize(b.context->stream()), "cudaStreamSynchronize"); }
    template <typename T>
    void* arg(const T& v) { return const_cast<void*>(static_cast<const void*>(&v)); }
    // The tiled attention kernels' index for a head size, or -1 where only the warp-per-row kernels serve
    template <std::size_t D>
    constexpr int tiled = D == 16 ? 0 : D == 32 ? 1 : D == 64 ? 2 : D == 128 ? 3 : -1;
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
    b.kernels->launch(b.kernels->norm_rows_, 32 * rows, args);   // a warp per row
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
    b.kernels->launch(b.kernels->norm_rows_, 32 * rows, a1);
    const float* g = dh.data();
    float* gx = dxs.data();
    void* a2[] = {&g, &xp, &wp, &sp, &gx, dc::arg(rows), dc::arg(d), &layer};
    b.kernels->launch(b.kernels->norm_rows_backward_add_, 32 * rows, a2);
    float* gw = dweight.empty() ? nullptr : dweight.data();
    float* gb = (bias.empty() || dbias.empty()) ? nullptr : dbias.data();
    void* a3[] = {&g, &xp, &sp, &gw, &gb, dc::arg(rows), dc::arg(d), &layer};
    if (gw != nullptr || gb != nullptr)
        b.kernels->launch(b.kernels->norm_param_grads_, d, a3);
    dc::sync(b);   // the scratch buffers are freed on return
}

namespace detail::cuda_ops {
    // The rope source's cosines and sines for positions 0 to n - 1, in device memory (the source lives on the host)
    template <ml::transformer_config C, typename R>
    struct rope_rows {
        device_buffer<float> cos, sin;
        rope_rows(const cuda_context& ctx, const R& rope_src, std::size_t n) : cos(ctx, n * (C.head_dim / 2)), sin(ctx, n * (C.head_dim / 2))
        {
            if constexpr (C.rope != ml::rope_kind::none) {
                std::vector<float> c(n * (C.head_dim / 2)), s(c.size());
                for (std::size_t t = 0; t < n; ++t) {
                    const auto ct = rope_src.cos_at(t);
                    const auto st = rope_src.sin_at(t);
                    for (std::size_t i = 0; i < C.head_dim / 2; ++i)
                        c[t * (C.head_dim / 2) + i] = ct[i], s[t * (C.head_dim / 2) + i] = st[i];
                }
                cos.upload(c);
                sin.upload(s);
            } else {
                (void)rope_src;
            }
        }
    };
    template <ml::transformer_config C>
    constexpr int qk_mode = C.qk_norm == ml::qk_norm_kind::none ? 0 : C.qk_norm == ml::qk_norm_kind::before_rope ? 1 : 2;
    template <typename S>
    const float* data_or_null(S s) { return s.empty() ? nullptr : s.data(); }
    template <typename S>
    float* mutable_or_null(S s) { return s.empty() ? nullptr : s.data(); }
} // namespace detail::cuda_ops

/// \brief QK norm and RoPE of every token's heads on the device, after the raw copies.
/// \tparam C The configuration. \param b The backend. \param w The weights (the norms', device memory).
/// \param act The activations (device memory). \param n The tokens. \param rope_src The rope source (host memory).
template <ml::transformer_config C, typename W, typename R>
void attention_position_rows(const cuda_training_backend& b, const ml::attention_weights<W, float>& w,
                             const ml::attention_activations<float>& act, std::size_t n, const R& rope_src)
{
    namespace dc = detail::cuda_ops;
    if constexpr (C.qk_norm == ml::qk_norm_kind::none && C.rope == ml::rope_kind::none) {
        (void)b, (void)w, (void)act, (void)n, (void)rope_src;   // nothing to do
    } else {
        static_assert(C.head_dim <= 256, "snicholls::gpu: the device kernels take heads of at most 256 channels");
        const dc::rope_rows<C, R> table(*b.context, rope_src, n);
        float* q = act.q.data();
        float* k = act.k.data();
        float* qr = dc::mutable_or_null(act.q_raw);
        float* kr = dc::mutable_or_null(act.k_raw);
        const float* qn = dc::data_or_null(w.q_norm);
        const float* kn = dc::data_or_null(w.k_norm);
        const float* cs = table.cos.span().data();
        const float* sn = table.sin.span().data();
        const dc::u64 nn = n;
        int H = int(C.n_heads), KVH = int(C.n_kv_heads), d = int(C.head_dim), qk = dc::qk_mode<C>, rope = C.rope == ml::rope_kind::none ? 0 : 1,
            half = C.layout() == ml::rope_layout::half ? 1 : 0;
        float eps = ml::norm_eps_v<C, float>;
        void* args[] = {&q, &k, &qr, &kr, &qn, &kn, &cs, &sn, dc::arg(nn), &H, &KVH, &d, &qk, &rope, &half, &eps};
        b.kernels->launch(b.kernels->position_, n * (C.n_heads + C.n_kv_heads), args);
        dc::sync(b);   // the uploaded table is freed on return
    }
}

namespace detail::cuda_ops {
    // A strided view of device memory for the backend's products
    inline linalg::matrix_view<const float> in(const float* p, std::size_t r, std::size_t c, std::size_t s0, std::size_t s1)
    {
        return linalg::matrix_view<const float>(p, {r, c}, {s0, s1});
    }
    inline linalg::matrix_view<float> out(float* p, std::size_t r, std::size_t c, std::size_t s0, std::size_t s1)
    {
        return linalg::matrix_view<float>(p, {r, c}, {s0, s1});
    }
} // namespace detail::cuda_ops

/// \brief Every token's heads over its visible keys on the device: `act.y` and `act.lse`. \tparam C The configuration.
/// \param b The backend. \param act The activations (device memory). \param n The tokens.
template <ml::transformer_config C>
void attention_attend_rows(const cuda_training_backend& b, const ml::attention_activations<float>& act, std::size_t n)
{
    namespace dc = detail::cuda_ops;
    static_assert(C.head_dim <= 256, "snicholls::gpu: the device kernels take heads of at most 256 channels");
    const float* q = act.q.data();
    const float* k = act.k.data();
    const float* v = act.v.data();
    float* y = act.y.data();
    float* lse = act.lse.data();
    const dc::u64 nn = n, window = C.window;
    int H = int(C.n_heads), KVH = int(C.n_kv_heads), d = int(C.head_dim);
    float scale = 1.0f / snicholls::math::sqrt(float(C.head_dim)), cap = ml::attention_softcap_v<C, float>;
    if (n <= b.gemm_attention_max) {
        // Per head: S = scale Q_h K^T, the softmax in place (lse written), Y_h = P V
        constexpr std::size_t dh = C.head_dim, hd = C.n_heads * C.head_dim, group = C.n_heads / C.n_kv_heads;
        float* S = b.kernels->scratch(n * n).data();
        for (std::size_t h = 0; h < C.n_heads; ++h) {
            const std::size_t kvh = h / group;
            b.context->gemm(scale, dc::in(q + h * dh, n, dh, hd, 1), dc::in(k + kvh * n * dh, dh, n, 1, dh), 0.0f, dc::out(S, n, n, n, 1), b.math);
            int hh = int(h);
            void* args[] = {&S, &lse, dc::arg(nn), &H, &hh, dc::arg(window), &cap};
            b.kernels->launch(b.kernels->attn_softmax_, 32 * n, args);
            b.context->gemm(1.0f, dc::in(S, n, n, n, 1), dc::in(v + kvh * n * dh, n, dh, dh, 1), 0.0f, dc::out(y + h * dh, n, dh, hd, 1), b.math);
        }
    } else if constexpr (dc::tiled<C.head_dim> >= 0) {
        void* args[] = {&q, &k, &v, &y, &lse, dc::arg(nn), &H, &KVH, dc::arg(window), &scale, &cap};
        b.kernels->launch_grid2(b.kernels->attend_tiled_[dc::tiled<C.head_dim>], unsigned((n + 63) / 64), C.n_heads, 64, args);
    } else {
        void* args[] = {&q, &k, &v, &y, &lse, dc::arg(nn), &H, &KVH, &d, dc::arg(window), &scale, &cap};
        b.kernels->launch(b.kernels->attend_, 32 * n * C.n_heads, args);   // a warp per (token, head)
    }
}

/// \brief The backward pass of the heads on the device, deterministic: the queries' gradients written, the keys' and
/// values' added in core's order. \tparam C The configuration. \param b The backend. \param act The activations.
/// \param dy The heads' output gradients. \param grads The gradients. \param n The tokens.
template <ml::transformer_config C>
void attention_attend_rows_backward(const cuda_training_backend& b, const ml::attention_activations<float>& act,
                                    std::type_identity_t<std::span<const float>> dy, const ml::attention_gradients<float>& grads, std::size_t n)
{
    namespace dc = detail::cuda_ops;
    if (n <= b.gemm_attention_max) {
        // Per head, from the saved lse (nothing else kept): S = scale Q_h K^T, P from S, dV += P^T dY_h, dP = dY_h V^T,
        // dS from dP, dQ_h = scale dS K, dK += scale dS^T Q_h. A group's heads add into its keys' and values'
        // gradients in order.
        constexpr std::size_t dh = C.head_dim, hd = C.n_heads * C.head_dim, group = C.n_heads / C.n_kv_heads;
        const auto space = b.kernels->scratch(3 * n * n);
        float* S = space.data();
        float* P = S + n * n;
        float* dP = P + n * n;
        const float* q = act.q.data();
        const float* k = act.k.data();
        const float* v = act.v.data();
        const float* y = act.y.data();
        const float* lse = act.lse.data();
        const float* g = dy.data();
        const dc::u64 nn = n, window = C.window;
        int H = int(C.n_heads), d = int(C.head_dim);
        const float scale = 1.0f / snicholls::math::sqrt(float(C.head_dim));
        float cap = ml::attention_softcap_v<C, float>;
        for (std::size_t h = 0; h < C.n_heads; ++h) {
            const std::size_t kvh = h / group;
            const float* qh = q + h * dh;
            const float* gh = g + h * dh;
            const float* kk = k + kvh * n * dh;
            const float* vv = v + kvh * n * dh;
            int hh = int(h);
            b.context->gemm(scale, dc::in(qh, n, dh, hd, 1), dc::in(kk, dh, n, 1, dh), 0.0f, dc::out(S, n, n, n, 1), b.math);
            const float* cS = S;
            void* a1[] = {&cS, &P, &lse, dc::arg(nn), &H, &hh, dc::arg(window), &cap};
            b.kernels->launch(b.kernels->attn_probs_, n * n, a1);
            b.context->gemm(1.0f, dc::in(P, n, n, 1, n), dc::in(gh, n, dh, hd, 1), 1.0f, dc::out(grads.v.data() + kvh * n * dh, n, dh, dh, 1), b.math);
            b.context->gemm(1.0f, dc::in(gh, n, dh, hd, 1), dc::in(vv, dh, n, 1, dh), 0.0f, dc::out(dP, n, n, n, 1), b.math);
            const float* cP = P;
            void* a2[] = {&dP, &cP, &cS, &g, &y, dc::arg(nn), &H, &hh, &d, &cap};
            b.kernels->launch(b.kernels->attn_dscores_, 32 * n, a2);
            b.context->gemm(scale, dc::in(dP, n, n, n, 1), dc::in(kk, n, dh, dh, 1), 0.0f, dc::out(grads.q.data() + h * dh, n, dh, hd, 1), b.math);
            b.context->gemm(scale, dc::in(dP, n, n, 1, n), dc::in(qh, n, dh, hd, 1), 1.0f, dc::out(grads.k.data() + kvh * n * dh, n, dh, dh, 1), b.math);
        }
        return;
    }
    // The heads' D = dy . y, then the key splits' partials: the kernels' scratch
    constexpr std::size_t D_ = C.head_dim;
    const std::size_t splits = std::clamp<std::size_t>(n / 128, 1, 8), part = C.n_kv_heads * n * D_;
    const auto space = b.kernels->scratch(n * C.n_heads + (dc::tiled<D_> >= 0 ? 2 * splits * part : 0));
    const float* q = act.q.data();
    const float* k = act.k.data();
    const float* v = act.v.data();
    const float* y = act.y.data();
    const float* lse = act.lse.data();
    const float* g = dy.data();
    float* gq = grads.q.data();
    float* gk = grads.k.data();
    float* gv = grads.v.data();
    float* dp = space.data();
    const dc::u64 nn = n, window = C.window;
    int H = int(C.n_heads), KVH = int(C.n_kv_heads), d = int(C.head_dim);
    float scale = 1.0f / snicholls::math::sqrt(float(C.head_dim)), cap = ml::attention_softcap_v<C, float>;
    const float* dc_ = dp;
    if constexpr (dc::tiled<C.head_dim> >= 0) {
        constexpr int i = dc::tiled<C.head_dim>;
        void* a1[] = {&q, &k, &v, &y, &lse, &g, &gq, &dp, dc::arg(nn), &H, &KVH, dc::arg(window), &scale, &cap};
        b.kernels->launch_grid2(b.kernels->attend_backward_queries_tiled_[i], unsigned((n + 63) / 64), C.n_heads, 64, a1);
        float* pk = space.data() + n * C.n_heads;
        float* pv = pk + splits * part;
        void* a2[] = {&q, &k, &v, &lse, &g, &dc_, &pk, &pv, dc::arg(nn), &H, &KVH, dc::arg(window), &scale, &cap};
        b.kernels->launch_grid3(b.kernels->attend_backward_keys_tiled_[i], unsigned((n + 31) / 32), C.n_kv_heads, splits, 32, a2);
        const float* cpk = pk;
        const float* cpv = pv;
        const dc::u64 m = part;
        int z = int(splits);
        void* a3[] = {&gk, &gv, &cpk, &cpv, dc::arg(m), &z};
        b.kernels->launch(b.kernels->attend_keys_reduce_, part, a3);
    } else {
        void* a1[] = {&q, &k, &v, &y, &lse, &g, &gq, &dp, dc::arg(nn), &H, &KVH, &d, dc::arg(window), &scale, &cap};
        b.kernels->launch(b.kernels->attend_backward_queries_, 32 * n * C.n_heads, a1);
        void* a2[] = {&q, &k, &v, &lse, &g, &dc_, &gk, &gv, dc::arg(nn), &H, &KVH, &d, dc::arg(window), &scale, &cap};
        b.kernels->launch(b.kernels->attend_backward_keys_, 32 * n * C.n_kv_heads, a2);
    }
}

/// \brief The backward pass of QK norm and RoPE on the device, in place on `grads.q` and `grads.k`; the norm weights'
/// gradients summed over heads and positions in core's order. \tparam C The configuration. \param b The backend.
/// \param w The weights. \param act The activations. \param grads The gradients. \param n The tokens.
/// \param rope_src The rope source (host memory).
template <ml::transformer_config C, typename W, typename R>
void attention_position_rows_backward(const cuda_training_backend& b, const ml::attention_weights<W, float>& w,
                                      const ml::attention_activations<float>& act, const ml::attention_gradients<float>& grads,
                                      std::size_t n, const R& rope_src)
{
    namespace dc = detail::cuda_ops;
    if constexpr (C.qk_norm == ml::qk_norm_kind::none && C.rope == ml::rope_kind::none) {
        (void)b, (void)w, (void)act, (void)grads, (void)n, (void)rope_src;
    } else {
        constexpr bool qk = C.qk_norm != ml::qk_norm_kind::none;
        const dc::rope_rows<C, R> table(*b.context, rope_src, n);
        device_buffer<float> cq(*b.context, qk && !grads.q_norm.empty() ? n * C.n_heads * C.head_dim : 0),
            ck(*b.context, qk && !grads.k_norm.empty() ? n * C.n_kv_heads * C.head_dim : 0);
        float* gq = grads.q.data();
        float* gk = grads.k.data();
        const float* qr = dc::data_or_null(act.q_raw);
        const float* kr = dc::data_or_null(act.k_raw);
        const float* qn = dc::data_or_null(w.q_norm);
        const float* kn = dc::data_or_null(w.k_norm);
        const float* cs = table.cos.span().data();
        const float* sn = table.sin.span().data();
        float* cqp = dc::mutable_or_null(cq.span());
        float* ckp = dc::mutable_or_null(ck.span());
        const dc::u64 nn = n;
        int H = int(C.n_heads), KVH = int(C.n_kv_heads), d = int(C.head_dim), mode = dc::qk_mode<C>, rope = C.rope == ml::rope_kind::none ? 0 : 1,
            half = C.layout() == ml::rope_layout::half ? 1 : 0;
        float eps = ml::norm_eps_v<C, float>;
        void* args[] = {&gq, &gk, &qr, &kr, &qn, &kn, &cs, &sn, &cqp, &ckp, dc::arg(nn), &H, &KVH, &d, &mode, &rope, &half, &eps};
        b.kernels->launch(b.kernels->position_backward_, n * (C.n_heads + C.n_kv_heads), args);
        if (cqp != nullptr)   // the rows are (token, head) in core's order: summed in that order
            sum_rows(b, std::span<const float>(cq.span()), grads.q_norm);
        if (ckp != nullptr)
            sum_rows(b, std::span<const float>(ck.span()), grads.k_norm);
        dc::sync(b);
    }
}

/// \brief A sequence's embeddings on the device. \param b The backend. \param table The token table. \param positions
/// The position rows, or empty. \param ids The token ids (device memory). \param out The rows.
inline void embed_rows(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> table,
                       std::type_identity_t<std::span<const float>> positions, std::span<const std::uint32_t> ids, std::span<float> out)
{
    const float* tp = table.data();
    const float* pp = positions.empty() ? nullptr : positions.data();
    const std::uint32_t* ip = ids.data();
    float* op = out.data();
    const detail::cuda_ops::u64 n = ids.size(), d = ids.empty() ? 0 : out.size() / ids.size();
    void* args[] = {&tp, &pp, &ip, &op, detail::cuda_ops::arg(n), detail::cuda_ops::arg(d)};
    b.kernels->launch(b.kernels->embed_, out.size(), args);
}

/// \brief The backward pass of `embed_rows` on the device, in token order. \param b The backend. \param dout The
/// gradients in the rows. \param ids The ids. \param dtable Added into. \param dpositions Added into, or empty.
inline void embed_rows_backward(const cuda_training_backend& b, std::type_identity_t<std::span<const float>> dout,
                                std::span<const std::uint32_t> ids, std::span<float> dtable, std::type_identity_t<std::span<float>> dpositions)
{
    const float* dp = dout.data();
    const std::uint32_t* ip = ids.data();
    float* tp = dtable.data();
    float* pp = dpositions.empty() ? nullptr : dpositions.data();
    const detail::cuda_ops::u64 n = ids.size(), d = ids.empty() ? 0 : dout.size() / ids.size();
    void* args[] = {&dp, &ip, &tp, &pp, detail::cuda_ops::arg(n), detail::cuda_ops::arg(d)};
    b.kernels->launch(b.kernels->embed_backward_, d, args);
}

/// \brief The fused classifier over a sequence on the device: each row's loss, the row replaced by its scaled gradient.
/// \param b The backend. \param logits `n · V`. \param targets `n` (device memory). \param losses `n`. \param scale The
/// gradient's factor.
inline void cross_entropy_rows(const cuda_training_backend& b, std::span<float> logits, std::type_identity_t<std::span<const std::uint32_t>> targets,
                               std::span<float> losses, std::type_identity_t<float> scale)
{
    float* lp = logits.data();
    const std::uint32_t* tp = targets.data();
    float* sp = losses.data();
    const detail::cuda_ops::u64 n = targets.size(), v = targets.empty() ? 0 : logits.size() / targets.size();
    void* args[] = {&lp, &tp, &sp, detail::cuda_ops::arg(n), detail::cuda_ops::arg(v), &scale};
    b.kernels->launch(b.kernels->cross_entropy_rows_, 32 * targets.size(), args);   // a warp per row
}

/// \brief One AdamW step over a tensor on the device. \param b The backend. \param p The parameters. \param g Their
/// gradients. \param m The first moments. \param v The second moments. \param step The step, from 1. \param config The settings.
inline void adamw_rows(const cuda_training_backend& b, std::span<float> p, std::type_identity_t<std::span<const float>> g,
                       std::type_identity_t<std::span<float>> m, std::type_identity_t<std::span<float>> v, std::uint64_t step,
                       const ml::adamw_config<float>& config)
{
    float* pp = p.data();
    const float* gp = g.data();
    float* mp = m.data();
    float* vp = v.data();
    const detail::cuda_ops::u64 n = p.size();
    float lr = config.learning_rate, b1 = config.beta1, b2 = config.beta2, eps = config.eps, wd = config.weight_decay;
    float c1 = 1.0f - snicholls::math::pow(config.beta1, float(step)), c2 = 1.0f - snicholls::math::pow(config.beta2, float(step));
    void* args[] = {&pp, &gp, &mp, &vp, detail::cuda_ops::arg(n), &lr, &b1, &b2, &eps, &wd, &c1, &c2};
    b.kernels->launch(b.kernels->adamw_, p.size(), args);
}

/// \brief `x *= alpha` on the device. \param b The backend. \param x The values. \param alpha The factor.
inline void scale_rows(const cuda_training_backend& b, std::span<float> x, std::type_identity_t<float> alpha)
{
    float* xp = x.data();
    const detail::cuda_ops::u64 n = x.size();
    void* args[] = {&xp, detail::cuda_ops::arg(n), &alpha};
    b.kernels->launch(b.kernels->scale_, x.size(), args);
}

namespace detail::cuda_ops {
    inline float reduce(const cuda_training_backend& b, std::span<const float> x, int squares)
    {
        if (x.empty())
            return 0.0f;
        const unsigned grid = static_cast<unsigned>(std::min<std::size_t>(1024, (x.size() + 255) / 256));
        device_buffer<float> partials(*b.context, grid);
        const float* xp = x.data();
        float* pp = partials.span().data();
        const u64 n = x.size();
        void* args[] = {&xp, arg(n), &squares, &pp};
        b.kernels->launch_grid(b.kernels->partial_sums_, grid, 256, args);
        std::vector<float> host(grid);
        partials.download(host);   // after the kernel: the copy waits for the stream
        float s = 0.0f;
        for (const float v : host)
            s += v;   // the partials in order
        return s;
    }
} // namespace detail::cuda_ops

/// \brief `Σ x` on the device, in a fixed order. \param b The backend. \param x The values. \return The sum.
template <typename E>
    requires std::is_same_v<std::remove_const_t<E>, float>
float sum_values(const cuda_training_backend& b, std::span<E> x)
{
    return detail::cuda_ops::reduce(b, std::span<const float>(x), 0);
}

/// \brief `Σ x²` on the device, in a fixed order. \param b The backend. \param x The values. \return The sum.
template <typename E>
    requires std::is_same_v<std::remove_const_t<E>, float>
float sum_squares(const cuda_training_backend& b, std::span<E> x)
{
    return detail::cuda_ops::reduce(b, std::span<const float>(x), 1);
}

/// @}

} // namespace snicholls::gpu

#endif /* snicholls_gpu_cuda_ops_hpp */
