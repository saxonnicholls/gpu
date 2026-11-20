//
//  rows_source.hpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The row operations of a transformer's forward pass over device memory, written once for OpenCL C 1.2 and for CUDA
//  C++ (NVRTC) in the macros of quant_source.hpp's preludes: fills, copies, sums, biases, the token table's rows, RMS
//  and layer norms, the gated activations, QK norm and RoPE, attention against a key-value cache (which, with the
//  sequence's own keys as the cache, is attention within a sequence), the cache's writes, and the cross-entropy rows.
//  Every buffer is a base and an element offset, so the host can address device memory without pointers (OpenCL 1.2
//  has none). Each operation follows core's host arithmetic in the same order where one thread computes it (the
//  norms' sums, a head's norm and rotation, each key's score); the reductions a work-group shares (attention over the
//  keys, the softmax of a row) are in a fixed order, so two runs give the same bits. Not public API: opencl_device.hpp
//  builds its program from it, cuda_rows.hpp the CUDA one.
//

#ifndef snicholls_gpu_detail_rows_source_hpp
#define snicholls_gpu_detail_rows_source_hpp

namespace snicholls::gpu::detail::rows {

// The kernels; the prelude (quant_source.hpp's opencl_prelude or cuda_prelude) comes first, then `math_opencl` or
// `math_cuda`, then this
inline constexpr const char* math_opencl = R"CLC(
#define EXPF(x) exp(x)
#define LOGF(x) log(x)
#define SQRTF(x) sqrt(x)
#define NEG_INF (-INFINITY)
)CLC";

inline constexpr const char* math_cuda = R"CU(
#define EXPF(x) expf(x)
#define LOGF(x) logf(x)
#define SQRTF(x) sqrtf(x)
#define NEG_INF (-__int_as_float(0x7f800000))
)CU";

inline constexpr const char* source = R"SRC(
#define GID(G) (GX * (u64)(G) + (u64)LX)

KERNEL GROUP1(256) void rows_fill(GLOBAL float* x, u64 ox, u64 n, float v)
{
    const u64 i = GID(256);
    if (i < n)
        x[ox + i] = v;
}

KERNEL GROUP1(256) void rows_copy(GLOBAL const float* a, u64 oa, GLOBAL float* b, u64 ob, u64 n)
{
    const u64 i = GID(256);
    if (i < n)
        b[ob + i] = a[oa + i];
}

// x += a
KERNEL GROUP1(256) void rows_add(GLOBAL const float* a, u64 oa, GLOBAL float* x, u64 ox, u64 n)
{
    const u64 i = GID(256);
    if (i < n)
        x[ox + i] = x[ox + i] + a[oa + i];
}

// y_t += bias for every row of r
KERNEL GROUP1(256) void rows_add_bias(GLOBAL float* y, u64 oy, GLOBAL const float* bias, u64 ob, u64 n, u64 r)
{
    const u64 i = GID(256);
    if (i < n)
        y[oy + i] = y[oy + i] + bias[ob + i % r];
}

// out_t = table[ids_t] (+ positions_t)
KERNEL GROUP1(256) void rows_embed(GLOBAL const float* table, u64 ot, GLOBAL const float* pos, u64 op, int has_pos, GLOBAL const uint* ids,
                                   u64 oi, GLOBAL float* out, u64 oo, u64 n, u64 d)
{
    const u64 i = GID(256);
    if (i >= n * d)
        return;
    const u64 t = i / d, c = i % d;
    const float e = table[ot + (u64)ids[oi + t] * d + c];
    out[oo + i] = has_pos ? e + pos[op + i] : e;
}

// One row's RMS norm (kind 1) or layer norm (kind 0) by one work-item, in core's order (norm.hpp): the sums from the
// first element to the last, then each output
KERNEL GROUP1(64) void rows_norm(GLOBAL const float* x, u64 ox, GLOBAL const float* w, u64 ow, int has_w, GLOBAL const float* b, u64 ob,
                                 int has_b, GLOBAL float* y, u64 oy, u64 rows, u64 d, float eps, int kind)
{
    const u64 r = GID(64);
    if (r >= rows)
        return;
    GLOBAL const float* in = x + ox + r * d;
    GLOBAL float* out = y + oy + r * d;
    if (kind == 1) {
        float ms = 0.0f;
        for (u64 i = 0; i < d; ++i)
            ms += in[i] * in[i];
        ms = d == 0 ? 0.0f : ms / (float)d;
        const float rstd = 1.0f / SQRTF(ms + eps);
        for (u64 i = 0; i < d; ++i)
            out[i] = in[i] * rstd * (has_w ? w[ow + i] : 1.0f);
    } else {
        float mean = 0.0f;
        for (u64 i = 0; i < d; ++i)
            mean += in[i];
        mean = mean / (float)d;
        float var = 0.0f;
        for (u64 i = 0; i < d; ++i)
            var += (in[i] - mean) * (in[i] - mean);
        var = var / (float)d;
        const float rstd = 1.0f / SQRTF(var + eps);
        for (u64 i = 0; i < d; ++i)
            out[i] = (in[i] - mean) * rstd * (has_w ? w[ow + i] : 1.0f) + (has_b ? b[ob + i] : 0.0f);
    }
}

// SwiGLU: hidden = silu(gate) * up, silu(x) = x sigmoid(x) as core's activations.hpp computes it
KERNEL GROUP1(256) void rows_swiglu(GLOBAL const float* gate, u64 og, GLOBAL const float* up, u64 ou, GLOBAL float* hidden, u64 oh, u64 n)
{
    const u64 i = GID(256);
    if (i >= n)
        return;
    const float v = gate[og + i];
    float s;
    if (v >= 0.0f) {
        s = 1.0f / (1.0f + EXPF(-v));
    } else {
        const float e = EXPF(v);
        s = e / (1.0f + e);
    }
    hidden[oh + i] = v * s * up[ou + i];
}

// QK norm and RoPE of one head in place, by one work-item, as core's attention_layer position(): qk 1 before RoPE,
// 2 after; rope 1 rotates by the table's row t (cos and sin, hd2 = d / 2 values each); half_layout pairs i with i + d/2,
// else 2i with 2i + 1. Items [0, rows * H) are the queries ([t][h] rows of d), then [rows * H, rows * (H + KVH)) the keys
// ([kvh][t] rows of d, each sequence's n rows together). The raw head is copied to q_raw / k_raw first when asked.
DEVICE void head_norm(GLOBAL float* h, GLOBAL const float* w, u64 d, float eps)
{
    float ms = 0.0f;
    for (u64 i = 0; i < d; ++i)
        ms += h[i] * h[i];
    ms = d == 0 ? 0.0f : ms / (float)d;
    const float rstd = 1.0f / SQRTF(ms + eps);
    for (u64 i = 0; i < d; ++i)
        h[i] = h[i] * rstd * (w ? w[i] : 1.0f);
}

DEVICE void head_rope(GLOBAL float* h, GLOBAL const float* c, GLOBAL const float* s, u64 hd2, int half_layout)
{
    for (u64 i = 0; i < hd2; ++i) {
        const u64 ia = half_layout ? i : 2 * i, ib = half_layout ? i + hd2 : 2 * i + 1;
        const float x1 = h[ia], x2 = h[ib], sn = s[i];
        h[ia] = x1 * c[i] - x2 * sn;
        h[ib] = x1 * sn + x2 * c[i];
    }
}

KERNEL GROUP1(64) void rows_position(GLOBAL float* q, u64 oq, GLOBAL float* k, u64 ok, GLOBAL float* qr, u64 oqr, GLOBAL float* kr, u64 okr,
                                     int raw, GLOBAL const float* qn, u64 oqn, int has_qn, GLOBAL const float* kn, u64 okn, int has_kn,
                                     GLOBAL const float* cs, GLOBAL const float* sn, u64 n, u64 sequences, int H, int KVH, u64 d, int qk,
                                     int rope, int half_layout, float eps)
{
    const u64 item = GID(64), rows = n * sequences, hd2 = d / 2;
    if (item >= rows * (u64)(H + KVH))
        return;
    GLOBAL float* head;
    GLOBAL float* copy;
    GLOBAL const float* w;
    u64 t;
    if (item < rows * (u64)H) {
        const u64 r = item / (u64)H, h = item % (u64)H;   // r = s * n + t
        t = r % n;
        head = q + oq + (r * (u64)H + h) * d;
        copy = raw ? qr + oqr + (r * (u64)H + h) * d : 0;
        w = has_qn ? qn + oqn : 0;
    } else {
        const u64 j = item - rows * (u64)H, s = j / (n * (u64)KVH), rest = j % (n * (u64)KVH), h = rest / n;
        t = rest % n;
        head = k + ok + ((s * (u64)KVH + h) * n + t) * d;
        copy = raw ? kr + okr + ((s * n + t) * (u64)KVH + h) * d : 0;
        w = has_kn ? kn + okn : 0;
    }
    if (copy)
        for (u64 i = 0; i < d; ++i)
            copy[i] = head[i];
    if (qk == 1)
        head_norm(head, w, d, eps);
    if (rope)
        head_rope(head, cs + t * hd2, sn + t * hd2, hd2, half_layout);
    if (qk == 2)
        head_norm(head, w, d, eps);
}

// Attention of n queries at positions first .. first + n - 1 over a cache of capacity rows a key-value head
// (keys and values [kvh][capacity][d]), each query seeing the keys from max(0, p - window + 1) (all when window is 0)
// to its own position p. A work-group of 64 per (query, head), in core's order (attention_layer.hpp's attention_head):
// the keys in tiles of 64, lane l scoring key l of a tile (scale times the dot product, summed from the first channel),
// then every lane folding the tile's scores in key order (attention.hpp's fold: the running maximum, sum and weighted
// values) for the channels it holds (l and l + 64). y is [t][h][d], lse [t][h].
KERNEL GROUP1(64) void rows_attend(GLOBAL const float* q, u64 oq, GLOBAL const float* keys, u64 okeys, GLOBAL const float* values, u64 ovalues,
                                   u64 first, u64 n, u64 capacity, u64 window, int H, int KVH, u64 d, float scale, GLOBAL float* y, u64 oy,
                                   GLOBAL float* lse, u64 olse)
{
    LOCAL float qs[128];
    LOCAL float sc[64];
    const u64 g = GX, t = g / (u64)H, h = g % (u64)H;
    const int lane = LX;
    if (t >= n)
        return;
    const u64 kvh = h / (u64)(H / KVH), p = first + t, lo = window != 0 && p + 1 > window ? p + 1 - window : 0;
    GLOBAL const float* qh = q + oq + (t * (u64)H + h) * d;
    GLOBAL const float* kb = keys + okeys + kvh * capacity * d;
    GLOBAL const float* vb = values + ovalues + kvh * capacity * d;
    for (u64 i = (u64)lane; i < d; i += 64)
        qs[i] = qh[i];
    BARRIER();
    const u64 c0 = (u64)lane, c1 = (u64)lane + 64;
    float a0 = 0.0f, a1 = 0.0f, mx = NEG_INF, sum = 0.0f;
    for (u64 tile = lo; tile <= p; tile += 64) {
        const u64 j = tile + (u64)lane;
        if (j <= p) {
            GLOBAL const float* kj = kb + j * d;
            float dot = 0.0f;
            for (u64 i = 0; i < d; ++i)
                dot += qs[i] * kj[i];
            sc[lane] = scale * dot;
        }
        BARRIER();
        const u64 count = p - tile + 1 < 64 ? p - tile + 1 : 64;
        for (u64 u = 0; u < count; ++u) {
            const float s = sc[u];
            GLOBAL const float* vj = vb + (tile + u) * d;
            const float v0 = c0 < d ? vj[c0] : 0.0f, v1 = c1 < d ? vj[c1] : 0.0f;
            if (s > mx) {
                const float f = EXPF(mx - s);
                a0 = a0 * f + v0, a1 = a1 * f + v1;
                sum = sum * f + 1.0f;
                mx = s;
            } else {
                const float w = EXPF(s - mx);
                a0 += w * v0, a1 += w * v1;
                sum += w;
            }
        }
        BARRIER();
    }
    GLOBAL float* out = y + oy + (t * (u64)H + h) * d;
    if (c0 < d)
        out[c0] = sum > 0.0f ? a0 / sum : 0.0f;
    if (c1 < d)
        out[c1] = sum > 0.0f ? a1 / sum : 0.0f;
    if (lane == 0)
        lse[olse + t * (u64)H + h] = sum > 0.0f ? mx + LOGF(sum) : NEG_INF;
}

// The cache's rows for n new tokens at positions first .. first + n - 1: k and v [kvh][n][d] (the sequence path's
// layout) into keys and values [kvh][capacity][d]
KERNEL GROUP1(256) void rows_store(GLOBAL const float* k, u64 ok, GLOBAL const float* v, u64 ov, GLOBAL float* keys, u64 okeys,
                                   GLOBAL float* values, u64 ovalues, u64 first, u64 n, u64 capacity, int KVH, u64 d)
{
    const u64 i = GID(256);
    if (i >= (u64)KVH * n * d)
        return;
    const u64 h = i / (n * d), t = (i / d) % n, c = i % d, to = (h * capacity + first + t) * d + c;
    keys[okeys + to] = k[ok + i];
    values[ovalues + to] = v[ov + i];
}

// Each row's cross-entropy over its first `classes` columns, the row replaced by scale (softmax - onehot) and the
// padding by 0 (core's cross_entropy_rows): a work-group of 256 per row, the maximum and the sum in a fixed tree
KERNEL GROUP1(256) void rows_cross_entropy(GLOBAL float* x, u64 ox, GLOBAL const uint* targets, u64 ot, GLOBAL float* losses, u64 ol,
                                           u64 v, u64 classes, float scale)
{
    LOCAL float red[256];
    const u64 r = GX;
    const int lane = LX;
    GLOBAL float* row = x + ox + r * v;
    float m = NEG_INF;
    for (u64 i = (u64)lane; i < classes; i += 256)
        m = row[i] > m ? row[i] : m;
    red[lane] = m;
    BARRIER();
    for (int w = 128; w > 0; w /= 2) {
        if (lane < w)
            red[lane] = red[lane + w] > red[lane] ? red[lane + w] : red[lane];
        BARRIER();
    }
    m = red[0];
    BARRIER();
    float s = 0.0f;
    for (u64 i = (u64)lane; i < classes; i += 256)
        s += EXPF(row[i] - m);
    red[lane] = s;
    BARRIER();
    for (int w = 128; w > 0; w /= 2) {
        if (lane < w)
            red[lane] = red[lane] + red[lane + w];
        BARRIER();
    }
    s = red[0];
    const uint target = targets[ot + r];
    const float xt = row[target];
    BARRIER();
    if (lane == 0)
        losses[ol + r] = (m - xt) + LOGF(s);
    const float inv = 1.0f / s;
    for (u64 i = (u64)lane; i < v; i += 256)
        row[i] = i < classes ? scale * (EXPF(row[i] - m) * inv - (i == (u64)target ? 1.0f : 0.0f)) : 0.0f;
}

// Y = X Wᵀ for a few rows of X (decode): each output summed from the first column, as core's reference product sums it
// (the bits of linalg::reference). A work-group of 64 takes 64 outputs; the weights come through local memory in
// 64 x 64 tiles loaded along their rows (coalesced), and each item adds its own output's terms in column order.
// x [m][k] rows (m at most 4), w [out][k] rows, y [m][out].
KERNEL GROUP1(64) void rows_matvec(GLOBAL const float* x, u64 ox, GLOBAL const float* w, u64 ow, GLOBAL float* y, u64 oy, u64 m, u64 k, u64 out,
                                   u64 sx, u64 sy)
{
    LOCAL float tile[64][65];
    LOCAL float xs[4][64];
    const int lane = LX;
    const u64 j0 = GX * 64, j = j0 + (u64)lane;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (u64 p0 = 0; p0 < k; p0 += 64) {
        const u64 cols = k - p0 < 64 ? k - p0 : 64;
        for (int r = 0; r < 64; ++r) {   // row j0 + r of the tile: item lane reads column p0 + lane
            const u64 jr = j0 + (u64)r;
            tile[r][lane] = jr < out && (u64)lane < cols ? w[ow + jr * k + p0 + (u64)lane] : 0.0f;
        }
        for (u64 r = 0; r < m; ++r)
            xs[r][lane] = (u64)lane < cols ? x[ox + r * sx + p0 + (u64)lane] : 0.0f;
        BARRIER();
        if (j < out)
            for (u64 r = 0; r < m; ++r)
                for (u64 p = 0; p < cols; ++p)
                    acc[r] += xs[r][p] * tile[lane][p];
        BARRIER();
    }
    if (j < out)
        for (u64 r = 0; r < m; ++r)
            y[oy + r * sy + j] = acc[r];
}

// The sum of n values into out[0]: one work-group, a fixed tree
KERNEL GROUP1(256) void rows_sum(GLOBAL const float* x, u64 ox, u64 n, int squares, GLOBAL float* out)
{
    LOCAL float red[256];
    const int lane = LX;
    float s = 0.0f;
    for (u64 i = (u64)lane; i < n; i += 256)
        s += squares ? x[ox + i] * x[ox + i] : x[ox + i];
    red[lane] = s;
    BARRIER();
    for (int w = 128; w > 0; w /= 2) {
        if (lane < w)
            red[lane] = red[lane] + red[lane + w];
        BARRIER();
    }
    if (lane == 0)
        out[0] = red[0];
}
)SRC";

} // namespace snicholls::gpu::detail::rows

#endif // snicholls_gpu_detail_rows_source_hpp
