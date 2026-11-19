//
//  quant_source.hpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

// The quantised-weight kernels' source, written once for OpenCL C 1.2 and for CUDA C++ (NVRTC): a prelude per
// language maps a handful of macros (KERNEL, GLOBAL, LOCAL, BARRIER, the thread indices, the bit casts), and the
// formats and kernels below use only those. Not public API: opencl_quant.hpp and cuda_quant.hpp build their programs
// from it.
//
// A format is one device function, `get_<format>(WARGS, r, c)`: the weight at row r, column c, dequantised exactly as
// core's host code does it (ml/quantize.hpp, ml/nf4.hpp, ml/minifloat.hpp), so a product that adds the same products in
// the same order gives core's bits. Its host side is `traits<W>` below. Adding a format is those two and its entry in
// `formats`.

#ifndef snicholls_gpu_detail_quant_source_hpp
#define snicholls_gpu_detail_quant_source_hpp

#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/nf4.hpp>
#include <snicholls/core/ml/quantize.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace snicholls::gpu::detail::quant {

// The formats, in the order of `formats`; the device source names them by `name`
enum class format : int { q8_0, q4_0, mxfp4, nf4, bf16, f16, e4m3, e5m2 };
inline constexpr std::size_t format_count = 8;
inline constexpr std::array<std::string_view, format_count> names{"q8_0", "q4_0", "mxfp4", "nf4", "bf16", "f16", "e4m3", "e5m2"};

// Whether a format has core's integer dot product with Q8_0 activations (ml/quantize.hpp `dot`)
constexpr bool has_q8_dot(format f) noexcept { return f == format::q8_0 || f == format::q4_0 || f == format::mxfp4; }

// The tokens a matrix-vector launch takes at once: as MV_TOKENS in the source
inline constexpr std::size_t mv_tokens = 8;
// Blocks of Q8_0 activations: 34 bytes of 32 values
inline constexpr std::size_t q8_bytes = sizeof(ml::block_q8_0);

// A weight storage's bytes on the device: the weights, the block scales and group scales of NF4 (empty otherwise),
// NF4's mean, the shape
struct parts {
    format fmt = format::q8_0;
    std::span<const std::byte> weights{}, block_scales{}, group_scales{};
    float mean = 0.0f;
    std::size_t rows = 0, cols = 0;
};

template <typename W>
struct traits;   // undefined: not a storage the kernels read

template <ml::quant_block B>
struct traits<ml::quantized_weights<B>> {
    static constexpr format fmt = std::is_same_v<B, ml::block_q8_0> ? format::q8_0 : std::is_same_v<B, ml::block_q4_0> ? format::q4_0 : format::mxfp4;
    static parts of(const ml::quantized_weights<B>& w) { return {fmt, std::as_bytes(w.blocks), {}, {}, 0.0f, w.rows, w.cols}; }
};

template <>
struct traits<ml::nf4_weights> {
    static constexpr format fmt = format::nf4;
    static parts of(const ml::nf4_weights& w)
    {
        return {fmt, std::as_bytes(w.codes), std::as_bytes(w.block_scales), std::as_bytes(w.group_scales), w.scale_mean, w.rows, w.cols};
    }
};

template <typename S>
    requires(std::is_same_v<S, ml::bfloat16> || std::is_same_v<S, ml::float16> || std::is_same_v<S, ml::float8_e4m3> || std::is_same_v<S, ml::float8_e5m2>)
struct traits<ml::dense_weights<S>> {
    static constexpr format fmt = std::is_same_v<S, ml::bfloat16> ? format::bf16
                                  : std::is_same_v<S, ml::float16> ? format::f16
                                  : std::is_same_v<S, ml::float8_e4m3> ? format::e4m3
                                                                       : format::e5m2;
    static parts of(const ml::dense_weights<S>& w) { return {fmt, std::as_bytes(w.data), {}, {}, 0.0f, w.rows, w.cols}; }
};

// A storage the quantised kernels read
template <typename W>
concept weights = requires(const W& w) { traits<W>::of(w); };

// The weights' bytes a storage of this shape must have, and whether the shape is one the format allows (a block never
// crosses a row): the overloads check both before a launch
inline bool valid(const parts& p) noexcept
{
    const std::size_t n = p.rows * p.cols;
    switch (p.fmt) {
    case format::q8_0: return p.cols % 32 == 0 && p.weights.size() == n / 32 * sizeof(ml::block_q8_0);
    case format::q4_0: return p.cols % 32 == 0 && p.weights.size() == n / 32 * sizeof(ml::block_q4_0);
    case format::mxfp4: return p.cols % 32 == 0 && p.weights.size() == n / 32 * sizeof(ml::block_mxfp4);
    case format::nf4:
        return p.cols % ml::nf4_block_size == 0 && p.weights.size() == ml::nf4_code_bytes(n) && p.block_scales.size() == ml::nf4_block_count(n)
               && p.group_scales.size() == ml::nf4_group_count(n) * sizeof(float);
    case format::bf16:
    case format::f16: return p.weights.size() == n * 2;
    case format::e4m3:
    case format::e5m2: return p.weights.size() == n;
    }
    return false;
}

// The languages' preludes: the macros the source below is written in
inline constexpr const char* opencl_prelude = R"CLC(
#pragma OPENCL FP_CONTRACT OFF
typedef ulong u64;
typedef long i64;
typedef char i8;
#define KERNEL __kernel
#define GROUP1(X) __attribute__((reqd_work_group_size(X, 1, 1)))
#define GROUP2(X, Y) __attribute__((reqd_work_group_size(X, Y, 1)))
#define DEVICE static
#define GLOBAL __global
#define LOCAL __local
#define CONSTANT __constant
#define BARRIER() barrier(CLK_LOCAL_MEM_FENCE)
#define LX ((int)get_local_id(0))
#define LY ((int)get_local_id(1))
#define GX ((u64)get_group_id(0))
#define GY ((u64)get_group_id(1))
#define AS_FLOAT(u) as_float((uint)(u))
#define AS_UINT(f) as_uint(f)
)CLC";

inline constexpr const char* cuda_prelude = R"CU(
typedef unsigned long long u64;
typedef long long i64;
typedef signed char i8;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
#define KERNEL extern "C" __global__
#define GROUP1(X) __launch_bounds__(X)
#define GROUP2(X, Y) __launch_bounds__((X) * (Y))
#define DEVICE __device__ __forceinline__
#define GLOBAL
#define LOCAL __shared__
#define CONSTANT __constant__
#define BARRIER() __syncthreads()
#define LX ((int)threadIdx.x)
#define LY ((int)threadIdx.y)
#define GX ((u64)blockIdx.x)
#define GY ((u64)blockIdx.y)
#define AS_FLOAT(u) __uint_as_float((uint)(u))
#define AS_UINT(f) __float_as_uint(f)
)CU";

// The formats: exact decoding, then one device function per format giving the weight at (r, c)
inline constexpr const char* formats_source = R"SRC(
#define WARGS GLOBAL const uchar* w, GLOBAL const uchar* sb, GLOBAL const float* gs, float mean, u64 cols
#define WPASS w, sb, gs, mean, cols
#define CAT2(a, b) a##_##b
#define CAT(a, b) CAT2(a, b)

// A minifloat's exact value (ml/minifloat.hpp to_float): E exponent bits, M mantissa bits, kind 0 IEEE (the all-ones
// exponent infinite or NaN), 1 finite with one NaN (E4M3)
DEVICE float mf_decode(uint b, int E, int M, int kind)
{
    const int bias = (1 << (E - 1)) - 1;
    const uint sign = (b >> (E + M)) & 1u, e = (b >> M) & ((1u << E) - 1u), m = b & ((1u << M) - 1u);
    float v;
    if (kind == 0 && e == (1u << E) - 1u) v = m == 0u ? AS_FLOAT(0x7f800000u) : AS_FLOAT(0x7fc00000u);
    else if (kind == 1 && e == (1u << E) - 1u && m == (1u << M) - 1u) v = AS_FLOAT(0x7fc00000u);
    else if (e == 0u) v = (float)m * AS_FLOAT((uint)(1 - bias - M + 127) << 23);   // exact: m below 2^M times a power of two
    else v = AS_FLOAT((((uint)((int)e - bias + 127)) << 23) | (m << (23 - M)));
    return sign ? -v : v;
}
DEVICE float half_at(GLOBAL const uchar* p) { return mf_decode((uint)p[0] | ((uint)p[1] << 8), 5, 10, 0); }
// 2^(e - 128), half the MX scale (ml/quantize.hpp e8m0_half)
DEVICE float e8m0_half(uint e) { return AS_FLOAT(e < 2u ? 0x00200000u << e : (e - 1u) << 23); }
// E2M1 doubled (ggml's kvalues_mxfp4) and NormalFloat4 (ml/nf4.hpp nf4_values)
CONSTANT int fp4_doubled[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
CONSTANT float nf4_values[16] = {-1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f, -0.28444138169288635f,
    -0.18477343022823334f, -0.09105003625154495f, 0.0f, 0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f,
    0.33791524171829224f, 0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f};

// Q8_0: 34-byte blocks, a float16 scale then 32 signed bytes; x = q d
DEVICE float get_q8_0(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = w + (r * (cols / 32) + c / 32) * 34;
    return (float)(i8)b[2 + c % 32] * half_at(b);
}
// Q4_0: 18-byte blocks, a float16 scale then element j in the low nibble of byte j, j + 16 in the high; x = (q - 8) d
DEVICE float get_q4_0(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = w + (r * (cols / 32) + c / 32) * 18;
    const uint j = (uint)(c % 32), q = b[2 + (j & 15u)];
    return (float)((int)(j < 16u ? q & 15u : q >> 4) - 8) * half_at(b);
}
// MXFP4: 17-byte blocks, an E8M0 exponent then E2M1 codes packed as Q4_0's; x = 2^(e - 128) 2v
DEVICE float get_mxfp4(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = w + (r * (cols / 32) + c / 32) * 17;
    const uint j = (uint)(c % 32), q = b[1 + (j & 15u)];
    return e8m0_half(b[0]) * (float)fp4_doubled[j < 16u ? q & 15u : q >> 4];
}
// NF4: two codes a byte, the first in the high nibble; a block of 64 scaled by e4m3(sb) s1 + mean, s1 its group's of 256
DEVICE float get_nf4(WARGS, u64 r, u64 c)
{
    const u64 i = r * cols + c, blk = i / 64;
    const uint q = w[i / 2], code = (i % 2 == 0) ? q >> 4 : q & 15u;
    const float s = mf_decode(sb[blk], 4, 3, 1) * gs[blk / 256];
    const float scale = s + mean;
    return nf4_values[code] * scale;
}
// Dense minifloats, one element each
DEVICE float get_bf16(WARGS, u64 r, u64 c) { return AS_FLOAT((uint)((GLOBAL const ushort*)w)[r * cols + c] << 16); }
DEVICE float get_f16(WARGS, u64 r, u64 c) { return mf_decode(((GLOBAL const ushort*)w)[r * cols + c], 5, 10, 0); }
DEVICE float get_e4m3(WARGS, u64 r, u64 c) { return mf_decode(w[r * cols + c], 4, 3, 1); }
DEVICE float get_e5m2(WARGS, u64 r, u64 c) { return mf_decode(w[r * cols + c], 5, 2, 0); }

// float to float16, to nearest even (ml/minifloat.hpp to_minifloat<float16>): Q8_0's scale
DEVICE uint f2h(float x)
{
    const uint u = AS_UINT(x), sign = (u >> 16) & 0x8000u, a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return sign | 0x7e00u;
    if (a >= 0x477ff000u) return sign | 0x7c00u;   // 65520 and above round to infinity
    if (a >= 0x38800000u) { const uint r = a - (112u << 23); return sign | ((r + 0x0fffu + ((r >> 13) & 1u)) >> 13); }
    if (a < 0x33000000u) return sign;              // at most 2^-25: zero (2^-25 itself ties to even)
    const uint e = a >> 23, m = (a & 0x7fffffu) | 0x800000u, shift = 126u - e;
    const uint q = m >> shift, rem = m & ((1u << shift) - 1u), tie = 1u << (shift - 1u);
    return sign | (q + ((rem > tie || (rem == tie && (q & 1u))) ? 1u : 0u));
}
// roundf for |x| below 2^31 (ml/quantize.hpp round_to_int)
DEVICE int round_to_int(float x)
{
    const int t = (int)x;
    const float frac = x - (float)t;
    return t + (frac >= 0.5f ? 1 : 0) - (frac <= -0.5f ? 1 : 0);
}

// core's block dot products with a Q8_0 block of activations (ml/quantize.hpp dot), one block's term
DEVICE float term_q8_0(GLOBAL const uchar* wb, GLOBAL const uchar* xb)
{
    int s = 0;
    for (int j = 0; j < 32; ++j) s += (int)(i8)wb[2 + j] * (int)(i8)xb[2 + j];
    const float scale = half_at(wb) * half_at(xb);
    return (float)s * scale;
}
DEVICE float term_q4_0(GLOBAL const uchar* wb, GLOBAL const uchar* xb)
{
    int s0 = 0, s1 = 0;
    for (int j = 0; j < 16; ++j) {
        s0 += ((int)(wb[2 + j] & 15u) - 8) * (int)(i8)xb[2 + j];
        s1 += ((int)(wb[2 + j] >> 4) - 8) * (int)(i8)xb[18 + j];
    }
    return (float)(s0 + s1) * half_at(wb) * half_at(xb);
}
DEVICE float term_mxfp4(GLOBAL const uchar* wb, GLOBAL const uchar* xb)
{
    const float d = half_at(xb) * e8m0_half(wb[0]);
    int s1 = 0, s2 = 0;
    for (int j = 0; j < 16; ++j) {
        s1 += (int)(i8)xb[2 + j] * fp4_doubled[wb[1 + j] & 15u];
        s2 += (int)(i8)xb[18 + j] * fp4_doubled[wb[1 + j] >> 4];
    }
    return d * (float)(s1 + s2);
}

// Rows of activations quantised to Q8_0 (ml/quantize.hpp quantize<block_q8_0>), a thread per block of 32
KERNEL GROUP1(64) void quantize_q8(GLOBAL const float* x, u64 blocks, GLOBAL uchar* out)
{
    const u64 b = GX * 64 + (u64)LX;
    if (b >= blocks) return;
    GLOBAL const float* v = x + b * 32;
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) { const float a = v[j] < 0.0f ? -v[j] : v[j]; amax = amax < a ? a : amax; }
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    GLOBAL uchar* o = out + b * 34;
    const uint h = f2h(d);
    o[0] = (uchar)(h & 0xffu);
    o[1] = (uchar)(h >> 8);
    for (int j = 0; j < 32; ++j) { const float scaled = v[j] * id; o[2 + j] = (uchar)(i8)round_to_int(scaled); }
}
)SRC";

// The kernels of one format, instantiated with FMT (its name) and GET(r, c) (its weight) defined
inline constexpr const char* kernels_source = R"SRC(
#define MV_TOKENS 8
// Matrix-vector (decode): y_t = W x_t + b for up to MV_TOKENS rows x_t. A group of 64 per output row; each item adds
// the products of eight consecutive columns at a time, 512 apart, then the 64 partial sums are added by a fixed tree, so
// the result is the same every run (not the reference's order: within rounding of it).
KERNEL GROUP1(64) void CAT(qmv, FMT)(WARGS, GLOBAL const float* x, u64 n, u64 rows, GLOBAL const float* bias, int has_bias, GLOBAL float* y)
{
    LOCAL float red[MV_TOKENS * 64];
    const int l = LX;
    const u64 r = GX;
    float acc[MV_TOKENS];
    for (int t = 0; t < MV_TOKENS; ++t) acc[t] = 0.0f;
    for (u64 c0 = (u64)l * 8; c0 < cols; c0 += 512)
        for (int j = 0; j < 8; ++j) {
            const u64 c = c0 + (u64)j;
            if (c < cols) {
                const float v = GET(r, c);
                for (int t = 0; t < MV_TOKENS; ++t)
                    if ((u64)t < n) { const float p = v * x[(u64)t * cols + c]; acc[t] += p; }
            }
        }
    for (int t = 0; t < MV_TOKENS; ++t) red[t * 64 + l] = acc[t];
    BARRIER();
    for (int s = 32; s > 0; s >>= 1) {
        if (l < s)
            for (int t = 0; t < MV_TOKENS; ++t) red[t * 64 + l] += red[t * 64 + l + s];
        BARRIER();
    }
    if (l == 0)
        for (int t = 0; t < MV_TOKENS; ++t)
            if ((u64)t < n) { float v = red[t * 64]; if (has_bias) v = v + bias[r]; y[(u64)t * rows + r] = v; }
}

// The tiled product C (m x nn) = A (m x kk, row-major) B + beta C, plus a bias over the columns, where B (kk x nn) is
// the weight read through BGET(p, c): Wᵀ for the forward (B(p, c) = W(c, p)), W for the input's gradient. The OpenCL
// gemm's tiling (opencl.hpp): 64 x 64 tiles, 16 deep, 16 x 16 items of 4 x 4 outputs, every output summed in index
// order, then s + beta c and + bias: core's reference product (linalg::reference) and add_bias_rows to the bit.
#define TILED(NAME, BGET, B_P, B_C)                                                                                    \
KERNEL GROUP2(16, 16) void NAME(WARGS, u64 m, u64 nn, u64 kk, GLOBAL const float* a, GLOBAL const float* bias,      \
                                int has_bias, float beta, GLOBAL float* cm)                                           \
{                                                                                                                     \
    LOCAL float as[16][64];                                                                                           \
    LOCAL float bs[16][64];                                                                                           \
    const int tc = LX, tr = LY, tid = tr * 16 + tc;                                                                   \
    const u64 col0 = GX * 64, row0 = GY * 64;                                                                         \
    float acc[4][4];                                                                                                  \
    for (int i = 0; i < 4; ++i)                                                                                       \
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;                                                                 \
    for (u64 t = 0; t < kk; t += 16) {                                                                                \
        for (int l = 0; l < 4; ++l) {                                                                                 \
            const int e = tid + l * 256;                                                                              \
            const int ar = e / 16, ap = e % 16;                                                                       \
            const u64 gr = row0 + (u64)ar, gp = t + (u64)ap;                                                          \
            as[ap][ar] = (gr < m && gp < kk) ? a[gr * kk + gp] : 0.0f;                                                \
            const int bp = B_P(e), bc = B_C(e);                                                                       \
            const u64 gq = t + (u64)bp, gc = col0 + (u64)bc;                                                          \
            bs[bp][bc] = (gq < kk && gc < nn) ? BGET(gq, gc) : 0.0f;                                                  \
        }                                                                                                             \
        BARRIER();                                                                                                    \
        const int last = kk - t < 16 ? (int)(kk - t) : 16;                                                            \
        for (int p = 0; p < last; ++p) {                                                                              \
            float ar[4], br[4];                                                                                       \
            for (int i = 0; i < 4; ++i) ar[i] = as[p][tr + i * 16];                                                   \
            for (int j = 0; j < 4; ++j) br[j] = bs[p][tc + j * 16];                                                   \
            for (int i = 0; i < 4; ++i)                                                                               \
                for (int j = 0; j < 4; ++j) { const float q = ar[i] * br[j]; acc[i][j] += q; }                        \
        }                                                                                                             \
        BARRIER();                                                                                                    \
    }                                                                                                                 \
    for (int i = 0; i < 4; ++i)                                                                                       \
        for (int j = 0; j < 4; ++j) {                                                                                 \
            const u64 row = row0 + (u64)(tr + i * 16), col = col0 + (u64)(tc + j * 16);                              \
            if (row < m && col < nn) {                                                                                \
                GLOBAL float* out = cm + row * nn + col;                                                              \
                float v = acc[i][j];                                                                                  \
                if (beta != 0.0f) { const float old = beta * *out; v = v + old; }                                     \
                if (has_bias) v = v + bias[col];                                                                      \
                *out = v;                                                                                             \
            }                                                                                                         \
        }                                                                                                             \
}
#define FWD_GET(p, c) GET(c, p)
#define FWD_P(e) ((e) % 16)
#define FWD_C(e) ((e) / 16)
#define BWD_GET(p, c) GET(p, c)
#define BWD_P(e) ((e) / 64)
#define BWD_C(e) ((e) % 64)
TILED(CAT(qmm, FMT), FWD_GET, FWD_P, FWD_C)
TILED(CAT(qmmt, FMT), BWD_GET, BWD_P, BWD_C)
#undef TILED
#undef FWD_GET
#undef FWD_P
#undef FWD_C
#undef BWD_GET
#undef BWD_P
#undef BWD_C
#undef MV_TOKENS
)SRC";

// The integer path of one format with core's dot product, instantiated with FMT, TERM and BYTES (its block's size)
inline constexpr const char* q8_source = R"SRC(
// y_t = b + Σ over blocks of core's term (W row r's block with x_t's Q8_0 block), the terms added in block order as
// core's dot does: a group per (row, token), 64 terms at a time into local memory, the first item adding them in order.
KERNEL GROUP1(64) void CAT(qdot, FMT)(GLOBAL const uchar* w, u64 cols, GLOBAL const uchar* xq, u64 rows, u64 t0, GLOBAL const float* bias, int has_bias,
                                      GLOBAL float* y)
{
    LOCAL float terms[64];
    const int l = LX;
    const u64 r = GX, t = t0 + GY, nb = cols / 32;
    float sum = 0.0f;
    for (u64 b0 = 0; b0 < nb; b0 += 64) {
        const u64 b = b0 + (u64)l;
        terms[l] = b < nb ? TERM(w + (r * nb + b) * BYTES, xq + (t * nb + b) * 34) : 0.0f;
        BARRIER();
        if (l == 0) { const u64 e = nb - b0 < 64 ? nb - b0 : 64; for (u64 i = 0; i < e; ++i) sum += terms[i]; }
        BARRIER();
    }
    if (l == 0) y[t * rows + r] = has_bias ? bias[r] + sum : sum;
}
)SRC";

// The whole program for one language: the prelude, the formats, each format's kernels, the integer paths, then `extra`
// (the CUDA tensor-core kernels)
inline std::string program(const char* prelude, const char* extra = "")
{
    std::string s = std::string(prelude) + formats_source;
    for (const auto name : names) {
        const std::string n(name);
        s += "#define FMT " + n + "\n#define GET(r, c) get_" + n + "(WPASS, r, c)\n" + kernels_source + "#undef GET\n#undef FMT\n";
    }
    for (const auto& [name, bytes] : {std::pair{"q8_0", 34}, std::pair{"q4_0", 18}, std::pair{"mxfp4", 17}}) {
        const std::string n(name);
        s += "#define FMT " + n + "\n#define TERM term_" + n + "\n#define BYTES " + std::to_string(bytes) + "\n" + q8_source
             + "#undef BYTES\n#undef TERM\n#undef FMT\n";
    }
    return s + extra;
}

} // namespace snicholls::gpu::detail::quant

#endif /* snicholls_gpu_detail_quant_source_hpp */
