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
#include <snicholls/gpu/quant_weights.hpp>
// The formats of core's next release (Q4_1, Q5_0, Q5_1, the K-quants, the ternary), where the pinned core has them
#if __has_include(<snicholls/core/ml/quant_k.hpp>) && __has_include(<snicholls/core/ml/quant_ternary.hpp>) && __has_include(<snicholls/core/ml/quant_iq.hpp>) \
    && __has_include(<snicholls/core/ml/bitnet_kernels.hpp>)
#include <snicholls/core/ml/quant_iq.hpp>
#include <snicholls/core/ml/quant_k.hpp>
#include <snicholls/core/ml/quant_ternary.hpp>
#define SNICHOLLS_GPU_QUANT_K 1
#endif

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

namespace snicholls::gpu::detail::quant {

// The formats, in the order of `formats`; the device source names them by `name`
enum class format : int {
    q8_0, q4_0, mxfp4, nf4, bf16, f16, e4m3, e5m2, q4_1, q5_0, q5_1, q2_k, q3_k, q4_k, q5_k, q6_k, tq1_0, tq2_0, iq4_nl, iq4_xs, nvfp4, q1_0, q2_0,
    iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq1_s, iq1_m, i2_s
};
inline constexpr std::size_t format_count = 31;
inline constexpr std::array<std::string_view, format_count> names{
    "q8_0", "q4_0", "mxfp4", "nf4",  "bf16",   "f16",    "e4m3",  "e5m2",   "q4_1",  "q5_0",  "q5_1",  "q2_k",  "q3_k",  "q4_k",  "q5_k",
    "q6_k", "tq1_0", "tq2_0", "iq4_nl", "iq4_xs", "nvfp4", "q1_0", "q2_0", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq1_s", "iq1_m", "i2_s"};
// The grid-based IQ formats read core's codebooks, which the device source embeds from core's tables: present only
// where the pinned core has them
constexpr bool grid_format(format f) noexcept { return f >= format::iq2_xxs && f <= format::iq1_m; }
#ifdef SNICHOLLS_GPU_QUANT_K
constexpr bool available(format) noexcept { return true; }
#else
constexpr bool available(format f) noexcept { return !grid_format(f); }
#endif
// The numbers and bytes of one block of each format (NF4: 64 numbers in 32 bytes, its scales apart; dense: one number)
inline constexpr std::array<std::size_t, format_count> block_elements{32,  32,  32,  64,  1,   1,   1,   1,   32,  32,  32,  256, 256, 256, 256,
                                                                    256, 256, 256, 32,  256, 64,  128, 64,  256, 256, 256, 256, 256, 256, 256, 128};
inline constexpr std::array<std::size_t, format_count> block_bytes{34,  18, 17, 32, 2,   2,  1,  1,  20, 22, 24, 84, 110, 144, 176,
                                                                 210, 54, 66, 18, 136, 36, 18, 18, 66, 74, 82, 98, 110, 50,  56, 32};

// Whether a format has core's integer dot product with Q8_0 activations (ml/quantize.hpp `dot`), or with Q8_K (the
// K-quants, ml/quant_k.hpp `dot`)
constexpr bool has_q8_0_dot(format f) noexcept
{
    return f == format::q8_0 || f == format::q4_0 || f == format::mxfp4 || f == format::q5_0 || f == format::iq4_nl || f == format::nvfp4 || f == format::q1_0
           || f == format::q2_0;
}
constexpr bool has_q8_1_dot(format f) noexcept { return f == format::q4_1 || f == format::q5_1; }
constexpr bool has_q8_k_dot(format f) noexcept
{
    return f == format::q2_k || f == format::q3_k || f == format::q4_k || f == format::q5_k || f == format::q6_k || f == format::tq1_0
           || f == format::tq2_0 || f == format::iq4_xs || grid_format(f);
}
// BitNet's I2_S: BitLinear's per-token absmax int8 activations (ml/bitnet.hpp bitlinear)
constexpr bool has_absmax_dot(format f) noexcept { return f == format::i2_s; }
constexpr bool has_q8_dot(format f) noexcept { return has_q8_0_dot(f) || has_q8_1_dot(f) || has_q8_k_dot(f) || has_absmax_dot(f); }
// Blocks of Q8_1 activations: Q8_0's with the float16 sum s = d Σ q after the scale
inline constexpr std::size_t q8_1_bytes = 36;
// Blocks of Q8_K activations: a float scale, 256 codes, 16 sums of 16 (ml/quantize.hpp block_q8_k)
inline constexpr std::size_t q8k_bytes = 292;

// The tokens a matrix-vector launch takes at once: as MV_TOKENS in the source
inline constexpr std::size_t mv_tokens = 8;
// The outputs a matrix-vector work-group makes: as MV_ROWS in the source
inline constexpr std::size_t mv_rows = 4;
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

// Core's quantized_weights over its 32-blocks of Q8_0 activations, named one by one: a block this does not know (core's
// quant_block concept admits more as core grows) is no storage the kernels read, never another format's
template <ml::quant_block B>
    requires(std::is_same_v<B, ml::block_q8_0> || std::is_same_v<B, ml::block_q4_0> || std::is_same_v<B, ml::block_mxfp4>)
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

#ifdef SNICHOLLS_GPU_QUANT_K
// The blocks of core's next release, over gpu::block_weights; their fields at ggml's offsets, as the device reads them
template <typename B>
inline constexpr int block_format = -1;
template <> inline constexpr int block_format<ml::block_q4_1> = int(format::q4_1);
template <> inline constexpr int block_format<ml::block_q5_0> = int(format::q5_0);
template <> inline constexpr int block_format<ml::block_q5_1> = int(format::q5_1);
template <> inline constexpr int block_format<ml::block_q2_k> = int(format::q2_k);
template <> inline constexpr int block_format<ml::block_q3_k> = int(format::q3_k);
template <> inline constexpr int block_format<ml::block_q4_k> = int(format::q4_k);
template <> inline constexpr int block_format<ml::block_q5_k> = int(format::q5_k);
template <> inline constexpr int block_format<ml::block_q6_k> = int(format::q6_k);
template <> inline constexpr int block_format<ml::block_tq1_0> = int(format::tq1_0);
template <> inline constexpr int block_format<ml::block_tq2_0> = int(format::tq2_0);
template <> inline constexpr int block_format<ml::block_iq4_nl> = int(format::iq4_nl);
template <> inline constexpr int block_format<ml::block_iq4_xs> = int(format::iq4_xs);
template <> inline constexpr int block_format<ml::block_nvfp4> = int(format::nvfp4);
template <> inline constexpr int block_format<ml::block_q1_0> = int(format::q1_0);
template <> inline constexpr int block_format<ml::block_q2_0> = int(format::q2_0);
template <> inline constexpr int block_format<ml::block_iq2_xxs> = int(format::iq2_xxs);
template <> inline constexpr int block_format<ml::block_iq2_xs> = int(format::iq2_xs);
template <> inline constexpr int block_format<ml::block_iq2_s> = int(format::iq2_s);
template <> inline constexpr int block_format<ml::block_iq3_xxs> = int(format::iq3_xxs);
template <> inline constexpr int block_format<ml::block_iq3_s> = int(format::iq3_s);
template <> inline constexpr int block_format<ml::block_iq1_s> = int(format::iq1_s);
template <> inline constexpr int block_format<ml::block_iq1_m> = int(format::iq1_m);

template <typename B>
    requires(block_format<B> >= 0)
struct traits<block_weights<B>> {
    static constexpr format fmt = format(block_format<B>);
    static_assert(sizeof(B) == block_bytes[std::size_t(fmt)] && B::elements == block_elements[std::size_t(fmt)],
                  "snicholls::gpu: a block's size differs from the device source's");
    static parts of(const block_weights<B>& w) { return {fmt, std::as_bytes(w.blocks), {}, {}, 0.0f, w.rows, w.cols}; }
};
// The newer 32-blocks core's quantized_weights also admits (Q5_0, IQ4_NL), as block_weights
template <ml::quant_block B>
    requires(block_format<B> >= 0)
struct traits<ml::quantized_weights<B>> {
    static constexpr format fmt = format(block_format<B>);
    static parts of(const ml::quantized_weights<B>& w) { return {fmt, std::as_bytes(w.blocks), {}, {}, 0.0f, w.rows, w.cols}; }
};
static_assert(offsetof(ml::block_q4_1, qs) == 4 && offsetof(ml::block_q5_0, qh) == 2 && offsetof(ml::block_q5_0, qs) == 6
                  && offsetof(ml::block_q5_1, qh) == 4 && offsetof(ml::block_q5_1, qs) == 8 && offsetof(ml::block_q2_k, qs) == 16
                  && offsetof(ml::block_q2_k, d) == 80 && offsetof(ml::block_q3_k, qs) == 32 && offsetof(ml::block_q3_k, scales) == 96
                  && offsetof(ml::block_q3_k, d) == 108 && offsetof(ml::block_q4_k, scales) == 4 && offsetof(ml::block_q4_k, qs) == 16
                  && offsetof(ml::block_q5_k, qh) == 16 && offsetof(ml::block_q5_k, qs) == 48 && offsetof(ml::block_q6_k, qh) == 128
                  && offsetof(ml::block_q6_k, scales) == 192 && offsetof(ml::block_q6_k, d) == 208 && offsetof(ml::block_tq1_0, qh) == 48
                  && offsetof(ml::block_tq1_0, d) == 52 && offsetof(ml::block_tq2_0, d) == 64 && offsetof(ml::block_iq4_nl, qs) == 2
                  && offsetof(ml::block_iq4_xs, scales_h) == 2 && offsetof(ml::block_iq4_xs, scales_l) == 4 && offsetof(ml::block_iq4_xs, qs) == 8
                  && offsetof(ml::block_nvfp4, qs) == 4 && offsetof(ml::block_q1_0, qs) == 2 && offsetof(ml::block_q2_0, qs) == 2
                  && offsetof(ml::block_iq2_xs, scales) == 66 && offsetof(ml::block_iq2_s, qh) == 66 && offsetof(ml::block_iq2_s, scales) == 74
                  && offsetof(ml::block_iq3_s, qh) == 66 && offsetof(ml::block_iq3_s, signs) == 74 && offsetof(ml::block_iq3_s, scales) == 106
                  && offsetof(ml::block_iq1_s, qh) == 34 && offsetof(ml::block_iq1_m, qh) == 32 && offsetof(ml::block_iq1_m, scales) == 48,
              "snicholls::gpu: a block's fields are not where the device source reads them");
#endif

// BitNet's I2_S: the codes as the weights, the scale where NF4 keeps its mean
template <>
struct traits<i2_s_weights> {
    static constexpr format fmt = format::i2_s;
    static parts of(const i2_s_weights& w)
    {
        const auto bytes = std::as_bytes(w.packed);
        const std::size_t codes = w.rows * w.cols / 4;
        return {fmt, bytes.size() == codes + 32 ? bytes.first(codes) : bytes, {}, {}, w.scale, w.rows, w.cols};
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
    default: {
        const std::size_t e = block_elements[std::size_t(p.fmt)];
        return p.cols % e == 0 && p.weights.size() == n / e * block_bytes[std::size_t(p.fmt)];
    }
    }
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
#define LOAD4(p) vload4(0, p)
#define U64C(x) x##UL
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
// A vector load where the address allows it (an odd row width or a span's offset may not), else four scalar loads
#define U64C(x) x##ULL
#define LOAD4(p) ((((u64)(p)) & 15ull) == 0 ? *(const float4*)(p) : make_float4((p)[0], (p)[1], (p)[2], (p)[3]))
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

// Eight consecutive weights of one row from column c0 (c0 a multiple of 8, c0 + 8 at most cols), the same values as
// get_<format>: the matrix-vector kernels' unit, decoding a block's scale once
DEVICE void get8_q8_0(WARGS, u64 r, u64 c0, float* v)
{
    GLOBAL const uchar* b = w + (r * (cols / 32) + c0 / 32) * 34;
    const float d = half_at(b);
    GLOBAL const ushort* q = (GLOBAL const ushort*)(b + 2 + c0 % 32);   // a block starts on an even byte
    for (int j = 0; j < 4; ++j) { const uint u = q[j]; v[2 * j] = (float)(i8)(u & 0xffu) * d; v[2 * j + 1] = (float)(i8)(u >> 8) * d; }
}
DEVICE void get8_q4_0(WARGS, u64 r, u64 c0, float* v)
{
    GLOBAL const uchar* b = w + (r * (cols / 32) + c0 / 32) * 18;
    const float d = half_at(b);
    const uint j0 = (uint)(c0 % 32), shift = j0 < 16u ? 0u : 4u;
    GLOBAL const ushort* q = (GLOBAL const ushort*)(b + 2 + (j0 & 15u));   // a block starts on an even byte
    for (int j = 0; j < 4; ++j) {
        const uint u = (uint)q[j] >> shift;
        v[2 * j] = (float)((int)(u & 15u) - 8) * d;
        v[2 * j + 1] = (float)((int)((u >> 8) & 15u) - 8) * d;
    }
}
DEVICE void get8_mxfp4(WARGS, u64 r, u64 c0, float* v)
{
    GLOBAL const uchar* b = w + (r * (cols / 32) + c0 / 32) * 17;
    const float h = e8m0_half(b[0]);
    const uint j0 = (uint)(c0 % 32), shift = j0 < 16u ? 0u : 4u;
    GLOBAL const uchar* q = b + 1 + (j0 & 15u);
    for (int j = 0; j < 8; ++j) v[j] = h * (float)fp4_doubled[(q[j] >> shift) & 15u];
}
DEVICE void get8_nf4(WARGS, u64 r, u64 c0, float* v)
{
    const u64 i = r * cols + c0, blk = i / 64;
    const float s = mf_decode(sb[blk], 4, 3, 1) * gs[blk / 256];
    const float scale = s + mean;
    GLOBAL const uchar* q = w + i / 2;
    for (int j = 0; j < 4; ++j) { v[2 * j] = nf4_values[q[j] >> 4] * scale; v[2 * j + 1] = nf4_values[q[j] & 15u] * scale; }
}
DEVICE void get8_bf16(WARGS, u64 r, u64 c0, float* v) { for (int j = 0; j < 8; ++j) v[j] = get_bf16(WPASS, r, c0 + (u64)j); }
DEVICE void get8_f16(WARGS, u64 r, u64 c0, float* v) { for (int j = 0; j < 8; ++j) v[j] = get_f16(WPASS, r, c0 + (u64)j); }
DEVICE void get8_e4m3(WARGS, u64 r, u64 c0, float* v) { for (int j = 0; j < 8; ++j) v[j] = get_e4m3(WPASS, r, c0 + (u64)j); }
DEVICE void get8_e5m2(WARGS, u64 r, u64 c0, float* v) { for (int j = 0; j < 8; ++j) v[j] = get_e5m2(WPASS, r, c0 + (u64)j); }

// The formats of core's next release (ml/quantize.hpp, ml/quant_k.hpp, ml/quant_ternary.hpp dequantize_row), each
// weight with core's operations in core's order. Block i of row r at w + (r cols / E + c / E) B.
DEVICE GLOBAL const uchar* block_at(GLOBAL const uchar* w, u64 cols, u64 r, u64 c, u64 e, u64 bytes) { return w + (r * (cols / e) + c / e) * bytes; }
DEVICE uint nibble(GLOBAL const uchar* qs, uint j) { const uint q = qs[j & 15u]; return j < 16u ? q & 15u : q >> 4; }
DEVICE uint word_at(GLOBAL const uchar* p) { return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24); }
// Q4_1 (20 bytes: d, m, qs): d q + m
DEVICE float get_q4_1(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 32, 20);
    const float a = (float)nibble(b + 4, (uint)(c % 32)) * half_at(b);
    return a + half_at(b + 2);
}
// Q5_0 (22: d, qh, qs): d (q - 16), the fifth bit of element j bit j of qh
DEVICE float get_q5_0(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 32, 22);
    const uint j = (uint)(c % 32), q = nibble(b + 6, j) | (((word_at(b + 2) >> j) & 1u) << 4);
    return (float)((int)q - 16) * half_at(b);
}
// Q5_1 (24: d, m, qh, qs): d q + m
DEVICE float get_q5_1(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 32, 24);
    const uint j = (uint)(c % 32), q = nibble(b + 8, j) | (((word_at(b + 4) >> j) & 1u) << 4);
    const float a = (float)q * half_at(b);
    return a + half_at(b + 2);
}
// Q2_K (84: scales[16], qs[64], d, dmin): (d s) q - (dmin m), element 128 h + 32 g + l in bits 2g of qs[32 h + l]
DEVICE float get_q2_k(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 256, 84);
    const uint e = (uint)(c % 256), h = e / 128, g = (e % 128) / 32, l = e % 32, q = (b[16 + 32 * h + l] >> (2 * g)) & 3u, sc = b[e / 16];
    const float dl = half_at(b + 80) * (float)(sc & 15u);
    const float ml = half_at(b + 82) * (float)(sc >> 4);
    const float v = dl * (float)q;
    return v - ml;
}
// Q3_K (110: hmask[32], qs[64], scales[12], d): (d (s - 32)) (q - 4 unless the high bit), the high bit of e bit e / 32 of hmask[e % 32]
DEVICE float get_q3_k(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 256, 110);
    const uint e = (uint)(c % 256), h = e / 128, g = (e % 128) / 32, l = e % 32, j = e / 16;
    const int q = (int)((b[32 + 32 * h + l] >> (2 * g)) & 3u) - (((b[l] >> (e / 32)) & 1u) ? 0 : 4);
    const uint low = j < 8u ? b[96 + j] & 15u : b[96 + j - 8] >> 4, high = (b[104 + j % 4] >> (2 * (j / 4))) & 3u;
    const float dl = half_at(b + 108) * (float)((int)(low | (high << 4)) - 32);
    return dl * (float)q;
}
// Q4_K's and Q5_K's 6-bit scale and minimum of sub-block j (ggml's get_scale_min_k4)
DEVICE void scale_min_k4(uint j, GLOBAL const uchar* q, uint* d, uint* m)
{
    if (j < 4u) { *d = q[j] & 63u; *m = q[j + 4] & 63u; }
    else { *d = (q[j + 4] & 15u) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
// Q4_K (144: d, dmin, scales[12], qs[128]): (d s) q - (dmin m) per 32, element 64 h + l low nibble of qs[32 h + l], 64 h + 32 + l high
DEVICE float get_q4_k(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 256, 144);
    const uint e = (uint)(c % 256), h = e / 64, l = e % 32, byte = b[16 + 32 * h + l], q = (e % 64) < 32u ? byte & 15u : byte >> 4;
    uint sc, mn;
    scale_min_k4(e / 32, b + 4, &sc, &mn);
    const float d1 = half_at(b) * (float)sc;
    const float m1 = half_at(b + 2) * (float)mn;
    const float v = d1 * (float)q;
    return v - m1;
}
// Q5_K (176: d, dmin, scales[12], qh[32], qs[128]): Q4_K plus 16 where bit e / 32 of qh[e % 32] is set
DEVICE float get_q5_k(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 256, 176);
    const uint e = (uint)(c % 256), h = e / 64, l = e % 32, byte = b[48 + 32 * h + l];
    const uint q = ((e % 64) < 32u ? byte & 15u : byte >> 4) + (((b[16 + l] >> (e / 32)) & 1u) ? 16u : 0u);
    uint sc, mn;
    scale_min_k4(e / 32, b + 4, &sc, &mn);
    const float d1 = half_at(b) * (float)sc;
    const float m1 = half_at(b + 2) * (float)mn;
    const float v = d1 * (float)q;
    return v - m1;
}
// Q6_K (210: ql[128], qh[64], scales[16] signed, d): (d s) (q - 32), the four quarters of 128 from ql's nibbles and qh's bit pairs
DEVICE float get_q6_k(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 256, 210);
    const uint e = (uint)(c % 256), n = e / 128, part = (e % 128) / 32, l = e % 32;
    const uint src = b[64 * n + 32 * (part & 1u) + l], nib = part < 2u ? src & 15u : src >> 4;
    const int q = (int)(nib | (((b[128 + 32 * n + l] >> (2 * part)) & 3u) << 4)) - 32;
    const float ds = half_at(b + 208) * (float)(i8)b[192 + e / 16];
    return ds * (float)q;
}
// TQ1_0 (54: qs[48], qh[4], d): d (t - 1), five trits a byte, trit k of byte x ((uchar)(x 3^k) 3) >> 8
CONSTANT uint pow3[5] = {1u, 3u, 9u, 27u, 81u};
DEVICE float get_tq1_0(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 256, 54);
    const uint e = (uint)(c % 256);
    uint byte, k;
    if (e < 160u) { byte = b[e % 32]; k = e / 32; }
    else if (e < 240u) { byte = b[32 + (e - 160) % 16]; k = (e - 160) / 16; }
    else { byte = b[48 + (e - 240) % 4]; k = (e - 240) / 4; }
    const uint t = (((byte * pow3[k]) & 255u) * 3u) >> 8;
    return (float)((int)t - 1) * half_at(b + 52);
}
// TQ2_0 (66: qs[64], d): d (q - 1), element 128 g + 32 l + m in bits 2l of qs[32 g + m]
DEVICE float get_tq2_0(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 256, 66);
    const uint e = (uint)(c % 256), q = (b[32 * (e / 128) + e % 32] >> (2 * ((e % 128) / 32))) & 3u;
    return (float)((int)q - 1) * half_at(b + 64);
}
// IQ4_NL (18: d, qs): d v[q], v ggml's 16 non-linear values; IQ4_XS (136: d, scales_h, scales_l[4], qs[128]): per 32
// (d (s - 32)) v[q], s six bits from scales_l's nibble and scales_h's pair
CONSTANT int iq4nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
DEVICE float get_iq4_nl(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 32, 18);
    return half_at(b) * (float)iq4nl_values[nibble(b + 2, (uint)(c % 32))];
}
DEVICE int iq4_xs_scale(GLOBAL const uchar* b, uint ib)
{
    const uint h = (uint)b[2] | ((uint)b[3] << 8);
    return (int)(((b[4 + ib / 2] >> (4 * (ib % 2))) & 15u) | (((h >> (2 * ib)) & 3u) << 4)) - 32;
}
DEVICE float get_iq4_xs(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 256, 136);
    const uint e = (uint)(c % 256), ib = e / 32;
    const float dl = half_at(b) * (float)iq4_xs_scale(b, ib);
    return dl * (float)iq4nl_values[nibble(b + 8 + 16 * ib, e % 32)];
}
// NVFP4 (36: d[4] UE4M3, qs[32]): per 16, ue4m3_half(d) 2v, element j of a sixteen the low nibble of byte j, j + 8 the
// high; Q1_0 (18: d, 128 sign bits): ±d; Q2_0 (18: d, qs[16]): d (q - 1)
DEVICE float ue4m3_half(uint x)
{
    if (x == 0u || x == 0x7fu) return 0.0f;
    const uint e = (x >> 3) & 15u, man = x & 7u;
    const float raw = e == 0u ? (float)man * AS_FLOAT(118u << 23) : AS_FLOAT((e - 7u + 127u) << 23) * (1.0f + (float)man / 8.0f);
    return raw * 0.5f;
}
DEVICE float get_nvfp4(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 64, 36);
    const uint e = (uint)(c % 64), sg = e / 16, j = e % 16, q = b[4 + 8 * sg + (j & 7u)];
    return ue4m3_half(b[sg]) * (float)fp4_doubled[j < 8u ? q & 15u : q >> 4];
}
DEVICE float get_q1_0(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 128, 18);
    const uint j = (uint)(c % 128);
    const float d = half_at(b);
    const float neg = -d;
    return ((b[2 + j / 8] >> (j % 8)) & 1u) ? d : neg;
}
DEVICE float get_q2_0(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 64, 18);
    const uint j = (uint)(c % 64);
    return (float)((int)((b[2 + j / 4] >> (2 * (j % 4))) & 3u) - 1) * half_at(b);
}
// BitNet's I2_S (bitnet.cpp's x86 layout): weight j of the tensor the 2-bit code at bits 6 - 2 ((j % 128) / 32) of byte
// j / 128 · 32 + j % 32, less 1, times the scale (passed as mean)
DEVICE int code_i2_s(GLOBAL const uchar* w, u64 j) { return (int)((w[j / 128 * 32 + j % 32] >> (6 - 2 * ((j % 128) / 32))) & 3u) - 1; }
DEVICE float get_i2_s(WARGS, u64 r, u64 c) { return (float)code_i2_s(w, r * cols + c) * mean; }
#define GET8_BY_ONE(F) DEVICE void get8_##F(WARGS, u64 r, u64 c0, float* v) { for (int j = 0; j < 8; ++j) v[j] = get_##F(WPASS, r, c0 + (u64)j); }
GET8_BY_ONE(i2_s)
GET8_BY_ONE(q4_1)
GET8_BY_ONE(q5_0)
GET8_BY_ONE(q5_1)
GET8_BY_ONE(q2_k)
GET8_BY_ONE(q3_k)
GET8_BY_ONE(q4_k)
GET8_BY_ONE(q5_k)
GET8_BY_ONE(q6_k)
GET8_BY_ONE(tq1_0)
GET8_BY_ONE(tq2_0)
GET8_BY_ONE(iq4_nl)
GET8_BY_ONE(iq4_xs)
GET8_BY_ONE(nvfp4)
GET8_BY_ONE(q1_0)
GET8_BY_ONE(q2_0)
#undef GET8_BY_ONE

// Rows of activations quantised to Q8_K (ml/quantize.hpp quantize_row<block_q8_k>), a thread per block of 256: the
// value of largest magnitude (the first), -127 / it, ggml's nearest_int, the sums of 16, then d = 1 / iscale
DEVICE int nearest_int(float x) { const float v = x + 12582912.0f; return (int)(AS_UINT(v) & 0x007fffffu) - 0x00400000; }
KERNEL GROUP1(64) void quantize_q8k(GLOBAL const float* x, u64 blocks, GLOBAL uchar* out)
{
    const u64 i = GX * 64 + (u64)LX;
    if (i >= blocks) return;
    GLOBAL const float* v = x + i * 256;
    GLOBAL uchar* o = out + i * 292;
    GLOBAL i8* qs = (GLOBAL i8*)(o + 4);
    GLOBAL short* bsums = (GLOBAL short*)(o + 260);
    float max = 0.0f, amax = 0.0f;
    for (int j = 0; j < 256; ++j) { const float a = v[j] < 0.0f ? -v[j] : v[j]; if (a > amax) { amax = a; max = v[j]; } }
    if (amax == 0.0f) {
        *(GLOBAL float*)o = 0.0f;
        for (int j = 0; j < 256; ++j) qs[j] = 0;
        for (int j = 0; j < 16; ++j) bsums[j] = 0;
        return;
    }
    const float iscale = -127.0f / max;
    for (int j = 0; j < 256; ++j) { const float s = iscale * v[j]; const int q = nearest_int(s); qs[j] = (i8)(q < 127 ? q : 127); }
    for (int j = 0; j < 16; ++j) { int sum = 0; for (int k = 0; k < 16; ++k) sum += qs[16 * j + k]; bsums[j] = (short)sum; }
    *(GLOBAL float*)o = 1.0f / iscale;
}
// BitLinear's activations (ml/bitnet.hpp absmax_quantize<8>): per token s = 127 / max(amax, 1e-5), codes
// nearest_int(clamp(x s, -128, 127)); a group of 64 per token, the token's s then its codes at t (cols + 4)
KERNEL GROUP1(64) void quantize_absmax(GLOBAL const float* x, u64 cols, GLOBAL uchar* out)
{
    LOCAL float red[64];
    const int l = LX;
    const u64 t = GX;
    GLOBAL const float* v = x + t * cols;
    float amax = 0.0f;
    for (u64 j = (u64)l; j < cols; j += 64) { const float a = v[j] < 0.0f ? -v[j] : v[j]; amax = amax > a ? amax : a; }
    red[l] = amax;
    BARRIER();
    for (int s = 32; s > 0; s >>= 1) { if (l < s) red[l] = red[l] > red[l + s] ? red[l] : red[l + s]; BARRIER(); }
    const float m = red[0];
    const float s = 127.0f / (m > 1e-5f ? m : 1e-5f);
    GLOBAL uchar* o = out + t * (cols + 4);
    if (l == 0) *(GLOBAL float*)o = s;
    for (u64 j = (u64)l; j < cols; j += 64) {
        const float q = v[j] * s;
        const float c = q < -128.0f ? -128.0f : q > 127.0f ? 127.0f : q;
        o[4 + j] = (uchar)(i8)nearest_int(c);
    }
}
// BitLinear (ml/bitnet.hpp bitlinear): y = (acc / s) γ, acc the exact sum of t q; a group per (row, token), the
// integer partial sums added by a tree (exact in any order)
KERNEL GROUP1(64) void qdot_i2_s(GLOBAL const uchar* w, u64 cols, GLOBAL const uchar* xq, u64 rows, u64 t0, float gamma, GLOBAL const float* bias,
                                 int has_bias, GLOBAL float* y)
{
    LOCAL int red[64];
    const int l = LX;
    const u64 r = GX, t = t0 + GY;
    GLOBAL const uchar* a = xq + t * (cols + 4);
    int acc = 0;
    for (u64 j = (u64)l; j < cols; j += 64) acc += code_i2_s(w, r * cols + j) * (int)(i8)a[4 + j];
    red[l] = acc;
    BARRIER();
    for (int s = 32; s > 0; s >>= 1) { if (l < s) red[l] += red[l + s]; BARRIER(); }
    if (l == 0) {
        const float q = (float)red[0] / *(GLOBAL const float*)a;
        const float o = q * gamma;
        y[t * rows + r] = has_bias ? bias[r] + o : o;
    }
}
// A K-quant block's integer code of element e (ml/quant_k.hpp unpack) and the scale of its sixteen (lanes' scale)
DEVICE int code_q3_k(GLOBAL const uchar* b, uint e) { const uint h = e / 128, g = (e % 128) / 32, l = e % 32; return (int)((b[32 + 32 * h + l] >> (2 * g)) & 3u) - (((b[l] >> (e / 32)) & 1u) ? 0 : 4); }
DEVICE int scale_q3_k(GLOBAL const uchar* b, uint j) { const uint low = j < 8u ? b[96 + j] & 15u : b[96 + j - 8] >> 4, high = (b[104 + j % 4] >> (2 * (j / 4))) & 3u; return (int)(low | (high << 4)) - 32; }
DEVICE int code_q4_k(GLOBAL const uchar* b, uint e) { const uint byte = b[16 + 32 * (e / 64) + e % 32]; return (int)((e % 64) < 32u ? byte & 15u : byte >> 4); }
DEVICE int scale_q4_k(GLOBAL const uchar* b, uint j) { uint sc, mn; scale_min_k4(j / 2, b + 4, &sc, &mn); return (int)sc; }
DEVICE int code_q5_k(GLOBAL const uchar* b, uint e) { const uint byte = b[48 + 32 * (e / 64) + e % 32]; return (int)(((e % 64) < 32u ? byte & 15u : byte >> 4) + (((b[16 + e % 32] >> (e / 32)) & 1u) ? 16u : 0u)); }
DEVICE int scale_q5_k(GLOBAL const uchar* b, uint j) { uint sc, mn; scale_min_k4(j / 2, b + 4, &sc, &mn); return (int)sc; }
DEVICE int code_q6_k(GLOBAL const uchar* b, uint e)
{
    const uint n = e / 128, part = (e % 128) / 32, l = e % 32, src = b[64 * n + 32 * (part & 1u) + l], nib = part < 2u ? src & 15u : src >> 4;
    return (int)(nib | (((b[128 + 32 * n + l] >> (2 * part)) & 3u) << 4)) - 32;
}
DEVICE int scale_q6_k(GLOBAL const uchar* b, uint j) { return (int)(i8)b[192 + j]; }
DEVICE float q8k_d(GLOBAL const uchar* y) { return *(GLOBAL const float*)y; }
DEVICE int q8k_q(GLOBAL const uchar* y, uint e) { return (int)(i8)y[4 + e]; }
DEVICE int q8k_bsum(GLOBAL const uchar* y, uint j) { return (int)((GLOBAL const short*)(y + 260))[j]; }
// ggml's eight lanes (ml/quant_k.hpp lanes): lane l sums scale(e / 16) q8[e] a[e] over e = 8 j + l, then d times it
#define K_LANES(F, B, D)                                                                                                \
    for (int l = 0; l < 8; ++l) {                                                                                       \
        int aux = 0;                                                                                                    \
        for (uint j = 0; j < 32; ++j) aux += scale_##F(B, j / 2) * (q8k_q(y, 8 * j + (uint)l) * code_##F(B, 8 * j + (uint)l)); \
        lanes[l] = (D) * (float)aux;                                                                                    \
    }
// A block's eight lane terms and its scalar term (Q2_K: added, Q4_K and Q5_K: subtracted, Q3_K and Q6_K: none)
DEVICE void kterm_q3_k(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t) { const float d = half_at(b + 108) * q8k_d(y); K_LANES(q3_k, b, d) *t = 0.0f; }
DEVICE void kterm_q6_k(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t) { const float d = half_at(b + 208) * q8k_d(y); K_LANES(q6_k, b, d) *t = 0.0f; }
DEVICE void kterm_k4(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t, int five)
{
    int sumi = 0;
    for (uint j = 0; j < 8; ++j) { uint sc, mn; scale_min_k4(j, b + 4, &sc, &mn); sumi += q8k_bsum(y, 2 * j) * (int)mn; sumi += q8k_bsum(y, 2 * j + 1) * (int)mn; }
    const float d = half_at(b) * q8k_d(y);
    if (five) { K_LANES(q5_k, b, d) } else { K_LANES(q4_k, b, d) }
    const float dmin = half_at(b + 2) * q8k_d(y);
    *t = dmin * (float)sumi;
}
DEVICE void kterm_q4_k(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t) { kterm_k4(b, y, lanes, t, 0); }
DEVICE void kterm_q5_k(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t) { kterm_k4(b, y, lanes, t, 1); }
// Q2_K (ml/quant_k.hpp dot): one term a block, (dy dx) isum - (dy dmin) summs
DEVICE void kterm_q2_k(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t)
{
    int summs = 0, isum = 0;
    for (uint j = 0; j < 16; ++j) summs += q8k_bsum(y, j) * (int)(b[j] >> 4);
    for (uint j = 0; j < 16; ++j) {
        int isuml = 0;
        for (uint l = 0; l < 16; ++l) { const uint e = 16 * j + l; isuml += q8k_q(y, e) * (int)((b[16 + 32 * (e / 128) + e % 32] >> (2 * ((e % 128) / 32))) & 3u); }
        isum += (int)(b[j] & 15u) * isuml;
    }
    const float dall = q8k_d(y) * half_at(b + 80);
    const float dmin = q8k_d(y) * half_at(b + 82);
    const float p = dall * (float)isum;
    const float q = dmin * (float)summs;
    *t = p - q;
    for (int l = 0; l < 8; ++l) lanes[l] = 0.0f;
}
#undef K_LANES
// IQ4_NL with Q8_0 (ml/quant_iq.hpp dot): (dy dx) (s1 + s2)
DEVICE float term_iq4_nl(GLOBAL const uchar* wb, GLOBAL const uchar* xb)
{
    const float d = half_at(xb) * half_at(wb);
    int s1 = 0, s2 = 0;
    for (uint j = 0; j < 16; ++j) { s1 += (int)(i8)xb[2 + j] * iq4nl_values[wb[2 + j] & 15u]; s2 += (int)(i8)xb[18 + j] * iq4nl_values[wb[2 + j] >> 4]; }
    return d * (float)(s1 + s2);
}
// NVFP4 with Q8_0 (two activation blocks a block): four terms, (dy d) (lo + hi) per sixteen, added in order
DEVICE void terms_nvfp4(GLOBAL const uchar* wb, GLOBAL const uchar* xb, float* sub)
{
    for (uint sg = 0; sg < 4; ++sg) {
        const float d = ue4m3_half(wb[sg]);
        GLOBAL const uchar* yb = xb + 34 * (sg / 2);
        const uint off = (sg % 2) * 16;
        const float dy = half_at(yb);
        int lo = 0, hi = 0;
        for (uint j = 0; j < 8; ++j) {
            const uint q = wb[4 + 8 * sg + j];
            lo += (int)(i8)yb[2 + off + j] * fp4_doubled[q & 15u];
            hi += (int)(i8)yb[2 + off + j + 8] * fp4_doubled[q >> 4];
        }
        const float dd = dy * d;
        sub[sg] = dd * (float)(lo + hi);
    }
}
// Q1_0 (four activation blocks) and Q2_0 (two): d_x Σ_k (d_k s_k), the inner sum from 0 in order
DEVICE float term_q1_0(GLOBAL const uchar* wb, GLOBAL const uchar* xb)
{
    float sumi = 0.0f;
    for (uint k = 0; k < 4; ++k) {
        GLOBAL const uchar* yb = xb + 34 * k;
        int s = 0;
        for (uint j = 0; j < 32; ++j) { const int q = (int)(i8)yb[2 + j]; s += ((wb[2 + 4 * k + j / 8] >> (j % 8)) & 1u) ? q : -q; }
        const float term = half_at(yb) * (float)s;
        sumi += term;
    }
    return half_at(wb) * sumi;
}
DEVICE float term_q2_0(GLOBAL const uchar* wb, GLOBAL const uchar* xb)
{
    float sumi = 0.0f;
    for (uint k = 0; k < 2; ++k) {
        GLOBAL const uchar* yb = xb + 34 * k;
        int s = 0;
        for (uint j = 0; j < 32; ++j) s += ((int)((wb[2 + 8 * k + j / 4] >> (2 * (j % 4))) & 3u) - 1) * (int)(i8)yb[2 + j];
        const float term = half_at(yb) * (float)s;
        sumi += term;
    }
    return half_at(wb) * sumi;
}
// IQ4_XS with Q8_K: eight terms a block, (dx dy (s - 32)) (s1 + s2) per 32, added in order (KIND 3)
DEVICE void kterm_iq4_xs(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t)
{
    const float d4d8 = half_at(b) * q8k_d(y);
    for (uint ib = 0; ib < 8; ++ib) {
        const float dl = d4d8 * (float)iq4_xs_scale(b, ib);
        int s1 = 0, s2 = 0;
        for (uint j = 0; j < 16; ++j) {
            s1 += q8k_q(y, 32 * ib + j) * iq4nl_values[b[8 + 16 * ib + j] & 15u];
            s2 += q8k_q(y, 32 * ib + 16 + j) * iq4nl_values[b[8 + 16 * ib + j] >> 4];
        }
        lanes[ib] = dl * (float)(s1 + s2);
    }
    *t = 0.0f;
}
// TQ1_0 and TQ2_0 (ml/quant_ternary.hpp dot): one integer sum a block over the elements in order, times dx dy
DEVICE int code_tq1_0(GLOBAL const uchar* b, uint e)
{
    uint byte, k;
    if (e < 160u) { byte = b[e % 32]; k = e / 32; }
    else if (e < 240u) { byte = b[32 + (e - 160) % 16]; k = (e - 160) / 16; }
    else { byte = b[48 + (e - 240) % 4]; k = (e - 240) / 4; }
    return (int)((((byte * pow3[k]) & 255u) * 3u) >> 8) - 1;
}
DEVICE void kterm_tq1_0(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t)
{
    int s = 0;
    for (uint e = 0; e < 256; ++e) s += code_tq1_0(b, e) * q8k_q(y, e);
    const float scale = half_at(b + 52) * q8k_d(y);
    *t = (float)s * scale;
    for (int l = 0; l < 8; ++l) lanes[l] = 0.0f;
}
DEVICE void kterm_tq2_0(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t)
{
    int s = 0;
    for (uint e = 0; e < 256; ++e) s += q8k_q(y, e) * ((int)((b[32 * (e / 128) + e % 32] >> (2 * ((e % 128) / 32))) & 3u) - 1);
    const float d = q8k_d(y) * half_at(b + 64);
    *t = (float)s * d;
    for (int l = 0; l < 8; ++l) lanes[l] = 0.0f;
}

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
// Rows of activations quantised to Q8_1 (ml/quantize.hpp quantize_row<block_q8_1>): Q8_0's codes, then s = (Σ q) d
KERNEL GROUP1(64) void quantize_q8_1(GLOBAL const float* x, u64 blocks, GLOBAL uchar* out)
{
    const u64 b = GX * 64 + (u64)LX;
    if (b >= blocks) return;
    GLOBAL const float* v = x + b * 32;
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) { const float a = v[j] < 0.0f ? -v[j] : v[j]; amax = amax > a ? amax : a; }
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    GLOBAL uchar* o = out + b * 36;
    const uint h = f2h(d);
    o[0] = (uchar)(h & 0xffu);
    o[1] = (uchar)(h >> 8);
    int sum = 0;
    for (int j = 0; j < 32; ++j) { const float scaled = v[j] * id; const int q = round_to_int(scaled); o[4 + j] = (uchar)(i8)q; sum += (int)(i8)q; }
    const float sd = (float)sum * d;
    const uint hs = f2h(sd);
    o[2] = (uchar)(hs & 0xffu);
    o[3] = (uchar)(hs >> 8);
}
// core's dot products of Q5_0 with Q8_0, and of Q4_1 and Q5_1 with Q8_1, one block's term
DEVICE float term_q5_0(GLOBAL const uchar* wb, GLOBAL const uchar* xb)
{
    const uint qh = word_at(wb + 2);
    int s0 = 0, s1 = 0;
    for (uint j = 0; j < 16; ++j) {
        s0 += ((int)((wb[6 + j] & 15u) | (((qh >> j) & 1u) << 4)) - 16) * (int)(i8)xb[2 + j];
        s1 += ((int)((wb[6 + j] >> 4) | (((qh >> (j + 16)) & 1u) << 4)) - 16) * (int)(i8)xb[18 + j];
    }
    const float dd = half_at(wb) * half_at(xb);
    return dd * (float)(s0 + s1);
}
DEVICE float term_q4_1(GLOBAL const uchar* wb, GLOBAL const uchar* xb)
{
    int s0 = 0, s1 = 0;
    for (uint j = 0; j < 16; ++j) { s0 += (int)(wb[4 + j] & 15u) * (int)(i8)xb[4 + j]; s1 += (int)(wb[4 + j] >> 4) * (int)(i8)xb[20 + j]; }
    const float dd = half_at(wb) * half_at(xb);
    const float a = dd * (float)(s0 + s1);
    const float b = half_at(wb + 2) * half_at(xb + 2);
    return a + b;
}
DEVICE float term_q5_1(GLOBAL const uchar* wb, GLOBAL const uchar* xb)
{
    const uint qh = word_at(wb + 4);
    int s0 = 0, s1 = 0;
    for (uint j = 0; j < 16; ++j) {
        s0 += (int)((wb[8 + j] & 15u) | (((qh >> j) & 1u) << 4)) * (int)(i8)xb[4 + j];
        s1 += (int)((wb[8 + j] >> 4) | (((qh >> (j + 16)) & 1u) << 4)) * (int)(i8)xb[20 + j];
    }
    const float dd = half_at(wb) * half_at(xb);
    const float a = dd * (float)(s0 + s1);
    const float b = half_at(wb + 2) * half_at(xb + 2);
    return a + b;
}

)SRC";

// The kernels of one format, instantiated with FMT (its name) and GET(r, c) (its weight) defined
inline constexpr const char* kernels_source = R"SRC(
#define MV_TOKENS 8
#define MV_ROWS 4
// Matrix-vector (decode): y_t = W x_t + b for up to MV_TOKENS rows x_t. A group of 64 per MV_ROWS outputs, sharing
// each load of x; each item decodes eight consecutive columns at a time (GET8), 512 apart, and adds their products, then the 64 partial sums are added by a fixed tree, so
// the result is the same every run (not the reference's order: within rounding of it).
KERNEL GROUP1(64) void CAT(qmv, FMT)(WARGS, GLOBAL const float* x, u64 n, u64 rows, GLOBAL const float* bias, int has_bias, GLOBAL float* y)
{
    LOCAL float red[MV_ROWS * 64];
    const int l = LX;
    const u64 r0 = GX * MV_ROWS;
    const int nr = rows - r0 < MV_ROWS ? (int)(rows - r0) : MV_ROWS, nt = (int)n;
    float acc[MV_ROWS][MV_TOKENS];
    for (int k = 0; k < MV_ROWS; ++k)
        for (int t = 0; t < MV_TOKENS; ++t) acc[k][t] = 0.0f;
    for (u64 c0 = (u64)l * 8; c0 < cols; c0 += 512) {
        float v[MV_ROWS][8];
        const int e = cols - c0 < 8 ? (int)(cols - c0) : 8;
        for (int k = 0; k < MV_ROWS; ++k)
            if (k < nr) {
                if (e == 8) GET8(r0 + (u64)k, c0, v[k]);
                else for (int j = 0; j < 8; ++j) v[k][j] = j < e ? GET(r0 + (u64)k, c0 + (u64)j) : 0.0f;
            }
        for (int t = 0; t < MV_TOKENS; ++t)
            if (t < nt) {
                GLOBAL const float* xt = x + (u64)t * cols + c0;
                float xv[8];
                if (e == 8) {   // 8 floats from a multiple of 8: two aligned vector loads
                    const float4 lo = LOAD4(xt), hi = LOAD4(xt + 4);
                    xv[0] = lo.x; xv[1] = lo.y; xv[2] = lo.z; xv[3] = lo.w; xv[4] = hi.x; xv[5] = hi.y; xv[6] = hi.z; xv[7] = hi.w;
                } else
                    for (int j = 0; j < 8; ++j) xv[j] = j < e ? xt[j] : 0.0f;
                for (int k = 0; k < MV_ROWS; ++k)
                    if (k < nr)
                        for (int j = 0; j < e; ++j) { const float p = v[k][j] * xv[j]; acc[k][t] += p; }
            }
    }
    // One token at a time through a small local array, so many groups fit on a compute unit
    for (int t = 0; t < MV_TOKENS; ++t) {
        if (t >= nt) break;
        for (int k = 0; k < MV_ROWS; ++k) red[k * 64 + l] = acc[k][t];
        BARRIER();
        for (int s = 32; s > 0; s >>= 1) {
            if (l < s)
                for (int k = 0; k < MV_ROWS; ++k) red[k * 64 + l] += red[k * 64 + l + s];
            BARRIER();
        }
        if (l < nr) {
            float o = red[l * 64];
            if (has_bias) o = o + bias[r0 + (u64)l];
            y[(u64)t * rows + r0 + (u64)l] = o;
        }
        BARRIER();
    }
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
#undef MV_ROWS
)SRC";

// The integer path of one format with core's dot product, instantiated with FMT, TERM and BYTES (its block's size)
inline constexpr const char* q8_source = R"SRC(
// y_t = b + Σ over blocks of core's terms (W row r's block of WE numbers with x_t's WE / 32 activation blocks), NSUB
// terms a block (NVFP4's four sixteens), added in order as core's dot does: a group per (row, token), 64 blocks at a
// time into local memory, the first item adding them in order.
KERNEL GROUP1(64) void CAT(qdot, FMT)(GLOBAL const uchar* w, u64 cols, GLOBAL const uchar* xq, u64 rows, u64 t0, GLOBAL const float* bias, int has_bias,
                                      GLOBAL float* y)
{
    LOCAL float terms[NSUB * 64];
    const int l = LX;
    const u64 r = GX, t = t0 + GY, nb = cols / WE, xb = cols / 32;
    float sum = 0.0f;
    for (u64 b0 = 0; b0 < nb; b0 += 64) {
        const u64 b = b0 + (u64)l;
        if (b < nb) {
            float sub[NSUB];
            TERM(w + (r * nb + b) * BYTES, xq + (t * xb + b * (WE / 32)) * XBYTES, sub);
            for (int k = 0; k < NSUB; ++k) terms[k * 64 + l] = sub[k];
        }
        BARRIER();
        if (l == 0) {
            const u64 e = nb - b0 < 64 ? nb - b0 : 64;
            for (u64 i = 0; i < e; ++i)
                for (int k = 0; k < NSUB; ++k) sum += terms[k * 64 + (int)i];
        }
        BARRIER();
    }
    if (l == 0) y[t * rows + r] = has_bias ? bias[r] + sum : sum;
}
)SRC";

// The grid-based IQ formats (ml/quant_iq.hpp dequantize_row and dot), after their codebooks (iq_tables()): each weight
// with core's operations in core's order; each dot one integer sum a block, d_x d_y times it, the blocks in order, then
// the format's final factor (FINAL)
inline constexpr const char* iq_source = R"SRC(
#define GET8_BY_ONE(F) DEVICE void get8_##F(WARGS, u64 r, u64 c0, float* v) { for (int j = 0; j < 8; ++j) v[j] = get_##F(WPASS, r, c0 + (u64)j); }
DEVICE uint u16_at(GLOBAL const uchar* p) { return (uint)p[0] | ((uint)p[1] << 8); }
DEVICE uint grid_byte(u64 g, uint j) { return (uint)((g >> (8 * j)) & 255u); }
DEVICE int grid_sbyte(u64 g, uint j) { return (int)(i8)(uchar)((g >> (8 * j)) & 255u); }
DEVICE float signed_by(float g, uint signs, uint j) { return ((signs >> j) & 1u) ? -g : g; }
DEVICE int sign_of(uint signs, uint j) { return ((signs >> j) & 1u) ? -1 : 1; }
// IQ2_XXS (66: d, qs[32] u16): per 32 the words a0 (four grid indices) and a1 (four 7-bit signs, the scale on top)
DEVICE float get_iq2_xxs(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 256, 66);
    const uint e = (uint)(c % 256), ib = e / 32, l = (e % 32) / 8, j = e % 8;
    const uint a0 = word_at(b + 2 + 8 * ib), a1 = word_at(b + 6 + 8 * ib);
    const float s = 0.5f + (float)(a1 >> 28);
    const float ds = half_at(b) * s;
    const float db = ds * 0.25f;
    const float g = db * (float)grid_byte(iq2xxs_grid[(a0 >> (8 * l)) & 255u], j);
    return signed_by(g, iq2_signs[(a1 >> (7 * l)) & 127u], j);
}
// IQ2_XS (74: d, qs[32] u16 of a 9-bit index and 7-bit signs, scales[8] two a byte)
DEVICE float get_iq2_xs(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 256, 74);
    const uint e = (uint)(c % 256), ib = e / 32, l = (e % 32) / 8, j = e % 8, q = u16_at(b + 2 + 2 * (4 * ib + l));
    const float s = 0.5f + (float)((b[66 + ib] >> (4 * (l / 2))) & 15u);
    const float ds = half_at(b) * s;
    const float db = ds * 0.25f;
    const float g = db * (float)grid_byte(iq2xs_grid[q & 511u], j);
    return signed_by(g, iq2_signs[q >> 9], j);
}
// IQ2_S (82: d, qs[64] indices' low bits then signs, qh[8], scales[8])
DEVICE float get_iq2_s(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 256, 82);
    const uint e = (uint)(c % 256), ib = e / 32, l = (e % 32) / 8, j = e % 8;
    const float s = 0.5f + (float)((b[74 + ib] >> (4 * (l / 2))) & 15u);
    const float ds = half_at(b) * s;
    const float db = ds * 0.25f;
    const uint idx = (uint)b[2 + 4 * ib + l] | (((uint)b[66 + ib] << (8 - 2 * l)) & 0x300u);
    const float g = db * (float)grid_byte(iq2s_grid[idx], j);
    return signed_by(g, b[2 + 32 + 4 * ib + l], j);
}
// IQ3_XXS (98: d, qs[96]: 64 indices of four, then per 32 a word of four 7-bit signs and the scale)
DEVICE float get_iq3_xxs(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 256, 98);
    const uint e = (uint)(c % 256), ib = e / 32, l = (e % 32) / 8, j = e % 8, aux = word_at(b + 2 + 64 + 4 * ib);
    const float s = 0.5f + (float)(aux >> 28);
    const float ds = half_at(b) * s;
    const float db = ds * 0.5f;
    const float g = db * (float)grid_byte(iq3xxs_grid[b[2 + 8 * ib + 2 * l + j / 4]], j % 4);
    return signed_by(g, iq2_signs[(aux >> (7 * l)) & 127u], j);
}
// IQ3_S (110: d, qs[64], qh[8] the 9th bits, signs[32], scales[4])
DEVICE float get_iq3_s(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 256, 110);
    const uint e = (uint)(c % 256), ib = e / 32, l = (e % 32) / 8, j = e % 8, h = j / 4;
    const float db = half_at(b) * (float)(1u + 2u * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 15u));
    const uint idx = (uint)b[2 + 8 * ib + 2 * l + h] | (((uint)b[66 + ib] << (8 - 2 * l - h)) & 256u);
    const float g = db * (float)grid_byte(iq3s_grid[idx], j % 4);
    return signed_by(g, b[74 + 4 * ib + l], j);
}
// IQ1_S (50: d, qs[32], qh[8] u16: index high bits, the 3-bit scale, the shift's sign)
DEVICE float get_iq1_s(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 256, 50);
    const uint e = (uint)(c % 256), ib = e / 32, l = (e % 32) / 8, j = e % 8, qh = u16_at(b + 34 + 2 * ib);
    const float dl = half_at(b) * (float)(2u * ((qh >> 12) & 7u) + 1u);
    const float delta = (qh & 0x8000u) ? -0.125f : 0.125f;
    const u64 g = iq1s_grid[(uint)b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7u) << 8)];
    const float v = (float)grid_sbyte(g, j) + delta;
    return dl * v;
}
// IQ1_M (56: qs[32], qh[16], scales[8]: four words of 3-bit scales, the float16 scale's nibbles on top)
DEVICE float iq1_m_d(GLOBAL const uchar* b)
{
    const uint s0 = u16_at(b + 48), s1 = u16_at(b + 50), s2 = u16_at(b + 52), s3 = u16_at(b + 54);
    return mf_decode((s0 >> 12) | ((s1 >> 8) & 0x00f0u) | ((s2 >> 4) & 0x0f00u) | (s3 & 0xf000u), 5, 10, 0);
}
DEVICE uint iq1_m_ls(GLOBAL const uchar* b, uint ib, uint second) { return 2u * ((u16_at(b + 48 + 2 * (ib / 2)) >> (6 * (ib % 2) + 3 * second)) & 7u) + 1u; }
DEVICE uint iq1_m_index(GLOBAL const uchar* b, uint ib, uint l)
{
    const uint h = b[32 + 2 * ib + l / 2];
    return (uint)b[4 * ib + l] | ((h << (8 - 4 * (l % 2))) & 0x700u);
}
DEVICE float get_iq1_m(WARGS, u64 r, u64 c)
{
    GLOBAL const uchar* b = block_at(w, cols, r, c, 256, 56);
    const uint e = (uint)(c % 256), ib = e / 32, l = (e % 32) / 8, j = e % 8;
    const float dl = iq1_m_d(b) * (float)iq1_m_ls(b, ib, l / 2);
    const uint h = b[32 + 2 * ib + l / 2];
    const float delta = (h & ((l % 2) ? 0x80u : 0x08u)) ? -0.125f : 0.125f;
    const float v = (float)grid_sbyte(iq1s_grid[iq1_m_index(b, ib, l)], j) + delta;
    return dl * v;
}
GET8_BY_ONE(iq2_xxs)
GET8_BY_ONE(iq2_xs)
GET8_BY_ONE(iq2_s)
GET8_BY_ONE(iq3_xxs)
GET8_BY_ONE(iq3_s)
GET8_BY_ONE(iq1_s)
GET8_BY_ONE(iq1_m)
#undef GET8_BY_ONE
// The dot products' terms with Q8_K
DEVICE void kterm_iq2_xxs(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t)
{
    const float d = half_at(b) * q8k_d(y);
    int bsum = 0;
    for (uint ib = 0; ib < 8; ++ib) {
        const uint a0 = word_at(b + 2 + 8 * ib), a1 = word_at(b + 6 + 8 * ib);
        int sumi = 0;
        for (uint l = 0; l < 4; ++l) {
            const u64 g = iq2xxs_grid[(a0 >> (8 * l)) & 255u];
            const uint sg = iq2_signs[(a1 >> (7 * l)) & 127u];
            for (uint j = 0; j < 8; ++j) sumi += (int)grid_byte(g, j) * q8k_q(y, 32 * ib + 8 * l + j) * sign_of(sg, j);
        }
        bsum += sumi * (int)(2u * (a1 >> 28) + 1u);
    }
    *t = d * (float)bsum;
    for (int k = 0; k < 8; ++k) lanes[k] = 0.0f;
}
DEVICE void kterm_iq2_xs(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t)
{
    const float d = half_at(b) * q8k_d(y);
    int bsum = 0;
    for (uint ib = 0; ib < 8; ++ib)
        for (uint h = 0; h < 2; ++h) {
            int sumi = 0;
            for (uint l = 2 * h; l < 2 * h + 2; ++l) {
                const uint q = u16_at(b + 2 + 2 * (4 * ib + l));
                const u64 g = iq2xs_grid[q & 511u];
                for (uint j = 0; j < 8; ++j) sumi += (int)grid_byte(g, j) * q8k_q(y, 32 * ib + 8 * l + j) * sign_of(iq2_signs[q >> 9], j);
            }
            bsum += sumi * (int)(2u * ((b[66 + ib] >> (4 * h)) & 15u) + 1u);
        }
    *t = d * (float)bsum;
    for (int k = 0; k < 8; ++k) lanes[k] = 0.0f;
}
DEVICE void kterm_iq2_s(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t)
{
    const float d = half_at(b) * q8k_d(y);
    int bsum = 0;
    for (uint ib = 0; ib < 8; ++ib) {
        int sumi[2] = {0, 0};
        for (uint l = 0; l < 4; ++l) {
            const u64 g = iq2s_grid[(uint)b[2 + 4 * ib + l] | (((uint)b[66 + ib] << (8 - 2 * l)) & 0x300u)];
            const uint sg = b[2 + 32 + 4 * ib + l];
            for (uint j = 0; j < 8; ++j) sumi[l / 2] += q8k_q(y, 32 * ib + 8 * l + j) * (int)grid_byte(g, j) * sign_of(sg, j);
        }
        bsum += (int)(1u + 2u * (b[74 + ib] & 15u)) * sumi[0] + (int)(1u + 2u * (b[74 + ib] >> 4)) * sumi[1];
    }
    *t = d * (float)bsum;
    for (int k = 0; k < 8; ++k) lanes[k] = 0.0f;
}
DEVICE void kterm_iq3_xxs(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t)
{
    const float d = half_at(b) * q8k_d(y);
    int bsum = 0;
    for (uint ib = 0; ib < 8; ++ib) {
        const uint aux = word_at(b + 2 + 64 + 4 * ib);
        int sumi = 0;
        for (uint l = 0; l < 4; ++l) {
            const uint g1 = iq3xxs_grid[b[2 + 8 * ib + 2 * l]], g2 = iq3xxs_grid[b[2 + 8 * ib + 2 * l + 1]], sg = iq2_signs[(aux >> (7 * l)) & 127u];
            for (uint j = 0; j < 4; ++j) {
                sumi += (int)grid_byte(g1, j) * q8k_q(y, 32 * ib + 8 * l + j) * sign_of(sg, j);
                sumi += (int)grid_byte(g2, j) * q8k_q(y, 32 * ib + 8 * l + j + 4) * sign_of(sg, j + 4);
            }
        }
        bsum += sumi * (int)(2u * (aux >> 28) + 1u);
    }
    *t = d * (float)bsum;
    for (int k = 0; k < 8; ++k) lanes[k] = 0.0f;
}
DEVICE void kterm_iq3_s(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t)
{
    const float d = half_at(b) * q8k_d(y);
    int bsum = 0;
    for (uint ib = 0; ib < 8; ++ib) {
        int sumi = 0;
        for (uint l = 0; l < 4; ++l) {
            const uint g1 = iq3s_grid[(uint)b[2 + 8 * ib + 2 * l] | (((uint)b[66 + ib] << (8 - 2 * l)) & 256u)];
            const uint g2 = iq3s_grid[(uint)b[2 + 8 * ib + 2 * l + 1] | (((uint)b[66 + ib] << (7 - 2 * l)) & 256u)];
            const uint sg = b[74 + 4 * ib + l];
            for (uint j = 0; j < 4; ++j) {
                sumi += (int)grid_byte(g1, j) * q8k_q(y, 32 * ib + 8 * l + j) * sign_of(sg, j);
                sumi += (int)grid_byte(g2, j) * q8k_q(y, 32 * ib + 8 * l + j + 4) * sign_of(sg, j + 4);
            }
        }
        bsum += sumi * (int)(2u * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 15u) + 1u);
    }
    *t = d * (float)bsum;
    for (int k = 0; k < 8; ++k) lanes[k] = 0.0f;
}
DEVICE void kterm_iq1_s(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t)
{
    int sumi = 0, sumi1 = 0;
    for (uint ib = 0; ib < 8; ++ib) {
        const uint qh = u16_at(b + 34 + 2 * ib);
        const int ls = (int)(2u * ((qh >> 12) & 7u) + 1u), delta = (qh & 0x8000u) ? -1 : 1;
        int lsum = 0;
        for (uint l = 0; l < 4; ++l) {
            const u64 g = iq1s_grid[(uint)b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7u) << 8)];
            for (uint j = 0; j < 8; ++j) lsum += q8k_q(y, 32 * ib + 8 * l + j) * grid_sbyte(g, j);
        }
        sumi += ls * lsum;
        sumi1 += ls * delta * (q8k_bsum(y, 2 * ib) + q8k_bsum(y, 2 * ib + 1));
    }
    const float dd = half_at(b) * q8k_d(y);
    const float shift = 0.125f * (float)sumi1;
    const float s = (float)sumi + shift;
    *t = dd * s;
    for (int k = 0; k < 8; ++k) lanes[k] = 0.0f;
}
DEVICE void kterm_iq1_m(GLOBAL const uchar* b, GLOBAL const uchar* y, float* lanes, float* t)
{
    int sumi1 = 0, sumi2 = 0;
    for (uint ib = 0; ib < 8; ++ib) {
        int sum1[2] = {0, 0}, sum2[2] = {0, 0};
        for (uint l = 0; l < 4; ++l) {
            const uint h = b[32 + 2 * ib + l / 2];
            const int delta = (h & ((l % 2) ? 0x80u : 0x08u)) ? -1 : 1;
            const u64 g = iq1s_grid[iq1_m_index(b, ib, l)];
            int lsum1 = 0, lsum2 = 0;
            for (uint j = 0; j < 8; ++j) { const int q = q8k_q(y, 32 * ib + 8 * l + j); lsum1 += q * grid_sbyte(g, j); lsum2 += q; }
            sum1[l / 2] += lsum1;
            sum2[l / 2] += lsum2 * delta;
        }
        const int ls1 = (int)iq1_m_ls(b, ib, 0), ls2 = (int)iq1_m_ls(b, ib, 1);
        sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
        sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
    }
    const float dd = iq1_m_d(b) * q8k_d(y);
    const float shift = 0.125f * (float)sumi2;
    const float s = (float)sumi1 + shift;
    *t = dd * s;
    for (int k = 0; k < 8; ++k) lanes[k] = 0.0f;
}
)SRC";

// The Q8_K path of one format, instantiated with FMT, BYTES and KIND (0: lanes only, 1: lanes less the scalar terms,
// 2: the scalar terms only, 3: the eight terms of each block added in order, IQ4_XS). As core's dot: the lanes' sums and the scalar sum each over the blocks in order, then the
// scalar sum plus the eight lanes in order. A group per (row, token), 64 blocks at a time; the first item adds.
inline constexpr const char* q8k_source = R"SRC(
KERNEL GROUP1(64) void CAT(qkdot, FMT)(GLOBAL const uchar* w, u64 cols, GLOBAL const uchar* xq, u64 rows, u64 t0, GLOBAL const float* bias, int has_bias,
                                       GLOBAL float* y)
{
    LOCAL float terms[9 * 64];
    const int l = LX;
    const u64 r = GX, t = t0 + GY, nb = cols / 256;
    float sums[8], sumf = 0.0f;
    for (int k = 0; k < 8; ++k) sums[k] = 0.0f;
    for (u64 b0 = 0; b0 < nb; b0 += 64) {
        const u64 b = b0 + (u64)l;
        if (b < nb) {
            float lanes[8], s;
            CAT(kterm, FMT)(w + (r * nb + b) * BYTES, xq + (t * nb + b) * 292, lanes, &s);
            for (int k = 0; k < 8; ++k) terms[k * 64 + l] = lanes[k];
            terms[8 * 64 + l] = s;
        }
        BARRIER();
        if (l == 0) {
            const int e = nb - b0 < 64 ? (int)(nb - b0) : 64;
            for (int i = 0; i < e; ++i) {
                if (KIND == 0 || KIND == 1) for (int k = 0; k < 8; ++k) sums[k] += terms[k * 64 + i];
                if (KIND == 3) for (int k = 0; k < 8; ++k) sumf += terms[k * 64 + i];
                if (KIND == 1) sumf -= terms[8 * 64 + i];
                if (KIND == 2) sumf += terms[8 * 64 + i];
            }
        }
        BARRIER();
    }
    if (l == 0) {
        if (KIND == 0 || KIND == 1) for (int k = 0; k < 8; ++k) sumf += sums[k];
        if (FINAL != 1.0f) sumf = FINAL * sumf;
        y[t * rows + r] = has_bias ? bias[r] + sumf : sumf;
    }
}
)SRC";

// Core's IQ codebooks as device constants, printed from core's own tables (ml/quant_iq_grids.hpp), or nothing where the
// pinned core has none
inline std::string iq_tables()
{
#ifdef SNICHOLLS_GPU_QUANT_K
    std::string s;
    char buf[32];
    const auto table = [&](const char* type, const char* name, const auto& t, bool wide) {
        s += std::string("CONSTANT ") + type + " " + name + "[" + std::to_string(t.size()) + "] = {";
        for (std::size_t i = 0; i < t.size(); ++i) {
            if (wide)
                std::snprintf(buf, sizeof buf, "U64C(0x%016llx),", static_cast<unsigned long long>(t[i]));
            else
                std::snprintf(buf, sizeof buf, "0x%08xu,", static_cast<unsigned>(t[i]));
            s += buf;
        }
        s += "};\n";
    };
    table("u64", "iq2xxs_grid", ml::iq2xxs_grid, true);
    table("u64", "iq2xs_grid", ml::iq2xs_grid, true);
    table("u64", "iq2s_grid", ml::iq2s_grid, true);
    table("u64", "iq1s_grid", ml::iq1s_grid, true);
    table("uint", "iq3xxs_grid", ml::iq3xxs_grid, false);
    table("uint", "iq3s_grid", ml::iq3s_grid, false);
    table("uint", "iq2_signs", ml::iq2_signs, false);
    return s;
#else
    return {};
#endif
}

// The whole program for one language: the prelude, the formats, each format's kernels, the integer paths, then `extra`
// (the CUDA tensor-core kernels)
inline std::string program(const char* prelude, const char* extra = "")
{
    std::string s = std::string(prelude) + formats_source + iq_tables() + (available(format::iq2_xxs) ? iq_source : "");
    for (std::size_t f = 0; f < format_count; ++f) {
        if (!available(format(f)))
            continue;
        const std::string n(names[f]);
        s += "#define FMT " + n + "\n#define GET(r, c) get_" + n + "(WPASS, r, c)\n#define GET8(r, c, v) get8_" + n + "(WPASS, r, c, v)\n" + kernels_source
             + "#undef GET8\n#undef GET\n#undef FMT\n";
    }
    // name, block bytes, activation block bytes, numbers a block, terms a block
    for (const auto& [name, bytes, xbytes, we, nsub] :
         {std::tuple{"q8_0", 34, 34, 32, 1}, std::tuple{"q4_0", 18, 34, 32, 1}, std::tuple{"mxfp4", 17, 34, 32, 1}, std::tuple{"q5_0", 22, 34, 32, 1},
          std::tuple{"q4_1", 20, 36, 32, 1}, std::tuple{"q5_1", 24, 36, 32, 1}, std::tuple{"iq4_nl", 18, 34, 32, 1}, std::tuple{"nvfp4", 36, 34, 64, 4},
          std::tuple{"q1_0", 18, 34, 128, 1}, std::tuple{"q2_0", 18, 34, 64, 1}}) {
        const std::string n(name);
        s += "#define FMT " + n + "\n#define BYTES " + std::to_string(bytes) + "\n#define XBYTES " + std::to_string(xbytes) + "\n#define WE "
             + std::to_string(we) + "\n#define NSUB " + std::to_string(nsub) + "\n"
             + (nsub == 1 ? "#define TERM(wb, xb, sub) (sub)[0] = term_" + n + "(wb, xb)\n" : "#define TERM terms_" + n + "\n") + q8_source
             + "#undef NSUB\n#undef WE\n#undef XBYTES\n#undef BYTES\n#undef TERM\n#undef FMT\n";
    }
    // name, block bytes, kind, final factor
    for (const auto& [name, bytes, kind, final] :
         {std::tuple{"q2_k", 84, 2, "1.0f"}, std::tuple{"q3_k", 110, 0, "1.0f"}, std::tuple{"q4_k", 144, 1, "1.0f"}, std::tuple{"q5_k", 176, 1, "1.0f"},
          std::tuple{"q6_k", 210, 0, "1.0f"}, std::tuple{"tq1_0", 54, 2, "1.0f"}, std::tuple{"tq2_0", 66, 2, "1.0f"}, std::tuple{"iq4_xs", 136, 3, "1.0f"},
          std::tuple{"iq2_xxs", 66, 2, "0.125f"}, std::tuple{"iq2_xs", 74, 2, "0.125f"}, std::tuple{"iq2_s", 82, 2, "0.125f"},
          std::tuple{"iq3_xxs", 98, 2, "0.25f"}, std::tuple{"iq3_s", 110, 2, "1.0f"}, std::tuple{"iq1_s", 50, 2, "1.0f"}, std::tuple{"iq1_m", 56, 2, "1.0f"}}) {
        const std::string n(name);
        if (!available(format(std::size_t(std::find(names.begin(), names.end(), std::string_view(name)) - names.begin()))))
            continue;
        s += "#define FMT " + n + "\n#define BYTES " + std::to_string(bytes) + "\n#define KIND " + std::to_string(kind) + "\n#define FINAL " + final + "\n"
             + q8k_source + "#undef FINAL\n#undef KIND\n#undef BYTES\n#undef FMT\n";
    }
    return s + extra;
}

} // namespace snicholls::gpu::detail::quant

#endif /* snicholls_gpu_detail_quant_source_hpp */
