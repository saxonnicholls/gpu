//
//  quant_cases.hpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Test and benchmark support (not part of the library): weight matrices in each of core's quantised formats, made
//  from random floats by core's own quantisers, with their dequantised floats (core's dequantisers) and the host
//  references every device product is checked against. Shared by the OpenCL and CUDA tests, so both backends are held
//  to the same cases.
//

#ifndef snicholls_gpu_tests_support_quant_cases_hpp
#define snicholls_gpu_tests_support_quant_cases_hpp

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/lora.hpp>
#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/nf4.hpp>
#include <snicholls/core/ml/quantize.hpp>
#include <snicholls/core/stats/random.hpp>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace snicholls::gpu::testing {

namespace qml = snicholls::ml;

inline std::vector<float> uniform(std::size_t n, std::uint64_t seed, float scale = 1.0f)
{
    std::vector<float> v(n);
    snicholls::stats::wyrand g(seed);
    for (float& e : v)
        e = scale * static_cast<float>(2 * snicholls::stats::uniform_real(g) - 1);
    return v;
}

// One weight matrix in format Tag: its storage, core's view over it, and its dequantised floats
template <typename Tag>
struct host_matrix;

// The formats, as tags: the view type, the column unit (a block never crosses a row) and whether core has an integer
// dot product for it
struct q8_0_tag { using view = qml::quantized_weights<qml::block_q8_0>; static constexpr std::size_t unit = 32; static constexpr bool q8 = true; static constexpr const char* name = "q8_0"; };
struct q4_0_tag { using view = qml::quantized_weights<qml::block_q4_0>; static constexpr std::size_t unit = 32; static constexpr bool q8 = true; static constexpr const char* name = "q4_0"; };
struct mxfp4_tag { using view = qml::quantized_weights<qml::block_mxfp4>; static constexpr std::size_t unit = 32; static constexpr bool q8 = true; static constexpr const char* name = "mxfp4"; };
struct nf4_tag { using view = qml::nf4_weights; static constexpr std::size_t unit = 64; static constexpr bool q8 = false; static constexpr const char* name = "nf4"; };
template <typename S>
struct dense_tag { using view = qml::dense_weights<S>; static constexpr std::size_t unit = 1; static constexpr bool q8 = false; };
struct bf16_tag : dense_tag<qml::bfloat16> { static constexpr const char* name = "bf16"; };
struct f16_tag : dense_tag<qml::float16> { static constexpr const char* name = "f16"; };
struct e4m3_tag : dense_tag<qml::float8_e4m3> { static constexpr const char* name = "e4m3"; };
struct e5m2_tag : dense_tag<qml::float8_e5m2> { static constexpr const char* name = "e5m2"; };

template <typename Block, typename Tag>
struct block_matrix {
    std::size_t rows = 0, cols = 0;
    std::vector<Block> blocks;
    std::vector<float> dense;
    block_matrix(std::size_t r, std::size_t c, std::uint64_t seed, float scale) : rows(r), cols(c), blocks(r * c / 32), dense(r * c)
    {
        const auto src = uniform(r * c, seed, scale);
        for (std::size_t i = 0; i < r; ++i) {
            const std::span<Block> row(blocks.data() + i * (c / 32), c / 32);
            qml::quantize(std::span<const float>(src.data() + i * c, c), row);
            qml::dequantize(std::span<const Block>(row), std::span<float>(dense.data() + i * c, c));
        }
    }
    [[nodiscard]] typename Tag::view view() const { return {std::span<const Block>(blocks), rows, cols}; }
};
template <>
struct host_matrix<q8_0_tag> : block_matrix<qml::block_q8_0, q8_0_tag> { using block_matrix::block_matrix; };
template <>
struct host_matrix<q4_0_tag> : block_matrix<qml::block_q4_0, q4_0_tag> { using block_matrix::block_matrix; };
template <>
struct host_matrix<mxfp4_tag> : block_matrix<qml::block_mxfp4, mxfp4_tag> { using block_matrix::block_matrix; };

template <>
struct host_matrix<nf4_tag> {
    std::size_t rows = 0, cols = 0;
    std::vector<std::uint8_t> codes;
    std::vector<qml::float8_e4m3> block_scales;
    std::vector<float> group_scales, dense;
    float mean = 0.0f;
    host_matrix(std::size_t r, std::size_t c, std::uint64_t seed, float scale)
        : rows(r), cols(c), codes(qml::nf4_code_bytes(r * c)), block_scales(qml::nf4_block_count(r * c)), group_scales(qml::nf4_group_count(r * c)),
          dense(r * c)
    {
        // Blocks of different magnitudes, so the double quantisation's mean and groups matter
        auto src = uniform(r * c, seed, scale);
        for (std::size_t i = 0; i < src.size(); ++i)
            src[i] *= 0.25f + float((i / qml::nf4_block_size) % 7) * 0.125f;
        mean = qml::nf4_quantize(std::span<const float>(src), std::span<std::uint8_t>(codes), std::span<qml::float8_e4m3>(block_scales),
                                 std::span<float>(group_scales));
        qml::nf4_dequantize(view(), std::span<float>(dense));
    }
    [[nodiscard]] qml::nf4_weights view() const { return {codes, block_scales, group_scales, mean, rows, cols}; }
};

template <typename S, typename Tag>
struct minifloat_matrix {
    std::size_t rows = 0, cols = 0;
    std::vector<S> data;
    std::vector<float> dense;
    minifloat_matrix(std::size_t r, std::size_t c, std::uint64_t seed, float scale) : rows(r), cols(c), data(r * c), dense(r * c)
    {
        const auto src = uniform(r * c, seed, scale);
        for (std::size_t i = 0; i < src.size(); ++i) {
            data[i] = qml::to_minifloat<S>(double(src[i]));
            dense[i] = qml::to_float(data[i]);
        }
    }
    [[nodiscard]] typename Tag::view view() const { return {std::span<const S>(data), rows, cols}; }
};
template <>
struct host_matrix<bf16_tag> : minifloat_matrix<qml::bfloat16, bf16_tag> { using minifloat_matrix::minifloat_matrix; };
template <>
struct host_matrix<f16_tag> : minifloat_matrix<qml::float16, f16_tag> { using minifloat_matrix::minifloat_matrix; };
template <>
struct host_matrix<e4m3_tag> : minifloat_matrix<qml::float8_e4m3, e4m3_tag> { using minifloat_matrix::minifloat_matrix; };
template <>
struct host_matrix<e5m2_tag> : minifloat_matrix<qml::float8_e5m2, e5m2_tag> { using minifloat_matrix::minifloat_matrix; };

// The references, on the host over the dequantised floats: core's reference product then add_bias_rows (what
// ml::linear_sequence does for dense weights), and the input's gradient dX += dY W (ml::linear_sequence_backward_input)
inline std::vector<float> reference_forward(const std::vector<float>& dense, std::size_t rows, std::size_t cols, std::span<const float> xs,
                                            std::span<const float> bias)
{
    std::vector<float> ys(xs.size() / (cols == 0 ? 1 : cols) * rows, 0.0f);
    if (cols == 0)
        return ys;
    qml::linear_sequence(snicholls::linalg::reference, qml::dense_weights<float>{dense, rows, cols}, xs, std::span<float>(ys), bias);
    return ys;
}

inline void reference_backward_input(const std::vector<float>& dense, std::size_t rows, std::size_t cols, std::span<const float> dys, std::span<float> dxs)
{
    qml::linear_sequence_backward_input(snicholls::linalg::reference, qml::dense_weights<float>{dense, rows, cols}, dys, dxs);
}

// The integer path's reference: each row of X quantised to Q8_0, each output core's dot product plus its bias, as
// ml::linear over quantized_weights
template <typename Block>
std::vector<float> reference_q8(const qml::quantized_weights<Block>& w, std::span<const float> xs, std::span<const float> bias)
{
    const std::size_t n = xs.size() / w.cols;
    std::vector<float> ys(n * w.rows);
    std::vector<qml::block_q8_0> q(w.cols / 32);
    for (std::size_t t = 0; t < n; ++t) {
        qml::quantize(xs.subspan(t * w.cols, w.cols), std::span<qml::block_q8_0>(q));
        for (std::size_t i = 0; i < w.rows; ++i) {
            const float d = qml::dot(w.row(i), std::span<const qml::block_q8_0>(q));
            ys[t * w.rows + i] = bias.empty() ? d : bias[i] + d;
        }
    }
    return ys;
}

// The bound of a sum of k products added in another order, per output: k 2^-23 Σ|w x| plus one rounding of the
// result (the bias added after)
inline std::vector<double> reorder_bound(const std::vector<float>& dense, std::size_t rows, std::size_t cols, std::span<const float> xs, double per_term)
{
    const std::size_t n = cols == 0 ? 0 : xs.size() / cols;
    std::vector<double> b(n * rows, 0.0);
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t i = 0; i < rows; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < cols; ++j)
                s += std::fabs(double(dense[i * cols + j]) * double(xs[t * cols + j]));
            b[t * rows + i] = per_term * s;
        }
    return b;
}

inline bool same_bits(std::span<const float> a, std::span<const float> b)
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return true;
}

// The first index where two results differ, for a failure's message, or the size
inline std::size_t first_difference(std::span<const float> a, std::span<const float> b)
{
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return i;
    return a.size();
}

} // namespace snicholls::gpu::testing

#endif /* snicholls_gpu_tests_support_quant_cases_hpp */
