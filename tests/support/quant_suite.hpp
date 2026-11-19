//
//  quant_suite.hpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Test support (not part of the library): the quantised-weight products' tests, written once as a type-parameterised
//  GoogleTest suite over (device, format) pairs, so the OpenCL and the CUDA backends pass the same cases. A device is a
//  type with a static `get()` returning an object with
//
//      name()                                   the device, for the log
//      exact_q8()                               whether the Q8_0 path is to the bit on it
//      set_matvec_max(n)                        rows up to which the forward takes the matrix-vector kernels
//      forward<Tag>(m, xs, ys, bias)            linear_sequence over host_matrix<Tag> m, host spans in and out
//      backward<Tag>(m, dys, dxs)               linear_sequence_backward_input
//      forward_q8<Tag>(m, xs, ys, bias)         linear_sequence_q8
//
//  Instantiate with INSTANTIATE_TYPED_TEST_SUITE_P(Prefix, QuantProduct, quant_cases_for<Device>).
//

#ifndef snicholls_gpu_tests_support_quant_suite_hpp
#define snicholls_gpu_tests_support_quant_suite_hpp

#include "quant_cases.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace snicholls::gpu::testing {

template <typename Device, typename Tag>
struct quant_case {
    using device = Device;
    using tag = Tag;
};

template <typename Device>
using quant_cases_for = ::testing::Types<quant_case<Device, q8_0_tag>, quant_case<Device, q4_0_tag>, quant_case<Device, mxfp4_tag>,
                                         quant_case<Device, nf4_tag>, quant_case<Device, bf16_tag>, quant_case<Device, f16_tag>,
                                         quant_case<Device, e4m3_tag>, quant_case<Device, e5m2_tag>>;

template <typename Case>
class QuantProduct : public ::testing::Test {
protected:
    using Tag = typename Case::tag;
    static auto& dev() { return Case::device::get(); }
    // Columns: `blocks` of the format's unit, or `blocks` plus an odd remainder for the dense formats
    static std::size_t cols(std::size_t blocks) { return Tag::unit == 1 ? blocks * 32 + 5 : blocks * Tag::unit; }
    void TearDown() override { dev().set_matvec_max(8); }
};
TYPED_TEST_SUITE_P(QuantProduct);

namespace quant_suite {
    inline void expect_within(std::span<const float> got, std::span<const float> want, const std::vector<double>& bound, const char* what)
    {
        ASSERT_EQ(got.size(), want.size());
        std::size_t bad = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const double d = std::fabs(double(got[i]) - double(want[i])), b = bound[i] + std::ldexp(std::fabs(double(want[i])), -22);
            if (!(d <= b) && bad++ < 3)
                ADD_FAILURE() << what << ": output " << i << " is " << got[i] << ", the reference " << want[i] << " (bound " << b << ")";
        }
        EXPECT_EQ(bad, 0u) << what;
    }
} // namespace quant_suite

TYPED_TEST_P(QuantProduct, TheTiledForwardGivesTheReferenceBits)
{
    using Tag = typename TestFixture::Tag;
    auto& d = TestFixture::dev();
    d.set_matvec_max(0);
    for (const std::size_t n : {std::size_t(9), std::size_t(70)})
        for (const bool with_bias : {false, true}) {
            const std::size_t rows = 67, cols = TestFixture::cols(3);
            const host_matrix<Tag> m(rows, cols, 11 + n, 0.75f);
            const auto xs = uniform(n * cols, 12 + n);
            const auto bias = with_bias ? uniform(rows, 13) : std::vector<float>{};
            std::vector<float> ys(n * rows, -1.0f);
            d.template forward<Tag>(m, xs, ys, bias);
            const auto want = reference_forward(m.dense, rows, cols, xs, bias);
            EXPECT_TRUE(same_bits(ys, want)) << Tag::name << " n " << n << " bias " << with_bias << ": first difference at "
                                             << first_difference(ys, want);
        }
}

TYPED_TEST_P(QuantProduct, TheBackwardToTheInputGivesTheReferenceBits)
{
    using Tag = typename TestFixture::Tag;
    auto& d = TestFixture::dev();
    for (const std::size_t n : {std::size_t(1), std::size_t(5), std::size_t(70)}) {
        const std::size_t rows = 75, cols = TestFixture::cols(5);
        const host_matrix<Tag> m(rows, cols, 21 + n, 0.5f);
        const auto dys = uniform(n * rows, 22 + n);
        auto dxs = uniform(n * cols, 23 + n, 0.25f);   // added into
        auto want = dxs;
        d.template backward<Tag>(m, dys, dxs);
        reference_backward_input(m.dense, rows, cols, dys, want);
        EXPECT_TRUE(same_bits(dxs, want)) << Tag::name << " n " << n << ": first difference at " << first_difference(dxs, want);
    }
}

TYPED_TEST_P(QuantProduct, TheMatrixVectorProductIsWithinTheBoundOfAReorderedSum)
{
    using Tag = typename TestFixture::Tag;
    auto& d = TestFixture::dev();
    for (const std::size_t n : {std::size_t(1), std::size_t(3), std::size_t(8)}) {
        const std::size_t rows = 45, cols = TestFixture::cols(Tag::unit == 64 ? 17 : 33);
        const host_matrix<Tag> m(rows, cols, 31 + n, 1.0f);
        const auto xs = uniform(n * cols, 32 + n);
        const auto bias = uniform(rows, 33);
        std::vector<float> ys(n * rows, -1.0f);
        d.template forward<Tag>(m, xs, ys, bias);
        const auto want = reference_forward(m.dense, rows, cols, xs, bias);
        quant_suite::expect_within(ys, want, reorder_bound(m.dense, rows, cols, xs, double(cols) * 0x1p-23), Tag::name);
    }
}

TYPED_TEST_P(QuantProduct, MoreRowsThanOneLaunchTakesAreAllWritten)
{
    using Tag = typename TestFixture::Tag;
    auto& d = TestFixture::dev();
    d.set_matvec_max(64);
    const std::size_t n = 19, rows = 6, cols = TestFixture::cols(2);   // three launches of at most 8 rows
    const host_matrix<Tag> m(rows, cols, 41, 1.0f);
    const auto xs = uniform(n * cols, 42);
    std::vector<float> ys(n * rows, -1.0f);
    d.template forward<Tag>(m, xs, ys, {});
    const auto want = reference_forward(m.dense, rows, cols, xs, {});
    quant_suite::expect_within(ys, want, reorder_bound(m.dense, rows, cols, xs, double(cols) * 0x1p-23), Tag::name);
}

TYPED_TEST_P(QuantProduct, TheEdgeCasesGiveTheReference)
{
    using Tag = typename TestFixture::Tag;
    auto& d = TestFixture::dev();
    for (const std::size_t mv : {std::size_t(0), std::size_t(8)}) {
        d.set_matvec_max(mv);
        // One output, one block (a dense format: one column)
        {
            const std::size_t cols = Tag::unit;
            const host_matrix<Tag> m(1, cols, 51, 1.0f);
            const auto xs = uniform(3 * cols, 52);
            const std::vector<float> bias{0.5f};
            std::vector<float> ys(3, -1.0f);
            d.template forward<Tag>(m, xs, ys, bias);
            const auto want = reference_forward(m.dense, 1, cols, xs, bias);
            quant_suite::expect_within(ys, want, reorder_bound(m.dense, 1, cols, xs, double(cols) * 0x1p-23), "one output");
            if (mv == 0)
                EXPECT_TRUE(same_bits(ys, want));
        }
        // Zero weights: the bias exactly; zero activations: the bias exactly
        {
            const std::size_t rows = 9, cols = TestFixture::cols(2);
            const host_matrix<Tag> zero(rows, cols, 53, 0.0f), m(rows, cols, 54, 1.0f);
            const auto xs = uniform(2 * cols, 55);
            const std::vector<float> zeros(2 * cols, 0.0f);
            const auto bias = uniform(rows, 56);
            std::vector<float> ys(2 * rows, -1.0f), yz(2 * rows, -1.0f);
            d.template forward<Tag>(zero, xs, ys, bias);
            d.template forward<Tag>(m, zeros, yz, bias);
            for (std::size_t i = 0; i < ys.size(); ++i) {
                EXPECT_EQ(ys[i], bias[i % rows]) << "zero weights, output " << i;
                EXPECT_EQ(yz[i], bias[i % rows]) << "zero activations, output " << i;
            }
        }
    }
    // No rows of activations: nothing is written
    {
        const host_matrix<Tag> m(4, Tag::unit, 57, 1.0f);
        std::vector<float> none;
        d.template forward<Tag>(m, none, none, {});
        d.template backward<Tag>(m, none, none);
    }
}

TYPED_TEST_P(QuantProduct, TwoRunsGiveTheSameBits)
{
    using Tag = typename TestFixture::Tag;
    auto& d = TestFixture::dev();
    const std::size_t rows = 130, cols = TestFixture::cols(9);
    const host_matrix<Tag> m(rows, cols, 61, 1.0f);
    for (const std::size_t n : {std::size_t(2), std::size_t(33)}) {
        const auto xs = uniform(n * cols, 62 + n);
        std::vector<float> a(n * rows), b(n * rows);
        d.template forward<Tag>(m, xs, a, {});
        d.template forward<Tag>(m, xs, b, {});
        EXPECT_TRUE(same_bits(a, b)) << "forward, n " << n;
        const auto dys = uniform(n * rows, 63 + n);
        std::vector<float> ga(n * cols, 0.0f), gb(n * cols, 0.0f);
        d.template backward<Tag>(m, dys, ga);
        d.template backward<Tag>(m, dys, gb);
        EXPECT_TRUE(same_bits(ga, gb)) << "backward, n " << n;
    }
}

TYPED_TEST_P(QuantProduct, Q8ActivationsGiveCoresDotProduct)
{
    using Tag = typename TestFixture::Tag;
    if constexpr (!Tag::q8) {
        GTEST_SKIP() << Tag::name << " has no integer dot product in core";
    } else {
        auto& d = TestFixture::dev();
        for (const std::size_t n : {std::size_t(1), std::size_t(5), std::size_t(70)})
            for (const bool with_bias : {false, true}) {
                const std::size_t rows = 37, cols = 32 * 70;   // more blocks than a work-group adds at once
                const host_matrix<Tag> m(rows, cols, 71 + n, 1.0f);
                const auto xs = uniform(n * cols, 72 + n, 3.0f);
                const auto bias = with_bias ? uniform(rows, 73) : std::vector<float>{};
                std::vector<float> ys(n * rows, -1.0f);
                d.template forward_q8<Tag>(m, xs, ys, bias);
                const auto want = reference_q8(m.view(), xs, bias);
                if (d.exact_q8())
                    EXPECT_TRUE(same_bits(ys, want)) << Tag::name << " n " << n << ": first difference at " << first_difference(ys, want);
                else
                    quant_suite::expect_within(ys, want, reorder_bound(m.dense, rows, cols, xs, 0x1p-8), Tag::name);
            }
    }
}

TYPED_TEST_P(QuantProduct, AShapeTheFormatDoesNotAllowIsRefused)
{
    using Tag = typename TestFixture::Tag;
    auto& d = TestFixture::dev();
    const host_matrix<Tag> m(4, TestFixture::cols(1), 81, 1.0f);
    std::vector<float> xs(m.cols + 1), ys(4);
    EXPECT_THROW(d.template forward<Tag>(m, xs, ys, {}), std::invalid_argument);   // not a whole number of rows
    std::vector<float> xs2(m.cols), bias(3);
    EXPECT_THROW(d.template forward<Tag>(m, xs2, ys, bias), std::invalid_argument);   // a bias of the wrong size
}

REGISTER_TYPED_TEST_SUITE_P(QuantProduct, TheTiledForwardGivesTheReferenceBits, TheBackwardToTheInputGivesTheReferenceBits,
                            TheMatrixVectorProductIsWithinTheBoundOfAReorderedSum, MoreRowsThanOneLaunchTakesAreAllWritten,
                            TheEdgeCasesGiveTheReference, TwoRunsGiveTheSameBits, Q8ActivationsGiveCoresDotProduct,
                            AShapeTheFormatDoesNotAllowIsRefused);

} // namespace snicholls::gpu::testing

#endif /* snicholls_gpu_tests_support_quant_suite_hpp */
