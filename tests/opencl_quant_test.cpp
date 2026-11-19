//
//  opencl_quant_test.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/gpu/opencl_quant.hpp>: every format's products on the device against core's references on the
//  host (support/quant_suite.hpp: the tiled kernels to the bit, the matrix-vector kernels within the bound of a
//  reordered sum, Q8_0 activations to core's dot product's bits, the edge cases, two runs bit for bit), then the
//  OpenCL specifics: weights kept on the device, a QLoRA layer through core's lora_sequence, the errors. They need an
//  OpenCL device: a GPU when there is one (the log names it), else a CPU implementation (PoCL in the Ubuntu image).
//

#include <snicholls/gpu/opencl_quant.hpp>

#include "support/quant_suite.hpp"

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/lora.hpp>
#include <snicholls/core/ml/nf4.hpp>
#include <snicholls/core/ml/quantize.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace gpu = snicholls::gpu;
namespace ml = snicholls::ml;
namespace la = snicholls::linalg;
using namespace snicholls::gpu::testing;

struct opencl_device {
    std::unique_ptr<gpu::opencl_context> context;
    std::unique_ptr<gpu::opencl_quant_kernels> kernels;

    static opencl_device& get()
    {
        static opencl_device d = [] {
            auto c = gpu::opencl_context::create(gpu::device_choice::any);
            if (c.is_err()) {
                std::fprintf(stderr, "opencl_quant_test: no OpenCL device\n");
                std::abort();
            }
            opencl_device out{std::move(c).unwrap(), nullptr};
            auto k = gpu::opencl_quant_kernels::create(*out.context);
            if (k.is_err()) {
                std::fprintf(stderr, "opencl_quant_test: the program did not build:\n%s\n", k.unwrap_err().log.c_str());
                std::abort();
            }
            out.kernels = std::move(k).unwrap();
            std::fprintf(stderr, "opencl_quant_test: on %s (%s), division %s\n", out.context->device_name().c_str(),
                         out.context->is_gpu() ? "a GPU" : "not a GPU", out.kernels->exact_q8() ? "correctly rounded" : "not correctly rounded");
            return out;
        }();
        return d;
    }

    [[nodiscard]] gpu::opencl_quant_backend backend() const { return {{context.get()}, kernels.get()}; }
    [[nodiscard]] bool exact_q8() const { return kernels->exact_q8(); }
    void set_matvec_max(std::size_t n) { kernels->matvec_max = n; }

    template <typename Tag>
    void forward(const host_matrix<Tag>& m, std::span<const float> xs, std::span<float> ys, std::span<const float> bias)
    {
        linear_sequence(backend(), m.view(), xs, ys, bias);
    }
    template <typename Tag>
    void backward(const host_matrix<Tag>& m, std::span<const float> dys, std::span<float> dxs)
    {
        linear_sequence_backward_input(backend(), m.view(), dys, dxs);
    }
    template <typename Tag>
    void forward_q8(const host_matrix<Tag>& m, std::span<const float> xs, std::span<float> ys, std::span<const float> bias)
    {
        linear_sequence_q8(backend(), m.view(), xs, ys, bias);
    }
};

} // namespace

INSTANTIATE_TYPED_TEST_SUITE_P(OpenCl, QuantProduct, quant_cases_for<opencl_device>);

namespace {

static_assert(std::is_trivially_copyable_v<gpu::opencl_quant_backend> && sizeof(gpu::opencl_quant_backend) == 2 * sizeof(void*));
static_assert(la::blas_backend<gpu::opencl_quant_backend, float>);
static_assert(!std::is_copy_constructible_v<gpu::opencl_quant_kernels> && !std::is_move_constructible_v<gpu::opencl_quant_kernels>);
static_assert(std::is_copy_constructible_v<gpu::opencl_quant_matrix> && std::is_default_constructible_v<gpu::opencl_quant_matrix>);

TEST(GpuOpenClQuant, WeightsKeptOnTheDeviceGiveTheBitsOfWeightsCopiedPerCall)
{
    auto& d = opencl_device::get();
    const host_matrix<q4_0_tag> m(50, 32 * 6, 101, 1.0f);
    const gpu::opencl_quant_matrix on_device = d.kernels->upload(m.view());
    EXPECT_TRUE(on_device.has_weights());
    EXPECT_EQ(on_device.rows, 50u);
    EXPECT_EQ(on_device.cols, 192u);
    EXPECT_EQ(on_device.bytes(), m.blocks.size() * sizeof(ml::block_q4_0));
    const auto b = d.backend();
    for (const std::size_t n : {std::size_t(1), std::size_t(40)}) {
        const auto xs = uniform(n * 192, 102 + n);
        std::vector<float> a(n * 50), c(n * 50);
        gpu::linear_sequence(b, on_device, xs, a);
        gpu::linear_sequence(b, m.view(), xs, c);
        EXPECT_TRUE(same_bits(a, c));
        std::vector<float> qa(n * 50), qc(n * 50);
        gpu::linear_sequence_q8(b, on_device, xs, qa);
        gpu::linear_sequence_q8(b, m.view(), xs, qc);
        EXPECT_TRUE(same_bits(qa, qc));
    }
    const gpu::opencl_quant_matrix copy = on_device;   // a handle: the same device memory
    EXPECT_EQ(copy.bytes(), on_device.bytes());
    EXPECT_FALSE(gpu::opencl_quant_matrix{}.has_weights());
    EXPECT_EQ(gpu::opencl_quant_matrix{}.bytes(), 0u);
}

TEST(GpuOpenClQuant, CoresLinearOverQuantisedWeightsIsTheQ8Path)
{
    // ml::linear with ml::prepare_input: core's own one-row product over quantised weights
    auto& d = opencl_device::get();
    constexpr std::size_t cols = 256;
    const host_matrix<mxfp4_tag> m(20, cols, 111, 2.0f);
    const auto x = uniform(cols, 112);
    const auto bias = uniform(20, 113);
    const auto q = ml::prepare_input<cols>(m.view(), std::span<const float, cols>(x.data(), cols));
    std::vector<float> want(20), got(20);
    ml::linear(m.view(), q, std::span<float>(want), std::span<const float>(bias));
    gpu::linear_sequence_q8(d.backend(), m.view(), x, got, bias);
    if (d.exact_q8())
        EXPECT_TRUE(same_bits(got, want));
}

TEST(GpuOpenClQuant, AQLoraLayerRunsThroughCoresLoraSequence)
{
    // core's lora_sequence and lora_sequence_backward over an NF4 base: the base's products on the device by these
    // kernels, the adapters' by opencl_backend's gemm; against the same calls through core's reference
    auto& d = opencl_device::get();
    const std::size_t rows = 96, cols = 128, rank = 4, n = 12;
    const host_matrix<nf4_tag> base(rows, cols, 121, 1.0f);
    const auto a = uniform(rank * cols, 122, 0.1f), bm = uniform(rows * rank, 123, 0.1f);
    const ml::lora_weights<ml::nf4_weights, float> host_w{base.view(), a, bm, rank, ml::lora_scaling(8.0f, rank)};
    const auto xs = uniform(n * cols, 124), bias = uniform(rows, 125), dys = uniform(n * rows, 126);

    std::vector<float> ys_ref(n * rows), us_ref(n * rank), ys(n * rows), us(n * rank);
    ml::lora_sequence(la::reference, host_w, xs, ys_ref, us_ref, bias);
    ml::lora_sequence(d.backend(), host_w, xs, ys, us, bias);
    // The base's sum in another order (core's NF4 product starts from the bias): within the reordered sum's bound
    const auto bound = reorder_bound(base.dense, rows, cols, xs, double(cols + rank + 2) * 0x1p-23);
    for (std::size_t i = 0; i < ys.size(); ++i)
        EXPECT_NEAR(ys[i], ys_ref[i], bound[i] + 1e-6) << i;
    EXPECT_TRUE(same_bits(us, us_ref));   // U = X Aᵀ: opencl_backend's gemm gives the reference's bits

    // The same with the base kept on the device
    const ml::lora_weights<gpu::opencl_quant_matrix, float> dev_w{d.kernels->upload(base.view()), a, bm, rank, host_w.scaling};
    std::vector<float> ys2(n * rows), us2(n * rank);
    ml::lora_sequence(d.backend(), dev_w, xs, ys2, us2, bias);
    EXPECT_TRUE(same_bits(ys2, ys));

    std::vector<float> dxs_ref(n * cols, 0.0f), da_ref(rank * cols, 0.0f), db_ref(rows * rank, 0.0f), dus_ref(n * rank);
    std::vector<float> dxs(n * cols, 0.0f), da(rank * cols, 0.0f), db(rows * rank, 0.0f), dus(n * rank);
    ml::lora_sequence_backward(la::reference, host_w, xs, us_ref, dys, dxs_ref, da_ref, db_ref, dus_ref);
    ml::lora_sequence_backward(d.backend(), dev_w, xs, us_ref, dys, dxs, da, db, dus);
    EXPECT_TRUE(same_bits(da, da_ref));
    EXPECT_TRUE(same_bits(db, db_ref));
    EXPECT_TRUE(same_bits(dus, dus_ref));
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t j = 0; j < cols; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < rows; ++i)
                s += std::fabs(double(base.dense[i * cols + j]) * double(dys[t * rows + i]));
            EXPECT_NEAR(dxs[t * cols + j], dxs_ref[t * cols + j], double(rows + rank + 2) * 0x1p-23 * s + 1e-6) << t << " " << j;
        }
}

TEST(GpuOpenClQuant, TheErrorsAreReported)
{
    auto& d = opencl_device::get();
    // Columns that are not a whole number of blocks
    const std::vector<ml::block_q8_0> blocks(3);
    EXPECT_THROW((void)d.kernels->upload(ml::quantized_weights<ml::block_q8_0>{blocks, 1, 33}), std::invalid_argument);
    EXPECT_THROW((void)d.kernels->upload(ml::quantized_weights<ml::block_q8_0>{blocks, 2, 32}), std::invalid_argument);
    // Q8_0 activations over a format without core's integer dot product
    const host_matrix<nf4_tag> nf4(2, 64, 131, 1.0f);
    const auto m = d.kernels->upload(nf4.view());
    std::vector<float> x(64), y(2);
    EXPECT_THROW(gpu::linear_sequence_q8(d.backend(), m, x, y), std::invalid_argument);
    // A matrix without weights, and a backend without kernels
    EXPECT_THROW(gpu::linear_sequence(d.backend(), gpu::opencl_quant_matrix{}, x, y), std::invalid_argument);
    const gpu::opencl_quant_backend bare{{d.context.get()}, nullptr};
    EXPECT_THROW(gpu::linear_sequence(bare, nf4.view(), x, y), std::logic_error);
}

} // namespace
