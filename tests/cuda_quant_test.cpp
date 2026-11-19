//
//  cuda_quant_test.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/gpu/cuda_quant.hpp>: the suite opencl_quant_test.cpp passes (support/quant_suite.hpp: every
//  format, the tiled kernels to the bit under cuda_math::float32, the matrix-vector kernels within the bound of a
//  reordered sum, Q8_0 activations to core's dot product's bits, the edge cases, two runs bit for bit), then the
//  tensor-core path under cuda_math::bf16 within its bound and deterministic, and a QLoRA layer through core's
//  lora_sequence with every span in device memory. They need a CUDA device.
//

#include <snicholls/gpu/cuda_quant.hpp>

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
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace gpu = snicholls::gpu;
namespace ml = snicholls::ml;
namespace la = snicholls::linalg;
using namespace snicholls::gpu::testing;

struct cuda_device;
const gpu::cuda_context& context();

// A host vector copied to device memory
template <typename T>
struct on_device {
    std::unique_ptr<gpu::device_buffer<T>> buffer;
    explicit on_device(std::span<const T> host) : buffer(std::make_unique<gpu::device_buffer<T>>(context(), host.size())) { buffer->upload(host); }
    [[nodiscard]] std::span<T> span() const { return buffer->span(); }
    [[nodiscard]] std::vector<T> back() const
    {
        context().synchronize();
        std::vector<T> out(buffer->size());
        buffer->download(out);
        return out;
    }
};

// Core's views over device copies of a host matrix's storage
template <typename Block>
struct device_weights_blocks {
    on_device<Block> blocks;
    std::size_t rows, cols;
    explicit device_weights_blocks(const ml::quantized_weights<Block>& w) : blocks(w.blocks), rows(w.rows), cols(w.cols) {}
    [[nodiscard]] ml::quantized_weights<Block> view() const { return {blocks.span(), rows, cols}; }
};
struct device_weights_nf4 {
    on_device<std::uint8_t> codes;
    on_device<ml::float8_e4m3> block_scales;
    on_device<float> group_scales;
    float mean;
    std::size_t rows, cols;
    explicit device_weights_nf4(const ml::nf4_weights& w)
        : codes(w.codes), block_scales(w.block_scales), group_scales(w.group_scales), mean(w.scale_mean), rows(w.rows), cols(w.cols)
    {
    }
    [[nodiscard]] ml::nf4_weights view() const { return {codes.span(), block_scales.span(), group_scales.span(), mean, rows, cols}; }
};
template <typename S>
struct device_weights_dense {
    on_device<S> data;
    std::size_t rows, cols;
    explicit device_weights_dense(const ml::dense_weights<S>& w) : data(w.data), rows(w.rows), cols(w.cols) {}
    [[nodiscard]] ml::dense_weights<S> view() const { return {data.span(), rows, cols}; }
};
template <typename Block>
device_weights_blocks<Block> to_device(const ml::quantized_weights<Block>& w) { return device_weights_blocks<Block>(w); }
inline device_weights_nf4 to_device(const ml::nf4_weights& w) { return device_weights_nf4(w); }
template <typename S>
device_weights_dense<S> to_device(const ml::dense_weights<S>& w) { return device_weights_dense<S>(w); }

struct cuda_device {
    std::unique_ptr<gpu::cuda_context> context;
    std::unique_ptr<gpu::cuda_kernels> kernels;
    std::unique_ptr<gpu::cuda_quant_kernels> quant;
    gpu::cuda_math math = gpu::cuda_math::float32;

    static cuda_device& get()
    {
        static cuda_device d = [] {
            auto c = gpu::cuda_context::create();
            if (c.is_err()) {
                std::fprintf(stderr, "cuda_quant_test: no CUDA device\n");
                std::abort();
            }
            cuda_device out{std::move(c).unwrap(), nullptr, nullptr};
            auto k = gpu::cuda_kernels::create(*out.context);
            auto q = gpu::cuda_quant_kernels::create(*out.context);
            if (k.is_err() || q.is_err()) {
                std::fprintf(stderr, "cuda_quant_test: the kernels did not compile\n");
                std::abort();
            }
            out.kernels = std::move(k).unwrap();
            out.quant = std::move(q).unwrap();
            std::fprintf(stderr, "cuda_quant_test: on %s\n", out.context->device_name().c_str());
            return out;
        }();
        return d;
    }

    [[nodiscard]] gpu::cuda_training_backend backend() const
    {
        gpu::cuda_training_backend b{{context.get(), math}, kernels.get()};
        b.quant = quant.get();
        return b;
    }
    [[nodiscard]] bool exact_q8() const { return true; }
    void set_matvec_max(std::size_t n) { quant->matvec_max = n; }

    template <typename Tag>
    void forward(const host_matrix<Tag>& m, std::span<const float> xs, std::span<float> ys, std::span<const float> bias)
    {
        const auto w = to_device(m.view());
        const on_device<float> x(xs), y{std::span<const float>(ys)}, b(bias);
        linear_sequence(backend(), w.view(), std::span<const float>(x.span()), y.span(), std::span<const float>(b.span()));
        const auto out = y.back();
        std::copy(out.begin(), out.end(), ys.begin());
    }
    template <typename Tag>
    void backward(const host_matrix<Tag>& m, std::span<const float> dys, std::span<float> dxs)
    {
        const auto w = to_device(m.view());
        const on_device<float> dy(dys), dx{std::span<const float>(dxs)};
        linear_sequence_backward_input(backend(), w.view(), std::span<const float>(dy.span()), dx.span());
        const auto out = dx.back();
        std::copy(out.begin(), out.end(), dxs.begin());
    }
    template <typename Tag>
    void forward_q8(const host_matrix<Tag>& m, std::span<const float> xs, std::span<float> ys, std::span<const float> bias)
    {
        const auto w = to_device(m.view());
        const on_device<float> x(xs), y{std::span<const float>(ys)}, b(bias);
        linear_sequence_q8(backend(), w.view(), std::span<const float>(x.span()), y.span(), std::span<const float>(b.span()));
        const auto out = y.back();
        std::copy(out.begin(), out.end(), ys.begin());
    }
};

const gpu::cuda_context& context() { return *cuda_device::get().context; }

} // namespace

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda, QuantProduct, quant_cases_for<cuda_device>);

namespace {

static_assert(std::is_trivially_copyable_v<gpu::cuda_training_backend>);
static_assert(!std::is_copy_constructible_v<gpu::cuda_quant_kernels> && !std::is_move_constructible_v<gpu::cuda_quant_kernels>);

// The tensor-core path: within 2^-7 Σ|w x| of the reference, the same every run
template <typename Tag>
void tensor_cores_within_bound()
{
    auto& d = cuda_device::get();
    d.math = gpu::cuda_math::bf16;
    d.set_matvec_max(0);
    const std::size_t rows = 131, cols = Tag::unit == 1 ? 32 * 7 + 3 : 7 * Tag::unit, n = 77;
    const host_matrix<Tag> m(rows, cols, 201, 1.0f);
    const auto xs = uniform(n * cols, 202), bias = uniform(rows, 203), dys = uniform(n * rows, 204);
    std::vector<float> ys(n * rows), again(n * rows);
    d.forward<Tag>(m, xs, ys, bias);
    d.forward<Tag>(m, xs, again, bias);
    EXPECT_TRUE(same_bits(ys, again)) << Tag::name;
    const auto want = reference_forward(m.dense, rows, cols, xs, bias);
    quant_suite::expect_within(ys, want, reorder_bound(m.dense, rows, cols, xs, 0x1p-7), Tag::name);
    auto dxs = uniform(n * cols, 205, 0.25f), dxs_ref = dxs;
    d.backward<Tag>(m, dys, dxs);
    reference_backward_input(m.dense, rows, cols, dys, dxs_ref);
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t j = 0; j < cols; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < rows; ++i)
                s += std::fabs(double(m.dense[i * cols + j]) * double(dys[t * rows + i]));
            EXPECT_NEAR(dxs[t * cols + j], dxs_ref[t * cols + j], 0x1p-7 * s + 1e-6) << Tag::name << " " << t << " " << j;
        }
    d.math = gpu::cuda_math::float32;
    d.set_matvec_max(8);
}

TEST(GpuCudaQuant, TheTensorCoresAreWithinTheirBoundAndDeterministic)
{
    tensor_cores_within_bound<q8_0_tag>();
    tensor_cores_within_bound<q4_0_tag>();
    tensor_cores_within_bound<mxfp4_tag>();
    tensor_cores_within_bound<nf4_tag>();
    tensor_cores_within_bound<bf16_tag>();
    tensor_cores_within_bound<f16_tag>();
    tensor_cores_within_bound<e4m3_tag>();
    tensor_cores_within_bound<e5m2_tag>();
}

TEST(GpuCudaQuant, AQLoraLayerRunsOnTheDeviceThroughCoresLoraSequence)
{
    auto& d = cuda_device::get();
    const std::size_t rows = 96, cols = 128, rank = 4, n = 12;
    const host_matrix<nf4_tag> base(rows, cols, 211, 1.0f);
    const auto a = uniform(rank * cols, 212, 0.1f), bm = uniform(rows * rank, 213, 0.1f);
    const auto xs = uniform(n * cols, 214), bias = uniform(rows, 215), dys = uniform(n * rows, 216);
    const ml::lora_weights<ml::nf4_weights, float> host_w{base.view(), a, bm, rank, ml::lora_scaling(8.0f, rank)};
    std::vector<float> ys_ref(n * rows), us_ref(n * rank);
    ml::lora_sequence(la::reference, host_w, xs, ys_ref, us_ref, bias);

    const auto dev_base = to_device(base.view());
    const on_device<float> da_{std::span<const float>(a)}, db_{std::span<const float>(bm)}, dx{std::span<const float>(xs)},
        dbias{std::span<const float>(bias)}, ddy{std::span<const float>(dys)};
    const std::vector<float> zeros_y(n * rows, 0.0f), zeros_u(n * rank, 0.0f), zeros_x(n * cols, 0.0f), zeros_a(rank * cols, 0.0f),
        zeros_b(rows * rank, 0.0f);
    const on_device<float> dys_out{std::span<const float>(zeros_y)}, dus{std::span<const float>(zeros_u)}, ddx{std::span<const float>(zeros_x)},
        dda{std::span<const float>(zeros_a)}, ddb{std::span<const float>(zeros_b)}, ddus{std::span<const float>(zeros_u)};
    const ml::lora_weights<ml::nf4_weights, float> dev_w{dev_base.view(), da_.span(), db_.span(), rank, host_w.scaling};
    const auto b = d.backend();
    ml::lora_sequence(b, dev_w, std::span<const float>(dx.span()), dys_out.span(), dus.span(), std::span<const float>(dbias.span()));
    const auto ys = dys_out.back();
    const auto bound = reorder_bound(base.dense, rows, cols, xs, double(cols + rank + 2) * 0x1p-23);
    for (std::size_t i = 0; i < ys.size(); ++i)
        EXPECT_NEAR(ys[i], ys_ref[i], bound[i] + 1e-5) << i;

    std::vector<float> dxs_ref(n * cols, 0.0f), da_ref(rank * cols, 0.0f), db_ref(rows * rank, 0.0f), dus_ref(n * rank);
    ml::lora_sequence_backward(la::reference, host_w, xs, us_ref, dys, dxs_ref, da_ref, db_ref, dus_ref);
    ml::lora_sequence_backward(b, dev_w, std::span<const float>(dx.span()), std::span<const float>(dus.span()), std::span<const float>(ddy.span()),
                               ddx.span(), dda.span(), ddb.span(), ddus.span());
    const auto dxs = ddx.back();
    for (std::size_t i = 0; i < dxs.size(); ++i)
        EXPECT_NEAR(dxs[i], dxs_ref[i], 1e-4 * (1 + std::fabs(dxs_ref[i]))) << i;
}

TEST(GpuCudaQuant, TheErrorsAreReported)
{
    auto& d = cuda_device::get();
    const std::vector<ml::block_q8_0> blocks(3);
    const auto dev = to_device(ml::quantized_weights<ml::block_q8_0>{blocks, 1, 33});
    const std::vector<float> host(66, 0.0f);
    const on_device<float> x{std::span<const float>(host)}, y{std::span<const float>(host)};
    EXPECT_THROW(gpu::linear_sequence(d.backend(), dev.view(), std::span<const float>(x.span()).first(33), y.span().first(1)), std::invalid_argument);
    gpu::cuda_training_backend bare = d.backend();
    bare.quant = nullptr;
    const auto ok = to_device(ml::quantized_weights<ml::block_q8_0>{std::span<const ml::block_q8_0>(blocks).first(1), 1, 32});
    EXPECT_THROW(gpu::linear_sequence(bare, ok.view(), std::span<const float>(x.span()).first(32), y.span().first(1)), std::logic_error);
}

} // namespace
