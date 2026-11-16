//
//  cuda_ops_test.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/gpu/cuda_ops.hpp>: each training step on the device against core's own, and whole training
//  steps (forward and backward) of GPT-2-, Llama- and nanochat-shaped blocks with every buffer in device memory,
//  against core's reference on the host. They need a CUDA device.
//

#include <snicholls/gpu/cuda_ops.hpp>

#include "support/block_step.hpp"

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/ml/transformer_block.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace gpu = snicholls::gpu;
namespace ml = snicholls::ml;
namespace la = snicholls::linalg;

struct device_t {
    std::unique_ptr<gpu::cuda_context> context;
    std::unique_ptr<gpu::cuda_kernels> kernels;
};

const device_t& device()
{
    static const device_t d = [] {
        auto c = gpu::cuda_context::create();
        if (c.is_err()) {
            std::fprintf(stderr, "cuda_ops_test: no CUDA device\n");
            std::abort();
        }
        device_t out{std::move(c).unwrap(), nullptr};
        auto k = gpu::cuda_kernels::create(*out.context);
        if (k.is_err()) {
            std::fprintf(stderr, "cuda_ops_test: the kernels did not compile\n");
            std::abort();
        }
        out.kernels = std::move(k).unwrap();
        return out;
    }();
    return d;
}

gpu::cuda_training_backend backend() { return {{device().context.get()}, device().kernels.get()}; }

using snicholls::gpu::testing::stream;
using snicholls::gpu::testing::step;

// A mirror on the shared device
struct mirror : snicholls::gpu::testing::mirror {
    explicit mirror(std::vector<float> h) : snicholls::gpu::testing::mirror(*device().context, std::move(h)) {}
};

void near(const std::vector<float>& got, const std::vector<float>& want, float tol, const std::string& what)
{
    ASSERT_EQ(got.size(), want.size()) << what;
    for (std::size_t i = 0; i < got.size(); ++i)
        ASSERT_NEAR(got[i], want[i], tol * (1 + std::fabs(want[i]))) << what << " at " << i;
}

constexpr ml::transformer_config small{.d_model = 64, .n_heads = 4, .n_kv_heads = 4, .head_dim = 16, .d_ff = 128,
                                       .norm = ml::norm_kind::layer, .norm_eps = 1e-5, .mlp = ml::mlp_kind::gelu_tanh, .bias = true};

TEST(GpuCudaOps, TheElementwiseStepsAreCoresToTheBit)
{
    const auto b = backend();
    mirror x(stream(1000, 1)), a(stream(1000, 2)), bias(stream(40, 3)), into(stream(40, 4));
    auto hx = x.host, hinto = into.host;
    gpu::add_rows(b, std::span<const float>(a.d()), x.d());
    ml::add_rows(la::reference, std::span<const float>(a.host), std::span<float>(hx));
    gpu::add_bias_rows(b, x.d(), std::span<const float>(bias.d()));
    ml::add_bias_rows(la::reference, std::span<float>(hx), std::span<const float>(bias.host));
    gpu::sum_rows(b, std::span<const float>(x.d()), into.d());
    ml::sum_rows(la::reference, std::span<const float>(hx), std::span<float>(hinto));
    cudaDeviceSynchronize();
    EXPECT_EQ(x.back(), hx);           // additions in the same order: the same bits
    EXPECT_EQ(into.back(), hinto);
    gpu::fill_rows(b, x.d(), 2.5f);
    gpu::copy_rows(b, std::span<const float>(x.d()), a.d());
    cudaDeviceSynchronize();
    EXPECT_EQ(a.back(), std::vector<float>(1000, 2.5f));
}

TEST(GpuCudaOps, TheNormsAndTheirBackwardAreCoresToRounding)
{
    constexpr std::size_t n = 20, d = small.d_model;
    const auto b = backend();
    mirror x(stream(n * d, 5, 2.0f)), w(stream(d, 6, 0.3f, 1.0f)), bias(stream(d, 7, 0.1f)), out(std::vector<float>(n * d)), dh(stream(n * d, 8)),
        dxs(stream(n * d, 9)), dw(stream(d, 10)), db(stream(d, 11));
    gpu::block_norm_rows<small>(b, std::span<const float>(x.d()), std::span<const float>(w.d()), std::span<const float>(bias.d()), out.d());
    auto hout = out.host, hdxs = dxs.host, hdw = dw.host, hdb = db.host;
    ml::block_norm_rows<small>(la::reference, std::span<const float>(x.host), std::span<const float>(w.host), std::span<const float>(bias.host),
                               std::span<float>(hout));
    gpu::block_norm_rows_backward_add<small>(b, std::span<const float>(dh.d()), std::span<const float>(x.d()), std::span<const float>(w.d()),
                                             std::span<const float>(bias.d()), dxs.d(), dw.d(), db.d());
    ml::block_norm_rows_backward_add<small>(la::reference, std::span<const float>(dh.host), std::span<const float>(x.host),
                                            std::span<const float>(w.host), std::span<const float>(bias.host), std::span<float>(hdxs),
                                            std::span<float>(hdw), std::span<float>(hdb));
    near(out.back(), hout, 1e-6f, "norm");
    near(dxs.back(), hdxs, 1e-5f, "dx");
    near(dw.back(), hdw, 1e-5f, "dweight");
    near(db.back(), hdb, 1e-6f, "dbias");
}

template <ml::transformer_config C>
void whole_step_agrees()
{
    static const ml::rope_table<float, 32, C.head_dim> table(ml::rope_base_v<C, float>);
    auto run = [&](auto& st, const auto& backend, bool dev) {
        const auto w = st.weights(dev);
        if constexpr (C.rope == ml::rope_kind::none) {
            ml::block_forward_sequence<C>(backend, w, st.s(st.xs, dev), ml::no_rope, st.act(dev));
            ml::block_backward<C>(backend, w, ml::no_rope, st.act(dev), st.s(st.dxs, dev), st.grads(dev));
        } else {
            ml::block_forward_sequence<C>(backend, w, st.s(st.xs, dev), table, st.act(dev));
            ml::block_backward<C>(backend, w, table, st.act(dev), st.s(st.dxs, dev), st.grads(dev));
        }
    };
    step<C> st(*device().context);
    run(st, backend(), true);    // on the device, every buffer in device memory
    run(st, la::reference, false);   // the reference, on the host copies
    cudaDeviceSynchronize();
    using S = step<C>;
    near(st.m[S::xs].back(), st.m[S::xs].host, 1e-4f, "output");
    near(st.m[S::dxs].back(), st.m[S::dxs].host, 1e-4f, "input gradient");
    for (int i = S::gn1w; i <= S::gbdown; ++i)
        near(st.m[std::size_t(i)].back(), st.m[std::size_t(i)].host, 1e-3f, "parameter gradient " + std::to_string(i - S::gn1w));
}

constexpr ml::transformer_config llama_like{.d_model = 64, .n_heads = 4, .n_kv_heads = 2, .head_dim = 16, .d_ff = 96, .norm = ml::norm_kind::rms,
                                            .norm_eps = 1e-5, .mlp = ml::mlp_kind::swiglu, .rope = ml::rope_kind::half, .rope_base = 10000};
constexpr ml::transformer_config nanochat_like{.d_model = 64, .n_heads = 4, .n_kv_heads = 4, .head_dim = 16, .d_ff = 128,
                                               .norm = ml::norm_kind::rms, .norm_eps = 1.1920928955078125e-07,
                                               .qk_norm = ml::qk_norm_kind::after_rope, .mlp = ml::mlp_kind::relu_squared,
                                               .rope = ml::rope_kind::half, .rope_base = 10000, .window = 8};

TEST(GpuCudaOps, AGpt2BlocksTrainingStepOnTheDeviceIsTheReferences)
{
    whole_step_agrees<small>();
}

TEST(GpuCudaOps, ALlamaBlocksTrainingStepOnTheDeviceIsTheReferences)
{
    whole_step_agrees<llama_like>();
}

TEST(GpuCudaOps, ANanochatBlocksTrainingStepOnTheDeviceIsTheReferences)
{
    whole_step_agrees<nanochat_like>();
}

} // namespace
