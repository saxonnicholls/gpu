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
#include <snicholls/core/ml/embedding.hpp>
#include <snicholls/core/ml/lm_head.hpp>
#include <snicholls/core/ml/optim.hpp>
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

// Normwise: the largest difference against the largest magnitude, for steps large enough that elementwise rounding
// varies (a different summation order is not a different answer)
void near_normwise(const std::vector<float>& got, const std::vector<float>& want, double tol, const std::string& what)
{
    ASSERT_EQ(got.size(), want.size()) << what;
    double diff = 0, scale = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
        diff = std::max(diff, double(std::fabs(got[i] - want[i]))), scale = std::max(scale, double(std::fabs(want[i])));
    std::printf("  %-22s normwise %.2e\n", what.c_str(), scale == 0 ? 0.0 : diff / scale);
    EXPECT_LE(diff, tol * scale) << what;
}

constexpr ml::transformer_config small{.d_model = 64, .n_heads = 4, .n_kv_heads = 4, .head_dim = 16, .d_ff = 128,
                                       .norm = ml::norm_kind::layer, .norm_eps = 1e-5, .mlp = ml::mlp_kind::gelu_tanh, .bias = true};

TEST(GpuCudaOps, TheElementwiseStepsAreCoresToTheBitAndTheSumsToRounding)
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
    near(into.back(), hinto, 1e-6f, "sum_rows");   // the rows summed lane by lane, then a tree: rounding
    gpu::fill_rows(b, x.d(), 2.5f);
    gpu::copy_rows(b, std::span<const float>(x.d()), a.d());
    cudaDeviceSynchronize();
    EXPECT_EQ(a.back(), std::vector<float>(1000, 2.5f));
}

TEST(GpuCudaOps, TheNormsAndTheirBackwardAreCoresToRounding)
{
    constexpr std::size_t n = 20, d = small.d_model;
    const auto b = backend();
    mirror x(stream(n * d, 5, 2.0f)), w(stream(d, 6, 0.3f, 1.0f)), bias(stream(d, 7, 0.1f)), out{std::vector<float>(n * d)}, dh(stream(n * d, 8)),
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

template <ml::transformer_config C, std::size_t N = 24>
void whole_step_agrees(std::size_t gemm_attention_max = 4096, gpu::cuda_math math = gpu::cuda_math::float32)
{
    static const ml::rope_table<float, (N > 32 ? N : 32), C.head_dim> table(ml::rope_base_v<C, float>);
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
    step<C, N> st(*device().context);
    auto b = backend();
    b.gemm_attention_max = gemm_attention_max;   // 0: the tiled kernels
    b.math = math;                               // TF32: the products and flash attention on the tensor cores
    run(st, b, true);    // on the device, every buffer in device memory
    run(st, la::reference, false);   // the reference, on the host copies
    cudaDeviceSynchronize();
    using S = step<C, N>;
    if (math != gpu::cuda_math::float32) {   // TF32 rounds the products' inputs to 10 bits, bfloat16 to 7: normwise
        const double f = math == gpu::cuda_math::bf16 ? 8.0 : 1.0;
        near_normwise(st.m[S::xs].back(), st.m[S::xs].host, 2e-3 * f, "output (reduced precision)");
        near_normwise(st.m[S::dxs].back(), st.m[S::dxs].host, 3e-3 * f, "input gradient (reduced precision)");
        for (int i = S::gn1w; i <= S::gbdown; ++i)
            if (i != S::gbk || C.attention_softcap > 0)
                near_normwise(st.m[std::size_t(i)].back(), st.m[std::size_t(i)].host, 5e-3 * f,
                              "parameter gradient (reduced precision) " + std::to_string(i - S::gn1w));
        return;
    }
    if constexpr (N <= 24) {
        near(st.m[S::xs].back(), st.m[S::xs].host, 1e-4f, "output");
        near(st.m[S::dxs].back(), st.m[S::dxs].host, 1e-4f, "input gradient");
        for (int i = S::gn1w; i <= S::gbdown; ++i)
            near(st.m[std::size_t(i)].back(), st.m[std::size_t(i)].host, 1e-3f, "parameter gradient " + std::to_string(i - S::gn1w));
    } else {
        near_normwise(st.m[S::xs].back(), st.m[S::xs].host, 1e-5, "output");
        near_normwise(st.m[S::dxs].back(), st.m[S::dxs].host, 1e-5, "input gradient");
        for (int i = S::gn1w; i <= S::gbdown; ++i)
            if (i != S::gbk || C.attention_softcap > 0)
                near_normwise(st.m[std::size_t(i)].back(), st.m[std::size_t(i)].host, 1e-4, "parameter gradient " + std::to_string(i - S::gn1w));
        // Without a soft cap the key bias's gradient is zero in exact arithmetic (a softmax does not see a shift common to
        // every key), so both sides are rounding: each negligible beside the query bias's
        if constexpr (C.attention_softcap > 0)
            return;
        const auto largest = [](const std::vector<float>& v) {
            float m = 0;
            for (const float e : v)
                m = std::max(m, std::fabs(e));
            return m;
        };
        const float q_bias = largest(st.m[S::gbq].host);
        EXPECT_LE(largest(st.m[S::gbk].back()), 1e-4f * q_bias);
        EXPECT_LE(largest(st.m[S::gbk].host), 1e-4f * q_bias);
    }
}

constexpr ml::transformer_config llama_like{.d_model = 64, .n_heads = 4, .n_kv_heads = 2, .head_dim = 16, .d_ff = 96, .norm = ml::norm_kind::rms,
                                            .norm_eps = 1e-5, .mlp = ml::mlp_kind::swiglu, .rope = ml::rope_kind::half, .rope_base = 10000};
constexpr ml::transformer_config nanochat_like{.d_model = 64, .n_heads = 4, .n_kv_heads = 4, .head_dim = 16, .d_ff = 128,
                                               .norm = ml::norm_kind::rms, .norm_eps = 1.1920928955078125e-07,
                                               .qk_norm = ml::qk_norm_kind::after_rope, .mlp = ml::mlp_kind::relu_squared,
                                               .rope = ml::rope_kind::half, .rope_base = 10000, .window = 8};

// The model's other steps on the device against core's own
TEST(GpuCudaOps, EmbeddingsTheClassifierAdamwAndTheReductionsAreCores)
{
    const auto b = backend();
    constexpr std::size_t vocab = 50, d = 24, n = 16;
    std::vector<std::uint32_t> ids(n);
    for (std::size_t t = 0; t < n; ++t)
        ids[t] = std::uint32_t((t * 7) % 11);   // repeats, so a table row gathers several tokens' gradients
    gpu::device_buffer<std::uint32_t> dids(*device().context, n);
    dids.upload(ids);
    const std::span<const std::uint32_t> ids_dev(dids.span().data(), n);
    mirror table(stream(vocab * d, 1)), pos(stream(n * d, 2)), out{std::vector<float>(n * d)}, dout(stream(n * d, 3)), dtable{std::vector<float>(vocab * d)},
        dpos{std::vector<float>(n * d)};
    gpu::embed_rows(b, std::span<const float>(table.d()), std::span<const float>(pos.d()), ids_dev, out.d());
    gpu::embed_rows_backward(b, std::span<const float>(dout.d()), ids_dev, dtable.d(), dpos.d());
    auto hout = out.host, hdtable = dtable.host, hdpos = dpos.host;
    ml::embed_rows(la::reference, std::span<const float>(table.host), std::span<const float>(pos.host), ids, std::span<float>(hout));
    ml::embed_rows_backward(la::reference, std::span<const float>(dout.host), ids, std::span<float>(hdtable), std::span<float>(hdpos));
    cudaDeviceSynchronize();
    EXPECT_EQ(out.back(), hout);         // additions in the same order: the same bits
    EXPECT_EQ(dtable.back(), hdtable);
    EXPECT_EQ(dpos.back(), hdpos);

    std::vector<std::uint32_t> targets(n);
    for (std::size_t t = 0; t < n; ++t)
        targets[t] = std::uint32_t((t * 13) % vocab);
    gpu::device_buffer<std::uint32_t> dtargets(*device().context, n);
    dtargets.upload(targets);
    mirror logits(stream(n * vocab, 4, 3.0f)), losses{std::vector<float>(n)};
    gpu::cross_entropy_rows(b, logits.d(), std::span<const std::uint32_t>(dtargets.span().data(), n), losses.d(), 0.25f);
    auto hlogits = logits.host, hlosses = losses.host;
    ml::cross_entropy_rows(la::reference, std::span<float>(hlogits), targets, std::span<float>(hlosses), 0.25f);
    near(losses.back(), hlosses, 1e-6f, "losses");
    near(logits.back(), hlogits, 1e-6f, "gradient");

    mirror p(stream(1000, 5)), g(stream(1000, 6)), m{std::vector<float>(1000)}, v{std::vector<float>(1000)};
    auto hp = p.host, hm = m.host, hv = v.host;
    const ml::adamw_config<float> c{.learning_rate = 0.01f, .beta2 = 0.95f, .weight_decay = 0.1f};
    for (std::uint64_t step = 1; step <= 3; ++step) {
        gpu::adamw_rows(b, p.d(), std::span<const float>(g.d()), m.d(), v.d(), step, c);
        ml::adamw_rows(la::reference, std::span<float>(hp), std::span<const float>(g.host), std::span<float>(hm), std::span<float>(hv), step, c);
    }
    near(p.back(), hp, 1e-6f, "adamw parameters");
    near(m.back(), hm, 1e-6f, "adamw first moments");

    mirror big(stream(100000, 7));
    // Against the exact sums (in double): the device's tree is within a few ulps; a sequential float sum of 10^5 values
    // (the host's) can be off by n eps |sum|, about 0.2 here, so it is not the yardstick
    double exact_squares = 0, exact_sum = 0;
    for (const float x : big.host)
        exact_squares += double(x) * double(x), exact_sum += double(x);
    EXPECT_NEAR(gpu::sum_squares(b, std::span<const float>(big.d())), exact_squares, 1e-5 * exact_squares);
    EXPECT_NEAR(gpu::sum_values(b, std::span<const float>(big.d())), exact_sum, 1e-2);
    const float again = gpu::sum_squares(b, std::span<const float>(big.d()));
    EXPECT_EQ(again, gpu::sum_squares(b, std::span<const float>(big.d())));   // a fixed order: the same bits every time
    gpu::scale_rows(b, big.d(), 0.5f);
    auto hbig = big.host;
    ml::scale_rows(la::reference, std::span<float>(hbig), 0.5f);
    cudaDeviceSynchronize();
    EXPECT_EQ(big.back(), hbig);
}

// The projections head by head as one product and a permutation, against core's per-head loop
TEST(GpuCudaOps, LinearHeadsAreTheReferencesPerHeadProducts)
{
    const auto b = backend();
    constexpr std::size_t heads = 4, d = 16, rows = heads * d, cols = 48, n = 37;
    mirror w(stream(rows * cols, 41, 0.3f)), xs(stream(n * cols, 42)), bias(stream(rows, 43, 0.1f)), out{std::vector<float>(n * rows)},
        douts(stream(n * rows, 44)), dxs(stream(n * cols, 45)), dw(stream(rows * cols, 46)), db(stream(rows, 47));
    const ml::dense_weights<float> dev{w.d(), rows, cols}, host{w.host, rows, cols};
    gpu::linear_heads(b, dev, std::span<const float>(xs.d()), out.d(), std::span<const float>(bias.d()), heads);
    gpu::linear_heads_backward(b, dev, std::span<const float>(xs.d()), std::span<const float>(douts.d()), dxs.d(), dw.d(), db.d(), heads);
    auto hout = out.host, hdxs = dxs.host, hdw = dw.host, hdb = db.host;
    ml::linear_heads(la::reference, host, std::span<const float>(xs.host), std::span<float>(hout), std::span<const float>(bias.host), heads);
    ml::linear_heads_backward(la::reference, host, std::span<const float>(xs.host), std::span<const float>(douts.host), std::span<float>(hdxs),
                              std::span<float>(hdw), std::span<float>(hdb), heads);
    near(out.back(), hout, 1e-5f, "outputs");
    near(dxs.back(), hdxs, 1e-5f, "input gradient");
    near(dw.back(), hdw, 1e-5f, "weight gradient");
    near(db.back(), hdb, 1e-5f, "bias gradient");
}

// The whole LM-head loss on the device: core's batched form through the CUDA backend's products and kernels
TEST(GpuCudaOps, TheSequencesLossThroughTheHeadIsTheReferences)
{
    const auto b = backend();
    constexpr std::size_t vocab = 512, d = 64, n = 32;
    mirror head(stream(vocab * d, 8, 0.2f)), hs(stream(n * d, 9)), logits{std::vector<float>(n * vocab)}, losses{std::vector<float>(n)},
        dhs{std::vector<float>(n * d)}, dhead{std::vector<float>(vocab * d)};
    std::vector<std::uint32_t> targets(n);
    for (std::size_t t = 0; t < n; ++t)
        targets[t] = std::uint32_t((t * 37) % vocab);
    gpu::device_buffer<std::uint32_t> dtargets(*device().context, n);
    dtargets.upload(targets);
    const float loss = ml::lm_head_loss_sequence(b, ml::dense_weights<float>{head.d(), vocab, d}, std::span<const float>(hs.d()),
                                                 std::span<const std::uint32_t>(dtargets.span().data(), n), logits.d(), losses.d(), dhs.d(), dhead.d());
    auto hlogits = logits.host, hlosses = losses.host, hdhs = dhs.host, hdhead = dhead.host;
    const float want = ml::lm_head_loss_sequence(la::reference, ml::dense_weights<float>{head.host, vocab, d}, std::span<const float>(hs.host), targets,
                                                 std::span<float>(hlogits), std::span<float>(hlosses), std::span<float>(hdhs), std::span<float>(hdhead));
    EXPECT_NEAR(loss, want, 1e-5f);
    near(dhs.back(), hdhs, 1e-4f, "dhs");
    near(dhead.back(), hdhead, 1e-4f, "dhead");
}

// Multi-query attention, QK norm before an interleaved RoPE, a soft cap, the exact GELU, layer norm with biases
constexpr ml::transformer_config mqa_capped{.d_model = 64, .n_heads = 4, .n_kv_heads = 1, .head_dim = 16, .d_ff = 128,
                                            .norm = ml::norm_kind::layer, .norm_eps = 1e-5, .qk_norm = ml::qk_norm_kind::before_rope,
                                            .mlp = ml::mlp_kind::gelu, .rope = ml::rope_kind::interleaved, .rope_base = 500, .window = 0,
                                            .attention_softcap = 5, .bias = true};

TEST(GpuCudaOps, AMultiQueryCappedBlocksTrainingStepOnTheDeviceIsTheReferences)
{
    whole_step_agrees<mqa_capped>();
}

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

// The tiled attention kernels over several tiles and a partial one: GPT-2's heads of 64, grouped heads of 128 under a
// window, and the capped multi-query heads of 16
constexpr ml::transformer_config gpt2_heads{.d_model = 128, .n_heads = 2, .n_kv_heads = 2, .head_dim = 64, .d_ff = 256,
                                            .norm = ml::norm_kind::layer, .norm_eps = 1e-5, .mlp = ml::mlp_kind::gelu_tanh, .bias = true};
constexpr ml::transformer_config wide_grouped{.d_model = 256, .n_heads = 4, .n_kv_heads = 2, .head_dim = 128, .d_ff = 128, .norm = ml::norm_kind::rms,
                                              .norm_eps = 1e-5, .mlp = ml::mlp_kind::swiglu, .rope = ml::rope_kind::half, .rope_base = 10000,
                                              .window = 40};
TEST(GpuCudaOps, AttentionAsMatrixProductsOverManyTokensIsTheReferences)
{
    whole_step_agrees<gpt2_heads, 150>();
    whole_step_agrees<wide_grouped, 100>();
    whole_step_agrees<mqa_capped, 130>();
}

TEST(GpuCudaOps, TiledAttentionOverManyTokensIsTheReferences)
{
    whole_step_agrees<gpt2_heads, 150>(0);
    whole_step_agrees<wide_grouped, 100>(0);
    whole_step_agrees<mqa_capped, 130>(0);
}

// Flash attention on the tensor cores (TF32): heads of 64 (GPT-2's) over several tiles and a partial one, and heads of
// 32 grouped, rotated, windowed and capped
constexpr ml::transformer_config flash_mix{.d_model = 128, .n_heads = 4, .n_kv_heads = 2, .head_dim = 32, .d_ff = 128, .norm = ml::norm_kind::rms,
                                           .norm_eps = 1e-5, .mlp = ml::mlp_kind::swiglu, .rope = ml::rope_kind::half, .rope_base = 10000,
                                           .window = 40, .attention_softcap = 5};
TEST(GpuCudaOps, FlashAttentionOnTheTensorCoresIsTheReferencesToTf32)
{
    whole_step_agrees<gpt2_heads, 150>(4096, gpu::cuda_math::tf32);
    whole_step_agrees<gpt2_heads, 24>(4096, gpu::cuda_math::tf32);
    whole_step_agrees<flash_mix, 130>(4096, gpu::cuda_math::tf32);
}

// bfloat16 products through the training backend: operands rounded by a kernel, so not TF32's bits, and within
// bfloat16's rounding of float, for one product and for whole training steps
TEST(GpuCudaOps, Bf16ProductsAreBfloat16AndWholeStepsWithinItsRounding)
{
    constexpr std::size_t m = 96, n = 80, k = 384;
    mirror a(stream(m * k, 21)), b(stream(k * n, 22)), want{std::vector<float>(m * n)}, got{std::vector<float>(m * n)}, tf{std::vector<float>(m * n)};
    auto be = backend();
    const auto run = [&](gpu::cuda_math math, mirror& out) {
        be.math = math;
        be.gemm(1.0f, la::matrix_view<const float>(a.d().data(), {m, k}, {k, 1}), la::matrix_view<const float>(b.d().data(), {k, n}, {n, 1}), 0.0f,
                la::matrix_view<float>(out.d().data(), {m, n}, {n, 1}));
    };
    run(gpu::cuda_math::float32, want);
    run(gpu::cuda_math::bf16, got);
    run(gpu::cuda_math::tf32, tf);
    const auto w = want.back(), g = got.back();
    double diff = 0, scale = 0;
    for (std::size_t i = 0; i < w.size(); ++i)
        diff = std::max(diff, double(std::fabs(g[i] - w[i]))), scale = std::max(scale, double(std::fabs(w[i])));
    std::printf("  bf16 against float: normwise %.2e\n", diff / scale);
    EXPECT_LE(diff, 2e-2 * scale);
    EXPECT_GT(diff, 1e-3 * scale);   // bfloat16's rounding, coarser than TF32's
    EXPECT_NE(g, tf.back());
    whole_step_agrees<gpt2_heads, 150>(4096, gpu::cuda_math::bf16);
    whole_step_agrees<flash_mix, 130>(4096, gpu::cuda_math::bf16);
}

TEST(GpuCudaOps, BothAttentionPathsGiveTheReferencesShortStep)
{
    whole_step_agrees<small>(0);   // the tiled kernels (heads of 16)
    whole_step_agrees<llama_like>(0);
    whole_step_agrees<nanochat_like>(0);   // a window
}

} // namespace
