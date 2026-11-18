//
//  cuda_ops_test.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/gpu/cuda_ops.hpp>: each training step on the device against core's own, and whole training
//  steps (forward and backward) of GPT-2-, Llama- and nanochat-shaped blocks with every buffer in device memory,
//  against core's reference on the host, and Muon's step on the device in float, TF32 and bfloat16. They need a CUDA
//  device.
//

#include <snicholls/gpu/cuda_ops.hpp>

#include "support/block_step.hpp"

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/activations.hpp>
#include <snicholls/core/ml/embedding.hpp>
#include <snicholls/core/ml/lm_head.hpp>
#include <snicholls/core/ml/mlp.hpp>
#include <snicholls/core/ml/muon.hpp>
#include <snicholls/core/ml/optim.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/ml/softmax.hpp>
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
#include <stdexcept>
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

// S sequences of N tokens in one call each way (core 0.13's batched passes): the device's against the reference's
template <ml::transformer_config C, std::size_t N = 24, std::size_t S = 1>
void whole_step_agrees(std::size_t gemm_attention_max = 4096, gpu::cuda_math math = gpu::cuda_math::float32)
{
    static const ml::rope_table<float, (N > 32 ? N : 32), C.head_dim> table(ml::rope_base_v<C, float>);
    auto run = [&](auto& st, const auto& backend, bool dev) {
        const auto w = st.weights(dev);
        if constexpr (C.rope == ml::rope_kind::none) {
            ml::block_forward_sequence<C>(backend, w, st.s(st.xs, dev), ml::no_rope, st.act(dev), S);
            ml::block_backward<C>(backend, w, ml::no_rope, st.act(dev), st.s(st.dxs, dev), st.grads(dev), S);
        } else {
            ml::block_forward_sequence<C>(backend, w, st.s(st.xs, dev), table, st.act(dev), S);
            ml::block_backward<C>(backend, w, table, st.act(dev), st.s(st.dxs, dev), st.grads(dev), S);
        }
    };
    step<C, N, S> st(*device().context);
    auto b = backend();
    b.gemm_attention_max = gemm_attention_max;   // 0: the tiled kernels
    b.math = math;                               // TF32: the products and flash attention on the tensor cores
    run(st, b, true);    // on the device, every buffer in device memory
    run(st, la::reference, false);   // the reference, on the host copies
    cudaDeviceSynchronize();
    using T = step<C, N, S>;
    if (math != gpu::cuda_math::float32) {   // TF32 rounds the products' inputs to 10 bits, bfloat16 to 7: normwise
        const double f = math == gpu::cuda_math::bf16 ? 8.0 : 1.0;
        near_normwise(st.m[T::xs].back(), st.m[T::xs].host, 2e-3 * f, "output (reduced precision)");
        near_normwise(st.m[T::dxs].back(), st.m[T::dxs].host, 3e-3 * f, "input gradient (reduced precision)");
        for (int i = T::gn1w; i <= T::gbdown; ++i)
            if (i != T::gbk || C.attention_softcap > 0)
                near_normwise(st.m[std::size_t(i)].back(), st.m[std::size_t(i)].host, 5e-3 * f,
                              "parameter gradient (reduced precision) " + std::to_string(i - T::gn1w));
        return;
    }
    if constexpr (N <= 24) {
        near(st.m[T::xs].back(), st.m[T::xs].host, 1e-4f, "output");
        near(st.m[T::dxs].back(), st.m[T::dxs].host, 1e-4f, "input gradient");
        for (int i = T::gn1w; i <= T::gbdown; ++i)
            near(st.m[std::size_t(i)].back(), st.m[std::size_t(i)].host, 1e-3f, "parameter gradient " + std::to_string(i - T::gn1w));
    } else {
        near_normwise(st.m[T::xs].back(), st.m[T::xs].host, 1e-5, "output");
        near_normwise(st.m[T::dxs].back(), st.m[T::dxs].host, 1e-5, "input gradient");
        for (int i = T::gn1w; i <= T::gbdown; ++i)
            if (i != T::gbk || C.attention_softcap > 0)
                near_normwise(st.m[std::size_t(i)].back(), st.m[std::size_t(i)].host, 1e-4, "parameter gradient " + std::to_string(i - T::gn1w));
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
        const float q_bias = largest(st.m[T::gbq].host);
        EXPECT_LE(largest(st.m[T::gbk].back()), 1e-4f * q_bias);
        EXPECT_LE(largest(st.m[T::gbk].host), 1e-4f * q_bias);
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

// The elementwise kernels take float4 for aligned spans (a quad a thread, then the tail) and one element a thread
// otherwise: both paths, with tails of every length, are core's bits
TEST(GpuCudaOps, TheElementwiseStepsAreCoresToTheBitAlignedOrNotWithAnyTail)
{
    const auto b = backend();
    for (const std::size_t offset : {0u, 1u, 4u})
        for (const std::size_t n : {0u, 1u, 3u, 4u, 7u, 1000u, 1003u}) {
            const std::string what = "offset " + std::to_string(offset) + ", n " + std::to_string(n);
            mirror x(stream(n + offset, 1)), a(stream(n + offset, 2));
            auto hx = x.host;
            const auto xs = x.d().subspan(offset), as = a.d().subspan(offset);
            const auto hxs = std::span<float>(hx).subspan(offset);
            const auto has = std::span<const float>(a.host).subspan(offset);
            gpu::add_rows(b, std::span<const float>(as), xs);
            ml::add_rows(la::reference, has, hxs);
            gpu::scale_rows(b, xs, 0.375f);
            ml::scale_rows(la::reference, hxs, 0.375f);
            cudaDeviceSynchronize();
            EXPECT_EQ(x.back(), hx) << what;
            gpu::copy_rows(b, std::span<const float>(xs), as);
            gpu::fill_rows(b, xs, -1.5f);
            cudaDeviceSynchronize();
            auto want_a = a.host;
            std::copy(hxs.begin(), hxs.end(), want_a.begin() + std::ptrdiff_t(offset));
            EXPECT_EQ(a.back(), want_a) << what;
            std::fill(hxs.begin(), hxs.end(), -1.5f);
            EXPECT_EQ(x.back(), hx) << what;
        }
    for (const std::size_t r : {4u, 6u, 64u})   // rows of a multiple of 4 take quads, others not
        for (const std::size_t offset : {0u, 2u}) {
            mirror y(stream(9 * r + offset, 3)), bias(stream(r, 4));
            auto hy = y.host;
            gpu::add_bias_rows(b, y.d().subspan(offset), std::span<const float>(bias.d()));
            ml::add_bias_rows(la::reference, std::span<float>(hy).subspan(offset), std::span<const float>(bias.host));
            cudaDeviceSynchronize();
            EXPECT_EQ(y.back(), hy) << "bias rows of " << r << ", offset " << offset;
        }
}

// The column sums of many rows: 32 row groups, each in order, then the groups in order
TEST(GpuCudaOps, SumRowsOverManyRowsAndOddColumnsIsCoresToRounding)
{
    const auto b = backend();
    for (const std::size_t r : {1u, 33u, 70u}) {
        constexpr std::size_t rows = 300;
        mirror ys(stream(rows * r, 5)), into(stream(r, 6));
        auto hinto = into.host;
        gpu::sum_rows(b, std::span<const float>(ys.d()), into.d());
        ml::sum_rows(la::reference, std::span<const float>(ys.host), std::span<float>(hinto));
        near(into.back(), hinto, 1e-5f, "sum_rows of " + std::to_string(r) + " columns");
        const auto first = into.back();
        into.dev->upload(into.host);
        gpu::sum_rows(b, std::span<const float>(ys.d()), into.d());
        EXPECT_EQ(into.back(), first);   // the same order every time
    }
}

// The classifier with float4 rows (a vocabulary padded to a multiple of 4) and without, against core's
TEST(GpuCudaOps, TheClassifierOverPaddedQuadRowsIsCores)
{
    const auto b = backend();
    constexpr std::size_t n = 12, V = 64, classes = 61;
    std::vector<std::uint32_t> targets(n);
    for (std::size_t t = 0; t < n; ++t)
        targets[t] = std::uint32_t((t * 17) % classes);
    targets[0] = classes - 1;   // the last class, in the tail after the quads
    gpu::device_buffer<std::uint32_t> dt(*device().context, n);
    dt.upload(targets);
    mirror logits(stream(n * V, 8, 4.0f)), losses{std::vector<float>(n)};
    auto hlogits = logits.host, hlosses = losses.host;
    gpu::cross_entropy_rows(b, logits.d(), std::span<const std::uint32_t>(dt.span()), losses.d(), 0.5f, classes);
    ml::cross_entropy_rows(la::reference, std::span<float>(hlogits), targets, std::span<float>(hlosses), 0.5f, classes);
    near(losses.back(), hlosses, 1e-6f, "losses");
    near(logits.back(), hlogits, 1e-6f, "gradient");
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t i = classes; i < V; ++i)
            ASSERT_EQ(logits.back()[t * V + i], 0.0f);   // the padding's gradient is 0
}

// Near the cancellations core 0.14.0 removed, against core in double (the truth) and in float: SiLU' for x in 8..17 and
// GELU-tanh' where 1 - sigmoid(2u) cancelled (x in 0.5..8; above 8 it is 1 in float either way) take 1 - s as
// sigmoid(-z) (8.2 and 16.5 ulps before, at most 2.4 on the host and 4 here with expf's own error)
float ulps(float got, double truth) { return float(std::fabs(double(got) - truth) / double(std::nextafter(float(truth), INFINITY) - float(truth))); }

template <ml::mlp_kind K>
void activation_derivative_near_cancellation(float lo, float hi)
{
    const auto b = backend();
    constexpr std::size_t m = 1001;
    constexpr ml::transformer_config C{.d_model = 4, .n_heads = 1, .n_kv_heads = 1, .head_dim = 4, .d_ff = 4, .mlp = K};
    constexpr bool swiglu = K == ml::mlp_kind::swiglu;
    std::vector<float> xs(m);
    for (std::size_t i = 0; i < m; ++i)
        xs[i] = lo + (hi - lo) * float(i) / float(m - 1);
    // dh = 1 and (SwiGLU) up = 1: the derivative itself comes back, in dh (GELU-tanh) or dgate (SwiGLU's SiLU')
    mirror up(swiglu ? std::vector<float>(m, 1.0f) : xs), gate(xs), dh{std::vector<float>(m, 1.0f)}, dgate{std::vector<float>(m)};
    const auto on = [](std::span<float> x) { return swiglu ? x : std::span<float>{}; };   // the gate's spans are SwiGLU's only
    gpu::mlp_activate_rows_backward<C>(b, ml::mlp_activations<float>{up.d(), on(gate.d()), {}}, dh.d(), on(dgate.d()));
    auto hdh = dh.host, hdgate = dgate.host, hup = up.host, hgate = gate.host;
    ml::mlp_activate_rows_backward<C>(la::reference, ml::mlp_activations<float>{std::span<float>(hup), on(std::span<float>(hgate)), {}}, std::span<float>(hdh),
                                      on(std::span<float>(hdgate)));
    const auto got = swiglu ? dgate.back() : dh.back();
    const auto& host = swiglu ? hdgate : hdh;
    near(got, host, 1e-6f, swiglu ? "silu'" : "gelu_tanh'");
    float worst = 0, worst_host = 0;
    for (std::size_t i = 0; i < m; ++i) {
        const double truth = swiglu ? ml::silu_derivative(double(xs[i])) : ml::gelu_tanh_derivative(double(xs[i]));
        worst = std::max(worst, ulps(got[i], truth)), worst_host = std::max(worst_host, ulps(host[i], truth));
    }
    std::printf("  %-12s on [%g, %g]: device %.2f ulps, core in float %.2f\n", swiglu ? "silu'" : "gelu_tanh'", double(lo), double(hi), double(worst),
                double(worst_host));
    EXPECT_LE(worst, 4.0f);
    EXPECT_LE(worst_host, 3.0f);
}

TEST(GpuCudaOps, SiluAndGeluTanhDerivativesKeepTheirDigitsWhereOneMinusSigmoidCancelled)
{
    activation_derivative_near_cancellation<ml::mlp_kind::swiglu>(8.0f, 17.0f);
    activation_derivative_near_cancellation<ml::mlp_kind::gelu_tanh>(0.5f, 8.0f);
}

// Logits near 1e4: the loss (max - x_t) + log(sum) and the gradient e^(x - max) (1 / sum) keep their digits (x - lse,
// rounded at 1e4, lost about 6e-4 before), against core in double and in float
TEST(GpuCudaOps, TheClassifierKeepsItsDigitsWithLogitsNear1e4)
{
    const auto b = backend();
    constexpr std::size_t n = 8, V = 50;
    std::vector<std::uint32_t> targets(n);
    for (std::size_t t = 0; t < n; ++t)
        targets[t] = std::uint32_t((t * 13) % V);
    gpu::device_buffer<std::uint32_t> dt(*device().context, n);
    dt.upload(targets);
    mirror logits(stream(n * V, 21, 3.0f, 1e4f)), losses{std::vector<float>(n)};
    auto hlogits = logits.host, hlosses = losses.host;
    std::vector<double> dlogits(logits.host.begin(), logits.host.end()), dlosses(n);
    gpu::cross_entropy_rows(b, logits.d(), std::span<const std::uint32_t>(dt.span()), losses.d(), 0.25f);
    ml::cross_entropy_rows(la::reference, std::span<float>(hlogits), targets, std::span<float>(hlosses), 0.25f);
    ml::cross_entropy_rows(la::reference, std::span<double>(dlogits), targets, std::span<double>(dlosses), 0.25);
    const auto got_losses = losses.back(), got = logits.back();
    near(got_losses, hlosses, 1e-6f, "losses against core in float");
    near(got, hlogits, 1e-6f, "gradient against core in float");
    near(got_losses, std::vector<float>(dlosses.begin(), dlosses.end()), 1e-6f, "losses against core in double");
    near(got, std::vector<float>(dlogits.begin(), dlogits.end()), 1e-6f, "gradient against core in double");
    for (std::size_t t = 0; t < n; ++t)
        EXPECT_LE(ulps(got_losses[t], dlosses[t]), 8.0f) << t;
}

// AdamW at beta2 = 0.9999 in its first steps, where 1 - pow(beta2, t) cancelled (830 ulps at t = 3): the device's
// factors are core's adam_bias_correction, so its update is core's, and both are the exact one's to rounding
TEST(GpuCudaOps, AdamwKeepsItsBiasCorrectionInTheFirstStepsAtBeta2NearOne)
{
    const auto b = backend();
    constexpr std::size_t n = 256;
    mirror p(stream(n, 31)), g(stream(n, 32)), m{std::vector<float>(n)}, v{std::vector<float>(n)};
    auto hp = p.host, hm = m.host, hv = v.host;
    std::vector<double> rp(p.host.begin(), p.host.end()), rm(n), rv(n);
    const ml::adamw_config<float> c{.learning_rate = 1.0f, .beta1 = 0.9f, .beta2 = 0.9999f, .eps = 0.0f, .weight_decay = 0.0f};
    const double b1 = c.beta1, b2 = c.beta2;
    for (std::uint64_t step = 1; step <= 3; ++step) {
        gpu::adamw_rows(b, p.d(), std::span<const float>(g.d()), m.d(), v.d(), step, c);
        ml::adamw_rows(la::reference, std::span<float>(hp), std::span<const float>(g.host), std::span<float>(hm), std::span<float>(hv), step, c);
        // the exact factors: b's powers up to the third are exact in double
        const double c1 = 1 - std::pow(b1, double(step)), c2 = 1 - std::pow(b2, double(step));
        for (std::size_t i = 0; i < n; ++i) {
            const double gi = g.host[i];
            rm[i] = b1 * rm[i] + double(1.0f - c.beta1) * gi;
            rv[i] = b2 * rv[i] + double(1.0f - c.beta2) * gi * gi;
            rp[i] -= (rm[i] / c1) / std::sqrt(rv[i] / c2);
        }
        const auto got = p.back();
        near(got, hp, 1e-6f, "adamw against core, step " + std::to_string(step));
        near(got, std::vector<float>(rp.begin(), rp.end()), 1e-5f, "adamw against the exact factors, step " + std::to_string(step));
    }
}

// One launch over many tensors with their own decays: each tensor's adamw_rows to the bit, the gradient scale as
// scale_rows before the update
TEST(GpuCudaOps, AdamwSegmentsIsAdamwRowsPerTensorInOneLaunch)
{
    const auto b = backend();
    for (const bool quads : {true, false}) {
        std::vector<gpu::adamw_segment> segs;
        for (std::size_t k = 0; k < 40; ++k)   // more than one launch's 32
            segs.push_back({quads ? 4 * (k % 5 + 1) : k % 7 + 1, k % 3 == 0 ? 0.1f : 0.0f});
        std::size_t total = 0;
        for (const auto& s : segs)
            total += s.size;
        mirror p(stream(total, 10)), g(stream(total, 11)), m{std::vector<float>(total)}, v{std::vector<float>(total)};
        mirror p2(p.host), g2(g.host), m2{std::vector<float>(total)}, v2{std::vector<float>(total)};
        const ml::adamw_config<float> c{.learning_rate = 0.01f, .beta2 = 0.95f, .weight_decay = 0.5f};   // not used by the segments
        for (std::uint64_t step = 1; step <= 3; ++step) {
            gpu::adamw_segments(b, p.d(), std::span<const float>(g.d()), m.d(), v.d(), step, c, segs, 0.25f);
            gpu::scale_rows(b, g2.d(), 0.25f);
            std::size_t at = 0;
            for (const auto& s : segs) {
                auto cs = c;
                cs.weight_decay = s.weight_decay;
                gpu::adamw_rows(b, p2.d().subspan(at, s.size), std::span<const float>(g2.d().subspan(at, s.size)), m2.d().subspan(at, s.size),
                                v2.d().subspan(at, s.size), step, cs);
                at += s.size;
            }
            g2.dev->upload(g2.host);   // the gradients unscaled again for the next step
        }
        EXPECT_EQ(p.back(), p2.back()) << (quads ? "quads" : "single elements");
        EXPECT_EQ(m.back(), m2.back());
        EXPECT_EQ(v.back(), v2.back());
    }
    mirror p(stream(8, 1)), g(stream(8, 2)), m{std::vector<float>(8)}, v{std::vector<float>(8)};
    const std::vector<gpu::adamw_segment> wrong{{4, 0.0f}, {3, 0.0f}};
    EXPECT_THROW(gpu::adamw_segments(b, p.d(), std::span<const float>(g.d()), m.d(), v.d(), 1, {}, wrong), std::invalid_argument);
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

// The same over a batch of sequences: one product and one permutation into each sequence's heads
TEST(GpuCudaOps, LinearHeadsOverSequencesAreTheReferences)
{
    const auto b = backend();
    constexpr std::size_t heads = 4, d = 16, rows = heads * d, cols = 48, n = 21, seqs = 3;
    mirror w(stream(rows * cols, 51, 0.3f)), xs(stream(seqs * n * cols, 52)), bias(stream(rows, 53, 0.1f)), out{std::vector<float>(seqs * n * rows)},
        douts(stream(seqs * n * rows, 54)), dxs(stream(seqs * n * cols, 55)), dw(stream(rows * cols, 56)), db(stream(rows, 57));
    const ml::dense_weights<float> dev{w.d(), rows, cols}, host{w.host, rows, cols};
    gpu::linear_heads(b, dev, std::span<const float>(xs.d()), out.d(), std::span<const float>(bias.d()), heads, seqs);
    gpu::linear_heads_backward(b, dev, std::span<const float>(xs.d()), std::span<const float>(douts.d()), dxs.d(), dw.d(), db.d(), heads, seqs);
    auto hout = out.host, hdxs = dxs.host, hdw = dw.host, hdb = db.host;
    ml::linear_heads(la::reference, host, std::span<const float>(xs.host), std::span<float>(hout), std::span<const float>(bias.host), heads, seqs);
    ml::linear_heads_backward(la::reference, host, std::span<const float>(xs.host), std::span<const float>(douts.host), std::span<float>(hdxs),
                              std::span<float>(hdw), std::span<float>(hdb), heads, seqs);
    near(out.back(), hout, 1e-5f, "outputs");
    near(dxs.back(), hdxs, 1e-5f, "input gradient");
    near(dw.back(), hdw, 1e-5f, "weight gradient");
    near(db.back(), hdb, 1e-5f, "bias gradient");
}

// A padded head on the device (core 0.12.0): the unpadded head's loss and gradients, nothing to the padding's rows
TEST(GpuCudaOps, APaddedHeadIsTheUnpaddedHeadsOnTheDevice)
{
    const auto b = backend();
    constexpr std::size_t vocab = 500, padded = 512, d = 64, n = 32;
    auto table = stream(padded * d, 8, 0.2f);
    mirror head(table), hs(stream(n * d, 9)), logits{std::vector<float>(n * padded)}, losses{std::vector<float>(n)}, dhs{std::vector<float>(n * d)},
        dhead{std::vector<float>(padded * d)};
    std::vector<std::uint32_t> targets(n);
    for (std::size_t t = 0; t < n; ++t)
        targets[t] = std::uint32_t((t * 37) % vocab);
    gpu::device_buffer<std::uint32_t> dtargets(*device().context, n);
    dtargets.upload(targets);
    const float loss = ml::lm_head_loss_sequence(b, ml::dense_weights<float>{head.d(), padded, d}, std::span<const float>(hs.d()),
                                                 std::span<const std::uint32_t>(dtargets.span().data(), n), logits.d(), losses.d(), dhs.d(), dhead.d(), vocab);
    std::vector<float> hl(n * vocab), hloss(n), hdhs(n * d), hdhead(vocab * d);
    const float want = ml::lm_head_loss_sequence(la::reference, ml::dense_weights<float>{std::span<const float>(table).first(vocab * d), vocab, d},
                                                 std::span<const float>(hs.host), targets, std::span<float>(hl), std::span<float>(hloss), std::span<float>(hdhs),
                                                 std::span<float>(hdhead));
    EXPECT_NEAR(loss, want, 1e-5f);
    near(dhs.back(), hdhs, 1e-4f, "dhs");
    const auto got = dhead.back();
    near(std::vector<float>(got.begin(), got.begin() + std::ptrdiff_t(vocab * d)), hdhead, 1e-4f, "dhead");
    EXPECT_TRUE(std::all_of(got.begin() + std::ptrdiff_t(vocab * d), got.end(), [](float e) { return e == 0.0f; }));
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

// A batch of sequences in one call (core's `sequences`): every attention path, QK norm, RoPE and a window restarting in
// each sequence, against the reference's batch (itself each sequence's own call)
TEST(GpuCudaOps, ABatchOfSequencesInOneCallIsTheReferences)
{
    whole_step_agrees<small, 24, 3>();          // matrix-product attention, sequence by sequence
    whole_step_agrees<nanochat_like, 24, 3>();  // QK norm after RoPE, a window
    whole_step_agrees<mqa_capped, 24, 2>(0);    // the tiled kernels, QK norm before RoPE, a soft cap
    whole_step_agrees<llama_like, 20, 4>(0);    // grouped heads
    whole_step_agrees<gpt2_heads, 150, 3>(4096, gpu::cuda_math::tf32);   // flash attention, the sequence the grid's z
    whole_step_agrees<flash_mix, 130, 2>(4096, gpu::cuda_math::tf32);
}

TEST(GpuCudaOps, BothAttentionPathsGiveTheReferencesShortStep)
{
    whole_step_agrees<small>(0);   // the tiled kernels (heads of 16)
    whole_step_agrees<llama_like>(0);
    whole_step_agrees<nanochat_like>(0);   // a window
}


// Muon (core's ml::muon_step) with the weights, gradient, momentum and working space in device memory: its elementwise
// steps by the kernels, its products by cuBLAS, against core's reference on the host. The momentum is the same
// arithmetic in the same order (the same bits); the update's products sum in cuBLAS's order (float), or round their
// operands to TF32 or bfloat16, so the weights' change is compared normwise
TEST(GpuCudaOps, AMuonStepOnTheDeviceIsTheReferences)
{
    struct shape {
        std::size_t rows, cols;
    };
    // Wide, tall, square, and one whose leading dimension is not a multiple of 8 (bf16 then runs the products with such an operand as TF32)
    for (const auto& [rows, cols] : {shape{64, 192}, shape{192, 64}, shape{96, 96}, shape{40, 52}})
        for (const auto& [math, tol] : {std::pair{gpu::cuda_math::float32, 1e-4}, std::pair{gpu::cuda_math::tf32, 1e-2},
                                       std::pair{gpu::cuda_math::bf16, 5e-2}}) {
            auto b = backend();
            b.math = math;
            const std::size_t n = rows * cols;
            mirror w(stream(n, 31, 0.1f)), momentum{std::vector<float>(n)}, work{std::vector<float>(ml::muon_work_size(rows, cols))};
            auto hw = w.host, hm = momentum.host, hwork = work.host;
            const ml::muon_config<float> config{.lr = 0.02f, .momentum = 0.95f, .nesterov = true};
            for (std::uint64_t t = 0; t < 3; ++t) {
                const mirror g(stream(n, 40 + t, 1.0f));
                ml::muon_step(b, w.d(), std::span<const float>(g.d()), momentum.d(), rows, cols, config, work.d());
                ml::muon_step(la::reference, std::span<float>(hw), std::span<const float>(g.host), std::span<float>(hm), rows, cols, config,
                              std::span<float>(hwork));
            }
            cudaDeviceSynchronize();
            const auto got = w.back();
            std::vector<float> dgot(n), dwant(n);
            for (std::size_t i = 0; i < n; ++i)
                dgot[i] = got[i] - w.host[i], dwant[i] = hw[i] - w.host[i];
            const std::string what = "muon " + std::to_string(rows) + "x" + std::to_string(cols) + " math " + std::to_string(int(math));
            EXPECT_EQ(momentum.back(), hm) << what;
            near_normwise(dgot, dwant, tol, what);
        }
}

} // namespace
