//
//  cuda_moe_test.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/gpu/cuda_moe.hpp>: each mixture-of-experts step on the device against core's reference on the
//  host (ml/moe.hpp), to the bit where the step is a permutation, a count or additions in core's order, within a stated
//  tolerance where it rounds differently; the edge cases (no rows, empty experts, every row to one expert, K = 1 and
//  K = E, segments over many blocks); two runs bit for bit; whole MoE layers forward and backward; a captured graph.
//  They need a CUDA device.
//

#include <snicholls/gpu/cuda_moe.hpp>

#include "support/block_step.hpp"

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/expert_parallel.hpp>
#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/moe.hpp>
#include <snicholls/core/ml/moe_block.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace gpu = snicholls::gpu;
namespace ml = snicholls::ml;
namespace la = snicholls::linalg;
using snicholls::gpu::testing::stream;

struct device_t {
    std::unique_ptr<gpu::cuda_context> context;
    std::unique_ptr<gpu::cuda_kernels> kernels;
    std::unique_ptr<gpu::cuda_moe_kernels> moe;
};

const device_t& device()
{
    static const device_t d = [] {
        auto c = gpu::cuda_context::create();
        if (c.is_err()) {
            std::fprintf(stderr, "cuda_moe_test: no CUDA device\n");
            std::abort();
        }
        device_t out{std::move(c).unwrap(), nullptr, nullptr};
        auto k = gpu::cuda_kernels::create(*out.context);
        auto m = gpu::cuda_moe_kernels::create(*out.context);
        if (k.is_err() || m.is_err()) {
            std::fprintf(stderr, "cuda_moe_test: the kernels did not compile\n");
            std::abort();
        }
        out.kernels = std::move(k).unwrap();
        out.moe = std::move(m).unwrap();
        return out;
    }();
    return d;
}

gpu::cuda_training_backend backend(gpu::cuda_math math = gpu::cuda_math::float32)
{
    gpu::cuda_training_backend b{{device().context.get(), math}, device().kernels.get()};
    b.moe = device().moe.get();
    return b;
}

// A host vector mirrored in device memory, of any trivially copyable element
template <typename T>
struct mirror {
    std::vector<T> host;
    std::unique_ptr<gpu::device_buffer<T>> dev;
    explicit mirror(std::vector<T> h) : host(std::move(h)), dev(std::make_unique<gpu::device_buffer<T>>(*device().context, host.size()))
    {
        dev->upload(host);
    }
    std::span<T> d() const { return dev->span(); }
    std::span<const T> c() const { return dev->span(); }
    std::vector<T> back() const
    {
        device().context->synchronize();
        std::vector<T> out(host.size());
        dev->download(out);
        return out;
    }
};
using fmirror = mirror<float>;
using umirror = mirror<std::uint32_t>;

// Normwise: the largest difference against the largest magnitude
double normwise(const std::vector<float>& got, const std::vector<float>& want)
{
    double diff = 0, scale = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
        diff = std::max(diff, double(std::fabs(got[i] - want[i]))), scale = std::max(scale, double(std::fabs(want[i])));
    return scale == 0 ? diff : diff / scale;
}

void near_normwise(const std::vector<float>& got, const std::vector<float>& want, double tol, const std::string& what)
{
    ASSERT_EQ(got.size(), want.size()) << what;
    const double e = normwise(got, want);
    std::printf("  %-34s normwise %.2e\n", what.c_str(), e);
    EXPECT_LE(e, tol) << what;
}

void near(const std::vector<float>& got, const std::vector<float>& want, float tol, const std::string& what)
{
    ASSERT_EQ(got.size(), want.size()) << what;
    for (std::size_t i = 0; i < got.size(); ++i)
        ASSERT_NEAR(got[i], want[i], tol * (1 + std::fabs(want[i]))) << what << " at " << i;
}

// Experts chosen at random (wyrand), or every one the same
std::vector<std::uint32_t> random_experts(std::size_t n, std::uint32_t E, std::uint64_t seed)
{
    std::vector<std::uint32_t> v(n);
    snicholls::stats::wyrand g(seed);
    for (auto& e : v)
        e = static_cast<std::uint32_t>(snicholls::stats::uniform_index(g, E));
    return v;
}

constexpr ml::transformer_config base{.d_model = 32, .n_heads = 2, .n_kv_heads = 2, .head_dim = 16, .d_ff = 64, .norm = ml::norm_kind::rms,
                                      .mlp = ml::mlp_kind::swiglu, .experts = 8, .top_k = 2, .expert_d_ff = 48};

template <std::size_t E, std::size_t K, ml::moe_scoring S, ml::mlp_kind M = ml::mlp_kind::swiglu>
constexpr ml::transformer_config with()
{
    ml::transformer_config c = base;
    c.experts = E, c.top_k = K, c.scoring = S, c.mlp = M;
    return c;
}

// ---------------------------------------------------------------------------------------------------------------
// Routing

template <ml::transformer_config C>
void route_case(std::size_t rows, bool with_bias, std::uint64_t seed)
{
    constexpr std::size_t E = C.experts, K = C.top_k;
    fmirror logits(stream(rows * E, seed, 3.0f)), bias(with_bias ? stream(E, seed + 1, 0.2f) : std::vector<float>{});
    umirror experts(std::vector<std::uint32_t>(rows * K));
    fmirror weights(std::vector<float>(rows * K));
    const auto b = backend();
    gpu::moe_route_rows<C>(b, logits.c(), bias.c(), experts.d(), weights.d());
    std::vector<std::uint32_t> he(rows * K);
    std::vector<float> hw(rows * K);
    ml::moe_route_rows<C>(la::reference, std::span<const float>(logits.host), std::span<const float>(bias.host), std::span<std::uint32_t>(he),
                          std::span<float>(hw));
    EXPECT_EQ(experts.back(), he) << "the chosen experts";
    near(weights.back(), hw, 2e-6f, "the weights");
    // the backward
    fmirror dw(stream(rows * K, seed + 2)), dz(std::vector<float>(rows * E, 7.0f));
    gpu::moe_route_rows_backward<C>(b, logits.c(), experts.c(), dw.c(), dz.d());
    std::vector<float> hdz(rows * E);
    ml::moe_route_rows_backward<C>(la::reference, std::span<const float>(logits.host), std::span<const std::uint32_t>(he), std::span<const float>(dw.host),
                                   std::span<float>(hdz));
    near(dz.back(), hdz, 4e-6f, "the gradient in the logits");
}

TEST(GpuCudaMoe, RoutingIsCoresTheExpertsToTheBitTheWeightsToRounding)
{
    route_case<with<8, 2, ml::moe_scoring::softmax_top_k>()>(300, false, 1);
    route_case<with<8, 2, ml::moe_scoring::softmax_top_k>()>(300, true, 2);
    route_case<with<16, 4, ml::moe_scoring::softmax>()>(300, true, 3);
    route_case<with<16, 4, ml::moe_scoring::sigmoid>()>(300, true, 4);
}

TEST(GpuCudaMoe, RoutingHandlesOneExpertPerRowEveryExpertAndNoRows)
{
    route_case<with<8, 1, ml::moe_scoring::softmax_top_k>()>(100, false, 5);
    route_case<with<8, 8, ml::moe_scoring::softmax_top_k>()>(100, true, 6);
    route_case<with<8, 8, ml::moe_scoring::softmax>()>(100, false, 7);
    route_case<with<8, 8, ml::moe_scoring::sigmoid>()>(100, true, 8);
    route_case<with<8, 2, ml::moe_scoring::sigmoid>()>(0, true, 9);
}

TEST(GpuCudaMoe, RoutingScalesTheWeightsAndBreaksTiesToTheLowerExpert)
{
    constexpr auto C = [] { auto c = with<8, 3, ml::moe_scoring::sigmoid>(); c.route_scale = 2.5; return c; }();
    route_case<C>(64, true, 10);
    // every logit equal: the lowest experts, in order
    constexpr auto T = with<8, 3, ml::moe_scoring::softmax_top_k>();
    fmirror logits(std::vector<float>(4 * 8, 0.5f));
    umirror experts(std::vector<std::uint32_t>(4 * 3));
    fmirror weights(std::vector<float>(4 * 3));
    gpu::moe_route_rows<T>(backend(), logits.c(), {}, experts.d(), weights.d());
    const auto e = experts.back();
    for (std::size_t r = 0; r < 4; ++r)
        EXPECT_EQ((std::vector<std::uint32_t>(e.begin() + 3 * r, e.begin() + 3 * r + 3)), (std::vector<std::uint32_t>{0, 1, 2}));
}

// ---------------------------------------------------------------------------------------------------------------
// Counting and the permutation

template <ml::transformer_config C>
void permutation_case(const std::vector<std::uint32_t>& chosen)
{
    constexpr std::size_t E = C.experts;
    const auto b = backend();
    umirror experts(chosen), counts(std::vector<std::uint32_t>(E, 99)), offsets(std::vector<std::uint32_t>(E + 1, 99)),
        order(std::vector<std::uint32_t>(chosen.size())), inverse(std::vector<std::uint32_t>(chosen.size()));
    gpu::moe_count_rows<C>(b, experts.c(), counts.d());
    gpu::moe_offsets<C>(b, counts.c(), offsets.d());
    gpu::moe_permutation<C>(b, experts.c(), offsets.c(), order.d(), inverse.d());
    std::vector<std::uint32_t> hc(E), ho(E + 1), hord(chosen.size()), hinv(chosen.size());
    ml::moe_count_rows<C>(la::reference, std::span<const std::uint32_t>(chosen), std::span<std::uint32_t>(hc));
    ml::moe_offsets<C>(la::reference, std::span<const std::uint32_t>(hc), std::span<std::uint32_t>(ho));
    ml::moe_permutation<C>(la::reference, std::span<const std::uint32_t>(chosen), std::span<const std::uint32_t>(ho), std::span<std::uint32_t>(hord),
                           std::span<std::uint32_t>(hinv));
    EXPECT_EQ(counts.back(), hc) << "counts";
    EXPECT_EQ(offsets.back(), ho) << "offsets";
    EXPECT_EQ(order.back(), hord) << "order";
    EXPECT_EQ(inverse.back(), hinv) << "inverse";
}

TEST(GpuCudaMoe, TheCountsOffsetsAndPermutationAreCoresToTheBit)
{
    constexpr auto C = with<8, 2, ml::moe_scoring::softmax_top_k>();
    permutation_case<C>(random_experts(2 * 37, 8, 11));
    permutation_case<C>(random_experts(2 * 5000, 8, 12));   // ten blocks of the counting sort
    constexpr auto F = with<64, 6, ml::moe_scoring::sigmoid>();
    permutation_case<F>(random_experts(6 * 3001, 64, 13));
    constexpr auto W = with<1024, 4, ml::moe_scoring::softmax>();   // the most experts the device takes
    permutation_case<W>(random_experts(4 * 2000, 1024, 14));
}

TEST(GpuCudaMoe, ThePermutationHandlesEmptyExpertsOneExpertAndNoRows)
{
    constexpr auto C = with<8, 2, ml::moe_scoring::softmax_top_k>();
    permutation_case<C>(std::vector<std::uint32_t>(2 * 3000, 5));          // every row to expert 5
    permutation_case<C>(random_experts(2 * 900, 3, 15));                   // experts 3 to 7 empty
    permutation_case<C>({});                                               // no rows
    constexpr auto one = with<8, 1, ml::moe_scoring::softmax_top_k>();
    permutation_case<one>(random_experts(1500, 8, 16));
    constexpr auto all = with<8, 8, ml::moe_scoring::softmax_top_k>();
    std::vector<std::uint32_t> every(8 * 700);
    for (std::size_t j = 0; j < every.size(); ++j)
        every[j] = static_cast<std::uint32_t>(7 - j % 8);   // each row to every expert, in reverse
    permutation_case<all>(every);
}

// ---------------------------------------------------------------------------------------------------------------
// Gather and combine

template <ml::transformer_config C>
void gather_combine_case(std::size_t rows, std::uint64_t seed)
{
    constexpr std::size_t E = C.experts, K = C.top_k, d = C.d_model;
    const auto chosen = [&] {   // distinct experts per row, as routing gives
        std::vector<std::uint32_t> v(rows * K);
        snicholls::stats::wyrand g(seed);
        for (std::size_t r = 0; r < rows; ++r) {
            std::vector<std::uint32_t> all(E);
            std::iota(all.begin(), all.end(), 0u);
            for (std::size_t s = 0; s < K; ++s) {
                std::swap(all[s], all[s + snicholls::stats::uniform_index(g, E - s)]);
                v[r * K + s] = all[s];
            }
        }
        return v;
    }();
    std::vector<std::uint32_t> hc(E), ho(E + 1), hord(rows * K), hinv(rows * K);
    ml::moe_count_rows<C>(la::reference, std::span<const std::uint32_t>(chosen), std::span<std::uint32_t>(hc));
    ml::moe_offsets<C>(la::reference, std::span<const std::uint32_t>(hc), std::span<std::uint32_t>(ho));
    ml::moe_permutation<C>(la::reference, std::span<const std::uint32_t>(chosen), std::span<const std::uint32_t>(ho), std::span<std::uint32_t>(hord),
                           std::span<std::uint32_t>(hinv));
    umirror order(hord), inverse(hinv);
    fmirror xs(stream(rows * d, seed + 1)), xp(std::vector<float>(rows * K * d)), yp(stream(rows * K * d, seed + 2)), w(stream(rows * K, seed + 3)),
        ys(std::vector<float>(rows * d)), dys(stream(rows * d, seed + 4)), dyp(std::vector<float>(rows * K * d)), dw(std::vector<float>(rows * K)),
        dxs(stream(rows * d, seed + 5));
    const auto b = backend();
    gpu::moe_gather_rows<C>(b, xs.c(), order.c(), xp.d());
    gpu::moe_combine_rows<C>(b, yp.c(), w.c(), inverse.c(), ys.d());
    gpu::moe_combine_rows_backward<C>(b, dys.c(), yp.c(), w.c(), inverse.c(), dyp.d(), dw.d());
    gpu::moe_gather_rows_backward<C>(b, yp.c(), inverse.c(), dxs.d());
    std::vector<float> hxp(rows * K * d), hys(rows * d), hdyp(rows * K * d), hdw(rows * K), hdxs = dxs.host;
    ml::moe_gather_rows<C>(la::reference, std::span<const float>(xs.host), std::span<const std::uint32_t>(hord), std::span<float>(hxp));
    ml::moe_combine_rows<C>(la::reference, std::span<const float>(yp.host), std::span<const float>(w.host), std::span<const std::uint32_t>(hinv),
                            std::span<float>(hys));
    ml::moe_combine_rows_backward<C>(la::reference, std::span<const float>(dys.host), std::span<const float>(yp.host), std::span<const float>(w.host),
                                     std::span<const std::uint32_t>(hinv), std::span<float>(hdyp), std::span<float>(hdw));
    ml::moe_gather_rows_backward<C>(la::reference, std::span<const float>(yp.host), std::span<const std::uint32_t>(hinv), std::span<float>(hdxs));
    EXPECT_EQ(xp.back(), hxp) << "gather";
    EXPECT_EQ(ys.back(), hys) << "combine";
    EXPECT_EQ(dyp.back(), hdyp) << "combine backward, dyp";
    EXPECT_EQ(dw.back(), hdw) << "combine backward, dweights";
    EXPECT_EQ(dxs.back(), hdxs) << "gather backward";
}

TEST(GpuCudaMoe, GatherAndCombineAndTheirBackwardAreCoresToTheBit)
{
    gather_combine_case<with<8, 2, ml::moe_scoring::softmax_top_k>()>(200, 21);
    gather_combine_case<with<8, 1, ml::moe_scoring::softmax_top_k>()>(50, 22);
    gather_combine_case<with<8, 8, ml::moe_scoring::softmax_top_k>()>(30, 23);
    constexpr auto odd = [] { auto c = with<4, 2, ml::moe_scoring::softmax>(); c.d_model = 30; c.head_dim = 15; c.rope = ml::rope_kind::none; return c; }();
    gather_combine_case<odd>(40, 24);   // rows of 30: no float4
    gather_combine_case<with<8, 2, ml::moe_scoring::softmax_top_k>()>(0, 25);
}

// ---------------------------------------------------------------------------------------------------------------
// The grouped products

struct grouped_case {
    std::size_t E, out, in;
    std::vector<std::uint32_t> counts;
    bool bias;
};

gpu::grouped_path gpu_path() { return device().moe->path; }

// The forward and the backward of one grouped product on the device against core's reference: per math, a normwise
// tolerance; returns the device's results for a determinism check
std::vector<float> grouped_check(const grouped_case& g, gpu::cuda_math math, double tol, std::uint32_t first = 0)
{
    std::vector<std::uint32_t> off(g.E + 1, first);
    for (std::size_t e = 0; e < g.E; ++e)
        off[e + 1] = off[e] + g.counts[e];
    const std::size_t n = off[g.E] - first;
    umirror offsets(off);
    fmirror w(stream(g.E * g.out * g.in, 31, 0.3f)), bias(g.bias ? stream(g.E * g.out, 32, 0.5f) : std::vector<float>{}), xs(stream(n * g.in, 33)),
        ys(std::vector<float>(n * g.out)), dys(stream(n * g.out, 34)), dxs(stream(n * g.in, 35)), dw(stream(g.E * g.out * g.in, 36)),
        db(g.bias ? stream(g.E * g.out, 37) : std::vector<float>{});
    auto b = backend(math);
    const ml::grouped_weights<float> dwts{w.c(), g.E, g.out, g.in}, hwts{std::span<const float>(w.host), g.E, g.out, g.in};
    gpu::grouped_linear(b, dwts, offsets.c(), xs.c(), ys.d(), bias.c());
    gpu::grouped_linear_backward(b, dwts, offsets.c(), xs.c(), dys.c(), dxs.d(), dw.d(), db.d());
    std::vector<float> hys(n * g.out), hdxs = dxs.host, hdw = dw.host, hdb = db.host;
    ml::grouped_linear(la::reference, hwts, std::span<const std::uint32_t>(off), std::span<const float>(xs.host), std::span<float>(hys),
                       std::span<const float>(bias.host));
    ml::grouped_linear_backward(la::reference, hwts, std::span<const std::uint32_t>(off), std::span<const float>(xs.host), std::span<const float>(dys.host),
                                std::span<float>(hdxs), std::span<float>(hdw), std::span<float>(hdb));
    const auto gys = ys.back(), gdxs = dxs.back(), gdw = dw.back(), gdb = db.back();
    near_normwise(gys, hys, tol, "grouped_linear");
    near_normwise(gdxs, hdxs, tol, "grouped_linear_backward dX");
    near_normwise(gdw, hdw, tol, "grouped_linear_backward dW");
    if (g.bias && gpu_path() == gpu::grouped_path::kernel) {
        EXPECT_EQ(gdb, hdb) << "the biases' gradient: core's sum_rows order";
    } else if (g.bias) {
        near_normwise(gdb, hdb, 1e-6, "the biases' gradient (sum_rows)");
    }
    std::vector<float> all = gys;
    all.insert(all.end(), gdxs.begin(), gdxs.end());
    all.insert(all.end(), gdw.begin(), gdw.end());
    return all;
}

// Tolerances, normwise against the largest element: float (three TF32 products) 2e-6; TF32 (10 bits of mantissa)
// 2e-3; bfloat16 (8 bits) 1e-2
constexpr double tol_float = 2e-6, tol_tf32 = 2e-3, tol_bf16 = 1e-2;

TEST(GpuCudaMoe, TheGroupedProductIsCoresToTheProductsRoundingInEachMath)
{
    const grouped_case g{8, 96, 64, {130, 0, 7, 300, 1, 129, 64, 2}, true};
    for (const auto& [math, tol] : {std::pair{gpu::cuda_math::float32, tol_float}, std::pair{gpu::cuda_math::tf32, tol_tf32},
                                    std::pair{gpu::cuda_math::bf16, tol_bf16}}) {
        const auto one = grouped_check(g, math, tol);
        EXPECT_EQ(grouped_check(g, math, tol), one) << "two runs, the same bits";
    }
}

TEST(GpuCudaMoe, TheGroupedProductHandlesEmptyExpertsOneExpertOddSizesAndOffsets)
{
    grouped_check({4, 64, 32, {0, 0, 0, 0}, true}, gpu::cuda_math::float32, tol_float);          // no rows
    grouped_check({4, 64, 32, {0, 700, 0, 0}, false}, gpu::cuda_math::float32, tol_float);       // one expert takes all
    grouped_check({3, 30, 18, {5, 17, 33}, true}, gpu::cuda_math::float32, tol_float);           // no float4: 30 and 18 columns
    grouped_check({3, 30, 18, {5, 17, 33}, true}, gpu::cuda_math::bf16, tol_bf16);
    grouped_check({5, 260, 140, {129, 1, 0, 257, 3}, false}, gpu::cuda_math::tf32, tol_tf32);    // partial tiles both ways
    grouped_check({2, 32, 16, {10, 20}, true}, gpu::cuda_math::float32, tol_float, 17);          // offsets from 17
}

TEST(GpuCudaMoe, TheCublasPathGivesTheSameProductsButCannotBeCaptured)
{
    device().moe->path = gpu::grouped_path::cublas;
    grouped_check({8, 96, 64, {130, 0, 7, 300, 1, 129, 64, 2}, true}, gpu::cuda_math::float32, 1e-6);
    grouped_check({8, 96, 64, {130, 0, 7, 300, 1, 129, 64, 2}, true}, gpu::cuda_math::bf16, tol_bf16);
    umirror offsets(std::vector<std::uint32_t>{0, 4});
    fmirror w(stream(16, 1)), xs(stream(16, 2)), ys(std::vector<float>(16));
    const auto b = backend();
    EXPECT_THROW((void)gpu::cuda_graph::capture(*device().context, [&] {
        gpu::grouped_linear(b, ml::grouped_weights<float>{w.c(), 1, 4, 4}, offsets.c(), xs.c(), ys.d());
    }), std::logic_error);
    device().moe->path = gpu::grouped_path::kernel;
}

// ---------------------------------------------------------------------------------------------------------------
// Balancing

template <ml::transformer_config C>
void balance_case(std::size_t rows, std::size_t groups, std::size_t denominator, std::uint64_t seed)
{
    constexpr std::size_t E = C.experts, K = C.top_k;
    fmirror logits(stream(rows * E, seed, 2.0f));
    std::vector<std::uint32_t> he(rows * K);
    std::vector<float> hw(rows * K);
    ml::moe_route_rows<C>(la::reference, std::span<const float>(logits.host), {}, std::span<std::uint32_t>(he), std::span<float>(hw));
    umirror experts(he);
    fmirror dz(stream(rows * E, seed + 1));
    const auto b = backend();
    const float lb = gpu::moe_balance_loss<C>(b, logits.c(), experts.c(), dz.d(), groups, denominator);
    const float lz = gpu::moe_z_loss<C>(b, logits.c(), dz.d(), denominator == 0 ? 0 : rows);
    const float lb_only = gpu::moe_balance_loss<C>(b, logits.c(), experts.c(), {}, groups, denominator);
    auto hdz = dz.host;
    const float want_b = ml::moe_balance_loss<C>(la::reference, std::span<const float>(logits.host), std::span<const std::uint32_t>(he),
                                                 std::span<float>(hdz), groups, denominator);
    const float want_z = ml::moe_z_loss<C>(la::reference, std::span<const float>(logits.host), std::span<float>(hdz), denominator == 0 ? 0 : rows);
    EXPECT_NEAR(lb, want_b, 2e-6 * std::fabs(want_b) + 1e-12) << "balance loss";
    EXPECT_EQ(lb, lb_only) << "the loss alone is the same number";
    EXPECT_NEAR(lz, want_z, 2e-6 * std::fabs(want_z) + 1e-12) << "z-loss";
    near(dz.back(), hdz, 2e-6f, "the losses' gradients");
}

TEST(GpuCudaMoe, TheBalanceLossAndZLossAreCoresToRounding)
{
    constexpr auto C = [] { auto c = with<8, 2, ml::moe_scoring::softmax>(); c.balance_loss_weight = 0.01; c.z_loss_weight = 0.001; return c; }();
    balance_case<C>(256, 1, 0, 41);
    balance_case<C>(256, 8, 0, 42);    // a loss per sequence of 32 rows
    balance_case<C>(256, 8, 32, 43);   // a worker's part of a batch of 32 sequences
    constexpr auto S = [] { auto c = with<16, 4, ml::moe_scoring::sigmoid>(); c.balance_loss_weight = 0.0001; c.z_loss_weight = 0.01; return c; }();
    balance_case<S>(300, 3, 0, 44);
    constexpr auto T = [] { auto c = with<4, 4, ml::moe_scoring::softmax_top_k>(); c.balance_loss_weight = 1; c.z_loss_weight = 1; return c; }();
    balance_case<T>(40, 1, 0, 45);
    balance_case<T>(0, 1, 0, 46);
}

TEST(GpuCudaMoe, TheBiasUpdateIsCoresToTheBit)
{
    constexpr auto C = [] { auto c = with<8, 2, ml::moe_scoring::sigmoid>(); c.bias_update_rate = 0.001; return c; }();
    umirror counts(std::vector<std::uint32_t>{10, 20, 30, 20, 0, 20, 40, 20});   // mean 20: below, at, above
    fmirror bias(stream(8, 51, 0.1f));
    gpu::moe_update_bias<C>(backend(), counts.c(), bias.d());
    auto h = bias.host;
    ml::moe_update_bias<C>(la::reference, std::span<const std::uint32_t>(counts.host), std::span<float>(h));
    EXPECT_EQ(bias.back(), h);
}

TEST(GpuCudaMoe, TheLossesStayOnTheDeviceWhenAsked)
{
    constexpr auto C = [] { auto c = with<8, 2, ml::moe_scoring::softmax>(); c.balance_loss_weight = 0.01; c.z_loss_weight = 0.001; return c; }();
    fmirror logits(stream(64 * 8, 61, 2.0f));
    umirror experts(random_experts(64 * 2, 8, 62));
    const auto b = backend();
    const float want = gpu::moe_z_loss<C>(b, logits.c(), {});
    device().moe->losses_on_device = true;
    const float got = gpu::moe_z_loss<C>(b, logits.c(), {});
    device().moe->losses_on_device = false;
    EXPECT_TRUE(std::isnan(got));
    std::vector<float> slots(2);
    device().context->synchronize();
    cudaMemcpy(slots.data(), device().moe->losses().data(), 2 * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(slots[1], want);
}

TEST(GpuCudaMoe, ABackendWithoutTheKernelsSaysSo)
{
    gpu::cuda_training_backend b{{device().context.get()}, device().kernels.get()};
    umirror counts(std::vector<std::uint32_t>(8)), offsets(std::vector<std::uint32_t>(9));
    EXPECT_THROW(gpu::moe_offsets<base>(b, counts.c(), offsets.d()), std::logic_error);
}

// ---------------------------------------------------------------------------------------------------------------
// Whole layers

// Every buffer of one MoE layer's forward and backward, host and device
template <ml::transformer_config C>
struct layer {
    static constexpr std::size_t d = C.d_model;
    std::size_t rows;
    fmirror params, sel, xs, ys, values, dys, dxs, grads, work;
    umirror indices;
    layer(std::size_t r, bool selection_bias, std::uint64_t seed)
        : rows(r), params(stream(ml::moe_parameter_count(C), seed, 0.2f)), sel(selection_bias ? stream(C.experts, seed + 1, 0.05f) : std::vector<float>{}),
          xs(stream(r * d, seed + 2)), ys(std::vector<float>(r * d)), values(std::vector<float>(ml::moe_activation_values(C, r))),
          dys(stream(r * d, seed + 3)), dxs(stream(r * d, seed + 4)), grads(stream(ml::moe_parameter_count(C), seed + 5, 0.01f)),
          work(std::vector<float>(ml::moe_gradient_work(C, r))), indices(std::vector<std::uint32_t>(ml::moe_activation_indices(C, r)))
    {
    }
    // The pass on the device (host = false) or on the host copies
    ml::moe_losses<float> run(const gpu::cuda_training_backend& b, std::size_t sequences, bool host)
    {
        auto f = [&](fmirror& m) { return host ? std::span<float>(m.host) : m.d(); };
        const auto w = ml::moe_weights_in<C>(std::span<const float>(f(params)), std::span<const float>(f(sel)));
        const auto act = ml::moe_activations_in<C>(f(values), host ? std::span<std::uint32_t>(indices.host) : indices.d(), rows);
        const auto g = ml::moe_gradients_in<C>(f(grads), f(work), rows);
        ml::moe_losses<float> l;
        if (host) {
            l = ml::moe_forward<C>(la::reference, w, std::span<const float>(f(xs)), f(ys), act, sequences);
            ml::moe_backward<C>(la::reference, w, std::span<const float>(f(xs)), act, std::span<const float>(f(dys)), f(dxs), g, sequences);
        } else {
            l = ml::moe_forward<C>(b, w, std::span<const float>(f(xs)), f(ys), act, sequences);
            ml::moe_backward<C>(b, w, std::span<const float>(f(xs)), act, std::span<const float>(f(dys)), f(dxs), g, sequences);
        }
        return l;
    }
};

template <ml::transformer_config C>
void layer_case(std::size_t rows, std::size_t sequences, bool selection_bias, std::uint64_t seed, double tol)
{
    layer<C> L(rows, selection_bias, seed);
    const auto b = backend();
    const auto got = L.run(b, sequences, false);
    const auto want = L.run(b, sequences, true);
    // the routing the device chose, to the bit (the logits agree to rounding; no key is within rounding of another here)
    const auto act_d = L.indices.back();
    EXPECT_EQ(std::vector<std::uint32_t>(act_d.begin(), act_d.begin() + rows * C.top_k),
              std::vector<std::uint32_t>(L.indices.host.begin(), L.indices.host.begin() + rows * C.top_k)) << "the chosen experts";
    near_normwise(L.ys.back(), L.ys.host, tol, "moe_forward ys");
    near_normwise(L.dxs.back(), L.dxs.host, tol, "moe_backward dxs");
    near_normwise(L.grads.back(), L.grads.host, tol, "moe_backward gradients");
    EXPECT_NEAR(got.balance, want.balance, 1e-5 * std::fabs(want.balance) + 1e-12);
    EXPECT_NEAR(got.z, want.z, 1e-5 * std::fabs(want.z) + 1e-12);
}

constexpr ml::transformer_config mixtral_like = with<8, 2, ml::moe_scoring::softmax_top_k>();
constexpr ml::transformer_config olmoe_like = [] {
    auto c = with<16, 4, ml::moe_scoring::softmax>();
    c.balance_loss_weight = 0.01, c.z_loss_weight = 0.001;
    return c;
}();
constexpr ml::transformer_config deepseek_like = [] {
    auto c = with<16, 3, ml::moe_scoring::sigmoid, ml::mlp_kind::gelu>();
    c.shared_experts = 1, c.route_scale = 2.5, c.bias_update_rate = 0.001, c.balance_loss_weight = 0.0001, c.bias = true;
    return c;
}();

TEST(GpuCudaMoe, WholeLayersForwardAndBackwardAreCoresToRounding)
{
    layer_case<mixtral_like>(96, 1, false, 71, 2e-5);
    layer_case<olmoe_like>(128, 4, false, 72, 2e-5);
    layer_case<deepseek_like>(120, 3, true, 73, 2e-5);
    layer_case<ml::tiny_moe>(40, 2, false, 74, 2e-5);
}

TEST(GpuCudaMoe, ALayerIsTheSameBitsEveryRunInEveryMath)
{
    for (const auto math : {gpu::cuda_math::float32, gpu::cuda_math::tf32, gpu::cuda_math::bf16}) {
        layer<olmoe_like> a(256, false, 81), c(256, false, 81);
        a.run(backend(math), 4, false);
        c.run(backend(math), 4, false);
        EXPECT_EQ(a.ys.back(), c.ys.back());
        EXPECT_EQ(a.dxs.back(), c.dxs.back());
        EXPECT_EQ(a.grads.back(), c.grads.back());
        EXPECT_EQ(a.indices.back(), c.indices.back());
    }
}

TEST(GpuCudaMoe, ALayerCapturedAsAGraphReplaysTheUncapturedBits)
{
    layer<deepseek_like> a(120, true, 91), c(120, true, 91);
    const auto b = backend(gpu::cuda_math::tf32);
    a.run(b, 3, false);   // uncaptured: sizes the working space
    device().moe->losses_on_device = false;
    auto graph = gpu::cuda_graph::capture(*device().context, [&] { c.run(b, 3, false); });
    ASSERT_TRUE(graph.is_ok());
    // the capture queued nothing: replay it once onto c's initial buffers
    c.grads.dev->upload(c.grads.host);
    c.dxs.dev->upload(c.dxs.host);
    const auto g = std::move(graph).unwrap();
    g.launch();
    EXPECT_EQ(c.ys.back(), a.ys.back());
    EXPECT_EQ(c.dxs.back(), a.dxs.back());
    EXPECT_EQ(c.grads.back(), a.grads.back());
}

// ---------------------------------------------------------------------------------------------------------------
// Expert parallelism's layouts and payloads, ranks in one process on one device

TEST(GpuCudaMoe, TheExpertParallelLayoutsAreCoresToTheBit)
{
    constexpr auto C = with<16, 3, ml::moe_scoring::softmax>();
    constexpr std::size_t E = C.experts, P = 4;
    const auto b = backend();
    // each rank's counts and offsets from its own routing
    std::vector<umirror> counts;
    std::vector<std::vector<std::uint32_t>> hcounts;
    for (std::size_t r = 0; r < P; ++r) {
        const auto chosen = random_experts(3 * (50 + 13 * r), 16, 100 + r);
        std::vector<std::uint32_t> c(E);
        ml::moe_count_rows<C>(la::reference, std::span<const std::uint32_t>(chosen), std::span<std::uint32_t>(c));
        if (r == 2)
            c[5] = 0, c[6] = 0, c[7] = 0;   // a rank that sends nothing to rank 1
        hcounts.push_back(c);
        counts.emplace_back(c);
    }
    std::vector<std::span<const std::uint32_t>> dc, hc;
    for (std::size_t r = 0; r < P; ++r)
        dc.push_back(counts[r].c()), hc.emplace_back(hcounts[r]);
    for (std::size_t r = 0; r < P; ++r) {
        std::vector<std::uint32_t> off(E + 1), send(P + 1), recv(P + 1), eo(E / P + 1);
        ml::moe_offsets<C>(la::reference, std::span<const std::uint32_t>(hcounts[r]), std::span<std::uint32_t>(off));
        ml::moe_send_offsets<C>(la::reference, std::span<const std::uint32_t>(off), P, std::span<std::uint32_t>(send));
        ml::moe_receive_offsets<C>(la::reference, std::span<const std::span<const std::uint32_t>>(hc), r, std::span<std::uint32_t>(recv));
        std::vector<std::uint32_t> order(recv[P]);
        ml::moe_receive_layout<C>(la::reference, std::span<const std::span<const std::uint32_t>>(hc), r, std::span<std::uint32_t>(eo),
                                  std::span<std::uint32_t>(order));
        umirror doff(off), dsend(std::vector<std::uint32_t>(P + 1)), drecv(std::vector<std::uint32_t>(P + 1)), deo(std::vector<std::uint32_t>(E / P + 1)),
            dorder(std::vector<std::uint32_t>(order.size()));
        gpu::moe_send_offsets<C>(b, doff.c(), P, dsend.d());
        gpu::moe_receive_offsets<C>(b, std::span<const std::span<const std::uint32_t>>(dc), r, drecv.d());
        gpu::moe_receive_layout<C>(b, std::span<const std::span<const std::uint32_t>>(dc), r, deo.d(), dorder.d());
        EXPECT_EQ(dsend.back(), send) << "rank " << r;
        EXPECT_EQ(drecv.back(), recv) << "rank " << r;
        EXPECT_EQ(deo.back(), eo) << "rank " << r;
        EXPECT_EQ(dorder.back(), order) << "rank " << r;
    }
}

TEST(GpuCudaMoe, TheAllToAllAndThePermutationsCopyCoresBits)
{
    constexpr std::size_t P = 3, width = 5;
    // source s sends rows [so[s][r], so[s][r + 1]) to r
    const std::vector<std::vector<std::uint32_t>> so = {{0, 4, 4, 9}, {0, 0, 7, 8}, {0, 2, 3, 6}};
    std::vector<fmirror> send;
    std::vector<umirror> dso;
    std::vector<std::vector<float>> hrecv(P);
    for (std::size_t s = 0; s < P; ++s) {
        send.emplace_back(stream(so[s][P] * width, 200 + s));
        dso.emplace_back(so[s]);
    }
    for (std::size_t r = 0; r < P; ++r) {
        std::size_t rows = 0;
        for (std::size_t s = 0; s < P; ++s)
            rows += so[s][r + 1] - so[s][r];
        hrecv[r].assign(rows * width, 0.0f);
    }
    std::vector<fmirror> recv;
    for (std::size_t r = 0; r < P; ++r)
        recv.emplace_back(std::vector<float>(hrecv[r].size(), -1.0f));
    std::vector<std::span<const float>> hs, ds;
    std::vector<std::span<const std::uint32_t>> hso, dsos;
    std::vector<std::span<float>> hr, dr;
    for (std::size_t s = 0; s < P; ++s) {
        hs.emplace_back(send[s].host), ds.push_back(send[s].c()), hso.emplace_back(so[s]), dsos.push_back(dso[s].c());
        hr.emplace_back(hrecv[s]), dr.push_back(recv[s].d());
    }
    const auto b = backend();
    ml::moe_all_to_all(la::reference, std::span<const std::span<const float>>(hs), std::span<const std::span<const std::uint32_t>>(hso), width,
                       std::span<const std::span<float>>(hr));
    gpu::moe_all_to_all(b, std::span<const std::span<const float>>(ds), std::span<const std::span<const std::uint32_t>>(dsos), width,
                        std::span<const std::span<float>>(dr));
    for (std::size_t r = 0; r < P; ++r)
        EXPECT_EQ(recv[r].back(), hrecv[r]) << "destination " << r;
    // rows permuted and back, float and bfloat16
    std::vector<std::uint32_t> order(9);
    std::iota(order.begin(), order.end(), 0u);
    std::reverse(order.begin(), order.begin() + 5);
    std::swap(order[6], order[8]);
    umirror dorder(order);
    fmirror from(stream(9 * width, 210)), to(std::vector<float>(9 * width)), back(std::vector<float>(9 * width));
    gpu::moe_permute_rows(b, from.c(), dorder.c(), to.d(), width);
    gpu::moe_unpermute_rows(b, to.c(), dorder.c(), back.d(), width);
    std::vector<float> hto(9 * width);
    ml::moe_permute_rows(la::reference, std::span<const float>(from.host), std::span<const std::uint32_t>(order), std::span<float>(hto), width);
    EXPECT_EQ(to.back(), hto);
    EXPECT_EQ(back.back(), from.host);
    std::vector<ml::bfloat16> hb(9 * width), hbt(9 * width);
    for (std::size_t i = 0; i < hb.size(); ++i)
        hb[i] = ml::to_bfloat16(from.host[i]);
    mirror<ml::bfloat16> db(hb), dbt(std::vector<ml::bfloat16>(9 * width));
    gpu::moe_permute_rows(b, db.c(), dorder.c(), dbt.d(), width);
    ml::moe_permute_rows(la::reference, std::span<const ml::bfloat16>(hb), std::span<const std::uint32_t>(order), std::span<ml::bfloat16>(hbt), width);
    const auto got = dbt.back();
    for (std::size_t i = 0; i < got.size(); ++i)
        EXPECT_EQ(got[i].bits, hbt[i].bits) << i;
}

TEST(GpuCudaMoe, TheBfloat16PayloadRoundsAndWidensAsCore)
{
    std::vector<float> v = stream(1000, 220, 100.0f);
    for (const float x : {0.0f, -0.0f, 1.0f, 1.00390625f, 1.01171875f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::denorm_min(),
                          3.4e38f, std::numeric_limits<float>::max(), 1e-40f})
        v.push_back(x);
    fmirror from(v), wide(std::vector<float>(v.size()));
    mirror<ml::bfloat16> to(std::vector<ml::bfloat16>(v.size()));
    const auto b = backend();
    gpu::to_bfloat16_rows(b, from.c(), to.d());
    gpu::to_float_rows(b, to.c(), wide.d());
    std::vector<ml::bfloat16> want(v.size());
    std::vector<float> wwide(v.size());
    ml::to_bfloat16_rows(la::reference, std::span<const float>(v), std::span<ml::bfloat16>(want));
    ml::to_float_rows(la::reference, std::span<const ml::bfloat16>(want), std::span<float>(wwide));
    const auto got = to.back();
    const auto gw = wide.back();
    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_EQ(got[i].bits, want[i].bits) << i;
        EXPECT_EQ(std::bit_cast<std::uint32_t>(gw[i]), std::bit_cast<std::uint32_t>(wwide[i])) << i;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// A block with an MoE MLP (ml/moe_block.hpp): attention, norms and the mixture of experts, all on the device

constexpr ml::transformer_config moe_block_cfg{.d_model = 32, .n_heads = 2, .n_kv_heads = 1, .head_dim = 16, .d_ff = 24, .mlp = ml::mlp_kind::swiglu,
                                               .experts = 4, .top_k = 2, .shared_experts = 1, .expert_d_ff = 24, .balance_loss_weight = 0.02,
                                               .z_loss_weight = 0.01};

template <ml::transformer_config C>
struct block {
    static constexpr std::size_t d = C.d_model, q = C.q_size(), kv = C.kv_size();
    std::size_t rows;
    std::vector<std::size_t> at;
    std::vector<fmirror> m;   // p, g, xs, x0, h1, x1, h2, aq, ak, av, ay, lse, mv, mw, scratch, dq, dk, dv, dy
    umirror mi;
    enum { p, g, xs, x0, h1, x1, h2, aq, ak, av, ay, lse, mv, mw, scratch, dq, dk, dv, dy };
    explicit block(std::size_t n, std::uint64_t seed) : rows(n), mi(std::vector<std::uint32_t>(ml::moe_activation_indices(C, n)))
    {
        const std::vector<std::size_t> sizes{d, 0, q * d, kv * d, kv * d, d * q, 0, 0, 0, 0, d, 0, ml::moe_parameter_count(C)};
        std::size_t total = 0;
        for (const auto s : sizes)
            at.push_back(total), total += s;
        at.push_back(total);
        auto params = stream(total, seed, 0.3f);
        for (const std::size_t norm : {0, 10})
            for (std::size_t i = at[norm]; i < at[norm + 1]; ++i)
                params[i] += 1.0f;
        m.emplace_back(params), m.emplace_back(stream(total, seed + 1, 0.01f)), m.emplace_back(stream(n * d, seed + 2));
        for (int i = 0; i < 4; ++i)
            m.emplace_back(std::vector<float>(n * d));
        m.emplace_back(std::vector<float>(n * q)), m.emplace_back(std::vector<float>(n * kv)), m.emplace_back(std::vector<float>(n * kv));
        m.emplace_back(std::vector<float>(n * q)), m.emplace_back(std::vector<float>(n * C.n_heads));
        m.emplace_back(std::vector<float>(ml::moe_activation_values(C, n))), m.emplace_back(std::vector<float>(ml::moe_gradient_work(C, n)));
        m.emplace_back(std::vector<float>(n * d));
        m.emplace_back(std::vector<float>(n * q)), m.emplace_back(std::vector<float>(n * kv)), m.emplace_back(std::vector<float>(n * kv));
        m.emplace_back(std::vector<float>(n * q));
    }
    // The pass over the device buffers (host = false) or the host copies
    ml::moe_losses<float> run(const gpu::cuda_training_backend& b, std::size_t sequences, bool host)
    {
        auto f = [&](int i) { return host ? std::span<float>(m[std::size_t(i)].host) : m[std::size_t(i)].d(); };
        auto param = [&](std::size_t i) { return std::span<const float>(f(p)).subspan(at[i], at[i + 1] - at[i]); };
        auto grad = [&](std::size_t i) { return f(g).subspan(at[i], at[i + 1] - at[i]); };
        const auto idx = host ? std::span<std::uint32_t>(mi.host) : mi.d();
        const ml::attention_weights<ml::dense_weights<float>, float> aw{{param(2), q, d}, {param(3), kv, d}, {param(4), kv, d}, {param(5), d, q},
                                                                         param(6), param(7), param(8), param(9)};
        const ml::moe_block_weights<float> w{param(0), param(1), aw, param(10), param(11), ml::moe_weights_in<C>(param(12))};
        const ml::moe_block_activations<float> act{f(x0), f(h1), f(x1), f(h2), {f(aq), f(ak), f(av), {}, {}, f(ay), f(lse)},
                                                   ml::moe_activations_in<C>(f(mv), idx, rows)};
        const ml::moe_block_gradients<float> gr{grad(0), grad(1), grad(10), grad(11),
                                                {f(dq), f(dk), f(dv), grad(2), grad(3), grad(4), grad(5), grad(6), grad(7), grad(8), grad(9), {}, {}, f(dy)},
                                                ml::moe_gradients_in<C>(grad(12), f(mw), rows), f(scratch)};
        ml::moe_losses<float> l;
        auto dys = stream(rows * d, 999);
        if (host) {
            l = ml::block_forward_sequence<C>(la::reference, w, f(xs), ml::no_rope, act, sequences);
            std::copy(dys.begin(), dys.end(), f(xs).begin());   // the output's gradient replaces the output
            ml::block_backward<C>(la::reference, w, ml::no_rope, act, f(xs), gr, sequences);
        } else {
            l = ml::block_forward_sequence<C>(b, w, f(xs), ml::no_rope, act, sequences);
            fmirror up(dys);
            gpu::copy_rows(b, up.c(), f(xs));
            ml::block_backward<C>(b, w, ml::no_rope, act, f(xs), gr, sequences);
            device().context->synchronize();
        }
        return l;
    }
};

TEST(GpuCudaMoe, ABlockWithAnMoeMlpIsCoresToRounding)
{
    block<moe_block_cfg> blk(64, 300);
    const auto b = backend();
    const auto got = blk.run(b, 2, false);
    const auto want = blk.run(b, 2, true);
    const auto mi = blk.mi.back();
    EXPECT_EQ(std::vector<std::uint32_t>(mi.begin(), mi.begin() + 64 * 2), std::vector<std::uint32_t>(blk.mi.host.begin(), blk.mi.host.begin() + 64 * 2));
    near_normwise(blk.m[blk.xs].back(), blk.m[blk.xs].host, 2e-5, "block_backward dxs");
    near_normwise(blk.m[blk.g].back(), blk.m[blk.g].host, 2e-5, "block_backward gradients");
    EXPECT_NEAR(got.balance, want.balance, 1e-5 * std::fabs(want.balance));
    EXPECT_NEAR(got.z, want.z, 1e-5 * std::fabs(want.z));
}

} // namespace
