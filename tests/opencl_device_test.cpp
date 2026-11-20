//
//  opencl_device_test.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/gpu/opencl_device.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/embedding.hpp>
#include <snicholls/core/ml/kv_cache.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/ml/softmax.hpp>
#include <snicholls/core/ml/transformer_block.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <vector>

namespace {

namespace gpu = snicholls::gpu;
namespace ml = snicholls::ml;

struct device {
    std::unique_ptr<gpu::opencl_context> context;
    std::unique_ptr<gpu::opencl_device_kernels> kernels;
    std::unique_ptr<gpu::opencl_arena> arena;

    static device& get()
    {
        static device d = [] {
            auto c = gpu::opencl_context::create(gpu::device_choice::any);
            if (c.is_err()) {
                std::fprintf(stderr, "opencl_device_test: no OpenCL device\n");
                std::abort();
            }
            device out{std::move(c).unwrap(), nullptr, nullptr};
            auto k = gpu::opencl_device_kernels::create(*out.context);
            if (k.is_err()) {
                std::fprintf(stderr, "opencl_device_test: the program did not build:\n%s\n", k.unwrap_err().log.c_str());
                std::abort();
            }
            out.kernels = std::move(k).unwrap();
            out.arena = gpu::opencl_arena::create(*out.context, std::size_t{4} << 30, std::size_t{64} << 20);
            return out;
        }();
        return d;
    }
    [[nodiscard]] gpu::opencl_device_backend backend() const { return {context.get(), arena.get(), kernels.get()}; }
    // A device copy of host values
    template <typename T>
    std::span<T> up(const std::vector<T>& v)
    {
        auto s = arena->allocate<T>(v.size());
        arena->write(s, std::span<const T>(v));
        return s;
    }
    template <typename T>
    std::vector<T> down(std::span<const T> s)
    {
        std::vector<T> v(s.size());
        arena->read(s, std::span<T>(v));
        return v;
    }
};

std::vector<float> random_values(std::size_t n, std::uint64_t seed, float scale = 0.5f, float offset = 0.0f)
{
    snicholls::stats::wyrand g(seed);
    std::vector<float> v(n);
    for (auto& x : v)
        x = offset + scale * float(2 * snicholls::stats::uniform_real(g) - 1);
    return v;
}

void expect_close(const std::vector<float>& got, const std::vector<float>& want, float tol, const char* what)
{
    ASSERT_EQ(got.size(), want.size()) << what;
    float worst = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
        worst = std::max(worst, std::abs(got[i] - want[i]) / std::max(1.0f, std::abs(want[i])));
    EXPECT_LE(worst, tol) << what;
}

TEST(GpuOpenClDevice, TheArenaHoldsWhatIsWrittenAndRefusesForeignSpans)
{
    auto& d = device::get();
    const auto mark = d.arena->mark();
    const std::vector<float> v = random_values(1000, 1);
    const auto s = d.up(v);
    EXPECT_TRUE(d.arena->owns(s.data()));
    EXPECT_EQ(d.down(std::span<const float>(s)), v);
    std::vector<float> host(4);
    EXPECT_FALSE(d.arena->owns(host.data()));
    EXPECT_THROW((void)d.arena->locate(host.data()), std::invalid_argument);
    // A larger allocation than a chunk gets a buffer of its own
    const auto big = d.arena->allocate<float>((std::size_t{64} << 20) / sizeof(float) + 10);
    EXPECT_NE(d.arena->locate(big.data()).mem, d.arena->locate(s.data()).mem);
    d.arena->release(mark);
}

TEST(GpuOpenClDevice, ProductsOverTheArenaAreTheReferencesBits)
{
    auto& d = device::get();
    const auto mark = d.arena->mark();
    const std::size_t n = 37, k = 70, m = 45;
    const auto x = random_values(n * k, 2), w = random_values(m * k, 3), bias = random_values(m, 4);
    std::vector<float> want(n * m);
    ml::linear_sequence(snicholls::linalg::reference, ml::dense_weights<float>{w, m, k}, x, want, bias);
    const auto dx = d.up(x), dw = d.up(w), db = d.up(bias);
    const auto dy = d.arena->allocate<float>(n * m);
    ml::linear_sequence(d.backend(), ml::dense_weights<float>{std::span<const float>(dw), m, k}, std::span<const float>(dx), dy,
                        std::span<const float>(db));
    EXPECT_EQ(d.down(std::span<const float>(dy)), want);
    d.arena->release(mark);
}

// A block of every kind the device runs: grouped-query attention, Qwen 2's biases, Qwen 3's QK norm, RoPE, SwiGLU
constexpr ml::transformer_config qwenish{.d_model = 64, .n_heads = 4, .n_kv_heads = 2, .head_dim = 16, .d_ff = 96, .norm = ml::norm_kind::rms,
                                         .norm_eps = 1e-6, .qk_norm = ml::qk_norm_kind::before_rope, .mlp = ml::mlp_kind::swiglu,
                                         .rope = ml::rope_kind::half, .rope_base = 1000000, .bias = true};
constexpr ml::transformer_config windowed = [] { auto c = qwenish; c.qk_norm = ml::qk_norm_kind::none; c.window = 5; return c; }();

template <ml::transformer_config C>
struct block {
    static constexpr std::size_t dm = C.d_model, qn = C.q_size(), kn = C.kv_size(), f = C.d_ff;
    std::vector<float> n1 = random_values(dm, 10, 0.2f, 1.0f), n2 = random_values(dm, 11, 0.2f, 1.0f), wq = random_values(qn * dm, 12),
                       wk = random_values(kn * dm, 13), wv = random_values(kn * dm, 14), wo = random_values(dm * qn, 15),
                       bq = random_values(qn, 16), bk = random_values(kn, 17), bv = random_values(kn, 18), qnw = random_values(C.head_dim, 19, 0.2f, 1.0f),
                       knw = random_values(C.head_dim, 20, 0.2f, 1.0f), wg = random_values(f * dm, 21), wu = random_values(f * dm, 22),
                       wd = random_values(dm * f, 23);
    // The weights as core's view, over whatever memory `at` gives each tensor
    template <typename At>
    ml::block_weights<ml::dense_weights<float>, float> weights(At&& at) const
    {
        using W = ml::dense_weights<float>;
        ml::block_weights<W, float> w{};
        w.norm1_weight = at(n1), w.norm2_weight = at(n2);
        w.attention.q = W{at(wq), qn, dm}, w.attention.k = W{at(wk), kn, dm}, w.attention.v = W{at(wv), kn, dm}, w.attention.o = W{at(wo), dm, qn};
        w.attention.q_bias = at(bq), w.attention.k_bias = at(bk), w.attention.v_bias = at(bv);
        if constexpr (C.qk_norm != ml::qk_norm_kind::none)
            w.attention.q_norm = at(qnw), w.attention.k_norm = at(knw);
        w.mlp.gate = W{at(wg), f, dm}, w.mlp.up = W{at(wu), f, dm}, w.mlp.down = W{at(wd), dm, f};
        return w;
    }
};

template <ml::transformer_config C>
ml::block_activations<float> activations(std::size_t n, auto&& alloc)
{
    constexpr std::size_t dm = C.d_model, qn = C.q_size(), kn = C.kv_size(), f = C.d_ff;
    ml::block_activations<float> a{};
    a.x0 = alloc(n * dm), a.h1 = alloc(n * dm), a.x1 = alloc(n * dm), a.h2 = alloc(n * dm);
    a.attention.q = alloc(n * qn), a.attention.k = alloc(n * kn), a.attention.v = alloc(n * kn), a.attention.y = alloc(n * qn);
    a.attention.lse = alloc(n * C.n_heads);
    if constexpr (C.qk_norm != ml::qk_norm_kind::none)
        a.attention.q_raw = alloc(n * qn), a.attention.k_raw = alloc(n * kn);
    a.mlp_up = alloc(n * f), a.mlp_gate = alloc(n * f), a.mlp_hidden = alloc(n * f);
    return a;
}

template <ml::transformer_config C>
void block_agrees(std::size_t n, std::size_t sequences)
{
    auto& d = device::get();
    const auto mark = d.arena->mark();
    const block<C> blk;
    static const ml::rope_table<float, 64, C.head_dim> rope(float(C.rope_base));
    const std::size_t rows = n * sequences;
    // The host: core's reference backend
    std::vector<std::vector<float>> host_store;
    const auto host_alloc = [&](std::size_t k) { host_store.emplace_back(k); return std::span<float>(host_store.back()); };
    host_store.reserve(64);
    auto x = random_values(rows * C.d_model, 30);
    const auto hw = blk.weights([](const std::vector<float>& v) { return std::span<const float>(v); });
    ml::block_forward_sequence<C>(snicholls::linalg::reference, hw, std::span<float>(x), rope, activations<C>(rows, host_alloc), sequences);
    // The device
    const auto dx = d.up(random_values(rows * C.d_model, 30));
    const auto dw = blk.weights([&](const std::vector<float>& v) { return std::span<const float>(d.up(v)); });
    const auto dev_alloc = [&](std::size_t k) { return d.arena->allocate<float>(k); };
    ml::block_forward_sequence<C>(d.backend(), dw, dx, rope, activations<C>(rows, dev_alloc), sequences);
    expect_close(d.down(std::span<const float>(dx)), x, 2e-5f, "block output");
    d.arena->release(mark);
}

TEST(GpuOpenClDevice, ABlocksForwardPassOnTheDeviceIsTheHostsWithinRounding)
{
    block_agrees<qwenish>(23, 1);
    block_agrees<qwenish>(9, 3);   // three sequences side by side, attention within each
    block_agrees<windowed>(20, 2);
}

// The cache: attention of new tokens against stored keys is core's per-token attention over kv_cache_view
TEST(GpuOpenClDevice, AttentionAgainstTheCacheIsCoresPerTokenAttention)
{
    auto& d = device::get();
    const auto mark = d.arena->mark();
    constexpr auto C = qwenish;
    constexpr std::size_t cap = 40, first = 17, n = 6, qn = C.q_size(), dh = C.head_dim;
    const auto keys = random_values(C.n_kv_heads * cap * dh, 40), values = random_values(keys.size(), 41), q = random_values(n * qn, 42);
    // The host: core's attention_head over the view's visible keys, query by query
    std::vector<float> hk = keys, hv = values;
    const ml::kv_cache_view<C, float> view(hk, hv, 1, cap);
    std::vector<float> want(n * qn);
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t h = 0; h < C.n_heads; ++h)
            ml::attention_head<C>(std::span<const float>(q).subspan(t * qn + h * dh, dh),
                                  view.visible(0, h / C.group(), ml::attention_window(first + t, C.window)), std::span<float>(want).subspan(t * qn + h * dh, dh));
    const auto dk = d.up(keys), dv = d.up(values), dq = d.up(q);
    const auto dy = d.arena->allocate<float>(n * qn), dl = d.arena->allocate<float>(n * C.n_heads);
    gpu::attention_attend_cached_rows<C>(d.backend(), std::span<const float>(dq), std::span<const float>(dk), std::span<const float>(dv), first, n, cap,
                                         dy, dl);
    expect_close(d.down(std::span<const float>(dy)), want, 2e-6f, "attention");
    EXPECT_THROW(gpu::attention_attend_cached_rows<C>(d.backend(), std::span<const float>(dq), std::span<const float>(dk), std::span<const float>(dv),
                                                      cap - 2, n, cap, dy, dl),
                 std::invalid_argument);

    // Storing: [kvh][n][d] rows into [kvh][cap][d] at first
    const auto k_new = random_values(C.n_kv_heads * n * dh, 43), v_new = random_values(k_new.size(), 44);
    gpu::kv_store_rows<C>(d.backend(), std::span<const float>(d.up(k_new)), std::span<const float>(d.up(v_new)), dk, dv, first, n, cap);
    const auto got = d.down(std::span<const float>(dk));
    for (std::size_t h = 0; h < C.n_kv_heads; ++h)
        for (std::size_t t = 0; t < n; ++t)
            for (std::size_t c = 0; c < dh; ++c)
                EXPECT_EQ(got[(h * cap + first + t) * dh + c], k_new[(h * n + t) * dh + c]);
    EXPECT_EQ(got[0], keys[0]);   // the rest untouched
    d.arena->release(mark);
}

TEST(GpuOpenClDevice, EmbeddingsCrossEntropyAndSumsAreTheHosts)
{
    auto& d = device::get();
    const auto mark = d.arena->mark();
    constexpr std::size_t v = 300, dm = 8, n = 5, classes = 290;
    const auto table = random_values(v * dm, 50, 3.0f);
    const std::vector<std::uint32_t> ids{4, 299, 0, 4, 17};
    std::vector<float> want(n * dm);
    ml::embed_rows(snicholls::linalg::reference, std::span<const float>(table), std::span<const float>{}, ids, std::span<float>(want));
    const auto dt = d.up(table);
    const auto di = d.up(ids);
    const auto dout = d.arena->allocate<float>(n * dm);
    gpu::embed_rows(d.backend(), std::span<const float>(dt), {}, std::span<const std::uint32_t>(di), dout);
    EXPECT_EQ(d.down(std::span<const float>(dout)), want);

    auto logits = random_values(n * v, 51, 6.0f);
    const auto dlog = d.up(logits);
    std::vector<float> losses(n);
    const std::vector<std::uint32_t> targets{1, 2, 289, 100, 0};
    ml::cross_entropy_rows(snicholls::linalg::reference, std::span<float>(logits), targets, std::span<float>(losses), 0.5f, classes);
    const auto dl = d.arena->allocate<float>(n);
    gpu::cross_entropy_rows(d.backend(), dlog, std::span<const std::uint32_t>(d.up(targets)), dl, 0.5f, classes);
    expect_close(d.down(std::span<const float>(dl)), losses, 1e-5f, "losses");
    expect_close(d.down(std::span<const float>(dlog)), logits, 1e-6f, "gradients");

    const auto x = random_values(10000, 52);
    double s = 0, s2 = 0;
    for (const float e : x)
        s += e, s2 += double(e) * e;
    const auto dxs = d.up(x);
    EXPECT_NEAR(gpu::sum_values(d.backend(), std::span<const float>(dxs)), s, 1e-3);
    EXPECT_NEAR(gpu::sum_squares(d.backend(), std::span<const float>(dxs)), s2, 1e-2);
    gpu::fill_rows(d.backend(), dxs, 2.5f);
    EXPECT_EQ(d.down(std::span<const float>(dxs)), std::vector<float>(x.size(), 2.5f));
    d.arena->release(mark);
}

} // namespace
