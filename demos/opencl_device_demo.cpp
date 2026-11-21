//
//  opencl_device_demo.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/gpu/opencl_device.hpp>: a transformer block (grouped-query attention, RoPE, SwiGLU) runs
//  through core's own block_forward_sequence with every operand on an OpenCL device, then the same block's attention
//  for four new tokens against a key-value cache; each is checked against core's reference backend on the host.
//

#include <snicholls/gpu/opencl_device.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/ml/transformer_block.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

namespace {
std::vector<float> values(std::size_t n, std::uint64_t seed, float offset = 0)
{
    snicholls::stats::wyrand g(seed);
    std::vector<float> v(n);
    for (auto& x : v)
        x = offset + 0.4f * float(2 * snicholls::stats::uniform_real(g) - 1);
    return v;
}
} // namespace

int main()
{
    namespace gpu = snicholls::gpu;
    namespace ml = snicholls::ml;
    std::printf("snicholls-gpu opencl_device demo\n");
    auto made = gpu::opencl_context::create(gpu::device_choice::any);
    if (made.is_err()) {
        std::printf("opencl_device demo: FAILED - no OpenCL device\n");
        return 1;
    }
    const auto context = std::move(made).unwrap();
    auto k = gpu::opencl_device_kernels::create(*context);
    if (k.is_err()) {
        std::printf("opencl_device demo: FAILED - %s\n", k.unwrap_err().log.c_str());
        return 1;
    }
    const auto kernels = std::move(k).unwrap();

    //! [device]
    const auto arena = gpu::opencl_arena::create(*context, std::size_t{1} << 30, std::size_t{16} << 20);
    const gpu::opencl_device_backend b{context.get(), arena.get(), kernels.get()};
    // Host values to device spans: the spans are addresses in the arena, as core's passes take them
    const auto up = [&](const std::vector<float>& v) {
        auto s = arena->allocate<float>(v.size());
        arena->write(s, std::span<const float>(v));
        return s;
    };
    constexpr ml::transformer_config c{.d_model = 32, .n_heads = 4, .n_kv_heads = 2, .head_dim = 8, .d_ff = 48, .norm = ml::norm_kind::rms,
                                       .norm_eps = 1e-5, .mlp = ml::mlp_kind::swiglu, .rope = ml::rope_kind::half, .rope_base = 10000};
    constexpr std::size_t n = 12, d = c.d_model, qn = c.q_size(), kn = c.kv_size(), f = c.d_ff, h = c.n_heads;
    const auto n1 = values(d, 1, 1), n2 = values(d, 2, 1), wq = values(qn * d, 3), wk = values(kn * d, 4), wv = values(kn * d, 5), wo = values(d * qn, 6),
               wg = values(f * d, 7), wu = values(f * d, 8), wd = values(d * f, 9), x = values(n * d, 10);
    using W = ml::dense_weights<float>;
    const auto weights = [&](auto&& at) {
        ml::block_weights<W, float> w{};
        w.norm1_weight = at(n1), w.norm2_weight = at(n2);
        w.attention.q = W{at(wq), qn, d}, w.attention.k = W{at(wk), kn, d}, w.attention.v = W{at(wv), kn, d}, w.attention.o = W{at(wo), d, qn};
        w.mlp.gate = W{at(wg), f, d}, w.mlp.up = W{at(wu), f, d}, w.mlp.down = W{at(wd), d, f};
        return w;
    };
    const auto acts = [&](auto&& alloc) {
        ml::block_activations<float> a{};
        a.x0 = alloc(n * d), a.h1 = alloc(n * d), a.x1 = alloc(n * d), a.h2 = alloc(n * d);
        a.attention.q = alloc(n * qn), a.attention.k = alloc(n * kn), a.attention.v = alloc(n * kn), a.attention.y = alloc(n * qn);
        a.attention.lse = alloc(n * h), a.mlp_up = alloc(n * f), a.mlp_gate = alloc(n * f), a.mlp_hidden = alloc(n * f);
        return a;
    };
    static const ml::rope_table<float, 16, c.head_dim> rope(10000.0f);
    const auto dx = up(x);
    ml::block_forward_sequence<c>(b, weights([&](const std::vector<float>& v) { return std::span<const float>(up(v)); }), dx, rope,
                                  acts([&](std::size_t m) { return arena->allocate<float>(m); }));
    std::vector<float> got(n * d);
    arena->read(std::span<const float>(dx), std::span<float>(got));
    //! [device]

    std::vector<std::vector<float>> store;
    store.reserve(32);
    auto hx = x;
    ml::block_forward_sequence<c>(snicholls::linalg::reference, weights([](const std::vector<float>& v) { return std::span<const float>(v); }),
                                  std::span<float>(hx), rope, acts([&](std::size_t m) { store.emplace_back(m); return std::span<float>(store.back()); }));
    float worst = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
        worst = std::max(worst, std::abs(got[i] - hx[i]));
    std::printf("  a block of %zu tokens on %s: largest difference from the host %.2e\n", n, context->device_name().c_str(), double(worst));
    const bool ok = worst < 1e-5f;
    std::printf(ok ? "opencl_device demo: ok\n" : "opencl_device demo: FAILED\n");
    return ok ? 0 : 1;
}
