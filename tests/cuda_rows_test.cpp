//
//  cuda_rows_test.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The key-value cache's operations on CUDA against core's per-token attention on the host: the same checks as
//  opencl_device_test.cpp's for the OpenCL build of the same kernels.
//

#include <snicholls/gpu/cuda_rows.hpp>

#include <snicholls/core/ml/kv_cache.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <vector>

namespace {

namespace gpu = snicholls::gpu;
namespace ml = snicholls::ml;

struct device_t {
    std::unique_ptr<gpu::cuda_context> context;
    std::unique_ptr<gpu::cuda_kernels> kernels;
    std::unique_ptr<gpu::cuda_rows_kernels> rows;
};

const device_t& device()
{
    static const device_t d = [] {
        auto c = gpu::cuda_context::create();
        if (c.is_err()) {
            std::fprintf(stderr, "cuda_rows_test: no CUDA device\n");
            std::abort();
        }
        device_t out{std::move(c).unwrap(), nullptr, nullptr};
        auto k = gpu::cuda_kernels::create(*out.context);
        auto r = gpu::cuda_rows_kernels::create(*out.context);
        if (k.is_err() || r.is_err()) {
            std::fprintf(stderr, "cuda_rows_test: the kernels did not compile\n");
            std::abort();
        }
        out.kernels = std::move(k).unwrap();
        out.rows = std::move(r).unwrap();
        return out;
    }();
    return d;
}

gpu::cuda_training_backend backend()
{
    gpu::cuda_training_backend b{{device().context.get()}, device().kernels.get()};
    b.rows = device().rows.get();
    return b;
}

std::vector<float> random_values(std::size_t n, std::uint64_t seed)
{
    snicholls::stats::wyrand g(seed);
    std::vector<float> v(n);
    for (auto& x : v)
        x = 0.5f * float(2 * snicholls::stats::uniform_real(g) - 1);
    return v;
}

constexpr ml::transformer_config qwenish{.d_model = 64, .n_heads = 4, .n_kv_heads = 2, .head_dim = 16, .d_ff = 96, .norm = ml::norm_kind::rms,
                                         .norm_eps = 1e-6, .qk_norm = ml::qk_norm_kind::before_rope, .mlp = ml::mlp_kind::swiglu,
                                         .rope = ml::rope_kind::half, .rope_base = 1000000, .bias = true};

TEST(GpuCudaRows, AttentionAgainstTheCacheIsCoresPerTokenAttention)
{
    constexpr auto C = qwenish;
    constexpr std::size_t cap = 40, first = 17, n = 6, qn = C.q_size(), dh = C.head_dim;
    const auto keys = random_values(C.n_kv_heads * cap * dh, 40), values = random_values(keys.size(), 41), q = random_values(n * qn, 42);
    std::vector<float> hk = keys, hv = values;
    const ml::kv_cache_view<C, float> view(hk, hv, 1, cap);
    std::vector<float> want(n * qn);
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t h = 0; h < C.n_heads; ++h)
            ml::attention_head<C>(std::span<const float>(q).subspan(t * qn + h * dh, dh),
                                  view.visible(0, h / C.group(), ml::attention_window(first + t, C.window)), std::span<float>(want).subspan(t * qn + h * dh, dh));
    const auto& ctx = *device().context;
    gpu::device_buffer<float> dk(ctx, keys.size()), dv(ctx, values.size()), dq(ctx, q.size()), dy(ctx, n * qn), dl(ctx, n * C.n_heads);
    dk.upload(keys), dv.upload(values), dq.upload(q);
    gpu::attention_attend_cached_rows<C>(backend(), std::span<const float>(dq.span()), std::span<const float>(dk.span()),
                                         std::span<const float>(dv.span()), first, n, cap, dy.span(), dl.span());
    std::vector<float> got(n * qn);
    dy.download(got);
    for (std::size_t i = 0; i < got.size(); ++i)
        EXPECT_NEAR(got[i], want[i], 2e-6f * std::max(1.0f, std::abs(want[i])));

    const auto k_new = random_values(C.n_kv_heads * n * dh, 43), v_new = random_values(k_new.size(), 44);
    gpu::device_buffer<float> dkn(ctx, k_new.size()), dvn(ctx, v_new.size());
    dkn.upload(k_new), dvn.upload(v_new);
    gpu::kv_store_rows<C>(backend(), std::span<const float>(dkn.span()), std::span<const float>(dvn.span()), dk.span(), dv.span(), first, n, cap);
    std::vector<float> stored(keys.size());
    dk.download(stored);
    for (std::size_t h = 0; h < C.n_kv_heads; ++h)
        for (std::size_t t = 0; t < n; ++t)
            for (std::size_t c = 0; c < dh; ++c)
                EXPECT_EQ(stored[(h * cap + first + t) * dh + c], k_new[(h * n + t) * dh + c]);
}

} // namespace
