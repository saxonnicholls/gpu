//
//  cuda_rows_demo.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/gpu/cuda_rows.hpp>: four new tokens attend to a key-value cache on a CUDA device with the
//  kernels opencl_device.hpp builds for OpenCL, checked against core's per-token attention on the host.
//

#include <snicholls/gpu/cuda_rows.hpp>

#include <snicholls/core/ml/kv_cache.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    namespace gpu = snicholls::gpu;
    namespace ml = snicholls::ml;
    std::printf("snicholls-gpu cuda_rows demo\n");
    auto c = gpu::cuda_context::create();
    if (c.is_err()) {
        std::printf("cuda_rows demo: FAILED - no CUDA device\n");
        return 1;
    }
    const auto context = std::move(c).unwrap();
    auto k = gpu::cuda_kernels::create(*context);
    auto r = gpu::cuda_rows_kernels::create(*context);
    if (k.is_err() || r.is_err()) {
        std::printf("cuda_rows demo: FAILED - the kernels did not compile\n");
        return 1;
    }
    const auto kernels = std::move(k).unwrap();
    const auto rows = std::move(r).unwrap();

    //! [rows]
    gpu::cuda_training_backend b{{context.get()}, kernels.get()};
    b.rows = rows.get();
    constexpr ml::transformer_config cfg{.d_model = 32, .n_heads = 4, .n_kv_heads = 2, .head_dim = 8, .d_ff = 48, .norm = ml::norm_kind::rms,
                                         .norm_eps = 1e-5, .mlp = ml::mlp_kind::swiglu, .rope = ml::rope_kind::half, .rope_base = 10000};
    constexpr std::size_t cap = 16, first = 6, n = 4, qn = cfg.q_size(), dh = cfg.head_dim;
    snicholls::stats::wyrand g(2026);
    std::vector<float> keys(cfg.n_kv_heads * cap * dh), values(keys.size()), q(n * qn);
    for (auto* v : {&keys, &values, &q})
        for (auto& x : *v)
            x = float(snicholls::stats::uniform_real(g)) - 0.5f;
    gpu::device_buffer<float> dk(*context, keys.size()), dv(*context, values.size()), dq(*context, q.size()), dy(*context, n * qn),
        dl(*context, n * cfg.n_heads);
    dk.upload(keys), dv.upload(values), dq.upload(q);
    gpu::attention_attend_cached_rows<cfg>(b, std::span<const float>(dq.span()), std::span<const float>(dk.span()), std::span<const float>(dv.span()),
                                           first, n, cap, dy.span(), dl.span());
    std::vector<float> y(n * qn);
    dy.download(y);
    //! [rows]

    const ml::kv_cache_view<cfg, float> view(keys, values, 1, cap);
    float worst = 0;
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t h = 0; h < cfg.n_heads; ++h) {
            std::vector<float> want(dh);
            ml::attention_head<cfg>(std::span<const float>(q).subspan(t * qn + h * dh, dh), view.visible(0, h / cfg.group(), ml::attention_window(first + t, 0)),
                                    std::span<float>(want));
            for (std::size_t i = 0; i < dh; ++i)
                worst = std::max(worst, std::abs(want[i] - y[t * qn + h * dh + i]));
        }
    std::printf("  four queries against the cache: largest difference from the host %.2e\n", double(worst));
    const bool ok = worst < 1e-5f;
    std::printf(ok ? "cuda_rows demo: ok\n" : "cuda_rows demo: FAILED\n");
    return ok ? 0 : 1;
}
