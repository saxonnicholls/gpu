//
//  cuda_ops_demo.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/gpu/cuda_ops.hpp>: a GPT-2-shaped block's forward pass through core's batched
//  ml::block_forward_sequence with every weight and activation in device memory, so each step runs on the GPU, against
//  core's reference on the host. Exits non-zero if a result is not what it documents, or if there is no CUDA device.
//

#include <snicholls/gpu/cuda_ops.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/transformer_block.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

int main()
{
    std::printf("snicholls-gpu cuda_ops demo\n");
    namespace gpu = snicholls::gpu;
    namespace ml = snicholls::ml;
    constexpr ml::transformer_config c{.d_model = 128, .n_heads = 4, .n_kv_heads = 4, .head_dim = 32, .d_ff = 512,
                                       .norm = ml::norm_kind::layer, .norm_eps = 1e-5, .mlp = ml::mlp_kind::gelu_tanh, .bias = true};
    constexpr std::size_t n = 64, d = c.d_model, f = c.d_ff;

    //! [cuda_ops]
    auto ctx = gpu::cuda_context::create();
    if (ctx.is_err()) {
        std::printf("cuda_ops demo: FAILED - no CUDA device\n");
        return 1;
    }
    const auto context = std::move(ctx).unwrap();
    auto made = gpu::cuda_kernels::create(*context);   // the training steps' kernels, compiled for this device
    if (made.is_err()) {
        std::printf("cuda_ops demo: FAILED - the kernels did not compile\n");
        return 1;
    }
    const auto kernels = std::move(made).unwrap();
    const gpu::cuda_training_backend device{{context.get()}, kernels.get()};

    // Every tensor once on the host and once on the device
    struct tensor {
        std::vector<float> host;
        std::unique_ptr<gpu::device_buffer<float>> dev;
    };
    std::vector<tensor> t;
    auto make = [&](std::size_t size, float scale, float offset) {
        tensor x{std::vector<float>(size), nullptr};
        for (std::size_t i = 0; i < size; ++i)
            x.host[i] = offset + scale * std::sin(0.37f * static_cast<float>(i) + static_cast<float>(t.size()));
        x.dev = std::make_unique<gpu::device_buffer<float>>(*context, size);
        x.dev->upload(x.host);
        t.push_back(std::move(x));
        return t.size() - 1;
    };
    const std::size_t n1w = make(d, 0.1f, 1), n1b = make(d, 0.1f, 0), wq = make(d * d, 0.1f, 0), wk = make(d * d, 0.1f, 0), wv = make(d * d, 0.1f, 0),
                      wo = make(d * d, 0.1f, 0), bq = make(d, 0.1f, 0), bk = make(d, 0.1f, 0), bv = make(d, 0.1f, 0), bo = make(d, 0.1f, 0),
                      n2w = make(d, 0.1f, 1), n2b = make(d, 0.1f, 0), up = make(f * d, 0.1f, 0), bup = make(f, 0.1f, 0), down = make(d * f, 0.05f, 0),
                      bdown = make(d, 0.1f, 0), xs = make(n * d, 1, 0), x0 = make(n * d, 0, 0), h1 = make(n * d, 0, 0), x1 = make(n * d, 0, 0),
                      h2 = make(n * d, 0, 0), q = make(n * d, 0, 0), k = make(n * d, 0, 0), v = make(n * d, 0, 0), y = make(n * d, 0, 0),
                      lse = make(n * c.n_heads, 0, 0), aup = make(n * f, 0, 0), hid = make(n * f, 0, 0);
    auto run = [&](const auto& backend, bool on_device) {
        auto s = [&](std::size_t i) { return on_device ? t[i].dev->span() : std::span<float>(t[i].host); };
        using W = ml::dense_weights<float>;
        ml::block_weights<W> w{};
        w.norm1_weight = s(n1w), w.norm1_bias = s(n1b), w.norm2_weight = s(n2w), w.norm2_bias = s(n2b);
        w.attention = {W{s(wq), d, d}, W{s(wk), d, d}, W{s(wv), d, d}, W{s(wo), d, d}, s(bq), s(bk), s(bv), s(bo)};
        w.mlp = {W{s(up), f, d}, W{}, W{s(down), d, f}, s(bup), {}, s(bdown)};
        const ml::block_activations<float> act{s(x0), s(h1), s(x1), s(h2), {s(q), s(k), s(v), {}, {}, s(y), s(lse)}, s(aup), {}, s(hid)};
        ml::block_forward_sequence<c>(backend, w, s(xs), ml::no_rope, act);
    };
    run(device, true);                        // the norms, residuals, activation and products on the GPU
    run(snicholls::linalg::reference, false); // core's reference on the host copies
    std::vector<float> got(n * d);
    t[xs].dev->download(got);
    //! [cuda_ops]

    float worst = 0.0f;
    for (std::size_t i = 0; i < got.size(); ++i)
        worst = std::fmax(worst, std::fabs(got[i] - t[xs].host[i]) / (1 + std::fabs(t[xs].host[i])));
    std::printf("  device: %s\n  a block's forward pass over %zu tokens on the device: within %.2g of core's reference\n",
                context->device_name().c_str(), n, static_cast<double>(worst));
    const bool ok = worst < 1e-4f;
    std::printf(ok ? "cuda_ops demo: ok\n" : "cuda_ops demo: FAILED\n");
    return ok ? 0 : 1;
}
