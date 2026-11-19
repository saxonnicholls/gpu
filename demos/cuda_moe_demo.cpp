//
//  cuda_moe_demo.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/gpu/cuda_moe.hpp>: a mixture-of-experts layer (16 experts, 4 per token, DeepSeek-V3's sigmoid
//  scoring with a selection bias and one shared expert) through core's ml::moe_forward and ml::moe_backward with every
//  weight and activation in device memory, against core's reference on the host; the bias rule on the device; the same
//  pass captured as a CUDA graph and replayed. Exits non-zero if a result is not what it documents, or if there is no
//  CUDA device.
//

#include <snicholls/gpu/cuda_moe.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/moe.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace gpu = snicholls::gpu;
namespace ml = snicholls::ml;

// A host vector and its copy in device memory
template <typename T>
struct tensor {
    std::vector<T> host;
    std::unique_ptr<gpu::device_buffer<T>> dev;
    tensor(const gpu::cuda_context& c, std::vector<T> h) : host(std::move(h)), dev(std::make_unique<gpu::device_buffer<T>>(c, host.size()))
    {
        dev->upload(host);
    }
    std::vector<T> back() const
    {
        std::vector<T> out(host.size());
        dev->download(out);
        return out;
    }
};

std::vector<float> wave(std::size_t n, float scale, float phase)
{
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = scale * std::sin(0.61f * static_cast<float>(i) + phase) * std::cos(0.0173f * static_cast<float>(i * i % 977));
    return v;
}

double normwise(const std::vector<float>& got, const std::vector<float>& want)
{
    double diff = 0, scale = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
        diff = std::max(diff, double(std::fabs(got[i] - want[i]))), scale = std::max(scale, double(std::fabs(want[i])));
    return scale == 0 ? diff : diff / scale;
}

} // namespace

int main()
{
    std::printf("snicholls-gpu cuda_moe demo\n");
    constexpr ml::transformer_config c{.d_model = 64, .n_heads = 4, .n_kv_heads = 4, .head_dim = 16, .d_ff = 96, .norm = ml::norm_kind::rms,
                                       .mlp = ml::mlp_kind::swiglu, .experts = 16, .top_k = 4, .shared_experts = 1, .expert_d_ff = 96,
                                       .scoring = ml::moe_scoring::sigmoid, .route_scale = 2.5, .bias_update_rate = 0.001,
                                       .balance_loss_weight = 0.0001, .z_loss_weight = 0.001};
    constexpr std::size_t rows = 256, sequences = 4, d = c.d_model;

    //! [moe]
    auto ctx = gpu::cuda_context::create();
    if (ctx.is_err()) {
        std::printf("cuda_moe demo: FAILED - no CUDA device\n");
        return 1;
    }
    const auto context = std::move(ctx).unwrap();
    auto dense = gpu::cuda_kernels::create(*context);    // the dense steps: products, norms, activations
    auto moe = gpu::cuda_moe_kernels::create(*context);  // routing, the permutation, the grouped products, the balance terms
    if (dense.is_err() || moe.is_err()) {
        std::printf("cuda_moe demo: FAILED - the kernels did not compile\n");
        return 1;
    }
    const auto kernels = std::move(dense).unwrap();
    const auto moe_kernels = std::move(moe).unwrap();
    gpu::cuda_training_backend device{{context.get()}, kernels.get()};
    device.moe = moe_kernels.get();

    // The layer's parameters, its selection bias, its input, and every buffer core's sizes say
    tensor<float> params(*context, wave(ml::moe_parameter_count(c), 0.2f, 0.0f)), bias(*context, wave(c.experts, 0.05f, 1.0f)),
        xs(*context, wave(rows * d, 1.0f, 2.0f)), ys(*context, std::vector<float>(rows * d)), values(*context, std::vector<float>(ml::moe_activation_values(c, rows))),
        dys(*context, wave(rows * d, 1.0f, 3.0f)), dxs(*context, std::vector<float>(rows * d)), grads(*context, std::vector<float>(ml::moe_parameter_count(c))),
        work(*context, std::vector<float>(ml::moe_gradient_work(c, rows)));
    tensor<std::uint32_t> indices(*context, std::vector<std::uint32_t>(ml::moe_activation_indices(c, rows)));
    auto pass = [&](const auto& backend, bool on_device) {
        auto s = [&](auto& t) { return on_device ? t.dev->span() : std::span(t.host); };
        const auto w = ml::moe_weights_in<c>(std::span<const float>(s(params)), std::span<const float>(s(bias)));
        const auto act = ml::moe_activations_in<c>(s(values), s(indices), rows);
        const auto losses = ml::moe_forward<c>(backend, w, std::span<const float>(s(xs)), s(ys), act, sequences);
        ml::moe_backward<c>(backend, w, std::span<const float>(s(xs)), act, std::span<const float>(s(dys)), s(dxs),
                            ml::moe_gradients_in<c>(s(grads), s(work), rows), sequences);
        return losses;
    };
    const auto on_gpu = pass(device, true);                                   // every step on the device
    const auto on_host = pass(snicholls::linalg::reference, false);           // core's reference
    //! [moe]

    context->synchronize();
    const auto chosen = indices.back();
    const bool same_route = std::equal(chosen.begin(), chosen.begin() + rows * c.top_k, indices.host.begin());
    const double ey = normwise(ys.back(), ys.host), ex = normwise(dxs.back(), dxs.host), eg = normwise(grads.back(), grads.host);
    std::printf("  %zu rows, %zu experts, %zu per token: the device chose the host's experts: %s\n", rows, c.experts, c.top_k, same_route ? "yes" : "NO");
    std::printf("  outputs %.1e, input gradient %.1e, weight gradients %.1e (normwise against the host)\n", ey, ex, eg);
    std::printf("  balance loss %.6g (host %.6g), z-loss %.6g (host %.6g)\n", on_gpu.balance, on_host.balance, on_gpu.z, on_host.z);
    bool ok = same_route && ey < 1e-5 && ex < 1e-5 && eg < 1e-5 && std::fabs(on_gpu.balance - on_host.balance) <= 1e-5 * on_host.balance
              && std::fabs(on_gpu.z - on_host.z) <= 1e-5 * on_host.z;

    // The selection bias moves toward balance: experts above the mean load less likely, below it more
    //! [bias]
    const auto act = ml::moe_activations_in<c>(values.dev->span(), indices.dev->span(), rows);
    gpu::moe_update_bias<c>(device, act.counts, bias.dev->span());
    //! [bias]
    std::vector<std::uint32_t> counts(c.experts);
    ml::moe_count_rows<c>(snicholls::linalg::reference, std::span<const std::uint32_t>(indices.host).first(rows * c.top_k), std::span<std::uint32_t>(counts));
    auto want = bias.host;
    ml::moe_update_bias<c>(snicholls::linalg::reference, std::span<const std::uint32_t>(counts), std::span<float>(want));
    const bool bias_ok = bias.back() == want;
    std::printf("  the bias rule on the device: core's bits: %s\n", bias_ok ? "yes" : "NO");
    ok = ok && bias_ok;
    bias.dev->upload(bias.host);

    // The same pass as one graph: nothing in it waits for the host, the losses stay on the device
    //! [graph]
    const auto grads_before = grads.back(), dxs_before = dxs.back();
    moe_kernels->losses_on_device = true;
    auto made = gpu::cuda_graph::capture(*context, [&] { pass(device, true); });
    moe_kernels->losses_on_device = false;
    //! [graph]
    if (made.is_err()) {
        std::printf("cuda_moe demo: FAILED - the capture failed\n");
        return 1;
    }
    const auto graph = std::move(made).unwrap();
    const auto ys_eager = ys.back();
    grads.dev->upload(grads_before);
    dxs.dev->upload(dxs_before);
    graph.launch();
    context->synchronize();
    const auto g2 = grads.back();
    grads.dev->upload(grads_before);
    dxs.dev->upload(dxs_before);
    pass(device, true);   // queued call by call, from the same state
    const bool replay = ys.back() == ys_eager && grads.back() == g2;
    std::printf("  a captured graph replays the queued pass's bits: %s\n", replay ? "yes" : "NO");
    ok = ok && replay;

    std::printf("cuda_moe demo: %s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
