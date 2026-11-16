//
//  cuda_ops_benchmark.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  A GPT-2-small layer's training step (forward and backward over 128 tokens) through core's batched passes with every
//  parameter, activation and gradient in device memory (cuda_training_backend: the products by cuBLAS, the other
//  steps by its kernels, attention still staged), against the same step from host memory (cuda_backend: the products
//  staged, the other steps on the host). Different paths in float, summing in different orders (the device's norms and
//  attention reduce across a warp), so main() checks they agree normwise: the largest difference within 1e-4 of each
//  tensor's largest value (measured 2e-7 to 1.8e-5, the input gradient through the layer norms' backward the largest).
//  Which of the two is closer to the exact result needs a double-precision reference, not yet here.
//
//  Then GPT-2 small's attention alone (12 heads of 64, causal, forward and backward) over 256, 1024 and 4096 tokens on
//  each device path: the matrix products in float, the same in TF32, and flash attention on the tensor cores (TF32).
//  main() first checks, at 1024 tokens, that the float path agrees with core's reference on the host (normwise 1e-4)
//  and the TF32 paths with the float path (5e-3).
//

#include <snicholls/gpu/cuda_ops.hpp>

#include "../tests/support/block_step.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace gpu = snicholls::gpu;
namespace ml = snicholls::ml;
using step = gpu::testing::step<ml::gpt2_small, 128>;

struct device_t {
    std::unique_ptr<gpu::cuda_context> context;
    std::unique_ptr<gpu::cuda_kernels> kernels;
};

const device_t& device()
{
    static const device_t d = [] {
        auto c = gpu::cuda_context::create();
        if (c.is_err())
            std::exit(1);
        device_t out{std::move(c).unwrap(), nullptr};
        auto k = gpu::cuda_kernels::create(*out.context);
        if (k.is_err())
            std::exit(1);
        out.kernels = std::move(k).unwrap();
        return out;
    }();
    return d;
}

template <typename B>
void train(step& st, const B& backend, bool dev)
{
    const auto w = st.weights(dev);
    ml::block_forward_sequence<ml::gpt2_small>(backend, w, st.s(step::xs, dev), ml::no_rope, st.act(dev));
    ml::block_backward<ml::gpt2_small>(backend, w, ml::no_rope, st.act(dev), st.s(step::dxs, dev), st.grads(dev));
    cudaDeviceSynchronize();
}

void BM_Resident(benchmark::State& state)
{
    step st(*device().context);
    const gpu::cuda_training_backend b{{device().context.get()}, device().kernels.get()};
    for (auto _ : state)
        train(st, b, true);   // the same step each time (the inputs are overwritten in place: the work is the same)
    state.SetItemsProcessed(state.iterations() * std::int64_t(step::n));
}
BENCHMARK(BM_Resident)->Name("BM_TrainGpt2SmallLayer_DeviceResident")->Unit(benchmark::kMillisecond);

void BM_Staged(benchmark::State& state)
{
    step st(*device().context);
    const gpu::cuda_backend b{device().context.get()};
    for (auto _ : state)
        train(st, b, false);
    state.SetItemsProcessed(state.iterations() * std::int64_t(step::n));
}
BENCHMARK(BM_Staged)->Name("BM_TrainGpt2SmallLayer_HostStaged")->Unit(benchmark::kMillisecond);

// Attention's activations and gradients for n tokens of GPT-2 small, on the device, with host copies
struct attention_case {
    static constexpr std::size_t d = ml::gpt2_small.head_dim, h = ml::gpt2_small.n_heads, w = h * d;
    std::size_t n;
    std::vector<gpu::testing::mirror> m;   // q, k, v, y, lse, dy, gq, gk, gv
    explicit attention_case(std::size_t tokens) : n(tokens)
    {
        const auto& c = *device().context;
        for (std::uint64_t i = 0; i < 3; ++i)
            m.emplace_back(c, gpu::testing::stream(n * w, 100 + i, 1.0f));
        m.emplace_back(c, std::vector<float>(n * w));
        m.emplace_back(c, std::vector<float>(n * h));
        m.emplace_back(c, gpu::testing::stream(n * w, 200, 1.0f));
        for (int i = 0; i < 3; ++i)
            m.emplace_back(c, std::vector<float>(n * w));
    }
    std::span<float> s(std::size_t i, bool dev) { return dev ? m[i].d() : std::span<float>(m[i].host); }
    ml::attention_activations<float> act(bool dev) { return {s(0, dev), s(1, dev), s(2, dev), {}, {}, s(3, dev), s(4, dev)}; }
    ml::attention_gradients<float> grads(bool dev) { return {s(6, dev), s(7, dev), s(8, dev), {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}}; }
    template <typename B>
    void run(const B& b, bool dev)
    {
        using ml::attention_attend_rows, ml::attention_attend_rows_backward, ml::fill_rows;
        fill_rows(b, s(7, dev), 0.0f);   // the keys' and values' gradients are added into
        fill_rows(b, s(8, dev), 0.0f);
        attention_attend_rows<ml::gpt2_small>(b, act(dev), n);
        attention_attend_rows_backward<ml::gpt2_small>(b, act(dev), std::span<const float>(s(5, dev)), grads(dev), n);
        if (dev)
            cudaDeviceSynchronize();
    }
};

enum class path { gemm_float, gemm_tf32, flash_tf32 };
gpu::cuda_training_backend backend_for(path p)
{
    gpu::cuda_training_backend b{{device().context.get()}, device().kernels.get()};
    b.math = p == path::gemm_float ? gpu::cuda_math::float32 : gpu::cuda_math::tf32;
    b.flash_attention = p == path::flash_tf32;
    return b;
}

void BM_Attention(benchmark::State& state)
{
    attention_case a(std::size_t(state.range(0)));
    const auto b = backend_for(path(state.range(1)));
    for (auto _ : state)
        a.run(b, true);
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_Attention)
    ->Name("BM_Gpt2SmallAttention_ForwardBackward")
    ->ArgNames({"tokens", "path"})   // path 0: matrix products in float, 1: the same in TF32, 2: flash attention (TF32)
    ->ArgsProduct({{256, 1024, 4096}, {0, 1, 2}})
    ->Unit(benchmark::kMillisecond);

double normwise(const std::vector<float>& got, const std::vector<float>& want)
{
    double diff = 0, scale = 0;
    for (std::size_t j = 0; j < got.size(); ++j)
        diff = std::max(diff, double(std::fabs(got[j] - want[j]))), scale = std::max(scale, double(std::fabs(want[j])));
    return scale == 0 ? 0 : diff / scale;
}

bool attention_fair()
{
    constexpr std::size_t n = 1024;
    attention_case ref(n), f(n);
    ref.run(snicholls::linalg::reference, false);
    f.run(backend_for(path::gemm_float), true);
    bool ok = true;
    for (const std::size_t i : {3u, 6u, 7u, 8u}) {
        const double e = normwise(f.m[i].back(), ref.m[i].host);
        std::printf("  attention, float against the host: tensor %zu normwise %.2g\n", i, e);
        ok &= e <= 1e-4;
    }
    for (const path p : {path::gemm_tf32, path::flash_tf32}) {
        attention_case t(n);
        t.run(backend_for(p), true);
        for (const std::size_t i : {3u, 6u, 7u, 8u}) {
            const double e = normwise(t.m[i].back(), f.m[i].back());
            std::printf("  attention, path %d against float: tensor %zu normwise %.2g\n", int(p), i, e);
            ok &= e <= 5e-3;
        }
    }
    return ok;
}

bool fair()
{
    step a(*device().context), b(*device().context);
    train(a, gpu::cuda_training_backend{{device().context.get()}, device().kernels.get()}, true);
    train(b, gpu::cuda_backend{device().context.get()}, false);
    // Normwise: the largest difference over the tensor's largest value. The two paths' float arithmetic differs (the
    // device's expf and erfcf, orders of summation), and a layer norm's backward amplifies last-bit differences by
    // cancellation, so an elementwise relative bound is the wrong test for a deep float pipeline
    for (const int i : {int(step::xs), int(step::dxs), int(step::gwq), int(step::gwo), int(step::gup), int(step::gdown), int(step::gn1w)}) {
        const auto got = a.m[std::size_t(i)].back();
        const auto& want = b.m[std::size_t(i)].host;
        float diff = 0.0f, scale = 0.0f;
        for (std::size_t j = 0; j < got.size(); ++j) {
            diff = std::fmax(diff, std::fabs(got[j] - want[j]));
            scale = std::fmax(scale, std::fabs(want[j]));
        }
        std::printf("  tensor %2d: normwise difference %.2g\n", i, double(diff / scale));
        if (diff > 1e-4f * scale)
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair() || !attention_fair()) {
        std::fprintf(stderr, "cuda_ops benchmark: fairness check FAILED; not benchmarking\n");
        return 1;
    }
    std::printf("device: %s\n", device().context->device_name().c_str());
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
