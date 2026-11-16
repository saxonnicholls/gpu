//
//  cuda_ops_benchmark.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  A GPT-2-small layer's training step (forward and backward over 128 tokens) through core's batched passes with every
//  parameter, activation and gradient in device memory (cuda_training_backend: the products by cuBLAS, the other
//  steps by its kernels, attention still staged), against the same step from host memory (cuda_backend: the products
//  staged, the other steps on the host). Different paths, so main() checks they agree normwise (the largest difference
//  within 1e-5 of each tensor's largest value; measured 1e-7 to 1e-6) before benchmarking.
//

#include <snicholls/gpu/cuda_ops.hpp>

#include "../tests/support/block_step.hpp"

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
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
        if (diff > 1e-5f * scale)
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
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
