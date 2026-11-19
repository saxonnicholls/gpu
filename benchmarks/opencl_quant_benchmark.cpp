//
//  opencl_quant_benchmark.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The quantised-weight products on an OpenCL device. Decode: one token through a matrix kept on the device in its
//  blocks (float activations, and Q8_0 activations), the rate in bytes of weights read per second; against core's own
//  one-row product on the host (ml::linear over quantized_weights, scalar, one thread) and against the dequantised float
//  matrix through opencl_backend's product, which copies the whole float matrix for each call (dequantise, then the dense
//  product). Prefill: 512 tokens through the tiled kernel, in GFLOP/s. Every device time includes the copies of the
//  activations and the results. A fairness check refuses to benchmark unless the device's results agree with core's
//  references: the tiled and Q8_0 paths to the bit, the matrix-vector path within the bound of a reordered sum.
//

#include <snicholls/gpu/opencl_quant.hpp>

#include "../tests/support/quant_cases.hpp"

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/quantize.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace gpu = snicholls::gpu;
namespace ml = snicholls::ml;
namespace la = snicholls::linalg;
using namespace snicholls::gpu::testing;

struct device_t {
    std::unique_ptr<gpu::opencl_context> context;
    std::unique_ptr<gpu::opencl_quant_kernels> kernels;
};

const device_t& device()
{
    static const device_t d = [] {
        auto c = gpu::opencl_context::create(gpu::device_choice::any);
        if (c.is_err()) {
            std::fprintf(stderr, "opencl_quant_benchmark: no OpenCL device\n");
            std::exit(1);
        }
        device_t out{std::move(c).unwrap(), nullptr};
        auto k = gpu::opencl_quant_kernels::create(*out.context);
        if (k.is_err()) {
            std::fprintf(stderr, "opencl_quant_benchmark: the program did not build\n");
            std::exit(1);
        }
        out.kernels = std::move(k).unwrap();
        std::printf("device: %s\n", out.context->device_name().c_str());
        return out;
    }();
    return d;
}

gpu::opencl_quant_backend backend() { return {{device().context.get()}, device().kernels.get()}; }

// One matrix per (format, shape), made once and kept on the device
template <typename Tag>
struct problem {
    host_matrix<Tag> host;
    gpu::opencl_quant_matrix on_device;
    std::vector<float> x;
    problem(std::size_t rows, std::size_t cols) : host(rows, cols, rows * 31 + cols, 1.0f), on_device(device().kernels->upload(host.view())), x(uniform(512 * cols, 7)) {}
};

// The fairness check: the device's paths against core's references, or no benchmark
template <typename Tag>
void fair(problem<Tag>& p)
{
    const std::size_t rows = p.host.rows, cols = p.host.cols, n = 3;
    const std::span<const float> xs(p.x.data(), n * cols);
    const auto want = reference_forward(p.host.dense, rows, cols, xs, {});
    std::vector<float> mv(n * rows), tiled(n * rows);
    gpu::linear_sequence(backend(), p.on_device, xs, mv);
    device().kernels->matvec_max = 0;
    gpu::linear_sequence(backend(), p.on_device, xs, tiled);
    device().kernels->matvec_max = 8;
    const auto bound = reorder_bound(p.host.dense, rows, cols, xs, double(cols) * 0x1p-23);
    bool ok = same_bits(tiled, want);
    for (std::size_t i = 0; i < mv.size(); ++i)
        ok = ok && std::fabs(double(mv[i]) - double(want[i])) <= bound[i] + std::ldexp(std::fabs(double(want[i])), -22);
    if constexpr (Tag::q8) {
        std::vector<float> q(n * rows);
        gpu::linear_sequence_q8(backend(), p.on_device, xs, q);
        ok = ok && (!device().kernels->exact_q8() || same_bits(q, reference_q8(p.host.view(), xs, {})));
    }
    if (!ok) {
        std::fprintf(stderr, "opencl_quant_benchmark: %s %zu x %zu: the device disagrees with core's reference; refusing to benchmark\n", Tag::name,
                     rows, cols);
        std::exit(1);
    }
}

template <typename Tag>
problem<Tag>& get(std::size_t rows, std::size_t cols)
{
    static std::map<std::pair<std::size_t, std::size_t>, std::unique_ptr<problem<Tag>>> made;
    auto& p = made[{rows, cols}];
    if (!p) {
        p = std::make_unique<problem<Tag>>(rows, cols);
        fair(*p);
    }
    return *p;
}

template <typename Tag, bool Q8>
void BM_Decode(benchmark::State& state)
{
    auto& p = get<Tag>(std::size_t(state.range(0)), std::size_t(state.range(1)));
    std::vector<float> y(p.host.rows);
    const std::span<const float> x(p.x.data(), p.host.cols);
    for (auto _ : state) {
        if constexpr (Q8)
            gpu::linear_sequence_q8(backend(), p.on_device, x, y);
        else
            gpu::linear_sequence(backend(), p.on_device, x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.counters["GB/s"] = benchmark::Counter(double(p.on_device.bytes()), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// core's one-row product over quantised weights on the host: the activations quantised to Q8_0, ggml's dot products
template <typename Tag>
void BM_DecodeHostCore(benchmark::State& state)
{
    constexpr std::size_t cols = 4096;
    auto& p = get<Tag>(std::size_t(state.range(0)), cols);
    std::vector<float> y(p.host.rows);
    for (auto _ : state) {
        const auto q = ml::prepare_input<cols>(p.host.view(), std::span<const float, cols>(p.x.data(), cols));
        ml::linear(p.host.view(), q, std::span<float>(y));
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.counters["GB/s"] = benchmark::Counter(double(p.on_device.bytes()), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// Dequantise, then the dense product: the float matrix through opencl_backend's product with one column, which copies it for each call
template <typename Tag>
void BM_DecodeDequantisedGemv(benchmark::State& state)
{
    auto& p = get<Tag>(std::size_t(state.range(0)), std::size_t(state.range(1)));
    const std::size_t rows = p.host.rows, cols = p.host.cols;
    std::vector<float> y(rows);
    const gpu::opencl_backend dense{device().context.get()};
    const la::matrix_view<const float, la::layout_right> a(p.host.dense.data(), {rows, cols}), x(p.x.data(), {cols, 1});
    for (auto _ : state) {
        [[maybe_unused]] const auto done = la::gemm(dense, 1.0f, a, x, 0.0f, la::matrix_view<float, la::layout_right>(y.data(), {rows, 1}));
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.counters["GB/s"] = benchmark::Counter(double(p.on_device.bytes()), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

template <typename Tag>
void BM_Prefill(benchmark::State& state)
{
    auto& p = get<Tag>(std::size_t(state.range(0)), std::size_t(state.range(1)));
    const std::size_t n = 512;
    std::vector<float> y(n * p.host.rows);
    for (auto _ : state) {
        gpu::linear_sequence(backend(), p.on_device, p.x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(2.0 * double(n * p.host.rows * p.host.cols), benchmark::Counter::kIsIterationInvariantRate,
                                                   benchmark::Counter::kIs1000);
}

void decode_shapes(benchmark::Benchmark* b) { b->Args({4096, 4096})->Args({14336, 4096})->Args({28672, 4096})->Unit(benchmark::kMicrosecond)->UseRealTime(); }

BENCHMARK(BM_Decode<q4_0_tag, false>)->Name("BM_Decode_Q4_0")->Apply(decode_shapes);
BENCHMARK(BM_Decode<q8_0_tag, false>)->Name("BM_Decode_Q8_0")->Apply(decode_shapes);
BENCHMARK(BM_Decode<nf4_tag, false>)->Name("BM_Decode_NF4")->Apply(decode_shapes);
BENCHMARK(BM_Decode<bf16_tag, false>)->Name("BM_Decode_BF16")->Apply(decode_shapes);
BENCHMARK(BM_Decode<q4_0_tag, true>)->Name("BM_DecodeQ8Activations_Q4_0")->Apply(decode_shapes);
BENCHMARK(BM_Decode<q8_0_tag, true>)->Name("BM_DecodeQ8Activations_Q8_0")->Apply(decode_shapes);
BENCHMARK(BM_DecodeHostCore<q4_0_tag>)->Name("BM_DecodeHostCore_Q4_0")->Arg(4096)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_DecodeHostCore<q8_0_tag>)->Name("BM_DecodeHostCore_Q8_0")->Arg(4096)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_DecodeDequantisedGemv<q4_0_tag>)->Name("BM_DecodeDequantisedGemv_Q4_0")->Args({4096, 4096})->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(BM_Prefill<q4_0_tag>)->Name("BM_Prefill512_Q4_0")->Args({4096, 4096})->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Prefill<nf4_tag>)->Name("BM_Prefill512_NF4")->Args({4096, 4096})->Unit(benchmark::kMillisecond)->UseRealTime();

} // namespace

BENCHMARK_MAIN();
