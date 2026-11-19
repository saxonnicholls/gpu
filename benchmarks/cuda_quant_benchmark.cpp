//
//  cuda_quant_benchmark.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The quantised-weight products on an NVIDIA GPU, over device memory. Decode: one token through a matrix in its
//  blocks (float activations, and Q8_0 activations), in bytes of weights read per second, against the matrix
//  dequantised to float once and multiplied by cuBLAS (the dequantise-then-BLAS way: 4 bytes a weight read for each
//  token). Prefill: 512 tokens through the tiled kernel on the CUDA cores and on the tensor cores (bfloat16), in
//  GFLOP/s, against cuBLAS over the dequantised matrix in float and in bfloat16. Each time includes waiting for the
//  stream. A fairness check refuses to benchmark unless the device's results agree with core's references (the
//  CUDA-core tiled and Q8_0 paths to the bit, the others within their stated bounds).
//

#include <snicholls/gpu/cuda_quant.hpp>

#include "../tests/support/quant_cases.hpp"

#include <snicholls/core/linalg/backend.hpp>
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
    std::unique_ptr<gpu::cuda_context> context;
    std::unique_ptr<gpu::cuda_kernels> kernels;
    std::unique_ptr<gpu::cuda_quant_kernels> quant;
};

const device_t& device()
{
    static const device_t d = [] {
        auto c = gpu::cuda_context::create();
        if (c.is_err()) {
            std::fprintf(stderr, "cuda_quant_benchmark: no CUDA device\n");
            std::exit(1);
        }
        device_t out{std::move(c).unwrap(), nullptr, nullptr};
        auto k = gpu::cuda_kernels::create(*out.context);
        auto q = gpu::cuda_quant_kernels::create(*out.context);
        if (k.is_err() || q.is_err()) {
            std::fprintf(stderr, "cuda_quant_benchmark: the kernels did not compile\n");
            std::exit(1);
        }
        out.kernels = std::move(k).unwrap();
        out.quant = std::move(q).unwrap();
        std::printf("device: %s\n", out.context->device_name().c_str());
        return out;
    }();
    return d;
}

gpu::cuda_training_backend backend(gpu::cuda_math math = gpu::cuda_math::float32)
{
    gpu::cuda_training_backend b{{device().context.get(), math}, device().kernels.get()};
    b.quant = device().quant.get();
    return b;
}

template <typename T>
std::unique_ptr<gpu::device_buffer<T>> upload(std::span<const T> host)
{
    auto b = std::make_unique<gpu::device_buffer<T>>(*device().context, host.size());
    b->upload(host);
    return b;
}

template <typename T>
std::vector<T> download(const gpu::device_buffer<T>& b)
{
    device().context->synchronize();
    std::vector<T> out(b.size());
    b.download(out);
    return out;
}

// One block-format matrix per (format, shape) on the device, with its dequantised floats there too
template <typename Tag>
struct problem {
    host_matrix<Tag> host;
    std::unique_ptr<gpu::device_buffer<typename decltype(host.blocks)::value_type>> blocks;
    std::unique_ptr<gpu::device_buffer<float>> dense, x, y;
    problem(std::size_t rows, std::size_t cols)
        : host(rows, cols, rows * 31 + cols, 1.0f), blocks(upload(std::span<const typename decltype(host.blocks)::value_type>(host.blocks))),
          dense(upload(std::span<const float>(host.dense))), x(upload(std::span<const float>(uniform(512 * cols, 7)))),
          y(std::make_unique<gpu::device_buffer<float>>(*device().context, 512 * rows))
    {
    }
    [[nodiscard]] typename Tag::view view() const { return {blocks->span(), host.rows, host.cols}; }
    [[nodiscard]] std::size_t bytes() const { return host.blocks.size() * sizeof(typename decltype(host.blocks)::value_type); }
};

template <typename Tag>
void fair(problem<Tag>& p)
{
    const std::size_t rows = p.host.rows, cols = p.host.cols, n = 3;
    const auto xs_host = download(*p.x);
    const std::span<const float> xs_h(xs_host.data(), n * cols);
    const std::span<const float> xs(p.x->span().data(), n * cols);
    const std::span<float> ys(p.y->span().data(), n * rows);
    const auto want = reference_forward(p.host.dense, rows, cols, xs_h, {});
    const auto get = [&] {
        auto all = download(*p.y);
        return std::vector<float>(all.begin(), all.begin() + std::ptrdiff_t(n * rows));
    };
    bool ok = true;
    device().quant->matvec_max = 0;
    gpu::linear_sequence(backend(), p.view(), xs, ys);
    ok = ok && same_bits(get(), want);
    gpu::linear_sequence(backend(gpu::cuda_math::bf16), p.view(), xs, ys);
    const auto tc = get();
    const auto loose = reorder_bound(p.host.dense, rows, cols, xs_h, 0x1p-7);
    for (std::size_t i = 0; i < tc.size(); ++i)
        ok = ok && std::fabs(double(tc[i]) - double(want[i])) <= loose[i] + 1e-6;
    device().quant->matvec_max = 8;
    gpu::linear_sequence(backend(), p.view(), xs, ys);
    const auto mv = get();
    const auto bound = reorder_bound(p.host.dense, rows, cols, xs_h, double(cols) * 0x1p-23);
    for (std::size_t i = 0; i < mv.size(); ++i)
        ok = ok && std::fabs(double(mv[i]) - double(want[i])) <= bound[i] + std::ldexp(std::fabs(double(want[i])), -22);
    gpu::linear_sequence_q8(backend(), p.view(), xs, ys);
    ok = ok && same_bits(get(), reference_q8(p.host.view(), xs_h, {}));
    if (!ok) {
        std::fprintf(stderr, "cuda_quant_benchmark: %s %zu x %zu: the device disagrees with core's reference; refusing to benchmark\n", Tag::name, rows,
                     cols);
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

// 0: float activations, 1: Q8_0 activations, 2: the dequantised float matrix through cuBLAS
template <typename Tag, int Path>
void BM_Decode(benchmark::State& state)
{
    auto& p = get<Tag>(std::size_t(state.range(0)), std::size_t(state.range(1)));
    const std::size_t rows = p.host.rows, cols = p.host.cols;
    const std::span<const float> x(p.x->span().data(), cols);
    const std::span<float> y(p.y->span().data(), rows);
    const auto b = backend();
    for (auto _ : state) {
        if constexpr (Path == 0)
            gpu::linear_sequence(b, p.view(), x, y);
        else if constexpr (Path == 1)
            gpu::linear_sequence_q8(b, p.view(), x, y);
        else
            ml::linear_sequence(b, ml::dense_weights<float>{p.dense->span(), rows, cols}, x, y);
        device().context->synchronize();
    }
    state.counters["GB/s"] = benchmark::Counter(double(Path == 2 ? rows * cols * sizeof(float) : p.bytes()), benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

// 0: CUDA cores, 1: tensor cores (bfloat16), 2: cuBLAS over the dequantised matrix in float, 3: the same in bfloat16
template <typename Tag, int Path>
void BM_Prefill(benchmark::State& state)
{
    auto& p = get<Tag>(std::size_t(state.range(0)), std::size_t(state.range(1)));
    const std::size_t rows = p.host.rows, cols = p.host.cols, n = 512;
    const std::span<const float> x = p.x->span();
    const std::span<float> y = p.y->span();
    const auto b = backend(Path == 1 || Path == 3 ? gpu::cuda_math::bf16 : gpu::cuda_math::float32);
    device().quant->matvec_max = 0;
    for (auto _ : state) {
        if constexpr (Path < 2)
            gpu::linear_sequence(b, p.view(), x, y);
        else
            ml::linear_sequence(b, ml::dense_weights<float>{p.dense->span(), rows, cols}, x, y);
        device().context->synchronize();
    }
    device().quant->matvec_max = 8;
    state.counters["GFLOP/s"] = benchmark::Counter(2.0 * double(n * rows * cols), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

void decode_shapes(benchmark::Benchmark* b) { b->Args({4096, 4096})->Args({14336, 4096})->Unit(benchmark::kMicrosecond)->UseRealTime(); }
void prefill_shapes(benchmark::Benchmark* b) { b->Args({4096, 4096})->Unit(benchmark::kMillisecond)->UseRealTime(); }

BENCHMARK(BM_Decode<q4_0_tag, 0>)->Name("BM_Decode_Q4_0")->Apply(decode_shapes);
BENCHMARK(BM_Decode<q8_0_tag, 0>)->Name("BM_Decode_Q8_0")->Apply(decode_shapes);
BENCHMARK(BM_Decode<mxfp4_tag, 0>)->Name("BM_Decode_MXFP4")->Apply(decode_shapes);
BENCHMARK(BM_Decode<q4_0_tag, 1>)->Name("BM_DecodeQ8Activations_Q4_0")->Apply(decode_shapes);
BENCHMARK(BM_Decode<q8_0_tag, 1>)->Name("BM_DecodeQ8Activations_Q8_0")->Apply(decode_shapes);
BENCHMARK(BM_Decode<q4_0_tag, 2>)->Name("BM_DecodeDequantisedCublas")->Apply(decode_shapes);
BENCHMARK(BM_Prefill<q4_0_tag, 0>)->Name("BM_Prefill512_Q4_0_CudaCores")->Apply(prefill_shapes);
BENCHMARK(BM_Prefill<q4_0_tag, 1>)->Name("BM_Prefill512_Q4_0_TensorCores")->Apply(prefill_shapes);
BENCHMARK(BM_Prefill<q4_0_tag, 2>)->Name("BM_Prefill512_DequantisedCublasFloat")->Apply(prefill_shapes);
BENCHMARK(BM_Prefill<q4_0_tag, 3>)->Name("BM_Prefill512_DequantisedCublasBf16")->Apply(prefill_shapes);

} // namespace

BENCHMARK_MAIN();
