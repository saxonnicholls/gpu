//
//  cuda_moe_benchmark.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The mixture of experts on the device (cuda_moe.hpp), at two shapes over 4096 tokens of 768: eight experts of 2048,
//  two per token (Mixtral's shape), and 64 fine-grained experts of 256, eight per token (DeepSeekMoE's), each routed
//  from random tokens through a random router, so the segments are as uneven as a router makes them.
//
//  1. The grouped product (the experts' up projection, every routed row times its expert's 2048 x 768 or 256 x 768
//     matrix), forward and backward, three ways: the grouped kernel (the segments read on the device), each expert
//     through cuBLAS as core's reference loops them (the offsets read back first: one wait), and one
//     cublasGemmGroupedBatchedEx (the shapes on the host, so the same wait, the pointer tables copied each call), in
//     float, TF32 and bfloat16 (cuBLAS's grouped call has no bfloat16 inputs with float outputs: not run).
//  2. The steps around the products: routing, the histogram, the offsets, the permutation, the gather and the combine.
//  3. A whole MoE layer's forward and backward (ml::moe_forward, ml::moe_backward) on the device, queued call by call,
//     replayed as a captured cuda_graph, and queued with the grouped products through cuBLAS expert by expert.
//
//  main() first checks each path's grouped product against core's reference on the host, normwise: float 2e-6, TF32
//  2e-3, bfloat16 1e-2, as tests/cuda_moe_test.cpp; it refuses to run otherwise.
//

#include <snicholls/gpu/cuda_moe.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/moe.hpp>

#include "../tests/support/block_step.hpp"

#include <benchmark/benchmark.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace gpu = snicholls::gpu;
namespace ml = snicholls::ml;
namespace la = snicholls::linalg;
using gpu::testing::stream;

struct device_t {
    std::unique_ptr<gpu::cuda_context> context;
    std::unique_ptr<gpu::cuda_kernels> kernels;
    std::unique_ptr<gpu::cuda_moe_kernels> moe;
    cublasHandle_t blas = nullptr;   // for cublasGemmGroupedBatchedEx, on the context's stream
};

const device_t& device()
{
    static const device_t d = [] {
        auto c = gpu::cuda_context::create();
        if (c.is_err())
            std::exit(1);
        device_t out{std::move(c).unwrap(), nullptr, nullptr, nullptr};
        auto k = gpu::cuda_kernels::create(*out.context);
        auto m = gpu::cuda_moe_kernels::create(*out.context);
        if (k.is_err() || m.is_err())
            std::exit(1);
        out.kernels = std::move(k).unwrap();
        out.moe = std::move(m).unwrap();
        cublasCreate(&out.blas);
        cublasSetStream(out.blas, out.context->stream());
        return out;
    }();
    return d;
}

gpu::cuda_training_backend backend(gpu::cuda_math math)
{
    gpu::cuda_training_backend b{{device().context.get(), math}, device().kernels.get()};
    b.moe = device().moe.get();
    return b;
}

constexpr std::size_t tokens = 4096;
constexpr ml::transformer_config wide{.d_model = 768, .n_heads = 12, .n_kv_heads = 12, .head_dim = 64, .d_ff = 2048, .norm = ml::norm_kind::rms,
                                      .mlp = ml::mlp_kind::swiglu, .experts = 8, .top_k = 2, .expert_d_ff = 2048, .z_loss_weight = 0.001};
constexpr ml::transformer_config fine{.d_model = 768, .n_heads = 12, .n_kv_heads = 12, .head_dim = 64, .d_ff = 256, .norm = ml::norm_kind::rms,
                                      .mlp = ml::mlp_kind::swiglu, .experts = 64, .top_k = 8, .shared_experts = 1, .expert_d_ff = 256,
                                      .scoring = ml::moe_scoring::sigmoid, .route_scale = 2.5, .balance_loss_weight = 0.0001};

template <typename T>
std::vector<T> zeros(std::size_t n)
{
    return std::vector<T>(n);
}

template <typename T>
struct buffer {
    std::vector<T> host;
    std::unique_ptr<gpu::device_buffer<T>> dev;
    explicit buffer(std::vector<T> h) : host(std::move(h)), dev(std::make_unique<gpu::device_buffer<T>>(*device().context, host.size())) { dev->upload(host); }
    std::span<T> d() const { return dev->span(); }
    std::vector<T> back() const
    {
        device().context->synchronize();
        std::vector<T> out(host.size());
        dev->download(out);
        return out;
    }
};

// One grouped product's operands, its segments from a real routing of random tokens
template <ml::transformer_config C>
struct grouped {
    static constexpr std::size_t E = C.experts, K = C.top_k, d = C.d_model, f = C.expert_d_ff, n = tokens * K;
    std::vector<std::uint32_t> offsets_host;
    buffer<std::uint32_t> offsets;
    buffer<float> w, xs, ys, dys, dxs, dw;
    std::vector<void*> ptrs;   // device pointer tables for cublasGemmGroupedBatchedEx
    void** table = nullptr;
    static std::vector<std::uint32_t> route()
    {
        const auto router = stream(E * d, 1, 0.05f), x = stream(tokens * d, 2);
        std::vector<float> logits(tokens * E), weights(tokens * K);
        std::vector<std::uint32_t> experts(tokens * K), counts(E), off(E + 1);
        ml::linear_sequence(la::reference, ml::dense_weights<float>{std::span<const float>(router), E, d}, std::span<const float>(x), std::span<float>(logits));
        ml::moe_route_rows<C>(la::reference, std::span<const float>(logits), {}, std::span<std::uint32_t>(experts), std::span<float>(weights));
        ml::moe_count_rows<C>(la::reference, std::span<const std::uint32_t>(experts), std::span<std::uint32_t>(counts));
        ml::moe_offsets<C>(la::reference, std::span<const std::uint32_t>(counts), std::span<std::uint32_t>(off));
        return off;
    }
    grouped()
        : offsets_host(route()), offsets(offsets_host), w(stream(E * f * d, 3, 0.05f)), xs(stream(n * d, 4)), ys(zeros<float>(n * f)),
          dys(stream(n * f, 5)), dxs(zeros<float>(n * d)), dw(zeros<float>(E * f * d))
    {
        cudaMalloc(reinterpret_cast<void**>(&table), 3 * E * sizeof(void*));
    }
    ~grouped() { cudaFree(table); }
    grouped(const grouped&) = delete;
    grouped& operator=(const grouped&) = delete;
    ml::grouped_weights<float> weights() const { return {std::span<const float>(w.d()), E, f, d}; }
    void forward(const gpu::cuda_training_backend& b) const
    {
        gpu::grouped_linear(b, weights(), std::span<const std::uint32_t>(offsets.d()), std::span<const float>(xs.d()), ys.d());
    }
    void backward(const gpu::cuda_training_backend& b) const
    {
        gpu::grouped_linear_backward(b, weights(), std::span<const std::uint32_t>(offsets.d()), std::span<const float>(xs.d()),
                                     std::span<const float>(dys.d()), dxs.d(), dw.d());
    }
    // One cublasGemmGroupedBatchedEx: the offsets read back (as a router's would be), the shapes and the pointer
    // tables built on the host each call. Ys (n x f, row-major) = Xs W_eᵀ: column-major Yᵀ = W_e Xᵀ with W_e (f x d
    // row-major) seen as d x f column-major, transposed
    bool grouped_batched(gpu::cuda_math math)
    {
        std::vector<std::uint32_t> off(E + 1);
        cudaMemcpyAsync(off.data(), offsets.d().data(), off.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, device().context->stream());
        cudaStreamSynchronize(device().context->stream());
        std::vector<cublasOperation_t> ta, tb;
        std::vector<int> m, nn, k, lda, ldb, ldc, sizes;
        std::vector<float> alpha, beta;
        std::vector<void*> host(3 * E);
        int groups = 0;
        for (std::size_t e = 0; e < E; ++e) {
            const int rows = int(off[e + 1] - off[e]);
            if (rows == 0)
                continue;
            ta.push_back(CUBLAS_OP_T), tb.push_back(CUBLAS_OP_N), m.push_back(int(f)), nn.push_back(rows), k.push_back(int(d));
            lda.push_back(int(d)), ldb.push_back(int(d)), ldc.push_back(int(f)), sizes.push_back(1), alpha.push_back(1.0f), beta.push_back(0.0f);
            host[std::size_t(groups)] = w.d().data() + e * f * d;
            host[E + std::size_t(groups)] = xs.d().data() + std::size_t(off[e]) * d;
            host[2 * E + std::size_t(groups)] = ys.d().data() + std::size_t(off[e]) * f;
            ++groups;
        }
        cudaMemcpyAsync(table, host.data(), host.size() * sizeof(void*), cudaMemcpyHostToDevice, device().context->stream());
        const auto compute = math == gpu::cuda_math::tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
        return cublasGemmGroupedBatchedEx(device().blas, ta.data(), tb.data(), m.data(), nn.data(), k.data(), alpha.data(), table, CUDA_R_32F, lda.data(),
                                          table + E, CUDA_R_32F, ldb.data(), beta.data(), table + 2 * E, CUDA_R_32F, ldc.data(), groups, sizes.data(),
                                          compute)
               == CUBLAS_STATUS_SUCCESS;
    }
};

double normwise(const std::vector<float>& got, const std::vector<float>& want)
{
    double diff = 0, scale = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
        diff = std::max(diff, double(std::fabs(got[i] - want[i]))), scale = std::max(scale, double(std::fabs(want[i])));
    return scale == 0 ? diff : diff / scale;
}

constexpr gpu::cuda_math maths[] = {gpu::cuda_math::float32, gpu::cuda_math::tf32, gpu::cuda_math::bf16};
constexpr const char* math_names[] = {"float", "tf32", "bf16"};
constexpr double tolerances[] = {2e-6, 2e-3, 1e-2};
enum path : int { kernel, cublas_loop, cublas_grouped };
constexpr const char* path_names[] = {"kernel", "cublas_loop", "cublas_grouped"};

template <ml::transformer_config C>
grouped<C>& operands()
{
    static grouped<C> g;
    return g;
}

// The fairness check: every path's forward product against core's reference on the host
template <ml::transformer_config C>
bool fair()
{
    auto& g = operands<C>();
    std::vector<float> want(g.ys.host.size());
    ml::grouped_linear(la::reference, ml::grouped_weights<float>{std::span<const float>(g.w.host), g.E, g.f, g.d}, std::span<const std::uint32_t>(g.offsets_host),
                       std::span<const float>(g.xs.host), std::span<float>(want));
    bool ok = true;
    for (int m = 0; m < 3; ++m)
        for (int p = 0; p < 3; ++p) {
            if (p == cublas_grouped && maths[m] == gpu::cuda_math::bf16)
                continue;
            const auto b = backend(maths[m]);
            device().moe->path = p == cublas_loop ? gpu::grouped_path::cublas : gpu::grouped_path::kernel;
            if (p == cublas_grouped)
                ok = g.grouped_batched(maths[m]) && ok;
            else
                g.forward(b);
            const double e = normwise(g.ys.back(), want);
            std::printf("fairness: E=%zu K=%zu %-5s %-14s normwise %.2e (limit %.0e)\n", g.E, g.K, math_names[m], path_names[p], e, tolerances[m]);
            ok = ok && e <= tolerances[m];
        }
    device().moe->path = gpu::grouped_path::kernel;
    return ok;
}

template <ml::transformer_config C>
void BM_Grouped(benchmark::State& state)
{
    const int m = int(state.range(0)), p = int(state.range(1)), back = int(state.range(2));
    auto& g = operands<C>();
    const auto b = backend(maths[m]);
    if (p == cublas_grouped && (maths[m] == gpu::cuda_math::bf16 || back)) {
        state.SkipWithError(back ? "the backward through cublasGemmGroupedBatchedEx is not measured" : "cuBLAS's grouped product has no bfloat16 x bfloat16 -> float");
        return;
    }
    device().moe->path = p == cublas_loop ? gpu::grouped_path::cublas : gpu::grouped_path::kernel;
    for (auto _ : state) {
        if (p == cublas_grouped)
            g.grouped_batched(maths[m]);
        else if (back)
            g.backward(b);
        else
            g.forward(b);
        device().context->synchronize();
    }
    device().moe->path = gpu::grouped_path::kernel;
    const double flop = 2.0 * double(g.n) * double(g.d) * double(g.f) * (back ? 2 : 1);
    state.counters["TFLOP/s"] = benchmark::Counter(flop * double(state.iterations()) / 1e12, benchmark::Counter::kIsRate);
    state.SetLabel(std::string(math_names[m]) + " " + path_names[p] + (back ? " backward" : " forward"));
}

// The steps around the products, forward: route, count, offsets, permutation, gather, combine
template <ml::transformer_config C>
void BM_Steps(benchmark::State& state)
{
    constexpr std::size_t E = C.experts, K = C.top_k, d = C.d_model, n = tokens * K;
    const auto b = backend(gpu::cuda_math::tf32);
    buffer<float> logits(stream(tokens * E, 6, 2.0f)), weights(zeros<float>(n)), xs(stream(tokens * d, 7)), xp(zeros<float>(n * d)),
        ys(zeros<float>(tokens * d));
    buffer<std::uint32_t> experts(zeros<std::uint32_t>(n)), counts(zeros<std::uint32_t>(E)), offsets(zeros<std::uint32_t>(E + 1)),
        order(zeros<std::uint32_t>(n)), inverse(zeros<std::uint32_t>(n));
    for (auto _ : state) {
        gpu::moe_route_rows<C>(b, std::span<const float>(logits.d()), {}, experts.d(), weights.d());
        gpu::moe_count_rows<C>(b, std::span<const std::uint32_t>(experts.d()), counts.d());
        gpu::moe_offsets<C>(b, std::span<const std::uint32_t>(counts.d()), offsets.d());
        gpu::moe_permutation<C>(b, std::span<const std::uint32_t>(experts.d()), std::span<const std::uint32_t>(offsets.d()), order.d(), inverse.d());
        gpu::moe_gather_rows<C>(b, std::span<const float>(xs.d()), std::span<const std::uint32_t>(order.d()), xp.d());
        gpu::moe_combine_rows<C>(b, std::span<const float>(xp.d()), std::span<const float>(weights.d()), std::span<const std::uint32_t>(inverse.d()), ys.d());
        device().context->synchronize();
    }
}

// A whole layer's forward and backward, queued call by call (graph = 0), replayed as a captured graph (graph = 1), or
// queued with the grouped products through cuBLAS expert by expert (graph = 2: a wait for the offsets per product)
template <ml::transformer_config C>
void BM_Layer(benchmark::State& state)
{
    const int m = int(state.range(0)), graph = int(state.range(1));
    const auto b = backend(maths[m]);
    constexpr std::size_t d = C.d_model;
    buffer<float> params(stream(ml::moe_parameter_count(C), 8, 0.03f)), xs(stream(tokens * d, 9)), ys(zeros<float>(tokens * d)),
        values(zeros<float>(ml::moe_activation_values(C, tokens))), dys(stream(tokens * d, 10)), dxs(zeros<float>(tokens * d)),
        grads(zeros<float>(ml::moe_parameter_count(C))), work(zeros<float>(ml::moe_gradient_work(C, tokens)));
    buffer<std::uint32_t> indices(zeros<std::uint32_t>(ml::moe_activation_indices(C, tokens)));
    const auto w = ml::moe_weights_in<C>(std::span<const float>(params.d()));
    const auto act = ml::moe_activations_in<C>(values.d(), indices.d(), tokens);
    const auto g = ml::moe_gradients_in<C>(grads.d(), work.d(), tokens);
    auto pass = [&] {
        ml::moe_forward<C>(b, w, std::span<const float>(xs.d()), ys.d(), act, 4);
        ml::moe_backward<C>(b, w, std::span<const float>(xs.d()), act, std::span<const float>(dys.d()), dxs.d(), g, 4);
    };
    device().moe->losses_on_device = graph == 1;
    device().moe->path = graph == 2 ? gpu::grouped_path::cublas : gpu::grouped_path::kernel;
    pass();   // sizes the working space
    device().context->synchronize();
    std::unique_ptr<gpu::cuda_graph> captured;
    if (graph == 1) {
        auto made = gpu::cuda_graph::capture(*device().context, pass);
        if (made.is_err()) {
            state.SkipWithError("the capture failed");
            device().moe->losses_on_device = false;
            return;
        }
        captured = std::make_unique<gpu::cuda_graph>(std::move(made).unwrap());
    }
    for (auto _ : state) {
        if (graph == 1)
            captured->launch();
        else
            pass();
        device().context->synchronize();
    }
    device().moe->losses_on_device = false;
    device().moe->path = gpu::grouped_path::kernel;
    state.SetLabel(std::string(math_names[m]) + (graph == 1 ? " graph" : graph == 2 ? " queued, cuBLAS per expert" : " queued"));
}

void grouped_args(benchmark::Benchmark* bm)
{
    for (int back = 0; back < 2; ++back)
        for (int m = 0; m < 3; ++m)
            for (int p = 0; p < 3; ++p)
                bm->Args({m, p, back});
}
void layer_args(benchmark::Benchmark* bm)
{
    for (int m = 0; m < 3; ++m)
        for (int graph = 0; graph < 3; ++graph)
            bm->Args({m, graph});
}

BENCHMARK(BM_Grouped<wide>)->Name("BM_MoeGroupedProduct_E8K2_2048")->Apply(grouped_args)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Grouped<fine>)->Name("BM_MoeGroupedProduct_E64K8_256")->Apply(grouped_args)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Steps<wide>)->Name("BM_MoeRoutePermuteGatherCombine_E8K2")->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Steps<fine>)->Name("BM_MoeRoutePermuteGatherCombine_E64K8")->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Layer<wide>)->Name("BM_MoeLayerForwardBackward_E8K2_2048")->Apply(layer_args)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Layer<fine>)->Name("BM_MoeLayerForwardBackward_E64K8_256_Shared1")->Apply(layer_args)->Unit(benchmark::kMillisecond)->UseRealTime();

} // namespace

int main(int argc, char** argv)
{
    std::printf("device: %s, %zu tokens of 768\n", device().context->device_name().c_str(), tokens);
    if (!fair<wide>() || !fair<fine>()) {
        std::fprintf(stderr, "cuda_moe_benchmark: a path disagrees with core's reference; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
