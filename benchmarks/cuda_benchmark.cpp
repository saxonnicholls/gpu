//
//  cuda_benchmark.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The CUDA backend: Y = X Wᵀ through core's linalg::gemm with the operands in device memory, against the same
//  cublasSgemm called directly (the interface's cost: the pair must agree to the bit); the same product from host
//  memory (staged: the transfers included); and a GPT-2-small layer's training step through core's batched passes
//  (still staged: the steps other than the products run on the host, the operands copied for every product). main()
//  refuses to benchmark unless the pair agrees to the bit and the staged forms agree with core's reference to rounding.
//

#include <snicholls/gpu/cuda.hpp>

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/transformer_block.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace gpu = snicholls::gpu;
namespace la = snicholls::linalg;
using right = la::layout_right;

const gpu::cuda_context& device()
{
    static const std::unique_ptr<gpu::cuda_context> c = [] {
        auto made = gpu::cuda_context::create();
        if (made.is_err()) {
            std::fprintf(stderr, "cuda benchmark: no CUDA device\n");
            std::exit(1);
        }
        return std::move(made).unwrap();
    }();
    return *c;
}

std::vector<float> stream(std::size_t size, std::uint64_t seed)
{
    std::vector<float> v(size);
    std::uint64_t x = seed * 0x9e3779b97f4a7c15ULL + 1;
    for (float& e : v) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        e = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5);
    }
    return v;
}

// Y (n x out) = X (n x d) Wᵀ, W out x d
struct product {
    std::size_t n, d, out;
    std::vector<float> x, w, y;
    gpu::device_buffer<float> dx, dw, dy;
    product(std::size_t n_, std::size_t d_, std::size_t o_)
        : n(n_), d(d_), out(o_), x(stream(n * d, 1)), w(stream(out * d, 2)), y(n * out), dx(device(), n * d), dw(device(), out * d),
          dy(device(), n * out)
    {
        dx.upload(x);
        dw.upload(w);
    }
};

void resident_core(product& p)
{
    (void)la::gemm(gpu::cuda_backend{&device()}, 1.0f, la::matrix_view<const float, right>(p.dx.span().data(), {p.n, p.d}),
                   la::transposed(la::matrix_view<const float, right>(p.dw.span().data(), {p.out, p.d})), 0.0f,
                   la::matrix_view<float, right>(p.dy.span().data(), {p.n, p.out}));
}

// By hand: row-major Y = X Wᵀ is column-major Yᵀ = W Xᵀ, one cublasSgemm, then the same synchronisation
void resident_by_hand(product& p)
{
    static cublasHandle_t h = [] {
        cublasHandle_t x = nullptr;
        cublasCreate(&x);
        return x;
    }();
    const float one = 1.0f, zero = 0.0f;
    cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, int(p.out), int(p.n), int(p.d), &one, p.dw.span().data(), int(p.d), p.dx.span().data(),
                int(p.d), &zero, p.dy.span().data(), int(p.out));
    cudaDeviceSynchronize();
}

void staged_core(product& p)
{
    (void)la::gemm(gpu::cuda_backend{&device()}, 1.0f, la::matrix_view<const float, right>(p.x.data(), {p.n, p.d}),
                   la::transposed(la::matrix_view<const float, right>(p.w.data(), {p.out, p.d})), 0.0f,
                   la::matrix_view<float, right>(p.y.data(), {p.n, p.out}));
}

template <void (*Run)(product&)>
void BM_Product(benchmark::State& state)
{
    product p(std::size_t(state.range(0)), std::size_t(state.range(1)), std::size_t(state.range(2)));
    for (auto _ : state) {
        Run(p);
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(2.0 * double(p.n * p.d * p.out), benchmark::Counter::kIsIterationInvariantRate,
                                                   benchmark::Counter::kIs1000);
}

void shapes(benchmark::Benchmark* b)
{
    b->Args({128, 768, 768})->Args({128, 768, 3072})->Args({512, 768, 3072})->Args({2048, 1024, 4096})->Unit(benchmark::kMillisecond);
}
BENCHMARK(BM_Product<resident_core>)->Name("BM_DeviceMemory_Core")->Apply(shapes);
BENCHMARK(BM_Product<resident_by_hand>)->Name("BM_DeviceMemory_CublasByHand")->Apply(shapes);
BENCHMARK(BM_Product<staged_core>)->Name("BM_HostMemory_Staged")->Apply(shapes);

// ---- a GPT-2-small layer's training step, staged ----

namespace ml = snicholls::ml;
constexpr ml::transformer_config tc = ml::gpt2_small;
constexpr std::size_t td = tc.d_model, tf = tc.d_ff, tn = 128;

struct layer_t {
    using v = std::vector<float>;
    v ln1w = stream(td, 3), ln1b = stream(td, 4), wq = stream(td * td, 5), wk = stream(td * td, 6), wv = stream(td * td, 7),
      bq = stream(td, 8), bk = stream(td, 9), bv = stream(td, 10), wo = stream(td * td, 11), bo = stream(td, 12),
      ln2w = stream(td, 13), ln2b = stream(td, 14), wup = stream(tf * td, 15), bup = stream(tf, 16), wdown = stream(td * tf, 17),
      bdown = stream(td, 18);
    layer_t()
    {
        for (float& x : ln1w)
            x += 1.0f;
        for (float& x : ln2w)
            x += 1.0f;
        for (auto* m : {&wq, &wk, &wv, &wo, &wup, &wdown})
            for (float& x : *m)
                x *= 0.1f;
    }
    ml::block_weights<ml::dense_weights<float>> weights() const
    {
        using W = ml::dense_weights<float>;
        ml::block_weights<W> w{};
        w.norm1_weight = ln1w;
        w.norm1_bias = ln1b;
        w.attention = {W{wq, td, td}, W{wk, td, td}, W{wv, td, td}, W{wo, td, td}, bq, bk, bv, bo};
        w.norm2_weight = ln2w;
        w.norm2_bias = ln2b;
        w.mlp = {W{wup, tf, td}, W{}, W{wdown, td, tf}, bup, {}, bdown};
        return w;
    }
};

struct step_t {
    using v = std::vector<float>;
    v xs = stream(tn * td, 20), dxs = stream(tn * td, 21);
    v x0 = v(tn * td), h1 = x0, x1 = x0, h2 = x0, q = x0, k = x0, val = x0, y = x0, lse = v(tn * tc.n_heads), up = v(tn * tf), hidden = up;
    v g1w = v(td), g1b = v(td), g2w = v(td), g2b = v(td), gq = v(tn * td), gk = gq, gv = gq, gwq = v(td * td), gwk = gwq, gwv = gwq,
      gwo = gwq, gbq = v(td), gbk = gbq, gbv = gbq, gbo = gbq, gup = v(tf * td), gupb = v(tf), gdown = v(td * tf), gdownb = v(td),
      scratch = v(tn * td), dy = v(tn * td), mlp_scratch = v(tn * tf);
    ml::block_activations<float> act() { return {x0, h1, x1, h2, {q, k, val, {}, {}, y, lse}, up, {}, hidden}; }
    ml::block_gradients<float> grads()
    {
        return {g1w, g1b, g2w, g2b, {gq, gk, gv, gwq, gwk, gwv, gwo, gbq, gbk, gbv, gbo, {}, {}, dy},
                {gup, {}, gdown, gupb, {}, gdownb, mlp_scratch}, scratch};
    }
};

template <typename B>
void train(const B& backend, step_t& s)
{
    static const layer_t layer;
    static const auto w = layer.weights();
    ml::block_forward_sequence<tc>(backend, w, std::span<float>(s.xs), ml::no_rope, s.act());
    ml::block_backward<tc>(backend, w, ml::no_rope, s.act(), std::span<float>(s.dxs), s.grads());
}

void BM_TrainStaged(benchmark::State& state)
{
    step_t s;
    const std::vector<float> x0 = s.xs, d0 = s.dxs;
    for (auto _ : state) {
        s.xs = x0;
        s.dxs = d0;
        train(gpu::cuda_backend{&device()}, s);
        benchmark::DoNotOptimize(s.dxs.data());
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(tn));
}
BENCHMARK(BM_TrainStaged)->Name("BM_TrainGpt2SmallLayer_CudaStaged")->Unit(benchmark::kMillisecond);

bool fair()
{
    for (const auto& [n, d, o] : {std::array<std::size_t, 3>{128, 768, 768}, {512, 768, 3072}}) {
        product a(n, d, o), b(n, d, o);
        resident_core(a);
        resident_by_hand(b);
        std::vector<float> ya(n * o), yb(n * o), yr(n * o);
        a.dy.download(ya);
        b.dy.download(yb);
        if (std::memcmp(ya.data(), yb.data(), ya.size() * sizeof(float)) != 0) {
            std::fprintf(stderr, "cuda benchmark: the resident pair differs at %zu x %zu x %zu\n", n, d, o);
            return false;   // the pair: the same cuBLAS call, the same bits
        }
        staged_core(a);
        (void)la::gemm(la::reference, 1.0f, la::matrix_view<const float, right>(a.x.data(), {n, d}),
                       la::transposed(la::matrix_view<const float, right>(a.w.data(), {o, d})), 0.0f, la::matrix_view<float, right>(yr.data(), {n, o}));
        float worst = 0.0f;
        for (std::size_t i = 0; i < yr.size(); ++i)
            worst = std::fmax(worst, std::fmax(std::fabs(a.y[i] - yr[i]), std::fabs(ya[i] - yr[i])));
        // the reference's to rounding: |error| <= d eps max|x||w| = d eps / 4 for these inputs in [-1/2, 1/2)
        if (worst > float(d) * 0x1.0p-23f / 4) {
            std::fprintf(stderr, "cuda benchmark: %zu x %zu x %zu differs from the reference by %g\n", n, d, o, double(worst));
            return false;
        }
    }
    step_t s, r;
    train(gpu::cuda_backend{&device()}, s);
    train(la::reference, r);
    float worst = 0.0f;
    for (std::size_t i = 0; i < s.dxs.size(); ++i)
        worst = std::fmax(worst, std::fabs(s.dxs[i] - r.dxs[i]) / (1 + std::fabs(r.dxs[i])));
    if (worst > 1e-4f) {
        std::fprintf(stderr, "cuda benchmark: the training step differs from the reference by %g\n", double(worst));
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "cuda benchmark: fairness check FAILED - the results disagree; not benchmarking\n");
        return 1;
    }
    std::printf("device: %s\n", device().device_name().c_str());
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
