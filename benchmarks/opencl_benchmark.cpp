//
//  opencl_benchmark.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The OpenCL backend against snicholls-core's reference backend on the host: C = A B and C = X Wᵀ (a transformer's
//  projection, as ml::linear_sequence asks for it) through linalg::gemm, the same call, for the shapes of training a
//  small transformer. Device times include both transfers. A third row calls the context's product directly, without
//  core's free function, to show the interface costs nothing. 1024 x 1024 x 1024 runs on the device only (the
//  reference takes seconds), checked against the reference once. A fairness check refuses to benchmark unless every
//  result agrees to the bit.
//
//  Training: one GPT-2-small layer (768 wide, 12 heads, MLP of 3072, layer norm, tanh GELU, biases) over 128 tokens,
//  forward with its activations saved and backward, through core's batched passes (ml::block_forward_sequence and
//  ml::block_backward with a backend): on the device and on the host's reference, which must agree to the bit; and,
//  for context, core's per-token path on the host (its own order of summation, so not a pair).
//

#include <snicholls/gpu/opencl.hpp>

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/transformer_block.hpp>

#include <benchmark/benchmark.h>

#include <array>
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

const gpu::opencl_context* device()
{
    static const std::unique_ptr<gpu::opencl_context> c = [] {
        auto made = gpu::opencl_context::create(gpu::device_choice::any);
        return made.is_ok() ? std::move(made).unwrap() : nullptr;
    }();
    return c.get();
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

struct problem {
    std::size_t m, n, k;
    bool transposed;   // C = X Wᵀ with W stored n x k, else C = A B with B stored k x n
    std::vector<float> a, b, c;
    problem(std::size_t m_, std::size_t n_, std::size_t k_, bool t)
        : m(m_), n(n_), k(k_), transposed(t), a(stream(m * k, 1)), b(stream(k * n, 2)), c(m * n)
    {
    }
    la::matrix_view<const float> lhs() const { return la::matrix_view<const float, right>(a.data(), {m, k}); }
    la::matrix_view<const float> rhs() const
    {
        return transposed ? la::matrix_view<const float>(la::transposed(la::matrix_view<const float, right>(b.data(), {n, k})))
                          : la::matrix_view<const float>(la::matrix_view<const float, right>(b.data(), {k, n}));
    }
    la::matrix_view<float> out() { return la::matrix_view<float, right>(c.data(), {m, n}); }
};

template <typename B>
void product(const B& backend, problem& p)
{
    [[maybe_unused]] const auto done = la::gemm(backend, 1.0f, p.lhs(), p.rhs(), 0.0f, p.out());
}

void on_host(problem& p) { product(la::reference, p); }
void on_device(problem& p) { product(gpu::opencl_backend{device()}, p); }
void on_device_directly(problem& p) { device()->gemm(1.0f, p.lhs(), p.rhs(), 0.0f, p.out()); }

template <void (*Run)(problem&)>
void BM_Gemm(benchmark::State& state)
{
    problem p(std::size_t(state.range(0)), std::size_t(state.range(1)), std::size_t(state.range(2)), state.range(3) != 0);
    for (auto _ : state) {
        Run(p);
        benchmark::DoNotOptimize(p.c.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(2.0 * double(p.m * p.n * p.k), benchmark::Counter::kIsIterationInvariantRate,
                                                   benchmark::Counter::kIs1000);
}

// m, n, k, and 1 for X Wᵀ
void shapes(benchmark::Benchmark* b)
{
    for (const int t : {0, 1}) {
        b->Args({64, 256, 256, t});    // one projection of 64 tokens, 256 wide
        b->Args({64, 1024, 256, t});   // the MLP's up projection
        b->Args({512, 768, 768, t});   // GPT-2 small's attention projection over 512 tokens
    }
    b->Unit(benchmark::kMillisecond);
}
BENCHMARK(BM_Gemm<on_host>)->Name("BM_Gemm_HostReference")->Apply(shapes);
BENCHMARK(BM_Gemm<on_device>)->Name("BM_Gemm_Device")->Apply(shapes);
BENCHMARK(BM_Gemm<on_device_directly>)->Name("BM_Gemm_DeviceDirectly")->Apply(shapes);
BENCHMARK(BM_Gemm<on_device>)->Name("BM_Gemm_DeviceOnly")->Args({1024, 1024, 1024, 0})->Args({1024, 1024, 1024, 1})->Unit(benchmark::kMillisecond);

// ---- training a GPT-2-small layer ----

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

const layer_t& layer()
{
    static const layer_t l;
    return l;
}

// One step's buffers: the sequence, what the forward saves, and every gradient with the batched path's working space
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
void train_batched(const B& backend, step_t& s)
{
    static const auto w = layer().weights();
    ml::block_forward_sequence<tc>(backend, w, std::span<float>(s.xs), ml::no_rope, s.act());
    ml::block_backward<tc>(backend, w, ml::no_rope, s.act(), std::span<float>(s.dxs), s.grads());
}

void train_on_device(step_t& s) { train_batched(gpu::opencl_backend{device()}, s); }
void train_on_host(step_t& s) { train_batched(la::reference, s); }
void train_per_token(step_t& s)
{
    static const auto w = layer().weights();
    ml::block_forward_sequence<tc>(w, s.xs, ml::no_rope, s.act());
    ml::block_backward<tc>(w, ml::no_rope, s.act(), s.dxs, s.grads());
}

template <void (*Train)(step_t&)>
void BM_Train(benchmark::State& state)
{
    step_t s;
    const std::vector<float> x0 = s.xs, d0 = s.dxs;
    for (auto _ : state) {
        s.xs = x0;
        s.dxs = d0;   // the same step every iteration; the weights' gradients keep accumulating
        Train(s);
        benchmark::DoNotOptimize(s.dxs.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(tn));   // tokens
}
BENCHMARK(BM_Train<train_on_device>)->Name("BM_TrainGpt2SmallLayer_Device")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Train<train_on_host>)->Name("BM_TrainGpt2SmallLayer_HostReference")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Train<train_per_token>)->Name("BM_TrainGpt2SmallLayer_HostPerToken_Context")->Unit(benchmark::kMillisecond);

bool training_agrees()
{
    step_t d, h;
    train_on_device(d);
    train_on_host(h);
    for (const auto m : {&step_t::xs, &step_t::dxs, &step_t::gwq, &step_t::gwo, &step_t::gup, &step_t::gdown, &step_t::g1w, &step_t::gbq})
        if (std::memcmp((d.*m).data(), (h.*m).data(), (d.*m).size() * sizeof(float)) != 0)
            return false;
    return true;
}

bool fair()
{
    if (!device())
        return false;
    for (const std::array<std::size_t, 3> s : {std::array<std::size_t, 3>{64, 256, 256}, {64, 1024, 256}, {512, 768, 768}, {1024, 1024, 1024}})
        for (const bool t : {false, true}) {
            problem h(s[0], s[1], s[2], t), d(s[0], s[1], s[2], t), dd(s[0], s[1], s[2], t);
            on_host(h);
            on_device(d);
            on_device_directly(dd);
            const std::size_t bytes = h.c.size() * sizeof(float);
            if (std::memcmp(h.c.data(), d.c.data(), bytes) != 0 || std::memcmp(h.c.data(), dd.c.data(), bytes) != 0)
                return false;
        }
    return training_agrees();
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "opencl benchmark: no device, or the fairness check FAILED - the results disagree; not benchmarking\n");
        return 1;
    }
    std::printf("device: %s\n", device()->device_name().c_str());
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
