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

#include <snicholls/gpu/opencl.hpp>

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/linalg/reference.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
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
    return true;
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
