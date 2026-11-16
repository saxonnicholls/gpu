//
//  cuda_test.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/gpu/cuda.hpp>: core's linalg BLAS conformance suite on the device, the reference's results to
//  rounding for every shape and layout from host memory and from device memory in place, the gaps of a strided result
//  kept, device buffers, threads sharing one context, host-only operations refusing device memory, and the context's
//  reports. They need a CUDA device.
//

#include <snicholls/gpu/cuda.hpp>

#include <snicholls/core/conformance/linalg.hpp>
#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace gpu = snicholls::gpu;
namespace la = snicholls::linalg;

const gpu::cuda_context& shared_context()
{
    static const auto c = [] {
        auto made = gpu::cuda_context::create();
        if (made.is_err()) {
            std::fprintf(stderr, "cuda_test: no CUDA device (error %d, code %d)\n", int(made.unwrap_err().kind), made.unwrap_err().code);
            std::abort();
        }
        return std::move(made).unwrap();
    }();
    return *c;
}

} // namespace

namespace snicholls::gpu::testing {
struct cuda_device_backend : cuda_backend {
    cuda_device_backend() : cuda_backend{&shared_context()} {}
};
static_assert(std::is_trivially_copyable_v<cuda_device_backend> && la::blas_backend<cuda_device_backend, float>);
} // namespace snicholls::gpu::testing

// The backend implements the BLAS part of the interface only; the other suites are not for it
namespace snicholls::conformance {
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LinalgFactorisationConformance);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LinalgDecompositionConformance);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LinalgBatchedConformance);
} // namespace snicholls::conformance

SNICHOLLS_LINALG_CONFORMANCE_BLAS(Cuda, snicholls::conformance::linalg_case<snicholls::gpu::testing::cuda_device_backend, float>,
                                  snicholls::conformance::linalg_case<snicholls::gpu::testing::cuda_device_backend, double>);

namespace {

static_assert(la::blas_backend<gpu::cuda_backend, float> && la::blas_backend<gpu::cuda_backend, double>);
static_assert(std::is_trivially_copyable_v<gpu::cuda_backend> && sizeof(gpu::cuda_backend) == 2 * sizeof(void*));   // the context, the math
static_assert(!std::is_copy_constructible_v<gpu::cuda_context> && !std::is_copy_constructible_v<gpu::device_buffer<float>>);

template <typename T>
std::vector<T> stream(std::size_t n, std::uint64_t seed)
{
    std::vector<T> v(n);
    snicholls::stats::wyrand g(seed);
    for (T& e : v)
        e = static_cast<T>(2 * snicholls::stats::uniform_real(g) - 1);
    return v;
}

template <typename T>
T tolerance(std::size_t k)
{
    return T(8) * std::numeric_limits<T>::epsilon() * T(std::max<std::size_t>(k, 1));   // |entries| <= 1: |error| <= k eps
}

// C = alpha op(A) op(B) + beta C on the device and through the reference, from host memory
template <typename T>
void agrees(std::size_t m, std::size_t n, std::size_t k, bool ta, bool tb, T beta)
{
    using R = la::layout_right;
    const auto a = stream<T>(m * k, 1 + m), b = stream<T>(k * n, 2 + n);
    auto c1 = stream<T>(m * n, 3 + k), c2 = c1;
    const la::matrix_view<const T> av = ta ? la::matrix_view<const T>(la::transposed(la::matrix_view<const T, R>(a.data(), {k, m})))
                                           : la::matrix_view<const T>(la::matrix_view<const T, R>(a.data(), {m, k}));
    const la::matrix_view<const T> bv = tb ? la::matrix_view<const T>(la::transposed(la::matrix_view<const T, R>(b.data(), {n, k})))
                                           : la::matrix_view<const T>(la::matrix_view<const T, R>(b.data(), {k, n}));
    ASSERT_TRUE(la::gemm(la::reference, T(0.75), av, bv, beta, la::matrix_view<T, R>(c1.data(), {m, n})).is_ok());
    ASSERT_TRUE(la::gemm(gpu::cuda_backend{&shared_context()}, T(0.75), av, bv, beta, la::matrix_view<T, R>(c2.data(), {m, n})).is_ok());
    for (std::size_t i = 0; i < c1.size(); ++i)
        ASSERT_NEAR(c1[i], c2[i], tolerance<T>(k) * 2) << m << "x" << n << "x" << k << (ta ? " A^T" : "") << (tb ? " B^T" : "") << " at " << i;
}

TEST(GpuCuda, GemmAgreesWithTheReferenceForEveryShapeAndLayout)
{
    for (const auto [m, n, k] : {std::array<std::size_t, 3>{1, 1, 1}, {3, 5, 7}, {64, 64, 16}, {65, 63, 17}, {130, 70, 200}, {7, 129, 64}})
        for (const bool ta : {false, true})
            for (const bool tb : {false, true}) {
                agrees<float>(m, n, k, ta, tb, 0.0f);
                agrees<float>(m, n, k, ta, tb, -1.5f);
                agrees<double>(m, n, k, ta, tb, 0.25);
            }
}

TEST(GpuCuda, DeviceMemoryIsUsedInPlace)
{
    // Y = X Wᵀ with every operand in device buffers, against the reference on host copies
    constexpr std::size_t n = 50, d = 70, out = 40;
    const auto x = stream<float>(n * d, 5), w = stream<float>(out * d, 6);
    const auto& ctx = shared_context();
    gpu::device_buffer<float> dx(ctx, n * d), dw(ctx, out * d), dy(ctx, n * out);
    dx.upload(x);
    dw.upload(w);
    using R = la::layout_right;
    const auto xs = dx.span(), ws = dw.span(), ys = dy.span();
    ASSERT_TRUE(la::gemm(gpu::cuda_backend{&ctx}, 1.0f, la::matrix_view<const float, R>(xs.data(), {n, d}),
                         la::transposed(la::matrix_view<const float, R>(ws.data(), {out, d})), 0.0f, la::matrix_view<float, R>(ys.data(), {n, out}))
                    .is_ok());
    std::vector<float> got(n * out), want(n * out);
    dy.download(got);
    ASSERT_TRUE(la::gemm(la::reference, 1.0f, la::matrix_view<const float, R>(x.data(), {n, d}),
                         la::transposed(la::matrix_view<const float, R>(w.data(), {out, d})), 0.0f, la::matrix_view<float, R>(want.data(), {n, out}))
                    .is_ok());
    for (std::size_t i = 0; i < got.size(); ++i)
        ASSERT_NEAR(got[i], want[i], tolerance<float>(d)) << i;
}

// On the tensor cores the inputs are rounded to TF32 (10 bits of mantissa): each element within a few parts in 10^4 of
// the largest, and not the float result's bits
TEST(GpuCuda, Tf32ProductsAreWithinTheirRoundingOfFloats)
{
    constexpr std::size_t m = 96, n = 80, k = 384;
    const auto a = stream<float>(m * k, 21), b = stream<float>(k * n, 22);
    using R = la::layout_right;
    std::vector<float> want(m * n), got(m * n);
    gpu::cuda_backend tf32{&shared_context()};
    tf32.math = gpu::cuda_math::tf32;
    ASSERT_TRUE(la::gemm(gpu::cuda_backend{&shared_context()}, 1.0f, la::matrix_view<const float, R>(a.data(), {m, k}),
                         la::matrix_view<const float, R>(b.data(), {k, n}), 0.0f, la::matrix_view<float, R>(want.data(), {m, n}))
                    .is_ok());
    ASSERT_TRUE(la::gemm(tf32, 1.0f, la::matrix_view<const float, R>(a.data(), {m, k}), la::matrix_view<const float, R>(b.data(), {k, n}), 0.0f,
                         la::matrix_view<float, R>(got.data(), {m, n}))
                    .is_ok());
    double diff = 0, scale = 0;
    for (std::size_t i = 0; i < want.size(); ++i)
        diff = std::max(diff, double(std::fabs(got[i] - want[i]))), scale = std::max(scale, double(std::fabs(want[i])));
    std::printf("  tf32 against float: normwise %.2e\n", diff / scale);
    EXPECT_LE(diff, 2e-3 * scale);
    EXPECT_GT(diff, 0.0);   // the tensor cores were used: TF32 is not float
}

// A product over device memory is queued, not waited for: a download after it, or synchronize(), sees its result
TEST(GpuCuda, ADeviceProductIsQueuedAndItsResultSeenAfterwards)
{
    constexpr std::size_t m = 256, n = 256, k = 256;
    const auto a = stream<float>(m * k, 31), b = stream<float>(k * n, 32);
    using R = la::layout_right;
    gpu::device_buffer<float> da(shared_context(), a.size()), db(shared_context(), b.size()), dc(shared_context(), m * n);
    da.upload(a), db.upload(b);
    std::vector<float> want(m * n), got(m * n), again(m * n);
    ASSERT_TRUE(la::gemm(la::reference, 1.0f, la::matrix_view<const float, R>(a.data(), {m, k}), la::matrix_view<const float, R>(b.data(), {k, n}),
                         0.0f, la::matrix_view<float, R>(want.data(), {m, n}))
                    .is_ok());
    for (int round = 0; round < 3; ++round)   // several queued products, each overwriting the last
        ASSERT_TRUE(la::gemm(gpu::cuda_backend{&shared_context()}, 1.0f, la::matrix_view<const float, R>(da.span().data(), {m, k}),
                             la::matrix_view<const float, R>(db.span().data(), {k, n}), 0.0f, la::matrix_view<float, R>(dc.span().data(), {m, n}))
                        .is_ok());
    dc.download(got);
    shared_context().synchronize();
    dc.download(again);
    EXPECT_EQ(got, again);
    for (std::size_t i = 0; i < want.size(); ++i)
        ASSERT_NEAR(got[i], want[i], tolerance<float>(k)) << i;
}

TEST(GpuCuda, AStridedResultKeepsItsGapsFromHostAndDeviceMemory)
{
    constexpr std::size_t m = 4, n = 3, k = 6;
    const auto a = stream<float>(m * k, 7), b = stream<float>(k * n, 8);
    using R = la::layout_right;
    std::vector<float> want(m * 2 * n, -7777.0f), host = want;
    auto strided = [&](float* p) { return la::matrix_view<float>(p, {m, n}, {2 * n, 2}); };
    ASSERT_TRUE(la::gemm(la::reference, 1.0f, la::matrix_view<const float, R>(a.data(), {m, k}), la::matrix_view<const float, R>(b.data(), {k, n}),
                         0.0f, strided(want.data()))
                    .is_ok());
    ASSERT_TRUE(la::gemm(gpu::cuda_backend{&shared_context()}, 1.0f, la::matrix_view<const float, R>(a.data(), {m, k}),
                         la::matrix_view<const float, R>(b.data(), {k, n}), 0.0f, strided(host.data()))
                    .is_ok());
    gpu::device_buffer<float> dc(shared_context(), want.size());
    dc.upload(std::vector<float>(want.size(), -7777.0f));
    ASSERT_TRUE(la::gemm(gpu::cuda_backend{&shared_context()}, 1.0f, la::matrix_view<const float, R>(a.data(), {m, k}),
                         la::matrix_view<const float, R>(b.data(), {k, n}), 0.0f, strided(dc.span().data()))
                    .is_ok());
    std::vector<float> device(want.size());
    dc.download(device);
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(host[i], want[i], 1e-6f) << i;
        EXPECT_NEAR(device[i], want[i], 1e-6f) << i;
    }
    for (std::size_t i = 1; i < want.size(); i += 2)
        EXPECT_TRUE(host[i] == -7777.0f && device[i] == -7777.0f) << i;
}

TEST(GpuCuda, GemvAgreesWithTheReference)
{
    constexpr std::size_t m = 70, k = 33;
    const auto a = stream<double>(m * k, 11), x = stream<double>(k, 12);
    auto y1 = stream<double>(m, 13), y2 = y1;
    using R = la::layout_right;
    const la::matrix_view<const double, R> av(a.data(), {m, k});
    ASSERT_TRUE(la::gemv(la::reference, 2.0, av, la::vector_view<const double, R>(x.data(), {k}), 0.5, la::vector_view<double, R>(y1.data(), {m})).is_ok());
    ASSERT_TRUE(la::gemv(gpu::cuda_backend{&shared_context()}, 2.0, av, la::vector_view<const double, R>(x.data(), {k}), 0.5,
                         la::vector_view<double, R>(y2.data(), {m}))
                    .is_ok());
    for (std::size_t i = 0; i < m; ++i)
        EXPECT_NEAR(y1[i], y2[i], tolerance<double>(k) * 4) << i;
}

TEST(GpuCuda, ThreadsSharingOneContextAgree)
{
    constexpr std::size_t m = 96, n = 80, k = 72;
    const auto a = stream<float>(m * k, 21), b = stream<float>(k * n, 22);
    using R = la::layout_right;
    std::vector<std::vector<float>> got(4, std::vector<float>(m * n));
    std::vector<std::thread> threads;
    for (auto& g : got)
        threads.emplace_back([&] {
            for (int r = 0; r < 5; ++r)
                (void)la::gemm(gpu::cuda_backend{&shared_context()}, 1.0f, la::matrix_view<const float, R>(a.data(), {m, k}),
                               la::matrix_view<const float, R>(b.data(), {k, n}), 0.0f, la::matrix_view<float, R>(g.data(), {m, n}));
        });
    for (auto& t : threads)
        t.join();
    for (const auto& g : got)
        EXPECT_EQ(g, got[0]);   // cuBLAS on one device and configuration: the same bits every time
}

TEST(GpuCuda, HostOnlyOperationsRefuseDeviceMemory)
{
    gpu::device_buffer<float> d(shared_context(), 4);
    std::vector<float> h(4, 1.0f);
    using R = la::layout_right;
    EXPECT_THROW((void)la::dot(gpu::cuda_backend{&shared_context()}, la::vector_view<const float, R>(d.span().data(), {4}),
                               la::vector_view<const float, R>(h.data(), {4})),
                 gpu::cuda_exception);
    EXPECT_FLOAT_EQ(la::dot(gpu::cuda_backend{&shared_context()}, la::vector_view<const float, R>(h.data(), {4}),
                            la::vector_view<const float, R>(h.data(), {4}))
                        .unwrap(),
                    4.0f);
}

TEST(GpuCuda, BuffersCopyBothWaysAndCheckTheirSizes)
{
    gpu::device_buffer<double> d(shared_context(), 3);
    EXPECT_EQ(d.size(), 3u);
    d.upload(std::vector<double>{1, 2, 3});
    std::vector<double> back(3);
    d.download(back);
    EXPECT_EQ(back, (std::vector<double>{1, 2, 3}));
    EXPECT_THROW(d.upload(std::vector<double>(2)), std::invalid_argument);
    gpu::device_buffer<float> empty(shared_context(), 0);
    EXPECT_TRUE(empty.span().empty());
}

TEST(GpuCuda, TheContextReportsItsDeviceAndAMissingOne)
{
    EXPECT_FALSE(shared_context().device_name().empty());
    const auto none = gpu::cuda_context::create(1000);
    ASSERT_TRUE(none.is_err());
    EXPECT_EQ(none.unwrap_err().kind, gpu::cuda_error::no_device);
}

} // namespace
