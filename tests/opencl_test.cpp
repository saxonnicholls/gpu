//
//  opencl_test.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/gpu/opencl.hpp>: core's linalg BLAS conformance suite on the device, the reference's bits for
//  every layout and shape, the gaps of a strided result kept, threads sharing one context, and the context's
//  reports. They need an OpenCL device: a GPU, or a CPU implementation (PoCL in the Ubuntu image and on CI).
//

#include <snicholls/gpu/opencl.hpp>

#include <snicholls/core/conformance/linalg.hpp>
#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace gpu = snicholls::gpu;
namespace la = snicholls::linalg;

// One context for the whole test binary, on any device
const gpu::opencl_context& shared_context()
{
    static const auto c = [] {
        auto made = gpu::opencl_context::create(gpu::device_choice::any);
        if (made.is_err()) {
            std::fprintf(stderr, "opencl_test: no OpenCL device (error %d, code %d)\n", int(made.unwrap_err().kind),
                         int(made.unwrap_err().code));
            std::abort();
        }
        return std::move(made).unwrap();
    }();
    return *c;
}

} // namespace

// The conformance suite value-initialises its backend, so it is registered as a backend whose default constructor
// takes the process's context: still one pointer, still trivially copyable.
namespace snicholls::gpu::testing {
struct device_backend : opencl_backend {
    device_backend() : opencl_backend{&shared_context()} {}
};
static_assert(std::is_trivially_copyable_v<device_backend> && la::blas_backend<device_backend, float>);
} // namespace snicholls::gpu::testing

// The backend implements the BLAS part of the interface only; the other suites are not for it
namespace snicholls::conformance {
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LinalgFactorisationConformance);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LinalgDecompositionConformance);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LinalgBatchedConformance);
} // namespace snicholls::conformance

SNICHOLLS_LINALG_CONFORMANCE_BLAS(OpenCl, snicholls::conformance::linalg_case<snicholls::gpu::testing::device_backend, float>,
                                  snicholls::conformance::linalg_case<snicholls::gpu::testing::device_backend, double>);

namespace {

static_assert(la::blas_backend<gpu::opencl_backend, float> && la::blas_backend<gpu::opencl_backend, double>);
static_assert(std::is_trivially_copyable_v<gpu::opencl_backend> && sizeof(gpu::opencl_backend) == sizeof(void*));
static_assert(std::is_base_of_v<std::runtime_error, gpu::opencl_exception>);
static_assert(!std::is_copy_constructible_v<gpu::opencl_context> && !std::is_move_constructible_v<gpu::opencl_context>);

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
bool same_bits(const std::vector<T>& a, const std::vector<T>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

// C = alpha op(A) op(B) + beta C through both backends, for row-major storage read directly or through transposed()
template <typename T>
void reference_bits(std::size_t m, std::size_t n, std::size_t k, bool ta, bool tb, T beta)
{
    using R = la::layout_right;
    const auto a = stream<T>(m * k, 1 + m), b = stream<T>(k * n, 2 + n);
    auto c1 = stream<T>(m * n, 3 + k), c2 = c1;
    const la::matrix_view<const T> av = ta ? la::matrix_view<const T>(la::transposed(la::matrix_view<const T, R>(a.data(), {k, m})))
                                           : la::matrix_view<const T>(la::matrix_view<const T, R>(a.data(), {m, k}));
    const la::matrix_view<const T> bv = tb ? la::matrix_view<const T>(la::transposed(la::matrix_view<const T, R>(b.data(), {n, k})))
                                           : la::matrix_view<const T>(la::matrix_view<const T, R>(b.data(), {k, n}));
    const T alpha = T(0.75);
    ASSERT_TRUE(la::gemm(la::reference, alpha, av, bv, beta, la::matrix_view<T, R>(c1.data(), {m, n})).is_ok());
    ASSERT_TRUE(la::gemm(gpu::opencl_backend{&shared_context()}, alpha, av, bv, beta, la::matrix_view<T, R>(c2.data(), {m, n})).is_ok());
    EXPECT_TRUE(same_bits(c1, c2)) << m << " x " << n << " x " << k << (ta ? " A^T" : "") << (tb ? " B^T" : "") << " beta " << beta;
}

TEST(GpuOpenCl, GemmGivesTheReferenceBitsForEveryShapeAndLayout)
{
    for (const auto [m, n, k] : {std::array<std::size_t, 3>{1, 1, 1}, {3, 5, 7}, {64, 64, 16}, {65, 63, 17}, {130, 70, 200}, {7, 129, 64}})
        for (const bool ta : {false, true})
            for (const bool tb : {false, true}) {
                reference_bits<float>(m, n, k, ta, tb, 0.0f);
                reference_bits<float>(m, n, k, ta, tb, -1.5f);
                reference_bits<double>(m, n, k, ta, tb, 0.25);
            }
}

TEST(GpuOpenCl, AnEmptyInnerDimensionScalesCAndAnEmptyResultIsANoOp)
{
    reference_bits<float>(5, 6, 0, false, false, 2.0f);   // C = beta C
    reference_bits<float>(5, 6, 0, false, false, 0.0f);   // C = 0
    std::vector<float> none;
    const std::vector<float> a(12, 1.0f);
    using R = la::layout_right;
    EXPECT_TRUE(la::gemm(gpu::opencl_backend{&shared_context()}, 1.0f, la::matrix_view<const float, R>(a.data(), {0, 12}),
                         la::matrix_view<const float, R>(a.data(), {12, 0}), 0.0f, la::matrix_view<float, R>(none.data(), {0, 0}))
                    .is_ok());
}

TEST(GpuOpenCl, AStridedResultKeepsItsGaps)
{
    // A 4 x 3 result inside a 4 x 5 buffer with a column gap between elements: the gaps must keep their sentinel
    constexpr std::size_t m = 4, n = 3, k = 6;
    const auto a = stream<float>(m * k, 7), b = stream<float>(k * n, 8);
    std::vector<float> c1(m * 2 * n, -7777.0f), c2 = c1;
    using R = la::layout_right;
    auto strided = [&](std::vector<float>& c) { return la::matrix_view<float>(c.data(), {m, n}, {2 * n, 2}); };
    ASSERT_TRUE(la::gemm(la::reference, 1.0f, la::matrix_view<const float, R>(a.data(), {m, k}),
                         la::matrix_view<const float, R>(b.data(), {k, n}), 0.0f, strided(c1))
                    .is_ok());
    ASSERT_TRUE(la::gemm(gpu::opencl_backend{&shared_context()}, 1.0f, la::matrix_view<const float, R>(a.data(), {m, k}),
                         la::matrix_view<const float, R>(b.data(), {k, n}), 0.0f, strided(c2))
                    .is_ok());
    EXPECT_TRUE(same_bits(c1, c2));
    for (std::size_t i = 1; i < c2.size(); i += 2)
        EXPECT_EQ(c2[i], -7777.0f) << i;
}

TEST(GpuOpenCl, GemvGivesTheReferenceBits)
{
    constexpr std::size_t m = 70, k = 33;
    const auto a = stream<float>(m * k, 11), x = stream<float>(k, 12);
    auto y1 = stream<float>(m, 13), y2 = y1;
    using R = la::layout_right;
    const la::matrix_view<const float, R> av(a.data(), {m, k});
    ASSERT_TRUE(la::gemv(la::reference, 2.0f, av, la::vector_view<const float, R>(x.data(), {k}), 0.5f,
                         la::vector_view<float, R>(y1.data(), {m}))
                    .is_ok());
    ASSERT_TRUE(la::gemv(gpu::opencl_backend{&shared_context()}, 2.0f, av, la::vector_view<const float, R>(x.data(), {k}), 0.5f,
                         la::vector_view<float, R>(y2.data(), {m}))
                    .is_ok());
    EXPECT_TRUE(same_bits(y1, y2));
}

#if defined(__SANITIZE_ADDRESS__)
#define SNICHOLLS_GPU_TEST_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define SNICHOLLS_GPU_TEST_ASAN 1
#endif
#endif

TEST(GpuOpenCl, ThreadsSharingOneContextGetTheSameBits)
{
#ifdef SNICHOLLS_GPU_TEST_ASAN
    // Under AddressSanitizer with PoCL's worker threads, ASan's own munmap fails (EINVAL) inside its runtime, PoCL
    // limited to one thread included; ThreadSanitizer is the tool for this test, and the Ubuntu gate runs it there
    GTEST_SKIP() << "threads are checked under ThreadSanitizer (scripts/test_ubuntu.sh), not AddressSanitizer";
#endif
    constexpr std::size_t m = 96, n = 80, k = 72;
    const auto a = stream<float>(m * k, 21), b = stream<float>(k * n, 22);
    using R = la::layout_right;
    std::vector<float> want(m * n);
    ASSERT_TRUE(la::gemm(la::reference, 1.0f, la::matrix_view<const float, R>(a.data(), {m, k}), la::matrix_view<const float, R>(b.data(), {k, n}),
                         0.0f, la::matrix_view<float, R>(want.data(), {m, n}))
                    .is_ok());
    std::vector<std::vector<float>> got(4, std::vector<float>(m * n));
    std::vector<std::thread> threads;
    for (auto& g : got)
        threads.emplace_back([&] {
            for (int r = 0; r < 5; ++r)
                (void)la::gemm(gpu::opencl_backend{&shared_context()}, 1.0f, la::matrix_view<const float, R>(a.data(), {m, k}),
                               la::matrix_view<const float, R>(b.data(), {k, n}), 0.0f, la::matrix_view<float, R>(g.data(), {m, n}));
        });
    for (auto& t : threads)
        t.join();
    for (const auto& g : got)
        EXPECT_TRUE(same_bits(g, want));
}

TEST(GpuOpenCl, AnUnsetContextRunsOnTheHost)
{
    // A default backend has no context: every operation is the reference's
    constexpr std::size_t m = 5, n = 4, k = 3;
    const auto a = stream<float>(m * k, 31), b = stream<float>(k * n, 32);
    std::vector<float> c1(m * n), c2(m * n);
    using R = la::layout_right;
    ASSERT_TRUE(la::gemm(la::reference, 1.0f, la::matrix_view<const float, R>(a.data(), {m, k}), la::matrix_view<const float, R>(b.data(), {k, n}),
                         0.0f, la::matrix_view<float, R>(c1.data(), {m, n}))
                    .is_ok());
    ASSERT_TRUE(la::gemm(gpu::opencl_backend{}, 1.0f, la::matrix_view<const float, R>(a.data(), {m, k}),
                         la::matrix_view<const float, R>(b.data(), {k, n}), 0.0f, la::matrix_view<float, R>(c2.data(), {m, n}))
                    .is_ok());
    EXPECT_TRUE(same_bits(c1, c2));
}

TEST(GpuOpenCl, TheContextReportsItsDevice)
{
    EXPECT_FALSE(shared_context().device_name().empty());
    // Asking for a GPU either finds one or reports that there is none
    auto g = gpu::opencl_context::create(gpu::device_choice::gpu);
    if (g.is_ok())
        EXPECT_TRUE(std::move(g).unwrap()->is_gpu());
    else
        EXPECT_EQ(g.unwrap_err().kind, gpu::opencl_error::no_device);
}

TEST(GpuOpenCl, TheNativeHandlesAreTheContextsOwn)
{
    const auto& c = shared_context();
    ASSERT_NE(c.native_device(), nullptr);
    ASSERT_NE(c.native_context(), nullptr);
    ASSERT_NE(c.native_queue(), nullptr);
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    ASSERT_EQ(clGetCommandQueueInfo(c.native_queue(), CL_QUEUE_DEVICE, sizeof(device), &device, nullptr), CL_SUCCESS);
    ASSERT_EQ(clGetCommandQueueInfo(c.native_queue(), CL_QUEUE_CONTEXT, sizeof(context), &context, nullptr), CL_SUCCESS);
    EXPECT_EQ(device, c.native_device());
    EXPECT_EQ(context, c.native_context());
}

TEST(GpuOpenCl, AnExceptionCarriesItsCode)
{
    const gpu::opencl_exception e("clEnqueueNDRangeKernel", CL_OUT_OF_RESOURCES);
    EXPECT_EQ(e.code(), CL_OUT_OF_RESOURCES);
    EXPECT_NE(std::string(e.what()).find("clEnqueueNDRangeKernel"), std::string::npos);
}

} // namespace
