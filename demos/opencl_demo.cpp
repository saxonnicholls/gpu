//
//  opencl_demo.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/gpu/opencl.hpp>: a matrix product on the GPU through core's linalg::gemm, the same call that
//  runs on the CPU with core's reference backend, and the same bits. Exits non-zero if a result is not what it
//  documents, or if there is no OpenCL device.
//

#include <snicholls/gpu/opencl.hpp>

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/linalg/reference.hpp>

#include <cstddef>
#include <cstdio>
#include <utility>
#include <vector>

int main()
{
    std::printf("snicholls-gpu opencl demo\n");
    namespace gpu = snicholls::gpu;
    namespace la = snicholls::linalg;

    //! [opencl]
    // One context per device, created once: it finds a GPU (or, with device_choice::any, any OpenCL device) and
    // compiles the kernels
    auto made = gpu::opencl_context::create(gpu::device_choice::any);
    if (made.is_err()) {
        std::printf("opencl demo: FAILED - no OpenCL device (error %d)\n", static_cast<int>(made.unwrap_err().kind));
        return 1;
    }
    const auto context = std::move(made).unwrap();
    const gpu::opencl_backend device{context.get()};   // one pointer: pass it by value

    // Y = X Wᵀ for 128 tokens of 256 channels into 512: a transformer's projection, as ml::linear_sequence asks for it
    constexpr std::size_t n = 128, d = 256, out = 512;
    std::vector<float> x(n * d), w(out * d), on_gpu(n * out), on_cpu(n * out);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = static_cast<float>(i % 7) * 0.05f - 0.15f;
    using right = la::layout_right;
    const la::matrix_view<const float, right> xv(x.data(), {n, d}), wv(w.data(), {out, d});
    const auto gpu_done = la::gemm(device, 1.0f, xv, la::transposed(wv), 0.0f, la::matrix_view<float, right>(on_gpu.data(), {n, out}));
    const auto cpu_done = la::gemm(la::reference, 1.0f, xv, la::transposed(wv), 0.0f, la::matrix_view<float, right>(on_cpu.data(), {n, out}));
    //! [opencl]

    std::printf("  device: %s (%s, double on the %s)\n", context->device_name().c_str(), context->is_gpu() ? "a GPU" : "not a GPU",
                context->has_fp64() ? "device" : "host");
    std::printf("  Y[0][0] %.6f, Y[127][511] %.6f\n", static_cast<double>(on_gpu[0]), static_cast<double>(on_gpu.back()));
    const bool same = on_gpu == on_cpu;
    std::printf("  the device and core's reference: %s\n", same ? "the same bits" : "DIFFERENT");
    const bool ok = gpu_done.is_ok() && cpu_done.is_ok() && same;
    std::printf(ok ? "opencl demo: ok\n" : "opencl demo: FAILED\n");
    return ok ? 0 : 1;
}
