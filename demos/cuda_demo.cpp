//
//  cuda_demo.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/gpu/cuda.hpp>: a projection Y = X Wᵀ on an NVIDIA GPU through core's linalg::gemm, from host
//  memory and with the operands kept in device memory, against core's reference; then the device product captured
//  as a CUDA graph and replayed. Exits non-zero if a result is not
//  what it documents, or if there is no CUDA device.
//

#include <snicholls/gpu/cuda.hpp>

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/linalg/reference.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <utility>
#include <vector>

int main()
{
    std::printf("snicholls-gpu cuda demo\n");
    namespace gpu = snicholls::gpu;
    namespace la = snicholls::linalg;
    using right = la::layout_right;

    //! [cuda]
    auto made = gpu::cuda_context::create();   // device 0: its cuBLAS handle and stream
    if (made.is_err()) {
        std::printf("cuda demo: FAILED - no CUDA device\n");
        return 1;
    }
    const auto context = std::move(made).unwrap();
    const gpu::cuda_backend device{context.get()};

    constexpr std::size_t n = 128, d = 256, out = 512;
    std::vector<float> x(n * d), w(out * d), from_host(n * out), reference(n * out);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = static_cast<float>(i % 7) * 0.05f - 0.15f;
    const la::matrix_view<const float, right> xv(x.data(), {n, d}), wv(w.data(), {out, d});

    // From host memory: staged to the device and back, as any backend
    const auto staged = la::gemm(device, 1.0f, xv, la::transposed(wv), 0.0f, la::matrix_view<float, right>(from_host.data(), {n, out}));

    // From device memory: the operands stay on the GPU; core's views point at it (never read on the host)
    gpu::device_buffer<float> dx(*context, x.size()), dw(*context, w.size()), dy(*context, n * out);
    dx.upload(x);
    dw.upload(w);
    const auto resident = la::gemm(device, 1.0f, la::matrix_view<const float, right>(dx.span().data(), {n, d}),
                                   la::transposed(la::matrix_view<const float, right>(dw.span().data(), {out, d})), 0.0f,
                                   la::matrix_view<float, right>(dy.span().data(), {n, out}));
    std::vector<float> from_device(n * out);
    dy.download(from_device);
    //! [cuda]

    //! [cuda_graph]
    // The device product captured once as a CUDA graph and replayed: new inputs go into the same buffers first
    const auto product = [&] {
        (void)la::gemm(device, 1.0f, la::matrix_view<const float, right>(dx.span().data(), {n, d}),
                       la::transposed(la::matrix_view<const float, right>(dw.span().data(), {out, d})), 0.0f,
                       la::matrix_view<float, right>(dy.span().data(), {n, out}));
    };
    auto captured = gpu::cuda_graph::capture(*context, product);
    if (captured.is_err()) {
        std::printf("cuda demo: FAILED - the capture failed with CUDA code %d\n", captured.unwrap_err().code);
        return 1;
    }
    const auto graph = std::move(captured).unwrap();
    dy.upload(std::vector<float>(n * out, 0.0f));
    graph.launch();   // queued on the context's stream, as the product itself would be
    std::vector<float> replayed(n * out);
    dy.download(replayed);
    //! [cuda_graph]
    std::printf("  the product captured as a graph of %zu node(s); its replay %s the direct product's bits\n", graph.nodes(),
                replayed == from_device ? "gives" : "does NOT give");

    (void)la::gemm(la::reference, 1.0f, xv, la::transposed(wv), 0.0f, la::matrix_view<float, right>(reference.data(), {n, out}));
    float worst = 0.0f;
    for (std::size_t i = 0; i < reference.size(); ++i)
        worst = std::fmax(worst, std::fmax(std::fabs(from_host[i] - reference[i]), std::fabs(from_device[i] - reference[i])));
    std::printf("  device: %s\n  both forms within %.2g of core's reference (256 products of order 1 per output)\n",
                context->device_name().c_str(), static_cast<double>(worst));
    const bool ok = staged.is_ok() && resident.is_ok() && worst < 1e-4f && replayed == from_device;
    std::printf(ok ? "cuda demo: ok\n" : "cuda demo: FAILED\n");
    return ok ? 0 : 1;
}
