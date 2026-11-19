//
//  opencl_quant_demo.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/gpu/opencl_quant.hpp>: a Q4_0 weight matrix kept on the device in its blocks, multiplied by
//  one token (decode) and by a batch (prefill), the input's gradient through it, and the integer path with Q8_0
//  activations; each checked against core on the host. Exits non-zero if a result is not what it documents, or if
//  there is no OpenCL device.
//

#include <snicholls/gpu/opencl_quant.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/lora.hpp>
#include <snicholls/core/ml/quantize.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <utility>
#include <vector>

int main()
{
    std::printf("snicholls-gpu opencl_quant demo\n");
    namespace gpu = snicholls::gpu;
    namespace ml = snicholls::ml;
    namespace la = snicholls::linalg;

    auto made = gpu::opencl_context::create(gpu::device_choice::any);
    if (made.is_err()) {
        std::printf("opencl_quant demo: FAILED - no OpenCL device\n");
        return 1;
    }
    const auto context = std::move(made).unwrap();
    constexpr std::size_t rows = 256, cols = 512, n = 64;
    std::vector<float> w(rows * cols), x(n * cols), dy(n * rows);
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = std::sin(0.37f * float(i)) * 0.5f;
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = std::cos(0.11f * float(i));
    for (std::size_t i = 0; i < dy.size(); ++i)
        dy[i] = std::sin(0.05f * float(i)) * 0.1f;

    //! [quant]
    // The kernels, built once for the context's device; the backend carries both
    auto built = gpu::opencl_quant_kernels::create(*context);
    if (built.is_err()) {
        std::printf("opencl_quant demo: FAILED - the program did not build\n%s\n", built.unwrap_err().log.c_str());
        return 1;
    }
    const auto kernels = std::move(built).unwrap();
    const gpu::opencl_quant_backend device{{context.get()}, kernels.get()};

    // Q4_0 weights, quantised by core; kept on the device, 4.5 bits a weight
    std::vector<ml::block_q4_0> blocks(rows * cols / 32);
    for (std::size_t i = 0; i < rows; ++i)
        ml::quantize(std::span<const float>(w.data() + i * cols, cols), std::span<ml::block_q4_0>(blocks.data() + i * cols / 32, cols / 32));
    const ml::quantized_weights<ml::block_q4_0> q4{blocks, rows, cols};
    const gpu::opencl_quant_matrix on_device = kernels->upload(q4);

    // Decode (one token: the matrix-vector kernel), prefill (64 tokens: the tiled kernel), the input's gradient
    std::vector<float> y1(rows), ys(n * rows), dx(n * cols, 0.0f), yq(rows);
    gpu::linear_sequence(device, on_device, std::span<const float>(x).first(cols), y1);
    gpu::linear_sequence(device, on_device, x, ys);
    gpu::linear_sequence_backward_input(device, on_device, dy, dx);
    // The integer path: the token quantised to Q8_0, ggml's dot products
    gpu::linear_sequence_q8(device, on_device, std::span<const float>(x).first(cols), yq);
    //! [quant]

    // Core on the host: the dequantised matrix through the reference product, and core's own Q8_0 product
    std::vector<float> dense(rows * cols);
    for (std::size_t i = 0; i < rows; ++i)
        ml::dequantize(q4.row(i), std::span<float>(dense.data() + i * cols, cols));
    std::vector<float> ys_ref(n * rows), dx_ref(n * cols, 0.0f), yq_ref(rows);
    ml::linear_sequence(la::reference, ml::dense_weights<float>{dense, rows, cols}, x, std::span<float>(ys_ref));
    ml::linear_sequence_backward_input(la::reference, ml::dense_weights<float>{dense, rows, cols}, dy, std::span<float>(dx_ref));
    ml::linear(q4, ml::prepare_input<cols>(q4, std::span<const float, cols>(x.data(), cols)), std::span<float>(yq_ref));
    double worst = 0.0;
    for (std::size_t i = 0; i < rows; ++i)
        worst = std::fmax(worst, std::fabs(double(y1[i]) - double(ys_ref[i])));

    std::printf("  device: %s; %zu x %zu Q4_0 weights, %zu bytes on the device (float: %zu)\n", context->device_name().c_str(), rows, cols,
                on_device.bytes(), rows * cols * sizeof(float));
    const bool prefill = ys == ys_ref, backward = dx == dx_ref, decode = worst < 1e-4, q8 = !kernels->exact_q8() || yq == yq_ref;
    std::printf("  prefill, 64 tokens: %s core's reference over the dequantised weights\n", prefill ? "the bits of" : "DIFFERENT from");
    std::printf("  the input's gradient: %s\n", backward ? "the reference's bits" : "DIFFERENT");
    std::printf("  decode, one token: within %.2g of the reference (a sum in another order)\n", worst);
    std::printf("  Q8_0 activations: %s core's ml::linear\n", q8 ? "the bits of" : "DIFFERENT from");
    const bool ok = prefill && backward && decode && q8;
    std::printf(ok ? "opencl_quant demo: ok\n" : "opencl_quant demo: FAILED\n");
    return ok ? 0 : 1;
}
