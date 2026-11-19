//
//  cuda_quant_demo.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/gpu/cuda_quant.hpp>: a Q4_0 weight matrix in device memory, multiplied by one token (decode) and
//  by a batch (prefill, on the CUDA cores and on the tensor cores), the input's gradient through it, and the integer
//  path with Q8_0 activations; each checked against core on the host. Exits non-zero if a result is not what it
//  documents, or if there is no CUDA device.
//

#include <snicholls/gpu/cuda_quant.hpp>

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
    std::printf("snicholls-gpu cuda_quant demo\n");
    namespace gpu = snicholls::gpu;
    namespace ml = snicholls::ml;
    namespace la = snicholls::linalg;

    auto made = gpu::cuda_context::create();
    if (made.is_err()) {
        std::printf("cuda_quant demo: FAILED - no CUDA device\n");
        return 1;
    }
    const auto context = std::move(made).unwrap();
    auto k = gpu::cuda_kernels::create(*context);
    if (k.is_err()) {
        std::printf("cuda_quant demo: FAILED - the kernels did not compile\n");
        return 1;
    }
    const auto ops = std::move(k).unwrap();
    constexpr std::size_t rows = 256, cols = 512, n = 64;
    std::vector<float> w(rows * cols), x(n * cols), dy(n * rows);
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = std::sin(0.37f * float(i)) * 0.5f;
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = std::cos(0.11f * float(i));
    for (std::size_t i = 0; i < dy.size(); ++i)
        dy[i] = std::sin(0.05f * float(i)) * 0.1f;
    std::vector<ml::block_q4_0> blocks(rows * cols / 32);
    for (std::size_t i = 0; i < rows; ++i)
        ml::quantize(std::span<const float>(w.data() + i * cols, cols), std::span<ml::block_q4_0>(blocks.data() + i * cols / 32, cols / 32));

    //! [quant]
    // The kernels, compiled once; the training backend carries them
    auto built = gpu::cuda_quant_kernels::create(*context);
    if (built.is_err()) {
        std::printf("cuda_quant demo: FAILED - the quantised kernels did not compile\n");
        return 1;
    }
    const auto quant = std::move(built).unwrap();
    gpu::cuda_training_backend device{{context.get(), gpu::cuda_math::float32}, ops.get()};
    device.quant = quant.get();

    // The Q4_0 blocks in device memory, and core's view over them
    gpu::device_buffer<ml::block_q4_0> on_device(*context, blocks.size());
    on_device.upload(blocks);
    const ml::quantized_weights<ml::block_q4_0> q4{on_device.span(), rows, cols};
    gpu::device_buffer<float> dx_in(*context, x.size()), ys(*context, n * rows), y1(*context, rows), yq(*context, rows), dyd(*context, dy.size()),
        dx(*context, n * cols), ys_tc(*context, n * rows);
    dx_in.upload(x);
    dyd.upload(dy);
    dx.upload(std::vector<float>(n * cols, 0.0f));
    const std::span<const float> xs(dx_in.span()), x1 = xs.first(cols);

    gpu::linear_sequence(device, q4, x1, y1.span());                                   // decode: one token
    gpu::linear_sequence(device, q4, xs, ys.span());                                   // prefill: 64 tokens, CUDA cores
    gpu::linear_sequence_backward_input(device, q4, std::span<const float>(dyd.span()), dx.span());   // the input's gradient
    gpu::linear_sequence_q8(device, q4, x1, yq.span());                                // Q8_0 activations, ggml's dot products
    gpu::cuda_training_backend tensor = device;
    tensor.math = gpu::cuda_math::bf16;
    gpu::linear_sequence(tensor, q4, xs, ys_tc.span());                                // prefill on the tensor cores
    //! [quant]

    context->synchronize();
    std::vector<float> h_ys(n * rows), h_y1(rows), h_yq(rows), h_dx(n * cols), h_tc(n * rows);
    ys.download(h_ys), y1.download(h_y1), yq.download(h_yq), dx.download(h_dx), ys_tc.download(h_tc);

    const ml::quantized_weights<ml::block_q4_0> host_q4{blocks, rows, cols};
    std::vector<float> dense(rows * cols);
    for (std::size_t i = 0; i < rows; ++i)
        ml::dequantize(host_q4.row(i), std::span<float>(dense.data() + i * cols, cols));
    std::vector<float> ys_ref(n * rows), dx_ref(n * cols, 0.0f), yq_ref(rows);
    ml::linear_sequence(la::reference, ml::dense_weights<float>{dense, rows, cols}, x, std::span<float>(ys_ref));
    ml::linear_sequence_backward_input(la::reference, ml::dense_weights<float>{dense, rows, cols}, dy, std::span<float>(dx_ref));
    ml::linear(host_q4, ml::prepare_input<cols>(host_q4, std::span<const float, cols>(x.data(), cols)), std::span<float>(yq_ref));
    double decode = 0.0, tc = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < rows; ++i)
        decode = std::fmax(decode, std::fabs(double(h_y1[i]) - double(ys_ref[i])));
    for (std::size_t i = 0; i < h_tc.size(); ++i)
        tc = std::fmax(tc, std::fabs(double(h_tc[i]) - double(ys_ref[i]))), scale = std::fmax(scale, std::fabs(double(ys_ref[i])));

    std::printf("  device: %s\n", context->device_name().c_str());
    const bool prefill = h_ys == ys_ref, backward = h_dx == dx_ref, q8 = h_yq == yq_ref, near = decode < 1e-4, tensor_ok = tc <= 1e-2 * scale;
    std::printf("  prefill, CUDA cores: %s core's reference over the dequantised weights\n", prefill ? "the bits of" : "DIFFERENT from");
    std::printf("  the input's gradient: %s\n", backward ? "the reference's bits" : "DIFFERENT");
    std::printf("  decode, one token: within %.2g of the reference (a sum in another order)\n", decode);
    std::printf("  prefill, tensor cores (bfloat16): within %.2g of the reference, largest output %.3g\n", tc, scale);
    std::printf("  Q8_0 activations: %s core's ml::linear\n", q8 ? "the bits of" : "DIFFERENT from");
    const bool ok = prefill && backward && q8 && near && tensor_ok;
    std::printf(ok ? "cuda_quant demo: ok\n" : "cuda_quant demo: FAILED\n");
    return ok ? 0 : 1;
}
