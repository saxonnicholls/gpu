# Changelog

All notable changes to snicholls-gpu. The format follows Keep a Changelog; versions follow Semantic Versioning.
A version is released when its tag is on a commit where every blocking CI leg is green.

## 0.5.0 - 2026-11-17

### Added

- **`cuda_math::bf16`**: through `cuda_training_backend`, each product's float operands are rounded to bfloat16 by a
  kernel (to nearest even) and multiplied on the tensor cores with float sums (`cuda_context::gemm_bf16`); master
  weights, gradients and optimiser state stay float, and attention takes the flash kernels. GPT-2 small's MLP and
  attention products 1.6 to 1.9 times TF32's speed, the conversions included; a product within 2.3e-3 normwise of float,
  a whole training step within about 1e-2. Products whose leading dimensions are not multiples of 8 (GPT-2's 50,257-row
  head) run as TF32, since cuBLAS's bfloat16 falls back to a slow kernel there. cuBLAS's own
  `CUBLAS_COMPUTE_32F_FAST_16BF` was tried first: on the Blackwell it chose TF32 kernels, giving TF32's bits.
- **The fused classifier over padded rows** (core 0.12.0): `cross_entropy_rows` with the classes the first of each row,
  the padding's gradient 0, so GPT-2's head runs as 50,304 rows and all three of its products in bfloat16. Against
  PyTorch on llm.c's reference batch, GPT-2 small in bfloat16 is as close as llm.c's own bfloat16 in every gradient
  tensor (within 1.5 times its error in each; lower in fifteen of sixteen), the loss 0.034 off against llm.c's 0.012.

## 0.4.0 - 2026-11-17

### Added

- **`linear_heads` and `linear_heads_backward` on the device** (core 0.11.0): the keys' and values' projections for
  every head in one product, permuted to the heads' layout, instead of a product per head. A whole GPT-2 small step
  over 1024 tokens with TF32: 67.5 to 63.2 ms.
- **Flash attention on the tensor cores** under `cuda_math::tf32`, for heads of 32 and 64 channels: `mma.sync`
  m16n8k8 in TF32, the online softmax in registers, no `n x n` scores, the masked tiles skipped; the backward in two
  kernels (queries; keys and values), no atomics, deterministic. GPT-2 small's attention at batch 4 x 1024 tokens:
  115 to 46 ms a step; the whole step with AdamW 245 to 230 ms. Against PyTorch on llm.c's reference batch: the loss
  within 5e-4, every gradient tensor within 1.2%.
- The documentation is built in CI and published to <https://saxonnicholls.github.io/gpu/>, as core's.

### Corrected

- 0.3.0's comparison with llm.c ("1.05x") was against llm.c measured beside another process on the same GPU, which
  halved llm.c's speed. On a free GPU, both with AdamW, over 1024 tokens: llm.c 33.6 ms at batch 1 and 105 ms at
  batch 4, against our 72.1 and 245 ms with TF32: 2.2 to 2.3 times slower.

## 0.3.0 - 2026-11-17

### Added

- **Attention as matrix products** up to `cuda_training_backend::gemm_attention_max` tokens (4096): per head the
  scores through cuBLAS, a softmax kernel writing the log-sum-exp, the output product; the backward from the saved
  log-sum-exp in four products and two kernels. With TF32 the products run on the tensor cores: a whole GPT-2 small
  step over 1024 tokens on the Blackwell in 70.2 ms (98.5 with the tiled kernels), against llm.c's 64.4 ms in
  bfloat16; 119 ms in float. Deterministic, within 1e-5 normwise of core's reference; the tiled kernels remain for
  longer sequences and are tested through the same steps.

### Changed

- **A product over device memory no longer waits for the stream**: `cuda_context::gemm` (and so `cuda_backend`'s
  `gemm` and `gemv`) returns once the product is queued when its operands and result are all device memory; it is
  ordered before later work on the context's stream and before a `download`, and `cuda_context::synchronize()` waits
  for it. A product touching host memory still returns finished. The whole GPT-2 step: 78.3 to 70.2 ms.
- **`sum_rows` a warp per column** (each lane's rows in order, then a fixed tree): deterministic, core's sequential sum
  to rounding rather than to the bit. The whole GPT-2 step: 70.2 to 67.7 ms.

## 0.2.0 - 2026-11-16

### Added

- **`cuda_training_backend`, `cuda_kernels`** (`include/snicholls/gpu/cuda_ops.hpp`): snicholls-core's batched
  training passes with every step on the GPU (core's TRAINING.md T9): CUDA overloads of all of core's training
  operations (fills, copies, residuals, bias rows and sums, the MLP activations, the norms and their backward passes,
  QK norm and RoPE, attention and its backward) as kernels compiled by NVRTC to the device's machine code. The norms
  and attention run a warp per row or head; the weight and key gradients are summed in core's order, with no atomics,
  so results do not depend on scheduling. Whole training steps of GPT-2-, Llama-, nanochat- and multi-query
  capped blocks (GQA, MQA, QK norm before and after RoPE, both RoPE layouts, a window, a soft cap, every activation)
  with every buffer in device memory agree with core's reference. A GPT-2-small layer's training step over 128
  tokens: 2.34 ms on an RTX PRO 4000 Blackwell, against 47.5 ms staged and 640 ms on its CPU. Needs core 0.10.0.
- **Tiled attention** for heads of 16, 32, 64 and 128 channels: a thread per query or key with its row in
  registers, the other side through shared memory a tile at a time; the keys' gradients split over the queries into
  partials added in a fixed order. Deterministic, and within 1e-5 normwise of core's reference over several tiles
  (GQA, a window, a soft cap). A whole GPT-2 small step over 1024 tokens fell from 288 to 137 ms.
- **`cuda_math::tf32`**: `cuda_backend::math` puts the `float` products on the tensor cores (inputs rounded to TF32);
  within 3e-4 normwise of float per product. A whole GPT-2 small step over 1024 tokens: 98.5 ms, against llm.c's 64.4
  ms in bfloat16 on the same GPU; its loss on llm.c's reference batch within 7e-4 of PyTorch's.
- The model's other steps on the device: the embeddings and their backward (in token order), the fused classifier
  over a sequence (a warp per row), AdamW, the rescale, and the sums and squared norms (a fixed grid and tree, the
  partials summed in order: deterministic, and closer to the exact sum than a sequential one). core's
  `lm_head_loss_sequence` through the CUDA backend agrees with the reference.
- **`cuda_backend`, `cuda_context`, `device_buffer`** (`include/snicholls/gpu/cuda.hpp`): snicholls-core's
  `linalg::blas_backend` on an NVIDIA GPU through cuBLAS. Device memory (`device_buffer<T>::span()`) is used in place,
  so a training step's data can stay on the GPU; host memory is staged, a row- or column-major operand copied as it is.
  Passes core's BLAS conformance suite in float and double and agrees with core's reference to rounding for every
  shape and layout. On an RTX PRO 4000 Blackwell: 9.0 TFLOP/s from device memory, level with `cublasSgemm` called by
  hand (the interface costs nothing), and a GPT-2-small layer's training step in 40.1 ms staged, against 53.1 ms on
  the OpenCL backend. Built where the CUDA toolkit is (`SNICHOLLS_GPU_WITH_CUDA`, target `snicholls::gpu_cuda`).

## 0.1.0 - 2026-11-16

### Added

- **The training benchmark**: a GPT-2-small layer's forward and backward over 128 tokens through core's batched passes,
  on the device and on the host's reference (the same bits, checked), and core's per-token path for context: 53 ms on
  an RTX PRO 4000 Blackwell against 633 ms on its CPU. Pins core v0.9.0 (`snicholls_core_require(0.9.0)`).
- `ROADMAP.md`: the plan to large-scale training on GPUs.
- **`opencl_backend`, `opencl_context`** (`include/snicholls/gpu/opencl.hpp`): snicholls-core's
  `linalg::blas_backend` on an OpenCL 1.2 device, a GPU by preference. `gemm` and `gemv` run on the device (64 x 64
  tiles, 4 x 4 outputs per work-item, loads along each operand's unit stride, contraction off), in float, and in
  double where the device has `cl_khr_fp64`; `dot`, `axpy`, `syrk` and `triangular_solve` on the host by core's
  reference. Passes core's linalg BLAS conformance suite and gives the reference's bits on an AMD Radeon Pro Vega II,
  an NVIDIA RTX PRO 4000 Blackwell and PoCL.
- `version.hpp`: the library's version and the core version it was built against.
