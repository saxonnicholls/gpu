# Changelog

All notable changes to snicholls-gpu. The format follows Keep a Changelog; versions follow Semantic Versioning.
A version is released when its tag is on a commit where every blocking CI leg is green.

## Unreleased

### Added

- **Products over quantised weights** (`opencl_quant.hpp`, `cuda_quant.hpp`, one kernel source in
  `detail/quant_source.hpp`): the weights stay in core's formats in device memory, ggml's Q8_0, Q4_0 and MXFP4 blocks,
  QLoRA's NF4 with double-quantised scales, and dense bfloat16, float16, FP8 E4M3 and E5M2, and each weight is
  dequantised in registers with core's arithmetic. Overloads of core's `linear_sequence` and
  `linear_sequence_backward_input` for `opencl_quant_backend` and `cuda_training_backend`, so `ml::lora_sequence` and
  its backward run a QLoRA layer on the device, and `linear_sequence_q8`, the activations quantised to Q8_0 on the
  device with core's integer block dot products. Matrix-vector kernels for decode (within the bound of a reordered
  sum), tiled kernels for prefill and the input's gradient (core's reference bits over the dequantised weights), the
  Q8_0 path to the bits of core's `ml::linear`; on CUDA under `cuda_math::bf16`, the tiled products on the tensor cores
  (`mma.sync m16n8k16`). Deterministic, no atomics. `opencl_quant_matrix` keeps frozen weights on an OpenCL device.
- The formats of core's next release where the pinned core has them (`block_weights<Block>`, `quant_weights.hpp`):
  Q4_1, Q5_0, Q5_1, Q2_K to Q6_K, TQ1_0, TQ2_0, IQ4_NL and IQ4_XS, with the integer path for each that core pairs with 8-bit
  activations: Q8_0 for Q5_0, Q8_1 for Q4_1 and Q5_1, Q8_K for the K-quants (core's eight-lane dot products) and the ternary, to the bit.
- `cuda_training_backend::quant`, after `moe`; `opencl_context::native_device`, `native_context`, `native_queue`.

## 0.8.0 - 2026-11-19

### Added

- **The mixture of experts on the device** (`cuda_moe.hpp`, `cuda_moe_kernels`): an overload for
  `cuda_training_backend` of every step of core's `ml/moe.hpp`, so `ml::moe_forward`, `ml::moe_backward` and the MoE
  block of `ml/moe_block.hpp` run on the GPU: routing with a selection-only bias under each scoring (softmax over the
  top K, softmax, normalised sigmoid) and its backward; the histogram, the offsets and the stable permutation (a
  counting sort from integer counts, so core's unique order to the bit); gather and the weighted combine in slot order
  and their backward (to the bit); the grouped products and their backward as one kernel each over every expert, the
  segments read from device memory, in TF32, bfloat16 or float's accuracy (three TF32 products); the bias rule (to the
  bit), the balance loss and the z-loss with their gradients; expert parallelism's layouts, the in-process all-to-all and
  the row permutations; the bfloat16 payload rows. Every kernel deterministic, no atomics in any floating-point sum;
  nothing waits for the host, so a whole MoE layer is captured as a `cuda_graph` (the losses then stay on the device).
  `grouped_path::cublas` makes the products expert by expert through cuBLAS instead (one wait per product).
- `cuda_training_backend::moe`: the backend's pointer to the mixture-of-experts kernels, the last member, so existing
  initialisers are unchanged.

### Changed

- Pins core v0.16.0 (was v0.15.0) and raises `snicholls_core_require` to 0.16.0: the device MoE kernels implement
  core 0.16.0's mixture-of-experts API (`ml/moe.hpp`, `ml/moe_block.hpp`).
- The Ubuntu gate (`make ubuntu`, `scripts/test_ubuntu.sh`) also builds with GCC 15 on Ubuntu 26.04
  (`docker/ubuntu-26.04`), as core's CI does with ubuntu-latest's GCC 15.2: GCC 15 Release with benchmark smoke runs
  and GCC 15 ASan+UBSan beside GCC 13 and Clang 18, nine configurations in all. `--quick` runs GCC 13 Release and
  GCC 15 Release. A failed configuration shows its first diagnostics above the log's tail.

## 0.7.1 - 2026-11-19

### Changed

- Pins core v0.15.0 (was v0.14.0), so a build that carries both this library and core 0.15.0 reaches one core
  version (core's diamond rule). No change to this library; `snicholls_core_require` stays at 0.14.0.

## 0.7.0 - 2026-11-18

### Fixed

- **The CUDA kernels take core 0.14.0's accurate forms** (`cuda_ops.hpp`), so the device keeps the digits core's
  reference now keeps:
  - GELU-tanh' and SiLU' (`mlp_activate_rows_backward`, and through it every block's backward pass) take
    `1 - sigmoid(z)` as `sigmoid(-z)`, both from one `expf(-fabsf(z))`, as core's `sigmoid_both`, instead of
    subtracting a sigmoid near 1 from 1 (about 8 ulps for SiLU' over x in 8..17 and 16 for GELU-tanh' below 8 before;
    within 4 ulps of the exact value now). The forward SiLU is unchanged.
  - `cross_entropy_rows` (and the classifier through it) computes the loss as `(max - x_t) + log(sum)` and the gradient as
    `e^(x - max) · (1 / sum)`, as core's `cross_entropy_with_gradient`: with logits near 1e4 the loss and the gradient
    no longer carry the rounding of the largest logit (about 5e-4 off before).
  - `adamw_rows` and `adamw_segments` take their bias corrections from core's `ml::adam_bias_correction` on the host,
    `-expm1(t · log1p(-(1 - β)))`, instead of `1 - pow(β, t)`, which cancelled for `β2` near 1 in the first steps.

  Results change in the last bits. Tested near each cancellation against core in float and in double. Needs core 0.14.0
  (`ml::adam_bias_correction`); `snicholls_core_require(0.14.0)`.

## 0.6.0 - 2026-11-18

### Added

- **`cuda_graph`**: the work a function queues on a context's stream captured once as a CUDA graph and `launch()`ed again
  as one launch. A GPT-2 training micro-batch's forward and backward passes (about 4,400 kernels and products) replay
  to the bit of the uncaptured pass; a capture that waits, allocates or reads back fails and says so.
- **`adamw_segments`**: one AdamW step over many tensors laid end to end in one span, each with its own weight decay, in
  one launch (one per 32 tensors), with an optional gradient scale (the clipping factor) applied as the gradients are
  read; each tensor's `adamw_rows` to the bit. GPT-2's sixteen launches and its clipping pass become one launch.
- `cuda_kernels::launch_tiles` (a row of two-dimensional blocks) and `cuda_kernels::partials` (the reductions' space,
  kept between calls).

### Changed

- **Fewer launches, no waits in a training step**: `block_norm_rows_backward_add` is two kernels instead of three (the
  backward recomputes the row's statistics itself), allocates nothing and no longer synchronises; `sum_values` and
  `sum_squares` keep their partial sums between calls instead of allocating. With these, a whole forward and backward
  pass can be captured.
- **Faster small kernels**, deterministic as before: float4 loads and stores in `fill_rows`, `copy_rows`, `add_rows`,
  `scale_rows`, `add_bias_rows`, AdamW and the bfloat16 conversion of the products' operands (core's bits); the column
  sums (`sum_rows`, the norms' parameter gradients) coalesced, a block per 32 columns with 32 row groups; the
  embeddings' backward a block per token instead of a thread per channel (still a sequential loop's bits); the
  classifier's maximum and sum of exponentials in one online pass. GPT-2 small at batch 4 in bfloat16 on the RTX PRO 4000
  Blackwell (shared with another job; the shortest of eleven steps per launch): these kernels 70 ms to 47 ms a step.

- **A batch of sequences in one call** (core's batched passes with `sequences`, on core's feature branch until its
  release): every overload of `cuda_training_backend` takes the sequence count. `linear_heads` and its backward are one
  product over every row and one permutation into each sequence's block of keys and values; QK norm and RoPE one launch
  over every (row, head), positions restarting in each sequence; flash attention's three kernels one launch each with the
  sequence as the grid's z; the matrix-product and tiled attention paths run sequence by sequence. Tested against core's
  reference over batches of two to four sequences on every attention path. A GPT-2-small layer's training step over 4 x
  1024 tokens (`BM_TrainGpt2SmallLayer_4x1024`) as one call against one call per sequence: level within the noise of a
  shared GPU (RTX PRO 4000 Blackwell with a training run on it, 2026-11-17, medians of 15 interleaved, cv about 20%),
  under TF32 and bfloat16 alike; at 1024 tokens a sequence's products already fill the device.

- **Muon on the device** (needs core's next release, whose `ml::muon_step` makes every step a call on the backend):
  `ml::muon_step(cuda_training_backend, ...)` with the weights, gradient, momentum and working space in device memory
  runs entirely on the GPU, the elementwise steps by the existing kernels and the Newton-Schulz products by cuBLAS
  (bfloat16 under `cuda_math::bf16`, as nanochat). Tested against core's reference in float (normwise 4e-6), TF32
  (5e-3) and bfloat16 (3.5e-2), the momentum to the bit. GPT-2 small's four block matrices: 12.1 ms per Muon step in
  bfloat16, 15.6 ms in TF32, against 2.0 ms for AdamW (`cuda_ops_benchmark.cpp`, a GPU shared with a training run).

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
