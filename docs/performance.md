# Performance {#performance}

How fast snicholls-gpu trains a transformer, how each figure was measured, and how it compares with llm.c, the
hand-written CUDA trainer the plan measures against. Every number here was taken on one machine: an NVIDIA RTX PRO
4000 Blackwell SFF (24 GB, sm_120), CUDA 13.3, a 13th-generation Core i5, Linux x86-64, Release builds, 2026-11-17,
with nothing else on the GPU. The figures attributed to the benchmarks are in `BENCHMARKS.md`, which
`scripts/run_benchmarks.sh` writes and nobody edits.

## The paths

The CUDA backend (`cuda.hpp`, `cuda_ops.hpp`) runs snicholls-core's batched training passes with every parameter,
activation and gradient in device memory. Two choices decide its speed and its rounding:

- **`cuda_backend::math`**: `cuda_math::float32` (the default: cuBLAS in float, every result within rounding of
  core's reference) or `cuda_math::tf32` (the products on the tensor cores, their inputs rounded to 10 bits of
  mantissa, the sums in float).
- **Attention**, chosen per call:
  - **flash attention on the tensor cores** under TF32, for heads of 32 or 64 channels
    (`cuda_training_backend::flash_attention`, on by default): `mma.sync m16n8k8`, the online softmax in registers, no
    `n x n` scores, the tiles past the causal diagonal or outside a window skipped; the backward in two kernels
    (the queries' gradients, then the keys' and values'), no atomics, so the result does not depend on scheduling;
  - otherwise **matrix products** up to `gemm_attention_max` tokens (4096): per head the scores through cuBLAS, a
    softmax kernel saving the log-sum-exp, the output product; the backward recomputing the probabilities;
  - beyond that, **tiled kernels** on the CUDA cores.

## Attention alone

GPT-2 small's attention (12 heads of 64, causal), forward and backward over one sequence, from
`BM_Gpt2SmallAttention_ForwardBackward` in `benchmarks/cuda_ops_benchmark.cpp`, medians of five. The benchmark first
checks, at 1024 tokens, that the float path agrees with core's reference on the host (measured: 2e-7 to 2e-6
normwise) and both TF32 paths with the float path (3e-4 to 6e-4); it refuses to run otherwise.

| tokens | matrix products, float | matrix products, TF32 | flash attention, TF32 |
|---|---|---|---|
| 256 | 0.579 ms | 0.423 ms | **0.164 ms** |
| 1024 | 1.88 ms | 1.12 ms | **0.820 ms** |
| 4096 | 38.8 ms | 27.8 ms | **9.54 ms** |

Two things were tried on the matrix-product path and taken out, because both made it slower: batching the products
of every head into one strided call (the `n x n` scores of twelve heads no longer fit in the L2 cache, and the
softmax kernels slowed by 3.5 times), and splitting the queries into blocks that skip the masked half (the smaller
products ran less efficiently than the saved work).

## A whole GPT-2 small training step

The whole model (124M parameters: embeddings, 12 blocks, the final norm, the tied head; forward, backward, gradient
clipping and AdamW) is composed in the private training library above this one; its benchmark and its comparison with
llm.c are recorded there. The step over sequences of 1024 tokens, as each change landed:

| change | batch 1 | batch 4 |
|---|---|---|
| attention a warp per (token, head), float | 288 ms | |
| tiled attention, float | 137 ms | |
| TF32 products, tiled attention (0.2.0) | 98.5 ms | |
| attention as matrix products (0.3.0) | 78.3 ms | |
| products over device memory queued, not waited for (0.3.0) | 70.2 ms | |
| bias sums a warp per column (0.3.0) | 67.7 ms | |
| `linear_heads`: the keys' and values' projections as one product (core 0.11.0) | 63.2 ms | |
| the same, AdamW included | 72.1 ms | 244.6 ms |
| **flash attention on the tensor cores, AdamW included** | **68.7 ms** | **230.4 ms** |

The rows without AdamW measure forward, backward and clipping only. The first six rows were measured while another
process (a 30B language model) held 21.6 GB of the GPU; our step was the same with the GPU free (67.5 ms), llm.c's
was not.

**Against llm.c** (`train_gpt2cu`, commit f1e2ace, bfloat16 with tensor cores, AdamW included, the same GPU free):
33.6 ms at batch 1 and 104.9 ms at batch 4. Our step is **2.0 times** llm.c's at batch 1 and **2.2 times** at batch
4; the milestone is 1.5 times.

Where the 230 ms at batch 4 goes (Nsight Systems, TF32): the matrix products 131 ms (the projections, the MLP and the
head; llm.c's are in bfloat16, about twice TF32's rate), flash attention 46 ms, the other kernels 57 ms (AdamW 9, the
fused classifier 9, the norms, the bias sums, the elementwise steps). Next: bfloat16 products, fewer and fused small
kernels, and each block over the whole batch at once.

## Accuracy

Against PyTorch's loss and gradients on llm.c's reference batch (4 x 64 tokens, `gpt2_124M_debug_state.bin`):

| path | loss (PyTorch 5.270009) | worst gradient tensor (largest difference / largest element) |
|---|---|---|
| float | 5.270005 | 0.07% |
| TF32 products, matrix-product attention | 5.269604 | 0.3% |
| TF32 products, flash attention | 5.269514 | 1.2% |
| llm.c's own float32 CPU code, for comparison | 5.269997 | 0.075% |

In float, ten AdamW steps on that batch give PyTorch's ten losses within 1.1e-4.

## A mixture of experts

`cuda_moe.hpp` runs core's mixture of experts (`ml/moe.hpp`) on the device. Measured with
`benchmarks/cuda_moe_benchmark.cpp` on the same RTX PRO 4000 Blackwell, CUDA 13.3, Release, 2026-11-19, **while another
training job kept the GPU fully busy**: every figure is an upper bound and the pairs are compared with each other, not
with the tables above. Medians of five where stated. Two shapes over 4096 tokens of 768: eight experts of 2048, two per
token (Mixtral's), and 64 experts of 256, eight per token, with one shared expert (DeepSeekMoE's), routed from random
tokens so the segments are as uneven as a router makes them.

**The grouped product.** The segments' sizes come from the routing and so live on the device. Three ways to multiply
them were measured; cuBLASLt's grouped layouts (`CUBLASLT_BATCH_MODE_GROUPED`) return `CUBLAS_STATUS_NOT_SUPPORTED` on
this device (sm_120) for float, TF32 and bfloat16 alike, and `cublasGemmGroupedBatchedEx` takes no bfloat16 inputs with
float outputs. The experts' up projection, forward:

| shape, math | grouped kernel | cuBLAS per expert | `cublasGemmGroupedBatchedEx` |
|---|---|---|---|
| 8 x 2048, float | 7.5 ms | 7.7 ms | 7.8 ms |
| 8 x 2048, TF32 | 3.8 ms | 3.5 ms | 3.4 ms |
| 8 x 2048, bfloat16 | 3.3 ms | 2.9 ms | (not supported) |
| 64 x 256, float | 4.0 ms | 6.9 ms | 4.0 ms |
| 64 x 256, TF32 | 3.3 ms | 3.3 ms | 3.4 ms |
| 64 x 256, bfloat16 | 2.9 ms | 3.3 ms | (not supported) |

Both cuBLAS ways need the segments' sizes on the host: one wait for the offsets per product, and nothing of it can be
captured in a graph. The grouped kernel reads them on the device: 128 x 128 tiles on the tensor cores (TF32, bfloat16,
or three TF32 products for float's accuracy), the fragments kept in shared memory in the order they are read (one
128-bit load per fragment; the first layout, a word per load, took 5.8 ms where this takes 3.8), no atomics. It is
within about 10% of cuBLAS for large segments and as fast or faster for many small ones.

**A whole layer**, `ml::moe_forward` and `ml::moe_backward` (router, routing, permutation, gather, the three grouped
products and their backward, the activations, combine, the shared expert, the balance losses), medians of five:

| shape, math | queued | captured graph | queued, cuBLAS per expert |
|---|---|---|---|
| 8 x 2048, float | 71.6 ms | 69.3 ms | 72.1 ms |
| 8 x 2048, TF32 | 39.8 ms | 35.3 ms | 43.0 ms |
| 8 x 2048, bfloat16 | 32.3 ms | 30.0 ms | 38.6 ms |
| 64 x 256 + shared, float | 53.9 ms | 51.5 ms | 66.2 ms |
| 64 x 256 + shared, TF32 | 35.0 ms | 30.6 ms | 39.1 ms |
| 64 x 256 + shared, bfloat16 | 31.3 ms | 29.1 ms | 41.0 ms |

So the grouped kernel is the default (`grouped_path::kernel`): with it nothing in a layer waits for the host except the
two auxiliary losses' values, which go to device memory instead when the step is captured
(`cuda_moe_kernels::losses_on_device`), and the whole layer replays as one graph. The steps around the products
(routing, the histogram, the permutation, gather and combine) take 2.4 ms for the first shape and 2.9 ms for the second.
`BENCHMARKS.md` is not rewritten from these contended figures; that waits for a free GPU.

## Quantised weights on OpenCL

`opencl_quant.hpp` on an AMD Radeon Pro Vega II (32 GB HBM2, about 1 TB/s; macOS 26, Apple's OpenCL 1.2, Release,
2026-11-20) **while other work kept the host busy** (load average 40 to 75): first figures, upper bounds on the times.
One token (decode) through a matrix kept on the device, the rate counting the bytes of weights read. The kernel alone,
from OpenCL's event profiling with the program of `detail/quant_source.hpp` built on its own profiling queue:

| format, activations | 4096 x 4096 | 14336 x 4096 |
|---|---|---|
| Q4_0, float | 129 us, 73 GB/s | 416 us, 80 GB/s |
| Q8_0, float | 126 us, 142 GB/s | 405 us, 154 GB/s |
| bfloat16, float | 305 us, 110 GB/s | 750 us, 157 GB/s |
| Q4_0, Q8_0 | 116 us, 81 GB/s | 380 us, 87 GB/s |
| Q8_0, Q8_0 | 165 us, 108 GB/s | 545 us, 115 GB/s |

Q4_0 and Q8_0 take the same time, so the matrix-vector kernel is bound by its instructions (decoding, the address
arithmetic), not by memory: about 15% of the device's bandwidth. Through the call (`benchmarks/opencl_quant_benchmark.cpp`,
medians of three, the copies of `x` and `y` and the buffers made for them included) each product costs about 550 us
more: 678 us and 980 us for Q8_0 at the two shapes. Against that, core's own one-row product on the host (scalar, one
thread) takes 21 ms for Q4_0 and 14 ms for Q8_0 at 4096 x 4096, and the dequantised float matrix through
`opencl_backend`'s product, which copies its 64 MB for each call, 46 ms. Prefill, 512 tokens through 4096 x 4096 on the
tiled kernel (the reference's bits): 10.9 ms for Q4_0 and 11.1 ms for NF4, 1.6 TFLOP/s. The CUDA kernels' figures wait
for the NVIDIA machine.
