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
