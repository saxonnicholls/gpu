# Roadmap

The aim: train transformer language models at scale on GPUs, with snicholls-core's checks and reproducibility. Core
holds the arithmetic and the interfaces and runs everything on the CPU as the reference; this library runs it on a
device. The milestones below are measured; nothing moves on until its numbers are written down.

## Now (0.3.0, core 0.10.0)

- Matrix products on any OpenCL 1.2 device, giving the bits of core's reference backend.
- A CUDA backend: cuBLAS over device memory (9 TFLOP/s on an NVIDIA RTX PRO 4000 Blackwell, against 1.4 for the
  OpenCL kernel), and every other step of core's batched training passes as a kernel (`cuda_ops.hpp`), so a training
  step's weights, activations and gradients stay on the GPU from start to end: one GPT-2-small layer's forward and
  backward over 128 tokens in 2.34 ms, against 640 ms on its CPU.
- Attention as matrix products on cuBLAS (tiled kernels for sequences beyond 4096 tokens), and TF32 tensor cores for
  the products as an option.
- Driven by a whole GPT-2 small model, the CUDA backend gives PyTorch's loss on llm.c's reference batch to 4e-6, and
  every gradient tensor to 0.07% of its largest element (TF32: 4e-4 and 0.3%), and ten AdamW steps give PyTorch's ten
  losses within 1.1e-4. The whole step with AdamW over 1024 tokens, on the GPU alone: 72 ms with TF32 at batch 1 and
  245 ms at batch 4, against llm.c's 34 and 105 ms in bfloat16 (2.2x).

## Next

1. **Attention without its masked half**, then **flash attention on the tensor cores**: attention's products are
   about half of a batch-4 step, full n x n though half is masked, in small tiles.
2. **Fewer, fused elementwise kernels**, and bfloat16 products.
3. **Muon on the device.**
4. **Mixed precision**: bfloat16 in the products (cuBLASLt), float master weights and optimiser state.
5. **The rest of the linear algebra on the device** (QR, the SVD, the batched solves), so core's truncated SVD runs
   there too, and every part registered in core's conformance suites.
6. **A faster OpenCL kernel**: more outputs per work-item, vector loads, double buffering; the bits unchanged.

## Milestone

A full GPT-2 small training step on one GPU within 1.5x of llm.c's CUDA build on the same GPU and batch, with the loss
curve matching llm.c's.

Training across machines (the communication, the data parallelism suited to the network between them) is built in
the libraries above this one; this library's part is the device.
