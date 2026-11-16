# Roadmap

The aim: train transformer language models at scale on GPUs, with snicholls-core's checks and reproducibility. Core
holds the arithmetic and the interfaces and runs everything on the CPU as the reference; this library runs it on a
device. The milestones below are measured; nothing moves on until its numbers are written down.

## Now (0.1.0, core 0.9.0)

- Matrix products on any OpenCL 1.2 device, giving the bits of core's reference backend.
- Core's batched training passes run through it: one GPT-2-small layer's forward and backward over 128 tokens in 53 ms
  on an NVIDIA RTX PRO 4000 Blackwell (633 ms on its CPU), 237 ms on an AMD Radeon Pro Vega II, the same bits as the
  host's reference (`benchmarks/opencl_benchmark.cpp`).

## Next

1. **Data that stays on the device.** Core's batched passes now make every step a call on the backend (core's
   TRAINING.md T9), and the CUDA backend takes device memory in place; what remains is the CUDA overloads of those
   steps (the norms, attention, activations, residuals), so a training step's weights, activations and gradients stay
   on the GPU from start to end.
2. **A CUDA backend** beside the OpenCL one. Done in 0.2.0 for the products (cuBLAS over device memory: 9 TFLOP/s on the
   Blackwell, against 1.4 for the OpenCL kernel); next, tensor cores through cuBLASLt (TF32, bfloat16), with a stated
   tolerance against core's reference per operation.
3. **Every training kernel on the device**: flash attention forward and backward, the norms and activations with their
   backward passes, softmax and the fused cross-entropy, AdamW and Muon; each checked against core's own function.
4. **Mixed precision**: bfloat16 in the products, float master weights and optimiser state.
5. **The rest of the linear algebra on the device** (QR, the SVD, the batched solves), so core's truncated SVD runs
   there too, and every part registered in core's conformance suites.
6. **A faster OpenCL kernel**: more outputs per work-item, vector loads, double buffering; the bits unchanged.

## Milestone

A full GPT-2 small training step on one GPU within 1.5x of llm.c's CUDA build on the same GPU and batch, with the loss
curve matching llm.c's.

Training across machines (the communication, the data parallelism suited to the network between them) is built in
the libraries above this one; this library's part is the device.
