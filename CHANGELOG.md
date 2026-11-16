# Changelog

All notable changes to snicholls-gpu. The format follows Keep a Changelog; versions follow Semantic Versioning.
A version is released when its tag is on a commit where every blocking CI leg is green.

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
