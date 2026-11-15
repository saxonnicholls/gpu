# Changelog

All notable changes to snicholls-gpu. The format follows Keep a Changelog; versions follow Semantic Versioning.
A version is released when its tag is on a commit where every blocking CI leg is green.

## Unreleased (0.1.0)

### Added

- **`opencl_backend`, `opencl_context`** (`include/snicholls/gpu/opencl.hpp`): snicholls-core's
  `linalg::blas_backend` on an OpenCL 1.2 device, a GPU by preference. `gemm` and `gemv` run on the device (64 x 64
  tiles, 4 x 4 outputs per work-item, loads along each operand's unit stride, contraction off), in float, and in
  double where the device has `cl_khr_fp64`; `dot`, `axpy`, `syrk` and `triangular_solve` on the host by core's
  reference. Passes core's linalg BLAS conformance suite and gives the reference's bits on an AMD Radeon Pro Vega II,
  an NVIDIA RTX PRO 4000 Blackwell and PoCL.
- `version.hpp`: the library's version and the core version it was built against.
