# snicholls-gpu

[![CI](https://github.com/saxonnicholls/gpu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/saxonnicholls/gpu/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-doxygen-blue)](https://saxonnicholls.github.io/gpu/)

**Documentation:** <https://saxonnicholls.github.io/gpu/>: the newest release at the root, every version under
`/vX.Y.Z/`, and the latest `main` under `/main/`. **Performance:** [docs/performance.md](docs/performance.md), how fast
a transformer trains on each path, how it was measured, and the comparison with llm.c; the raw figures in
[BENCHMARKS.md](BENCHMARKS.md).

Backends of [snicholls-core](https://github.com/saxonnicholls/core)'s interfaces on GPUs. Core is standard-library
only and never includes a device header; it states the interfaces (here `linalg::blas_backend`) and checks every
implementation with its conformance suites. This library implements them on a device.

- **`snicholls::gpu::cuda_backend`** (`include/snicholls/gpu/cuda.hpp`): the NVIDIA backend, cuBLAS over device memory
  in place (`device_buffer<T>`) or host memory staged; built where the CUDA toolkit is (`snicholls::gpu_cuda`).
- **`snicholls::gpu::cuda_training_backend`** (`cuda_ops.hpp`, `cuda_moe.hpp`): core's batched training passes with
  every step a kernel over device memory, dense blocks and mixtures of experts alike.
- **`snicholls::gpu::opencl_backend`** (`include/snicholls/gpu/opencl.hpp`): matrix products on any OpenCL 1.2
  device, a GPU by preference: AMD, NVIDIA, Intel, Apple. Code written against core's `linalg::gemm` (and through it
  core's batched transformer training) runs on the GPU by changing the backend it is handed. The products give the
  bits of core's reference backend.

```cpp
auto context = snicholls::gpu::opencl_context::create().unwrap();   // finds a GPU, compiles the kernels
const snicholls::gpu::opencl_backend device{context.get()};
snicholls::linalg::gemm(device, 1.0f, x, snicholls::linalg::transposed(w), 0.0f, y);   // Y = X Wᵀ on the GPU
```

See `ROADMAP.md` for where this is going: large-scale training of transformers on GPUs.

## Use

As a submodule, after core:

    add_subdirectory(third_party/gpu EXCLUDE_FROM_ALL)
    target_link_libraries(my_target PRIVATE snicholls::gpu)

It pins core as `third_party/core` (a release tag) and follows core's diamond rules: a consumer's core wins, and
`snicholls_core_require` states the minimum. It needs an OpenCL runtime and headers: macOS has them; on Linux, the
GPU vendor's driver (or PoCL, which runs OpenCL on the CPU) and `ocl-icd-opencl-dev` / `opencl-headers`.

## Build and test

    make test          # configure, build, every CTest (needs an OpenCL device)
    make test-werror   # as CI does it
    make docs          # Doxygen, warnings are errors
    make check         # the version and mirror checks
    make ubuntu        # every configuration on Ubuntu 24.04 and 26.04 in Docker (GCC 13 and 15, Clang), on PoCL
    make benchmarks    # Release benchmarks; rewrites BENCHMARKS.md

## Licence

MIT; see `LICENSE`.
