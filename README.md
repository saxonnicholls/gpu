# snicholls-gpu

Backends of [snicholls-core](https://github.com/saxonnicholls/core)'s interfaces on GPUs. Core is standard-library
only and never includes a device header; it states the interfaces (here `linalg::blas_backend`) and checks every
implementation with its conformance suites. This library implements them on a device.

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
    make ubuntu        # every configuration on Ubuntu 24.04 in Docker, on PoCL
    make benchmarks    # Release benchmarks; rewrites BENCHMARKS.md

## Licence

MIT; see `LICENSE`.
