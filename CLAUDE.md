# snicholls-gpu

Backends of snicholls-core's interfaces on GPUs. Header-only C++20, namespace `snicholls::gpu`, include root
`<snicholls/gpu/...>`, CMake target `snicholls::gpu`. It pins snicholls-core as `third_party/core` (a release tag);
core's CLAUDE.md is the standard this repository follows, and where this file is silent, core's rule applies.

## Authorship and commits

Commits are authored by the repository owner and carry **no AI attribution**: no `Co-Authored-By` trailers, no
"Generated with" lines, in commit messages, pull request descriptions, file headers, README, CHANGELOG or this file.
Every source file starts with the banner

    //
    //  <name>.hpp
    //  snicholls-gpu
    //
    //  Copyright 2010-2026 Saxon Herschel Nicholls
    //

## What belongs here (MUST)

- **Implementations of core's interfaces that need a device runtime**, and nothing else. The interface (the concept,
  its error type, its reference implementation and its conformance suite) stays in core; this repository implements
  it. A new capability's contract goes to core first.
- **The reference's bits are the contract.** A device product gives exactly what core's reference backend gives:
  the same operations in the same order, `#pragma OPENCL FP_CONTRACT OFF`, `-ffp-contract=off` on the host. A
  faster kernel that changes a bit is not admitted; a device that cannot keep the bits (one that flushes
  subnormals) is documented, not papered over.
- **Every implementation passes core's conformance suite** for the parts it implements, registered in its test, and
  says which parts it does not (`GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST`).
- **What the device does not do runs on the host by core's reference**, documented per operation, never silently
  skipped.
- **Portable APIs first.** OpenCL 1.2 runs on AMD, NVIDIA, Intel and Apple devices and on PoCL, so one kernel is
  tested everywhere; a vendor API (CUDA, Metal, HIP) joins as a second backend only with a measured reason.

## Layout

- `include/snicholls/gpu/<name>.hpp`; `version.hpp` holds the version and the Doxygen group.
- `tests/<name>_test.cpp`, `demos/<name>_demo.cpp`, `benchmarks/<name>_benchmark.cpp` mirror the headers;
  `scripts/check_mirror.py` fails on a missing or unlisted one, and on one without a header.
- Header guards `snicholls_gpu_<path>_hpp`; macros `SNICHOLLS_GPU_`; Boost / standard library naming, as core.

## Versioning

Semantic versioning. The version is written in `include/snicholls/gpu/version.hpp`, `project(... VERSION)` in
`CMakeLists.txt` and the tag; `scripts/check_version.py` fails if they disagree. Core is pinned by tag and required
with `snicholls_core_require(<minimum>)`; the top-level build ends with `snicholls_core_assert_single_version()`.

## Tests, demos, benchmarks, documentation (MUST)

As core's CLAUDE.md: every public entity tested (GoogleTest, core's pinned copy), every header demonstrated by a demo
that exits non-zero when its result is not what it documents, Doxygen in core's twelve sections with warnings as
errors. Benchmarks compare the device with core's reference on the host through the same call, transfers included,
after a fairness check that refuses to run unless the results agree to the bit; `BENCHMARKS.md` is written by
`scripts/run_benchmarks.sh`, never by hand. State the device, driver, build type and date with every number.

## Build and CI

    make test          # configure, build, every CTest (needs an OpenCL device)
    make test-werror   # warnings as errors, as CI
    make asan          # Address + UB sanitizers
    make docs          # Doxygen, warnings are errors
    make check         # version and mirror
    make ubuntu        # nine configurations on Ubuntu 24.04 and 26.04 in Docker (GCC 15, TSan included), on PoCL
    make benchmarks    # Release benchmarks; rewrites BENCHMARKS.md

The local gate is macOS (its GPU) plus `make ubuntu` (PoCL). `make ubuntu` builds two images, as core's does:
`docker/ubuntu` (Ubuntu 24.04: GCC 13, Clang 18, Doxygen) runs GCC and Clang Release and Debug, ASan+UBSan under both
and Clang TSan; `docker/ubuntu-26.04` (Ubuntu 26.04: GCC 15, the `g++` of `ubuntu-latest`, which core's CI builds with) runs GCC 15 Release
with benchmark smoke runs and GCC 15 ASan+UBSan, because GCC 15 warns where GCC 13 does not. `--quick` runs GCC 13
Release and GCC 15 Release. On a macOS 26 host every AddressSanitizer binary traps
before `main` (a hello world included), so sanitizer evidence comes from the Ubuntu configurations. A change to a
kernel is also run on an NVIDIA GPU before it is pushed: the tests and the benchmark, with their bits checked. The
CUDA backend builds only where the toolkit is (`SNICHOLLS_GPU_WITH_CUDA`), so it is built and tested on the NVIDIA
machine, every test and the benchmark, before a push; CI runs Linux with PoCL and macOS, without CUDA.

CI also builds the documentation on every run (warnings are errors, the CUDA headers included: Doxygen reads them
without the toolkit) and publishes it to GitHub Pages, as core does: <https://saxonnicholls.github.io/gpu/>, the newest
release at the root, every version under `/vX.Y.Z/`, the latest `main` under `/main/`. The `gh-pages` branch holds
generated HTML only; never edit it by hand.
