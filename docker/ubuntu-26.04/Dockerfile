# syntax=docker/dockerfile:1.6
#
# snicholls-gpu - the GCC 15 build and test image, after core's (docker/ubuntu-26.04/Dockerfile in core).
#
# Core's CI builds on ubuntu-latest, whose g++ is GCC 15, and GCC 15 warns where GCC 13 does not (a false
# -Warray-bounds in core, for one); this repository's CI builds on ubuntu-24.04 with GCC 13. This image puts that compiler in the local gate: scripts/test_ubuntu.sh
# builds it beside docker/ubuntu/Dockerfile (Ubuntu 24.04, GCC 13 and Clang 18) and runs the GCC 15 configurations
# in it, on PoCL as there.
#
# Ubuntu 26.04 LTS ships GCC 15 in its own archive, so the image needs no PPA and no compiler built from source: the
# base image is pinned by its release, and the compiler by its major version, as docker/ubuntu/Dockerfile pins
# GCC 13. Compilers, build tools and OpenCL on PoCL only; the documentation is checked in the other image.

FROM ubuntu:26.04

LABEL org.opencontainers.image.title="snicholls-gpu Ubuntu GCC 15 builder" \
      org.opencontainers.image.description="Ubuntu 26.04: GCC 15, CMake, Ninja, OpenCL on PoCL"

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential gcc-15 g++-15 \
      cmake ninja-build \
      python3-minimal git ca-certificates \
      ocl-icd-opencl-dev opencl-headers pocl-opencl-icd clinfo && \
    rm -rf /var/lib/apt/lists/* && \
    g++-15 --version | head -1

# Sources are mounted read-only by scripts/test_ubuntu.sh and copied in, so a
# build inside the container never writes into the checkout.
WORKDIR /work
CMD ["bash"]
