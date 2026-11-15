# snicholls-gpu - the short way to drive CMake and the scripts.
#
#   make test          configure, build and run every CTest in build/ (Debug)
#   make test-werror   the same with warnings as errors, as CI does it
#   make asan          Address + UB sanitizers
#   make docs          Doxygen, warnings are errors; output build-docs/docs/html
#   make check         the static checks: version and mirror
#   make ubuntu        every configuration on Ubuntu 24.04 in Docker (scripts/test_ubuntu.sh), on PoCL
#   make benchmarks    Release benchmarks; rewrites BENCHMARKS.md (scripts/run_benchmarks.sh)

.PHONY: test test-werror asan docs check ubuntu benchmarks clean

test:
	cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
	cmake --build build
	ctest --test-dir build --output-on-failure

test-werror:
	cmake -S . -B build-werror -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNICHOLLS_GPU_WERROR=ON
	cmake --build build-werror
	ctest --test-dir build-werror --output-on-failure

asan:
	cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSNICHOLLS_GPU_WERROR=ON -DSNICHOLLS_GPU_SANITIZE=address,undefined
	cmake --build build-asan
	ctest --test-dir build-asan --output-on-failure

docs:
	cmake -S . -B build-docs -G Ninja -DSNICHOLLS_GPU_BUILD_TESTS=OFF -DSNICHOLLS_GPU_BUILD_DOCS=ON
	cmake --build build-docs --target docs

check:
	python3 scripts/check_version.py
	python3 scripts/check_mirror.py

ubuntu:
	scripts/test_ubuntu.sh

benchmarks:
	scripts/run_benchmarks.sh

clean:
	rm -rf build build-*
