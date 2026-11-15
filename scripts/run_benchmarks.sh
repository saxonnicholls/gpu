#!/usr/bin/env bash
#
#  run_benchmarks.sh
#  snicholls-gpu
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Build the benchmarks in Release and run them, writing BENCHMARKS.md (never edited by hand). Each benchmark refuses
#  to run unless its fairness check passes.
#
#      scripts/run_benchmarks.sh
#

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build-bench"
GENERATOR=()
command -v ninja >/dev/null && GENERATOR=(-G Ninja)   # else CMake's default
cmake -S "$ROOT" -B "$BUILD" "${GENERATOR[@]}" -DCMAKE_BUILD_TYPE=Release -DSNICHOLLS_GPU_BUILD_BENCHMARKS=ON >/dev/null
cmake --build "$BUILD" -j >/dev/null
OUT="$ROOT/BENCHMARKS.md"
{
    echo "# Benchmarks"
    echo
    echo "Written by \`scripts/run_benchmarks.sh\`; never edited by hand. Release, $(uname -sm), $(date -u +%Y-%m-%d)."
    echo "Medians of five. Device times include both transfers; GFLOP/s counts 2 m n k."
    echo
    echo '```'
    "$BUILD/benchmarks/snicholls_gpu_opencl_benchmark" --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
        --benchmark_min_time=0.2s 2>&1 | grep -E "^device|median|^Benchmark|^-"
    echo '```'
} > "$OUT"
echo "run_benchmarks: wrote $OUT"
