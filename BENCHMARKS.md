# Benchmarks

Written by `scripts/run_benchmarks.sh`; never edited by hand. Release, Darwin x86_64, 2026-11-15.
Medians of five. Device times include both transfers; GFLOP/s counts 2 m n k.

```
device: AMD Radeon Pro Vega II Compute Engine
------------------------------------------------------------------------------------------------------
Benchmark                                            Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------------------------
BM_Gemm_HostReference/64/256/256/0_median         3.74 ms         3.74 ms            5 GFLOP/s=2.24018G/s
BM_Gemm_HostReference/64/1024/256/0_median        19.2 ms         19.2 ms            5 GFLOP/s=1.74834G/s
BM_Gemm_HostReference/512/768/768/0_median         337 ms          337 ms            5 GFLOP/s=1.7898G/s
BM_Gemm_HostReference/64/256/256/1_median         3.67 ms         3.67 ms            5 GFLOP/s=2.28614G/s
BM_Gemm_HostReference/64/1024/256/1_median        14.8 ms         14.8 ms            5 GFLOP/s=2.27237G/s
BM_Gemm_HostReference/512/768/768/1_median         292 ms          292 ms            5 GFLOP/s=2.07079G/s
BM_Gemm_Device/64/256/256/0_median               0.986 ms        0.563 ms            5 GFLOP/s=14.888G/s
BM_Gemm_Device/64/1024/256/0_median              0.897 ms        0.528 ms            5 GFLOP/s=63.6046G/s
BM_Gemm_Device/512/768/768/0_median               2.41 ms         1.47 ms            5 GFLOP/s=410.548G/s
BM_Gemm_Device/64/256/256/1_median               0.635 ms        0.330 ms            5 GFLOP/s=25.4586G/s
BM_Gemm_Device/64/1024/256/1_median              0.896 ms        0.515 ms            5 GFLOP/s=65.1512G/s
BM_Gemm_Device/512/768/768/1_median               2.02 ms         1.10 ms            5 GFLOP/s=547.954G/s
BM_Gemm_DeviceDirectly/64/256/256/0_median       0.605 ms        0.311 ms            5 GFLOP/s=26.9957G/s
BM_Gemm_DeviceDirectly/64/1024/256/0_median      0.850 ms        0.486 ms            5 GFLOP/s=69.01G/s
BM_Gemm_DeviceDirectly/512/768/768/0_median       2.05 ms         1.14 ms            5 GFLOP/s=527.672G/s
BM_Gemm_DeviceDirectly/64/256/256/1_median       0.606 ms        0.310 ms            5 GFLOP/s=27.038G/s
BM_Gemm_DeviceDirectly/64/1024/256/1_median      0.877 ms        0.510 ms            5 GFLOP/s=65.7546G/s
BM_Gemm_DeviceDirectly/512/768/768/1_median       2.16 ms         1.27 ms            5 GFLOP/s=474.385G/s
BM_Gemm_DeviceOnly/1024/1024/1024/0_median        4.66 ms         2.74 ms            5 GFLOP/s=784.49G/s
BM_Gemm_DeviceOnly/1024/1024/1024/1_median        5.45 ms         3.42 ms            5 GFLOP/s=627.003G/s
```
