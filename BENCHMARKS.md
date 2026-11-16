# Benchmarks

Written by `scripts/run_benchmarks.sh`; never edited by hand. Release, Linux x86_64, 2026-11-16.
Medians of five. Device times include both transfers; GFLOP/s counts 2 m n k.

```
device: NVIDIA RTX PRO 4000 Blackwell SFF Edition
-------------------------------------------------------------------------------------------------------------
Benchmark                                                   Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------------------------
BM_Gemm_HostReference/64/256/256/0_median                1.96 ms         1.96 ms            5 GFLOP/s=4.27757G/s
BM_Gemm_HostReference/64/1024/256/0_median               13.7 ms         13.7 ms            5 GFLOP/s=2.45277G/s
BM_Gemm_HostReference/512/768/768/0_median                181 ms          181 ms            5 GFLOP/s=3.33606G/s
BM_Gemm_HostReference/64/256/256/1_median                1.21 ms         1.21 ms            5 GFLOP/s=6.93991G/s
BM_Gemm_HostReference/64/1024/256/1_median               4.83 ms         4.83 ms            5 GFLOP/s=6.94717G/s
BM_Gemm_HostReference/512/768/768/1_median                113 ms          113 ms            5 GFLOP/s=5.33346G/s
BM_Gemm_Device/64/256/256/0_median                      0.143 ms        0.143 ms            5 GFLOP/s=58.5092G/s
BM_Gemm_Device/64/1024/256/0_median                     0.285 ms        0.285 ms            5 GFLOP/s=117.869G/s
BM_Gemm_Device/512/768/768/0_median                     0.962 ms        0.962 ms            5 GFLOP/s=627.736G/s
BM_Gemm_Device/64/256/256/1_median                      0.148 ms        0.148 ms            5 GFLOP/s=56.6307G/s
BM_Gemm_Device/64/1024/256/1_median                     0.285 ms        0.285 ms            5 GFLOP/s=117.546G/s
BM_Gemm_Device/512/768/768/1_median                     0.981 ms        0.981 ms            5 GFLOP/s=615.773G/s
BM_Gemm_DeviceDirectly/64/256/256/0_median              0.143 ms        0.143 ms            5 GFLOP/s=58.539G/s
BM_Gemm_DeviceDirectly/64/1024/256/0_median             0.286 ms        0.286 ms            5 GFLOP/s=117.448G/s
BM_Gemm_DeviceDirectly/512/768/768/0_median             0.968 ms        0.968 ms            5 GFLOP/s=623.965G/s
BM_Gemm_DeviceDirectly/64/256/256/1_median              0.149 ms        0.149 ms            5 GFLOP/s=56.3354G/s
BM_Gemm_DeviceDirectly/64/1024/256/1_median             0.288 ms        0.288 ms            5 GFLOP/s=116.505G/s
BM_Gemm_DeviceDirectly/512/768/768/1_median             0.986 ms        0.986 ms            5 GFLOP/s=612.477G/s
BM_Gemm_DeviceOnly/1024/1024/1024/0_median               2.06 ms         2.06 ms            5 GFLOP/s=1.0412T/s
BM_Gemm_DeviceOnly/1024/1024/1024/1_median               2.11 ms         2.11 ms            5 GFLOP/s=1.0178T/s
BM_TrainGpt2SmallLayer_Device_median                     53.4 ms         53.4 ms            5 items_per_second=2.39846k/s
BM_TrainGpt2SmallLayer_HostReference_median              2286 ms         2286 ms            5 items_per_second=55.9821/s
BM_TrainGpt2SmallLayer_HostPerToken_Context_median        640 ms          640 ms            5 items_per_second=200.1/s
```
