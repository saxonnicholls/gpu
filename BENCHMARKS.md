# Benchmarks

Written by `scripts/run_benchmarks.sh`; never edited by hand. Release, Linux x86_64, 2026-11-16.
Medians of five. Device times include both transfers; GFLOP/s counts 2 m n k.

```
device: NVIDIA RTX PRO 4000 Blackwell SFF Edition
-------------------------------------------------------------------------------------------------------------
Benchmark                                                   Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------------------------
BM_DeviceMemory_Core/128/768/768_median                 0.032 ms        0.032 ms            5 GFLOP/s=4.73652T/s
BM_DeviceMemory_Core/128/768/3072_median                0.084 ms        0.084 ms            5 GFLOP/s=7.21756T/s
BM_DeviceMemory_Core/512/768/3072_median                0.266 ms        0.266 ms            5 GFLOP/s=9.07395T/s
BM_DeviceMemory_Core/2048/1024/4096_median               1.90 ms         1.90 ms            5 GFLOP/s=9.04072T/s
BM_DeviceMemory_CublasByHand/128/768/768_median         0.031 ms        0.031 ms            5 GFLOP/s=4.79533T/s
BM_DeviceMemory_CublasByHand/128/768/3072_median        0.084 ms        0.084 ms            5 GFLOP/s=7.20045T/s
BM_DeviceMemory_CublasByHand/512/768/3072_median        0.269 ms        0.269 ms            5 GFLOP/s=8.97054T/s
BM_DeviceMemory_CublasByHand/2048/1024/4096_median       1.93 ms         1.93 ms            5 GFLOP/s=8.92089T/s
BM_HostMemory_Staged/128/768/768_median                 0.397 ms        0.397 ms            5 GFLOP/s=380.448G/s
BM_HostMemory_Staged/128/768/3072_median                 1.20 ms         1.20 ms            5 GFLOP/s=504.295G/s
BM_HostMemory_Staged/512/768/3072_median                 1.92 ms         1.92 ms            5 GFLOP/s=1.2601T/s
BM_HostMemory_Staged/2048/1024/4096_median               6.87 ms         6.87 ms            5 GFLOP/s=2.49995T/s
BM_TrainGpt2SmallLayer_CudaStaged_median                 40.1 ms         40.1 ms            5 items_per_second=3.19541k/s
device: NVIDIA RTX PRO 4000 Blackwell SFF Edition
-------------------------------------------------------------------------------------------------------------
Benchmark                                                   Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------------------------
BM_Gemm_HostReference/64/256/256/0_median                1.98 ms         1.98 ms            5 GFLOP/s=4.24171G/s
BM_Gemm_HostReference/64/1024/256/0_median               11.9 ms         11.9 ms            5 GFLOP/s=2.81624G/s
BM_Gemm_HostReference/512/768/768/0_median                176 ms          176 ms            5 GFLOP/s=3.44057G/s
BM_Gemm_HostReference/64/256/256/1_median                1.21 ms         1.21 ms            5 GFLOP/s=6.94362G/s
BM_Gemm_HostReference/64/1024/256/1_median               4.83 ms         4.83 ms            5 GFLOP/s=6.94772G/s
BM_Gemm_HostReference/512/768/768/1_median                113 ms          113 ms            5 GFLOP/s=5.34123G/s
BM_Gemm_Device/64/256/256/0_median                      0.146 ms        0.146 ms            5 GFLOP/s=57.327G/s
BM_Gemm_Device/64/1024/256/0_median                     0.293 ms        0.293 ms            5 GFLOP/s=114.614G/s
BM_Gemm_Device/512/768/768/0_median                     0.973 ms        0.973 ms            5 GFLOP/s=620.479G/s
BM_Gemm_Device/64/256/256/1_median                      0.151 ms        0.151 ms            5 GFLOP/s=55.5732G/s
BM_Gemm_Device/64/1024/256/1_median                     0.294 ms        0.294 ms            5 GFLOP/s=114.26G/s
BM_Gemm_Device/512/768/768/1_median                     0.993 ms        0.993 ms            5 GFLOP/s=607.958G/s
BM_Gemm_DeviceDirectly/64/256/256/0_median              0.146 ms        0.146 ms            5 GFLOP/s=57.451G/s
BM_Gemm_DeviceDirectly/64/1024/256/0_median             0.293 ms        0.293 ms            5 GFLOP/s=114.686G/s
BM_Gemm_DeviceDirectly/512/768/768/0_median             0.974 ms        0.974 ms            5 GFLOP/s=620.362G/s
BM_Gemm_DeviceDirectly/64/256/256/1_median              0.151 ms        0.151 ms            5 GFLOP/s=55.5263G/s
BM_Gemm_DeviceDirectly/64/1024/256/1_median             0.294 ms        0.294 ms            5 GFLOP/s=114.3G/s
BM_Gemm_DeviceDirectly/512/768/768/1_median             0.995 ms        0.995 ms            5 GFLOP/s=607.233G/s
BM_Gemm_DeviceOnly/1024/1024/1024/0_median               2.07 ms         2.07 ms            5 GFLOP/s=1.03651T/s
BM_Gemm_DeviceOnly/1024/1024/1024/1_median               2.12 ms         2.12 ms            5 GFLOP/s=1.014T/s
BM_TrainGpt2SmallLayer_Device_median                     53.1 ms         53.1 ms            5 items_per_second=2.41259k/s
BM_TrainGpt2SmallLayer_HostReference_median              2345 ms         2345 ms            5 items_per_second=54.5785/s
BM_TrainGpt2SmallLayer_HostPerToken_Context_median        637 ms          637 ms            5 items_per_second=200.81/s
```
