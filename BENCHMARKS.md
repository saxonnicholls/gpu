# Benchmarks

Written by `scripts/run_benchmarks.sh`; never edited by hand. Release, Linux x86_64, 2026-11-16.
Medians of five. Device times include both transfers; GFLOP/s counts 2 m n k.

```
device: NVIDIA RTX PRO 4000 Blackwell SFF Edition
-------------------------------------------------------------------------------------------------------------
Benchmark                                                   Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------------------------
BM_DeviceMemory_Core/128/768/768_median                 0.026 ms        0.026 ms            5 GFLOP/s=5.88697T/s
BM_DeviceMemory_Core/128/768/3072_median                0.076 ms        0.076 ms            5 GFLOP/s=7.98732T/s
BM_DeviceMemory_Core/512/768/3072_median                0.241 ms        0.241 ms            5 GFLOP/s=10.0296T/s
BM_DeviceMemory_Core/2048/1024/4096_median               1.88 ms         1.88 ms            5 GFLOP/s=9.13584T/s
BM_DeviceMemory_CublasByHand/128/768/768_median         0.033 ms        0.033 ms            5 GFLOP/s=4.59368T/s
BM_DeviceMemory_CublasByHand/128/768/3072_median        0.087 ms        0.087 ms            5 GFLOP/s=6.97454T/s
BM_DeviceMemory_CublasByHand/512/768/3072_median        0.281 ms        0.281 ms            5 GFLOP/s=8.59999T/s
BM_DeviceMemory_CublasByHand/2048/1024/4096_median       2.06 ms         2.06 ms            5 GFLOP/s=8.32609T/s
BM_HostMemory_Staged/128/768/768_median                 0.396 ms        0.396 ms            5 GFLOP/s=381.573G/s
BM_HostMemory_Staged/128/768/3072_median                 1.20 ms         1.20 ms            5 GFLOP/s=501.521G/s
BM_HostMemory_Staged/512/768/3072_median                 1.94 ms         1.94 ms            5 GFLOP/s=1.24561T/s
BM_HostMemory_Staged/2048/1024/4096_median               6.97 ms         6.97 ms            5 GFLOP/s=2.46393T/s
BM_TrainGpt2SmallLayer_CudaStaged_median                 40.2 ms         40.2 ms            5 items_per_second=3.18249k/s
device: NVIDIA RTX PRO 4000 Blackwell SFF Edition
--------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------------------------------------------
BM_TrainGpt2SmallLayer_DeviceResident_median                          1.42 ms         1.42 ms            5 items_per_second=89.9936k/s
BM_TrainGpt2SmallLayer_HostStaged_median                              61.5 ms         61.5 ms            5 items_per_second=2.08148k/s
BM_Gpt2SmallAttention_ForwardBackward/tokens:256/path:0_median       0.579 ms        0.579 ms            5 items_per_second=442.12k/s
BM_Gpt2SmallAttention_ForwardBackward/tokens:1024/path:0_median       1.88 ms         1.88 ms            5 items_per_second=545.245k/s
BM_Gpt2SmallAttention_ForwardBackward/tokens:4096/path:0_median       38.8 ms         38.8 ms            5 items_per_second=105.612k/s
BM_Gpt2SmallAttention_ForwardBackward/tokens:256/path:1_median       0.423 ms        0.423 ms            5 items_per_second=604.597k/s
BM_Gpt2SmallAttention_ForwardBackward/tokens:1024/path:1_median       1.12 ms         1.12 ms            5 items_per_second=911.059k/s
BM_Gpt2SmallAttention_ForwardBackward/tokens:4096/path:1_median       27.8 ms         27.8 ms            5 items_per_second=147.22k/s
BM_Gpt2SmallAttention_ForwardBackward/tokens:256/path:2_median       0.164 ms        0.164 ms            5 items_per_second=1.56283M/s
BM_Gpt2SmallAttention_ForwardBackward/tokens:1024/path:2_median      0.820 ms        0.820 ms            5 items_per_second=1.24827M/s
BM_Gpt2SmallAttention_ForwardBackward/tokens:4096/path:2_median       9.54 ms         9.54 ms            5 items_per_second=429.391k/s
device: NVIDIA RTX PRO 4000 Blackwell SFF Edition
-------------------------------------------------------------------------------------------------------------
Benchmark                                                   Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------------------------
BM_Gemm_HostReference/64/256/256/0_median                1.96 ms         1.96 ms            5 GFLOP/s=4.27318G/s
BM_Gemm_HostReference/64/1024/256/0_median               12.7 ms         12.7 ms            5 GFLOP/s=2.64046G/s
BM_Gemm_HostReference/512/768/768/0_median                176 ms          176 ms            5 GFLOP/s=3.42486G/s
BM_Gemm_HostReference/64/256/256/1_median                1.21 ms         1.21 ms            5 GFLOP/s=6.90785G/s
BM_Gemm_HostReference/64/1024/256/1_median               4.84 ms         4.84 ms            5 GFLOP/s=6.93314G/s
BM_Gemm_HostReference/512/768/768/1_median                113 ms          113 ms            5 GFLOP/s=5.34123G/s
BM_Gemm_Device/64/256/256/0_median                      0.146 ms        0.146 ms            5 GFLOP/s=57.2941G/s
BM_Gemm_Device/64/1024/256/0_median                     0.291 ms        0.291 ms            5 GFLOP/s=115.369G/s
BM_Gemm_Device/512/768/768/0_median                     0.977 ms        0.977 ms            5 GFLOP/s=618.036G/s
BM_Gemm_Device/64/256/256/1_median                      0.152 ms        0.152 ms            5 GFLOP/s=55.106G/s
BM_Gemm_Device/64/1024/256/1_median                     0.292 ms        0.292 ms            5 GFLOP/s=114.922G/s
BM_Gemm_Device/512/768/768/1_median                     0.992 ms        0.992 ms            5 GFLOP/s=608.6G/s
BM_Gemm_DeviceDirectly/64/256/256/0_median              0.146 ms        0.146 ms            5 GFLOP/s=57.4231G/s
BM_Gemm_DeviceDirectly/64/1024/256/0_median             0.290 ms        0.290 ms            5 GFLOP/s=115.609G/s
BM_Gemm_DeviceDirectly/512/768/768/0_median             0.977 ms        0.977 ms            5 GFLOP/s=618.426G/s
BM_Gemm_DeviceDirectly/64/256/256/1_median              0.152 ms        0.152 ms            5 GFLOP/s=55.3249G/s
BM_Gemm_DeviceDirectly/64/1024/256/1_median             0.291 ms        0.291 ms            5 GFLOP/s=115.125G/s
BM_Gemm_DeviceDirectly/512/768/768/1_median             0.993 ms        0.993 ms            5 GFLOP/s=608.01G/s
BM_Gemm_DeviceOnly/1024/1024/1024/0_median               2.06 ms         2.06 ms            5 GFLOP/s=1.04081T/s
BM_Gemm_DeviceOnly/1024/1024/1024/1_median               2.11 ms         2.11 ms            5 GFLOP/s=1.01961T/s
BM_TrainGpt2SmallLayer_Device_median                     53.2 ms         53.2 ms            5 items_per_second=2.40776k/s
BM_TrainGpt2SmallLayer_HostReference_median              2339 ms         2339 ms            5 items_per_second=54.7276/s
BM_TrainGpt2SmallLayer_HostPerToken_Context_median        637 ms          637 ms            5 items_per_second=200.998/s
```
