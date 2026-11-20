# Benchmarks (Google Benchmark, core's pinned copy). Only Release numbers mean anything; scripts/run_benchmarks.sh
# builds them in Release and rewrites BENCHMARKS.md. A short smoke run of each is a CTest (label benchmark).

add_executable(snicholls_gpu_opencl_benchmark opencl_benchmark.cpp)
target_link_libraries(snicholls_gpu_opencl_benchmark PRIVATE snicholls::gpu benchmark::benchmark)
snicholls_gpu_compile_options(snicholls_gpu_opencl_benchmark)
add_executable(snicholls_gpu_opencl_quant_benchmark opencl_quant_benchmark.cpp)
target_link_libraries(snicholls_gpu_opencl_quant_benchmark PRIVATE snicholls::gpu benchmark::benchmark)
snicholls_gpu_compile_options(snicholls_gpu_opencl_quant_benchmark)
if(SNICHOLLS_GPU_BUILD_TESTS)
    add_test(NAME benchmark.opencl COMMAND snicholls_gpu_opencl_benchmark --benchmark_min_time=0.01s
             --benchmark_filter=/64/)
    add_test(NAME benchmark.opencl_quant COMMAND snicholls_gpu_opencl_quant_benchmark --benchmark_min_time=0.01s
             --benchmark_filter=4096/4096$)
    set_tests_properties(benchmark.opencl benchmark.opencl_quant PROPERTIES LABELS benchmark FIXTURES_REQUIRED opencl_programs)
endif()

if(SNICHOLLS_GPU_WITH_CUDA)
    add_executable(snicholls_gpu_cuda_ops_benchmark cuda_ops_benchmark.cpp)
    target_link_libraries(snicholls_gpu_cuda_ops_benchmark PRIVATE snicholls::gpu_cuda benchmark::benchmark)
    snicholls_gpu_compile_options(snicholls_gpu_cuda_ops_benchmark)
    add_executable(snicholls_gpu_cuda_moe_benchmark cuda_moe_benchmark.cpp)
    target_link_libraries(snicholls_gpu_cuda_moe_benchmark PRIVATE snicholls::gpu_cuda benchmark::benchmark)
    snicholls_gpu_compile_options(snicholls_gpu_cuda_moe_benchmark)
    add_executable(snicholls_gpu_cuda_quant_benchmark cuda_quant_benchmark.cpp)
    target_link_libraries(snicholls_gpu_cuda_quant_benchmark PRIVATE snicholls::gpu_cuda benchmark::benchmark)
    snicholls_gpu_compile_options(snicholls_gpu_cuda_quant_benchmark)
    add_executable(snicholls_gpu_cuda_benchmark cuda_benchmark.cpp)
    target_link_libraries(snicholls_gpu_cuda_benchmark PRIVATE snicholls::gpu_cuda benchmark::benchmark)
    snicholls_gpu_compile_options(snicholls_gpu_cuda_benchmark)
    if(SNICHOLLS_GPU_BUILD_TESTS)
        add_test(NAME benchmark.cuda COMMAND snicholls_gpu_cuda_benchmark --benchmark_min_time=0.01s --benchmark_filter=/128/768/768)
        add_test(NAME benchmark.cuda_ops COMMAND snicholls_gpu_cuda_ops_benchmark --benchmark_min_time=0.01s)
        add_test(NAME benchmark.cuda_moe COMMAND snicholls_gpu_cuda_moe_benchmark --benchmark_min_time=0.01s
                 --benchmark_filter=Steps|/0/0/0)
        add_test(NAME benchmark.cuda_quant COMMAND snicholls_gpu_cuda_quant_benchmark --benchmark_min_time=0.01s --benchmark_filter=4096/4096$)
        set_tests_properties(benchmark.cuda benchmark.cuda_ops benchmark.cuda_moe benchmark.cuda_quant PROPERTIES LABELS benchmark)
    endif()
endif()
