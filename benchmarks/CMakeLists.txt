# Benchmarks (Google Benchmark, core's pinned copy). Only Release numbers mean anything; scripts/run_benchmarks.sh
# builds them in Release and rewrites BENCHMARKS.md. A short smoke run of each is a CTest (label benchmark).

add_executable(snicholls_gpu_opencl_benchmark opencl_benchmark.cpp)
target_link_libraries(snicholls_gpu_opencl_benchmark PRIVATE snicholls::gpu benchmark::benchmark)
snicholls_gpu_compile_options(snicholls_gpu_opencl_benchmark)
if(SNICHOLLS_GPU_BUILD_TESTS)
    add_test(NAME benchmark.opencl COMMAND snicholls_gpu_opencl_benchmark --benchmark_min_time=0.01s
             --benchmark_filter=/64/)
    set_tests_properties(benchmark.opencl PROPERTIES LABELS benchmark)
endif()

if(SNICHOLLS_GPU_WITH_CUDA)
    add_executable(snicholls_gpu_cuda_benchmark cuda_benchmark.cpp)
    target_link_libraries(snicholls_gpu_cuda_benchmark PRIVATE snicholls::gpu_cuda benchmark::benchmark)
    snicholls_gpu_compile_options(snicholls_gpu_cuda_benchmark)
    if(SNICHOLLS_GPU_BUILD_TESTS)
        add_test(NAME benchmark.cuda COMMAND snicholls_gpu_cuda_benchmark --benchmark_min_time=0.01s --benchmark_filter=/128/768/768)
        set_tests_properties(benchmark.cuda PROPERTIES LABELS benchmark)
    endif()
endif()
