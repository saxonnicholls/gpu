# Tests. The layout mirrors include/snicholls/gpu/ (scripts/check_mirror.py enforces it): <name>.hpp has
# <name>_test.cpp, a GoogleTest file in snicholls_gpu_tests (cuda_test.cpp in its own, built only where CUDA is). The OpenCL tests need a device: a GPU, or a CPU
# implementation (PoCL in the Ubuntu image and on CI).

add_executable(snicholls_gpu_tests
    opencl_quant_test.cpp
    opencl_test.cpp
    version_test.cpp
)
target_link_libraries(snicholls_gpu_tests PRIVATE snicholls::gpu snicholls::core_conformance GTest::gtest_main)
snicholls_gpu_compile_options(snicholls_gpu_tests)
snicholls_gpu_sanitize(snicholls_gpu_tests)
include(GoogleTest)
gtest_discover_tests(snicholls_gpu_tests PROPERTIES LABELS unit DISCOVERY_TIMEOUT 60)

# The CUDA backend's tests, where CUDA is
if(SNICHOLLS_GPU_WITH_CUDA)
    add_executable(snicholls_gpu_cuda_tests cuda_moe_test.cpp cuda_ops_test.cpp cuda_quant_test.cpp cuda_test.cpp)
    target_link_libraries(snicholls_gpu_cuda_tests PRIVATE snicholls::gpu_cuda snicholls::core_conformance GTest::gtest_main)
    snicholls_gpu_compile_options(snicholls_gpu_cuda_tests)
    snicholls_gpu_sanitize(snicholls_gpu_cuda_tests)
    gtest_discover_tests(snicholls_gpu_cuda_tests PROPERTIES LABELS unit DISCOVERY_TIMEOUT 60)
endif()
