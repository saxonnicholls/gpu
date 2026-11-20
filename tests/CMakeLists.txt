# Tests. The layout mirrors include/snicholls/gpu/ (scripts/check_mirror.py enforces it): <name>.hpp has
# <name>_test.cpp, a GoogleTest file in snicholls_gpu_tests (cuda_test.cpp in its own, built only where CUDA is). The OpenCL tests need a device: a GPU, or a CPU
# implementation (PoCL in the Ubuntu image and on CI).

add_executable(snicholls_gpu_tests
    opencl_device_test.cpp
    opencl_quant_test.cpp
    opencl_test.cpp
    quant_weights_test.cpp
    version_test.cpp
)
target_link_libraries(snicholls_gpu_tests PRIVATE snicholls::gpu snicholls::core_conformance GTest::gtest_main)
snicholls_gpu_compile_options(snicholls_gpu_tests)
snicholls_gpu_sanitize(snicholls_gpu_tests)
include(GoogleTest)
# PoCL 6 keeps a program's build files under its cache directory at paths given by the program, so processes that
# compile the same program at once (ctest -j) can read each other's half-written files and run wrong kernels. One test
# compiles every OpenCL program first, in one process; the others need it, so they find the programs in the cache.
add_test(NAME opencl.compile_programs COMMAND snicholls_gpu_tests --gtest_filter=GpuOpenCl.TheContextReportsItsDevice:GpuOpenClQuant.TheErrorsAreReported)
set_tests_properties(opencl.compile_programs PROPERTIES LABELS unit FIXTURES_SETUP opencl_programs)
gtest_discover_tests(snicholls_gpu_tests PROPERTIES LABELS unit FIXTURES_REQUIRED opencl_programs DISCOVERY_TIMEOUT 60)

# The CUDA backend's tests, where CUDA is
if(SNICHOLLS_GPU_WITH_CUDA)
    add_executable(snicholls_gpu_cuda_tests cuda_moe_test.cpp cuda_ops_test.cpp cuda_quant_test.cpp cuda_test.cpp)
    target_link_libraries(snicholls_gpu_cuda_tests PRIVATE snicholls::gpu_cuda snicholls::core_conformance GTest::gtest_main)
    snicholls_gpu_compile_options(snicholls_gpu_cuda_tests)
    snicholls_gpu_sanitize(snicholls_gpu_cuda_tests)
    gtest_discover_tests(snicholls_gpu_cuda_tests PROPERTIES LABELS unit DISCOVERY_TIMEOUT 60)
endif()
