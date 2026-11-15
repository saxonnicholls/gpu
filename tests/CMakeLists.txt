# Tests. The layout mirrors include/snicholls/gpu/ (scripts/check_mirror.py enforces it): <name>.hpp has
# <name>_test.cpp, a GoogleTest file in snicholls_gpu_tests. The OpenCL tests need a device: a GPU, or a CPU
# implementation (PoCL in the Ubuntu image and on CI).

add_executable(snicholls_gpu_tests
    opencl_test.cpp
    version_test.cpp
)
target_link_libraries(snicholls_gpu_tests PRIVATE snicholls::gpu snicholls::core_conformance GTest::gtest_main)
snicholls_gpu_compile_options(snicholls_gpu_tests)
snicholls_gpu_sanitize(snicholls_gpu_tests)
include(GoogleTest)
gtest_discover_tests(snicholls_gpu_tests PROPERTIES LABELS unit DISCOVERY_TIMEOUT 60)
