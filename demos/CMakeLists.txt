# Demos. The layout mirrors include/snicholls/gpu/ (scripts/check_mirror.py enforces it): <name>.hpp has
# <name>_demo.cpp. Each prints what it does and exits non-zero if the result is not what it documents, so every demo
# is also a CTest (label demo).

function(snicholls_gpu_add_demo name source)
    add_executable(snicholls_gpu_${name} ${source})
    target_link_libraries(snicholls_gpu_${name} PRIVATE snicholls::gpu ${ARGN})
    snicholls_gpu_compile_options(snicholls_gpu_${name})
    snicholls_gpu_sanitize(snicholls_gpu_${name})
    add_test(NAME demo.${name} COMMAND snicholls_gpu_${name})
    set_tests_properties(demo.${name} PROPERTIES LABELS demo)
endfunction()

if(SNICHOLLS_GPU_WITH_CUDA)
    snicholls_gpu_add_demo(cuda_demo cuda_demo.cpp snicholls::gpu_cuda)
    snicholls_gpu_add_demo(cuda_moe_demo cuda_moe_demo.cpp snicholls::gpu_cuda)
    snicholls_gpu_add_demo(cuda_ops_demo cuda_ops_demo.cpp snicholls::gpu_cuda)
    snicholls_gpu_add_demo(cuda_quant_demo cuda_quant_demo.cpp snicholls::gpu_cuda)
    snicholls_gpu_add_demo(cuda_rows_demo cuda_rows_demo.cpp snicholls::gpu_cuda)
endif()
snicholls_gpu_add_demo(opencl_demo opencl_demo.cpp)
snicholls_gpu_add_demo(opencl_device_demo opencl_device_demo.cpp)
snicholls_gpu_add_demo(opencl_quant_demo opencl_quant_demo.cpp)
snicholls_gpu_add_demo(quant_weights_demo quant_weights_demo.cpp)
# The OpenCL demos compile their programs after tests/CMakeLists.txt's opencl.compile_programs, not beside each other
set_tests_properties(demo.opencl_demo demo.opencl_device_demo demo.opencl_quant_demo PROPERTIES FIXTURES_REQUIRED opencl_programs)
snicholls_gpu_add_demo(version_demo version_demo.cpp)
