# snicholls-gpu - backends of snicholls-core's interfaces on GPUs.
#
# snicholls-core is standard-library-only and never includes a device header; the implementations of its
# interfaces that need a GPU runtime live here, and pin core like any other member of the family:
#
#   snicholls::gpu   the OpenCL backend of linalg::blas_backend (include/snicholls/gpu/opencl.hpp), header-only,
#                    linking OpenCL::OpenCL.
#
# Driven two ways: as the top-level project (tests, demos, benchmarks, docs), and as a subdirectory of a consumer
# (the target only).

cmake_minimum_required(VERSION 3.20)

project(snicholls_gpu
    VERSION 0.2.0
    DESCRIPTION "snicholls-gpu - GPU backends (OpenCL) of snicholls-core's interfaces"
    LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SNICHOLLS_GPU_TOP_LEVEL ON)
else()
    set(SNICHOLLS_GPU_TOP_LEVEL OFF)
endif()

option(SNICHOLLS_GPU_BUILD_TESTS "Build the tests and demos" ${SNICHOLLS_GPU_TOP_LEVEL})
option(SNICHOLLS_GPU_BUILD_BENCHMARKS "Build the benchmarks (Google Benchmark)" OFF)
option(SNICHOLLS_GPU_BUILD_DOCS "Build the Doxygen documentation; warnings are errors" OFF)
option(SNICHOLLS_GPU_WERROR "Treat warnings as errors in everything this project compiles" OFF)
set(SNICHOLLS_GPU_SANITIZE "" CACHE STRING "Sanitizers for the tests and demos, e.g. address,undefined")

if(SNICHOLLS_GPU_TOP_LEVEL AND NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# ---------------------------------------------------------------------------
# snicholls-core, pinned to a release tag (third_party/core). The diamond rules (core's CLAUDE.md "Versioning and
# the diamond"): when a consumer already added core, core's own CMakeLists records this copy and skips it.

set(SNICHOLLS_GPU_CORE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/core)
if(NOT EXISTS "${SNICHOLLS_GPU_CORE_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "Submodule 'third_party/core' is not checked out.\n"
                        "Run: git submodule update --init --recursive")
endif()
add_subdirectory(${SNICHOLLS_GPU_CORE_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/third_party/core EXCLUDE_FROM_ALL)
snicholls_core_require(0.10.0)

find_package(OpenCL REQUIRED)

# CUDA: the NVIDIA backend (include/snicholls/gpu/cuda.hpp), where the toolkit is (cuBLAS, the runtime)
find_package(CUDAToolkit QUIET)
option(SNICHOLLS_GPU_WITH_CUDA "Build the CUDA backend, its tests and demo (needs the CUDA toolkit)" ${CUDAToolkit_FOUND})
if(SNICHOLLS_GPU_WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

function(snicholls_gpu_compile_options target)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    if(SNICHOLLS_GPU_WERROR)
        target_compile_options(${target} PRIVATE -Werror)
    endif()
endfunction()

function(snicholls_gpu_sanitize target)
    if(SNICHOLLS_GPU_SANITIZE)
        target_compile_options(${target} PRIVATE -fsanitize=${SNICHOLLS_GPU_SANITIZE} -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=${SNICHOLLS_GPU_SANITIZE})
    endif()
endfunction()

add_library(snicholls_gpu INTERFACE)
add_library(snicholls::gpu ALIAS snicholls_gpu)
target_include_directories(snicholls_gpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(snicholls_gpu INTERFACE snicholls::core OpenCL::OpenCL)
target_compile_features(snicholls_gpu INTERFACE cxx_std_20)

if(SNICHOLLS_GPU_WITH_CUDA)
    add_library(snicholls_gpu_cuda INTERFACE)
    add_library(snicholls::gpu_cuda ALIAS snicholls_gpu_cuda)
    target_include_directories(snicholls_gpu_cuda INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(snicholls_gpu_cuda INTERFACE snicholls::core CUDA::cudart CUDA::cublas CUDA::nvrtc)
    target_compile_features(snicholls_gpu_cuda INTERFACE cxx_std_20)
endif()

# ---------------------------------------------------------------------------
# Tests, demos, benchmarks, docs

if(SNICHOLLS_GPU_BUILD_TESTS)
    enable_testing()
    if(NOT TARGET GTest::gtest_main)
        # GoogleTest: core's pinned copy (third_party/core/third_party/googletest, v1.18.0)
        set(_gtest_root ${SNICHOLLS_GPU_CORE_ROOT}/third_party/googletest)
        if(NOT EXISTS ${_gtest_root}/CMakeLists.txt)
            message(FATAL_ERROR "${_gtest_root} is empty - run: git submodule update --init --recursive")
        endif()
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        add_subdirectory(${_gtest_root} ${CMAKE_CURRENT_BINARY_DIR}/third_party/googletest EXCLUDE_FROM_ALL)
        # Plain -I, never SYSTEM: a Homebrew googletest in /usr/local/include would shadow it otherwise
        set_property(TARGET gtest gtest_main gmock gmock_main PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")
    endif()

    # Every header compiles on its own, twice (a broken include guard is caught too)
    file(GLOB_RECURSE _gpu_headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include CONFIGURE_DEPENDS
         ${CMAKE_CURRENT_SOURCE_DIR}/include/snicholls/*.hpp)
    set(_gpu_check_sources)
    foreach(_header IN LISTS _gpu_headers)
        if(_header MATCHES "/cuda(_ops)?\\.hpp$" AND NOT SNICHOLLS_GPU_WITH_CUDA)
            continue()
        endif()
        string(MAKE_C_IDENTIFIER "${_header}" _id)
        set(HEADER "${_header}")
        configure_file(cmake/header_check.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/header_check/${_id}.cpp @ONLY)
        list(APPEND _gpu_check_sources ${CMAKE_CURRENT_BINARY_DIR}/header_check/${_id}.cpp)
    endforeach()
    add_library(snicholls_gpu_header_check OBJECT ${_gpu_check_sources})
    target_link_libraries(snicholls_gpu_header_check PRIVATE snicholls::gpu)
    if(SNICHOLLS_GPU_WITH_CUDA)
        target_link_libraries(snicholls_gpu_header_check PRIVATE snicholls::gpu_cuda)
    endif()
    snicholls_gpu_compile_options(snicholls_gpu_header_check)

    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME check.version COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_version.py)
    add_test(NAME check.mirror COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_mirror.py)
    set_tests_properties(check.version check.mirror PROPERTIES LABELS boundary)

    add_subdirectory(tests)
    add_subdirectory(demos)
endif()

if(SNICHOLLS_GPU_BUILD_BENCHMARKS)
    if(NOT TARGET benchmark::benchmark)
        set(_bench_root ${SNICHOLLS_GPU_CORE_ROOT}/third_party/benchmark)
        if(NOT EXISTS ${_bench_root}/CMakeLists.txt)
            message(FATAL_ERROR "${_bench_root} is empty - run: git submodule update --init --recursive")
        endif()
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
        add_subdirectory(${_bench_root} ${CMAKE_CURRENT_BINARY_DIR}/third_party/benchmark EXCLUDE_FROM_ALL)
    endif()
    add_subdirectory(benchmarks)
endif()

if(SNICHOLLS_GPU_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs)
    configure_file(docs/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Doxygen: every public entity documented, warnings are errors"
        VERBATIM)
    if(SNICHOLLS_GPU_BUILD_TESTS)
        add_test(NAME docs.build COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(docs.build PROPERTIES LABELS docs)
    endif()
endif()

# snicholls-core diamond rule 3: exactly one version of core in this build. Only the top-level project asserts it;
# as a subdirectory, the consumer's last line does.
if(SNICHOLLS_GPU_TOP_LEVEL)
    snicholls_core_assert_single_version()
endif()
