cmake_minimum_required(VERSION 3.12.0)
project(KernelSbKernel)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES "*.cpp")

# Default settings if not provided
set(SOC_VERSION "Ascend910B4" CACHE STRING "system on chip type")
# D set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package installation directory")
set(ASCEND_CANN_PACKAGE_PATH "/home/ma-user/anaconda3/envs/npu/Ascend/ascend-toolkit/latest/" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug" FORCE)
set(LIBRARY_TYPE "SHARED" CACHE STRING "library type: SHARED or STATIC")

# Prefer conda's Python (3.10) for AscendC tooling when available.
if(DEFINED ENV{CONDA_PREFIX})
    set(ASCEND_PYTHON_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python" CACHE FILEPATH "Conda Python executable" FORCE)
endif()
if(NOT ASCEND_PYTHON_EXECUTABLE)
    find_program(ASCEND_PYTHON_EXECUTABLE NAMES python3 python)
endif()

# CMake < 3.13 does not provide target_link_options/target_link_directories; use no-op fallbacks.
if(NOT COMMAND target_link_options)
    function(target_link_options)
    endfunction()
endif()
if(NOT COMMAND target_link_directories)
    function(target_link_directories)
    endfunction()
endif()


# Find ascendc cmake
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(kernels SHARED ${SOURCES})
