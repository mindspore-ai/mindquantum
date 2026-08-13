cmake_minimum_required(VERSION 3.12)
project(KernelSbHost)

# - set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE STRING "ASCEND CANN package installation directory")
set(ASCEND_CANN_PACKAGE_PATH "/home/ma-user/anaconda3/envs/npu/Ascend/ascend-toolkit/latest/" CACHE STRING "ASCEND CANN package installation directory")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

# --- Original Executable ---
add_executable(kernel_sb_main main.cpp)

target_compile_options(kernel_sb_main PRIVATE -O2 -std=c++17 -Wall)

target_include_directories(kernel_sb_main PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../out/include/kernels
)

target_link_directories(kernel_sb_main PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${CMAKE_CURRENT_SOURCE_DIR}/../out/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../out/lib64
)

target_link_libraries(kernel_sb_main PRIVATE
    ascendcl
    kernels
)

# --- Python Module ---
# Use FindPython and prefer the active conda environment to bind against the current Python.
set(PYBIND11_FINDPYTHON ON)
# Prefer the active conda environment when available (Python 3.10 here).
if(DEFINED ENV{CONDA_PREFIX})
    set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python" CACHE FILEPATH "Conda Python executable" FORCE)
    set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}" CACHE PATH "Conda Python root" FORCE)
endif()
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Automatically find pybind11
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(PYBIND11_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
endif()
find_package(pybind11 REQUIRED)

pybind11_add_module(kernel_sb kernel_sb_wrapper.cpp)

target_compile_options(kernel_sb PRIVATE -O2 -std=c++17 -Wall)

target_include_directories(kernel_sb PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../out/include/kernels
)

target_link_directories(kernel_sb PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${CMAKE_CURRENT_SOURCE_DIR}/../out/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../out/lib64
)

target_link_libraries(kernel_sb PRIVATE
    ascendcl
    kernels
)
