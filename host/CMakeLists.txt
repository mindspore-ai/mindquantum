cmake_minimum_required(VERSION 3.22)
project(KernelSbHost LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN 9.1 installation")
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED PATHS "${PYBIND11_CMAKE_DIR}" NO_DEFAULT_PATH)

pybind11_add_module(_kernel_sb kernel_sb_wrapper.cpp)
target_compile_options(_kernel_sb PRIVATE -Wall -Wextra)
target_include_directories(_kernel_sb PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../out/include/kernels)
target_link_directories(_kernel_sb PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${CMAKE_CURRENT_SOURCE_DIR}/../out/lib)
target_link_libraries(_kernel_sb PRIVATE kernels ascendcl opapi nnopbase)
set_target_properties(_kernel_sb PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    BUILD_RPATH "$ORIGIN/out/lib;${ASCEND_CANN_PACKAGE_PATH}/lib64")
