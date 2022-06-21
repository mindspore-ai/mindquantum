# ==============================================================================
#
# Copyright 2020 <Huawei Technologies Co., Ltd>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

# lint_cmake: -whitespace/indent

# C++ standard flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Always generate position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# RPATH settings... Funadamentally, we do not want to use RPATH but RUNPATH. In order to achieve this, we use a
# combination of these CMake options, some target properties (namely INSTALL_RPATH; see *_set_rpath macros in
# macros.cmake) and some linker flags (see linker_flags.cmake)
#
# All of this should achieve the desired effect on all platforms and compilers

set(CMAKE_BUILD_SKIP_RPATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# CMake usually does not add /usr/local/include to any compiler commands. This can lead to some issues on Mac OS when
# using the -isysroot option so we allow for explicit -I/usr/local/include on the command line.
if(APPLE)
  list(REMOVE_ITEM CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES /usr/local/include)
  list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES /usr/local/include)
  list(REMOVE_ITEM CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES /usr/local/include)
  list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_DIRECTORIES /usr/local/lib)
  list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES /usr/local/lib)
endif()

# ------------------------------------------------------------------------------

if(MSVC)
  test_compile_option(
    compile_msvc_flags
    LANGS CXX
    FLAGS "/Zc:__cplusplus")

  test_compile_option(
    compile_msvc_mt_flags
    LANGS C CXX
    FLAGS "/MT"
    GENEX "$<AND:$<CONFIG:RELEASE>,$<BOOL:${ENABLE_MT}>>"
    NO_TRYCOMPILE_TARGET NO_TRYCOMPILE_FLAGCHECK_TARGET)
  test_compile_option(
    compile_msvc_mt_flags
    LANGS C CXX
    FLAGS "/MTd"
    GENEX "$<AND:$<OR:$<CONFIG:DEBUG>,$<CONFIG:RELWITHDEBINFO>>,$<BOOL:${ENABLE_MT}>>"
    NO_TRYCOMPILE_TARGET NO_TRYCOMPILE_FLAGCHECK_TARGET)

  test_compile_option(
    compile_msvc_md_flags
    LANGS C CXX
    FLAGS "/MD"
    GENEX "$<AND:$<CONFIG:RELEASE>,$<BOOL:${ENABLE_MD}>>"
    NO_TRYCOMPILE_TARGET NO_TRYCOMPILE_FLAGCHECK_TARGET)
  test_compile_option(
    compile_msvc_md_flags
    LANGS C CXX
    FLAGS "/MDd"
    GENEX "$<AND:$<OR:$<CONFIG:DEBUG>,$<CONFIG:RELWITHDEBINFO>>,$<BOOL:${ENABLE_MD}>>"
    NO_TRYCOMPILE_TARGET NO_TRYCOMPILE_FLAGCHECK_TARGET)
endif()

# ------------------------------------------------------------------------------

if(ENABLE_CUDA)
  test_compile_option(
    cuda_allow_unsupported_flag FLAGCHECK
    LANGS CUDA
    FLAGS "-allow-unsupported-compiler"
    CMAKE_OPTION CUDA_ALLOW_UNSUPPORTED_COMPILER)

  test_compile_option(
    nvhpc_flagcheck_flags FLAGCHECK
    LANGS NVCXX
    FLAGS "--flagcheck"
    NO_MQ_TARGET NO_TRYCOMPILE_TARGET)

  test_compile_option(
    nvhpc_cxx_standard_flags FLAGCHECK
    LANGS NVCXX
    FLAGS "-std=c++20 -std=c++17")

  set(_flag -gpu=cuda${MQ_CUDA_VERSION})
  test_compile_option(
    nvhpc_cuda_version_flags FLAGCHECK
    LANGS NVCXX
    FLAGS "${_flag}")

  if(NOT nvhpc_cuda_version_flags_NVCXX)
    disable_cuda("NVHPC not supporting ${_flag}")
  endif()
  unset(_flag)

  set(_args)
  foreach(_cc ${CMAKE_CUDA_ARCHITECTURES})
    test_compile_option(
      nvhpc_gpu_compute_capability FLAGCHECK
      LANGS NVCXX
      FLAGS "-gpu=cc${_cc}" ${_args})
    # Only add multiple -gpu=ccXX for the "real" target and no the try_compile ones in order to speed up the
    # compilations in the case of calls to try_compile()
    set(_args NO_TRYCOMPILE_TARGET NO_TRYCOMPILE_FLAGCHECK_TARGET)
  endforeach()

  test_compile_option(
    nvhpc_cuda_flags FLAGCHECK
    LANGS NVCXX
    FLAGS "-stdpar" "-cuda")

  if(NOT nvhpc_cuda_flags_NVCXX)
    disable_cuda("NVHPC not supporting -stdpar -cuda")
  endif()

  # For all the languages except NVCXX, use NVHPC's filename extension detection for the language
  target_compile_options(NVCXX_mindquantum INTERFACE "$<$<AND:$<OR:$<C_COMPILER_ID:NVHPC>,\
$<CXX_COMPILER_ID:NVHPC>,$<CUDA_COMPILER_ID:NVHPC>>,$<NOT:$<COMPILE_LANGUAGE:NVCXX>>>:-x none>")
endif()

# ------------------------------------------------------------------------------

test_compile_option(
  compile_flags_release
  LANGS C CXX DPCXX
  FLAGS "-ffast-math /fp:fast -fast" "-O3 /Ox"
  GENEX "$<OR:$<CONFIG:RELEASE>,$<CONFIG:RELWITHDEBINFO>>")

# --------------------------------------

if(X86_64)
  test_compile_option(
    intrin_flag
    LANGS C CXX DPCXX
    NO_MQ_TARGET NO_TRYCOMPILE_TARGET NO_TRYCOMPILE_FLAGCHECK_TARGET
    FLAGS "-mavx2 -xCORE-AVX2 /QxCORE-AVX2 /arch:AVX2")
elseif(AARCH64)
  test_compile_option(
    intrin_flag
    LANGS C CXX DPCXX
    NO_MQ_TARGET NO_TRYCOMPILE_TARGET NO_TRYCOMPILE_FLAGCHECK_TARGET
    FLAGS "-march=armv8.5-a -march=armv8.4-a -march=armv8.3-a -march=armv8.2-a")
endif()

foreach(_lang C CXX DPCXX)
  if(TARGET intrin_flag_${_lang})
    append_to_property(mq_install_targets GLOBAL intrin_flag_${_lang})
  endif()
endforeach()

# --------------------------------------

test_compile_option(
  dpcpp_flags
  LANGS DPCXX
  FLAGS "-fsycl")

# --------------------------------------

test_compile_option(
  profiling_flags
  LANGS C CXX DPCXX
  FLAGS "-pg -prof-gen /Qprof-gen" "-fprofile-instr-generate"
  CMAKE_OPTION ENABLE_PROFILING)

# --------------------------------------

test_compile_option(
  stack_protection
  LANGS C CXX DPCXX
  FLAGS "-fstack-protector-all"
  CMAKE_OPTION ENABLE_STACK_PROTECTION)

# ------------------------------------------------------------------------------

if(NOT VERSION_INFO)
  execute_process(
    COMMAND ${Python_EXECUTABLE} setup.py --version
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE _version_info
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  set(VERSION_INFO "\"${_version_info}\"")
endif()

if(VERSION_INFO MATCHES [=["(.*)\.dev[0-9]+"$]=])
  message(STATUS "Version info for source code: ${VERSION_INFO} -> \"${CMAKE_MATCH_1}\"")
  set(VERSION_INFO "\"${CMAKE_MATCH_1}\"")
endif()

if(VERSION_INFO MATCHES [=["(.*)"]=])
  set(MQ_VERSION ${CMAKE_MATCH_1})
else()
  set(MQ_VERSION ${VERSION_INFO})
endif()

if("${MQ_VERSION}" STREQUAL "" AND EXISTS "${PROJECT_SOURCE_DIR}/VERSION.txt")
  file(STRINGS "${PROJECT_SOURCE_DIR}/VERSION.txt" MQ_VERSION)
endif()

if("${MQ_VERSION}" STREQUAL "")
  message(FATAL_ERROR "Unable to get MindQuantum version number!")
else()
  message(STATUS "MindQuantum version: ${MQ_VERSION}")
endif()

# --------------------------------------

include(compiler_test)

# --------------------------------------

mq_add_compile_definitions("$<$<BOOL:${USE_OPENMP}>:USE_OPENMP>" "$<$<BOOL:${USE_PARALLEL_STL}>:USE_PARALLEL_STL>"
                           "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CXX>>:_FORTIFY_SOURCE=2>")

# ==============================================================================
# Platform specific flags

if(MSVC)
  if(NOT "${CMAKE_C_COMPILER_LAUNCHER}" STREQUAL "" AND NOT "${CMAKE_CXX_COMPILER_LAUNCHER}" STREQUAL "")
    message(STATUS "Replacing /Zi with /Z7 in compiler flags")
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_MINSIZEREL}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}")
  endif()
  mq_add_compile_definitions(_USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS WIN32_LEAN_AND_MEAN _ITERATOR_DEBUG_LEVEL=0)
elseif(MINGW)
  mq_add_compile_definitions(_USE_MATH_DEFINES)
endif()

# ==============================================================================

configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake_config.hpp.in ${PROJECT_BINARY_DIR}/core/cmake_config.hpp)

add_library(cmake_config INTERFACE)
target_include_directories(cmake_config INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)

# ------------------------------------------------------------------------------

append_to_property(mq_install_targets GLOBAL cmake_config)
install(FILES ${PROJECT_BINARY_DIR}/core/cmake_config.hpp DESTINATION ${MQ_INSTALL_INCLUDEDIR}/experimental/core)

# ==============================================================================
