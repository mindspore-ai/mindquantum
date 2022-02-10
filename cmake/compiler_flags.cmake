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

# C++ standard flags
set(CMAKE_CXX_STANDARD 17)
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

if(WIN32)
  test_compile_option(
    _compile_win32_flags
    LANGS CXX
    FLAGS "/Zc:__cplusplus"
    AUTO_ADD_CO
  )
endif()

# ------------------------------------------------------------------------------

test_compile_option(
  _compile_flags_release
  LANGS CXX DPCXX
  FLAGS "-ffast-math /fp:fast -fast" "-O3 /Ox"
  AUTO_ADD_CO
  GENEX "$<AND:$<OR:$<CONFIG:RELEASE>,$<CONFIG:RELWITHDEBINFO>>,$<COMPILE_LANGUAGE:@lang@>>"
)

# --------------------------------------

if(X86_64)
  test_compile_option(
    _intrin_flag
    LANGS CXX DPCXX
    FLAGS "-mavx2 -xCORE-AVX2 /QxCORE-AVX2 /arch:AVX2"
  )
elseif(AARCH64)
  test_compile_option(
    _intrin_flag
    LANGS CXX DPCXX
    FLAGS "-march=armv8.5-a -march=armv8.4-a -march=armv8.3-a -march=armv8.2-a"
  )
endif()

# --------------------------------------

test_compile_option(
  _dpcpp_flags
  LANGS DPCXX
  FLAGS "-fsycl"
  AUTO_ADD_CO)

# --------------------------------------

if(ENABLE_PROFILING)
  test_compile_option(
    _profiling_flags
    LANGS CXX DPCXX
    FLAGS "-pg -prof-gen /Qprof-gen" "-fprofile-instr-generate"
    AUTO_ADD_CO)
endif()

# --------------------------------------

if(ENABLE_STACK_PROTECTION)
  test_compile_option(
    _stack_protection
    LANGS CXX DPCXX
    FLAGS "-fstack-protector-all"
    AUTO_ADD_CO)
endif()

# ------------------------------------------------------------------------------

if(NOT VERSION_INFO)
  execute_process(
    COMMAND ${Python_EXECUTABLE} setup.py --version
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE _version_info
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  set(VERSION_INFO "\"${_version_info}\"")
endif()

# --------------------------------------

add_compile_definitions(
  "$<$<BOOL:${ENABLE_OPENMP}>:ENABLE_OPENMP>" "$<$<BOOL:${VERSION_INFO}>:VERSION_INFO=${VERSION_INFO}>"
  "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CXX>>:_FORTIFY_SOURCE=2>")

# ==============================================================================
# Platform specific flags

if(WIN32)
  add_compile_definitions(_USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS WIN32_LEAN_AND_MEAN)
endif()

# ==============================================================================
