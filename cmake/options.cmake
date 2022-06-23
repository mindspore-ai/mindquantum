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

include(CMakeDependentOption)

# ==============================================================================
# MindQuantum feature selection

option(ENABLE_PROJECTQ "Enable ProjectQ support" ON)
option(ENABLE_GITEE "Use Gitee instead of GitHub for checking out third-party dependencies" OFF)
option(ENABLE_CXX_EXPERIMENTAL "Enable the new (experimental) C++ backend" OFF)
option(ENABLE_DOCUMENTATION "Enable building of the documentation using Doxygen" OFF)

# ==============================================================================
# Python related options

if(APPLE)
  option(PYTHON_VIRTUALENV_COMPAT "(Mac OS X) Make CMake search for Python Framework *after* any available\
  unix-style package. Can be useful in case of virtual environments." ON)
else()
  option(PYTHON_VIRTUALENV_COMPAT "(Mac OS X) Make CMake search for Python Framework *after* any available\
  unix-style package. Can be useful in case of virtual environments." OFF)
endif()

option(IS_PYTHON_BUILD "Is CMake called from setup.py? (e.g. python3 setup.py install?)" OFF)
option(IN_PLACE_BUILD "Are we building in-place for testing/development?" ON)

# ==============================================================================
# CUDA related options

if(DEFINED ENABLE_CUDA)
  set(_enable_cuda_init ${ENABLE_CUDA})
elseif(DEFINED GPUACCELERATED)
  set(_enable_cuda_init ${GPUACCELERATED})
else()
  set(_enable_cuda_init OFF)
endif()

option(ENABLE_CUDA "Enable building of CUDA libraries" _enable_cuda_init)
option(CUDA_ALLOW_UNSUPPORTED_COMPILER "Allow the use of an unsupported compiler version" OFF)
option(CUDA_STATIC "Use static version of Nvidia CUDA libraries during linking (also applies to nvc++)" OFF)

# ==============================================================================
# Compilation options

option(USE_OPENMP "Enable the use of OpenMP throughout the code" ON)

# cmake-lint: disable=C0103
set(_USE_PARALLEL_STL OFF)
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC"
   OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
   OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"
   OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
  set(_USE_PARALLEL_STL ON)
endif()
option(USE_PARALLEL_STL
       "Use parallel STL algorithms (GCC, Intel, IntelLLVM and MSVC only for now) over OpenMP if possible."
       ${_USE_PARALLEL_STL})

# ------------------------------------------------------------------------------

if(MSVC)
  option(ENABLE_MD "Enable compilation using the /MD,/MDd flags" OFF)
  option(ENABLE_MT "Enable compilation using the /MT,/MTd flags" OFF)
  option(DISABLE_FORTRAN_COMPILER "Forcefully disable the Fortran compiler for some 3rd party libraries" ON)
endif()

# ------------------------------------------------------------------------------

option(ENABLE_PROFILING "Enable compilation with profiling flags." OFF)
option(ENABLE_STACK_PROTECTION "Enable the use of -fstack-protector during compilation" ON)

# ==============================================================================
# Linking options

option(ENABLE_RUNPATH "Prefer RUNPATH over RPATH when linking" ON)

option(LINKER_DTAGS "Use --enable-new-dtags or --disable-new-dtags during linking" ON)
option(LINKER_NOEXECSTACK "Use -z,noexecstack during linking" ON)
option(LINKER_RELRO "Use -z,relro during linking for certain targets" ON)
option(LINKER_RPATH "Enable the use of RPATH/RUNPATH related flags during linking" ON)
option(LINKER_STRIP_ALL "Use --strip-all during linking" ON)

# ==============================================================================
# Package related options

# ==============================================================================
# Other CMake related options

option(BUILD_SHARED_LIBS "Build shared libs" OFF)
option(BUILD_TESTING "Build the test suite?" OFF)
option(CLEAN_3RDPARTY_INSTALL_DIR "Clean third-party installation directory" OFF)
option(ENABLE_CMAKE_DEBUG "Enable verbose output to debug CMake issues" OFF)
option(USE_VERBOSE_MAKEFILE "Use verbose Makefiles" ON)

# ==============================================================================
# ==============================================================================
# Python related options

if(PYTHON_VIRTUALENV_COMPAT)
  set(CMAKE_FIND_FRAMEWORK LAST)
endif()

# ------------------------------------------------------------------------------

if(DEFINED ENABLE_OPENMP) # For backwards compatibility
  set(USE_OPENMP ${ENABLE_OPENMP})
endif()

if(IS_PYTHON_BUILD AND IN_PLACE_BUILD)
  message(FATAL_ERROR "Cannot specify both IS_PYTHON_BUILD=ON and IN_PLACE_BUILD=ON!")
endif()

# ==============================================================================
# CUDA related options

include(CheckLanguage)

set(_mq_added_nvcxx_module_path FALSE)
if(ENABLE_CUDA)
  set(_mq_added_nvcxx_module_path TRUE)
  list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/NVCXX)

  # NB: NVHPC < 20.11 will fail this test since they do not support -x c++
  check_language(NVCXX)

  if(CMAKE_NVCXX_COMPILER)
    enable_language(NVCXX)

    if(NOT CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
      # Default architectures list supported by NVHPC when using -stdpar -cuda -gpu=ccXX (taken from NVHPC 22.3)
      set(CMAKE_CUDA_ARCHITECTURES
          60
          61
          62
          70
          72
          75
          80)
      if(CMAKE_NVCXX_COMPILER_VERSION VERSION_GREATER_EQUAL 21.5)
        list(APPEND CMAKE_CUDA_ARCHITECTURES 86)
      endif()

      # NB: CUDAARCHS requires CMake 3.20+
      message(STATUS "Neither of CMAKE_CUDA_ARCHITECTURES (CMake variable) or CUDAARCHS (env. variable; CMake 3.20+) "
                     "have been defined. Defaulting to ${CMAKE_CUDA_ARCHITECTURES}")
    elseif(NOT "$ENV{CUDAARCHS}" STREQUAL "")
      message(STATUS "CUDAARCHS environment variable present: $ENV{CUDAARCHS}")
    endif()
    list(SORT CMAKE_CUDA_ARCHITECTURES ORDER DESCENDING)

    setup_language(NVCXX)
    enable_language(CUDA)
    setup_language(CUDA)

    if(CMAKE_NVCXX_COMPILER_VERSION VERSION_LESS 21.5)
      # * NVCXX < 20.11 : missing '-x c++' argument for CMake flag detection
      # * NVCXX < 21.3  : can only specify one CUDA_ARCHITECTURE
      # * NVCXX < 21.5  : extraction of GPU kernels from shared library is broken
      message(
        FATAL_ERROR "MindQuantum is not compatible with the current version of NVHPC (${CMAKE_NVCXX_COMPILER_VERSION})"
                    "Required is at least 21.5.")
    endif()
  else()
    disable_cuda()
  endif()
endif()

# ==============================================================================
# Compilation options

# ==============================================================================
# Other CMake related options

if(USE_VERBOSE_MAKEFILE)
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

# ==============================================================================
