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

include(debug_print)
include(CMakeDependentOption)

# ==============================================================================
# MindQuantum feature selection

option(ENABLE_GITEE "Use Gitee instead of GitHub for checking out third-party dependencies" OFF)
option(ENABLE_DOCUMENTATION "Enable building of the documentation using Doxygen" OFF)
option(ENABLE_LOGGING "Enable the use of logging in C++" OFF)
cmake_dependent_option(ENABLE_LOGGING_TRACE_LEVEL "If logging is enabled, log everything down to the TRACE level" OFF
                       "ENABLE_LOGGING" OFF)
cmake_dependent_option(ENABLE_LOGGING_DEBUG_LEVEL "If logging is enabled, log everything down to the DEBUG level" OFF
                       "ENABLE_LOGGING" OFF)
option(ENABLE_LONG_DOUBLE "Enable the use of the `long double` type in certain areas" OFF)
option(ENABLE_COLOR_COMPILER "Use color in compiler diagnostic messages" ON)

# ==============================================================================
# Python related options

if(APPLE)
  option(PYTHON_VIRTUALENV_COMPAT "(Mac OS X) Make CMake search for Python Framework *after* any available\
  unix-style package. Can be useful in case of virtual environments." ON)
else()
  option(PYTHON_VIRTUALENV_COMPAT "(Mac OS X) Make CMake search for Python Framework *after* any available\
  unix-style package. Can be useful in case of virtual environments." OFF)
endif()

option(PYTHON_VIRTUALENV_OVER_ROOT_DIR
       "Ignore Python_ROOT_DIR if present at the same time as the VIRTUAL_ENV env. variable." ON)

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
  option(ENABLE_ITERATOR_DEBUG
         "Enable the definition of _ITERATOR_DEBUG compiler defines (use MQ_ITERATOR_DEBUG to specify the value)" OFF)
  option(DISABLE_FORTRAN_COMPILER "Forcefully disable the Fortran compiler for some 3rd party libraries" ON)
endif()

# ------------------------------------------------------------------------------

option(ENABLE_PROFILING "Enable compilation with profiling flags." OFF)
option(ENABLE_STACK_PROTECTION "Enable the use of -fstack-protector during compilation" ON)

option(ENABLE_GCC_DEBUG_MODE "Enable the debug mode for GCC and libstdc++" OFF)

option(ENABLE_ANALYZER "Enable compiler static analysis tools (e.g. -fanalyzer for GCC)" OFF)

option(ENABLE_SANITIZERS "Enable additional CMake build types for sanitizers" ON)
cmake_dependent_option(ENABLE_SANITIZER_ADDRESS "Enable the address sanitizer" ON "ENABLE_SANITIZERS" OFF)
cmake_dependent_option(ENABLE_SANITIZER_UNDEFINED "Enable the undefined behavior sanitizer" ON "ENABLE_SANITIZERS" OFF)
cmake_dependent_option(SANITIZER_USE_O1 "Use -O1 when the build type is for a sanitizer" OFF "ENABLE_SANITIZERS" OFF)
cmake_dependent_option(SANITIZER_USE_Og "Use -O1 when the build type is for a sanitizer" OFF "ENABLE_SANITIZERS" OFF)

# ==============================================================================
# Linking options

option(ENABLE_RUNPATH "Prefer RUNPATH over RPATH when linking" ON)

option(LINKER_DTAGS "Use --enable-new-dtags or --disable-new-dtags during linking" ON)
option(LINKER_NOEXECSTACK "Use -z,noexecstack during linking" ON)
option(LINKER_NOW "Use -z,now during linking for certain targets" ON)
option(LINKER_RELRO "Use -z,relro during linking for certain targets" ON)
option(LINKER_RPATH "Enable the use of RPATH/RUNPATH related flags during linking" ON)
option(LINKER_STRIP_ALL "Use --strip-all during linking" ON)

# ==============================================================================
# Package related options

set(_enable_abseil_cpp OFF)
if("${OS_NAME}" STREQUAL "Cygwin" OR "${OS_NAME}" STREQUAL "MSYS-MSYS")
  if(ENABLE_ABSEIL_CPP)
    message(WARNING "Unable to use abseil-cpp with Cygwin or MSYS2-MSYS")
  endif()
  set(ENABLE_ABSEIL_CPP
      OFF
      CACHE BOOL "Enable the use of the abseil-cpp library" FORCE)
  set(_enable_abseil_cpp OFF)
endif()
option(ENABLE_ABSEIL_CPP "Enable the use of the abseil-cpp library" _enable_abseil_cpp)

# ==============================================================================
# Other CMake related options

option(BUILD_SHARED_LIBS "Build shared libs" OFF)
option(BUILD_TESTING "Build the test suite?" OFF)
option(CLEAN_3RDPARTY_INSTALL_DIR "Clean third-party installation directory" OFF)
option(ENABLE_CMAKE_DEBUG "Enable verbose output to debug CMake issues" OFF)
option(USE_VERBOSE_MAKEFILE "Use verbose Makefiles" ON)

# ==============================================================================
# ==============================================================================
# MindQuantum feature selection

if(ENABLE_LOGGING_TRACE_LEVEL AND ENABLE_LOGGING_DEBUG_LEVEL)
  message(
    FATAL_ERROR "Cannot have *both* ENABLE_LOGGING_TRACE_LEVEL and ENABLE_LOGGING_DEBUG_LEVEL set at the same time")
endif()

# ==============================================================================
# Python related options

if(PYTHON_VIRTUALENV_COMPAT)
  set(CMAKE_FIND_FRAMEWORK LAST)
endif()

debug_print(STATUS "ENV{Python_ROOT_DIR} = $ENV{Python_ROOT_DIR}")
debug_print(STATUS "ENV{VIRTUAL_ENV} = $ENV{VIRTUAL_ENV}")

if(PYTHON_VIRTUALENV_OVER_ROOT_DIR)
  if(DEFINED ENV{Python_ROOT_DIR} AND DEFINED ENV{VIRTUAL_ENV})
    message(STATUS "Both Python_ROOT_DIR and VIRTUAL_ENV environment variables are defined")
    message(STATUS "Removing Python_ROOT_DIR to favour VIRTUAL_ENV")
    unset(ENV{Python_ROOT_DIR})
  endif()
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

  set(_default_cudaarchs FALSE)
  if(NOT CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(_default_cudaarchs TRUE)
    # Default architectures list supported by NVHPC when using -stdpar -cuda -gpu=ccXX (taken from NVHPC 22.3)
    set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75)

    # NB: CUDAARCHS requires CMake 3.20+
    message(STATUS "Neither of CMAKE_CUDA_ARCHITECTURES (CMake variable) or CUDAARCHS (env. variable; CMake 3.20+) "
                   "have been defined. Defaulting to ${CMAKE_CUDA_ARCHITECTURES}")
  elseif(NOT "$ENV{CUDAARCHS}" STREQUAL "")
    message(STATUS "CUDAARCHS environment variable present: $ENV{CUDAARCHS}")
  endif()
  list(SORT CMAKE_CUDA_ARCHITECTURES ORDER DESCENDING)

  # First try to activate CUDA
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    setup_language(CUDA)
  else()
    disable_cuda("missing/unable to locate CUDA compiler")
  endif()

  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 80)
    list(SORT CMAKE_CUDA_ARCHITECTURES ORDER DESCENDING)
  elseif(ENABLE_CUDA)
    disable_cuda("CUDA compiler version is too old (${CMAKE_CUDA_COMPILER_VERSION} < 11.0)")
  endif()

  # Now look if we find NVHPC
  if(ENABLE_CUDA)
    # NB: NVHPC < 20.11 will fail this test since they do not support -x c++
    check_language(NVCXX)

    if(CMAKE_NVCXX_COMPILER)
      enable_language(NVCXX)

      if(_default_cudaarchs AND CMAKE_NVCXX_COMPILER_VERSION VERSION_GREATER_EQUAL 21.5)
        list(APPEND CMAKE_CUDA_ARCHITECTURES 86)
        list(SORT CMAKE_CUDA_ARCHITECTURES ORDER DESCENDING)
      endif()

      setup_language(NVCXX)

      if(CMAKE_NVCXX_COMPILER_VERSION VERSION_LESS 21.5)
        # * NVCXX < 20.11 : missing '-x c++' argument for CMake flag detection
        # * NVCXX < 21.3  : can only specify one CUDA_ARCHITECTURE
        # * NVCXX < 21.5  : extraction of GPU kernels from shared library is broken
        message(
          FATAL_ERROR
            "MindQuantum is not compatible with the current version of NVHPC (${CMAKE_NVCXX_COMPILER_VERSION})"
            "Required is at least 21.5.")
      endif()
    else()
      message(STATUS "NVHPC compiler not found. NVHPC features will be disabled")
    endif()
  endif()
endif()

# ==============================================================================
# Compilation options

if(ENABLE_GCC_DEBUG_MODE)
  message(
    WARNING "Support for ENABLE_GCC_DEBUG_MODE is still experimental and may lead to compilation or link issues!"
            "You might also want to force all third-party libraries to be locally built and force a complete rebuild.")
endif()

if(ENABLE_SANITIZERS)
  if(SANITIZER_USE_O1 AND SANITIZER_USE_Og)
    message(FATAL_ERROR "Cannot define SANITIZER_USE_O1=ON and SANITIZER_USE_Og=ON at the same time!")
  endif()
endif()

if(MSVC)
  if(NOT DEFINED MQ_ITERATOR_DEBUG)
    set(MQ_ITERATOR_DEBUG
        2
        CACHE STRING "Value to define the _ITERATOR_DEBUG compiler define to \
(requires ENABLE_ITERATOR_DEBUG to have an effect)")
  endif()
endif()

# ==============================================================================
# Other CMake related options

if(USE_VERBOSE_MAKEFILE)
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

# ==============================================================================
