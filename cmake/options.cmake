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

include(CMakeDependentOption)

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

option(ENABLE_OPENMP "Use OpenMP for multi-threading" ON)

option(ENABLE_PROJECTQ "Enable ProjectQ support" ON)
option(ENABLE_QUEST "Enable QuEST support" ON)

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

option(BUILD_TESTING "Build the test suite?" OFF)

# NB: most if not all of our libraries have the type explicitly specified.
option(BUILD_SHARED_LIBS "Build shared libs" OFF)

option(USE_VERBOSE_MAKEFILE "Use verbose Makefiles" ON)

# ==============================================================================
# ==============================================================================
# Python related options

if(PYTHON_VIRTUALENV_COMPAT)
  set(CMAKE_FIND_FRAMEWORK LAST)
endif()

# ------------------------------------------------------------------------------

if(IS_PYTHON_BUILD AND IN_PLACE_BUILD)
  message(FATAL_ERROR "Cannot specify both IS_PYTHON_BUILD=ON and IN_PLACE_BUILD=ON!")
endif()

# ==============================================================================
# CUDA related options

if(CUDA_ALLOW_UNSUPPORTED_COMPILER)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
endif()

if(ENABLE_CUDA)
  enable_language(CUDA)

  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    find_package(CUDAToolkit REQUIRED)
  else()
    find_package(CUDA REQUIRED)

    if(CUDA_LIBRARIES)
      if(NOT TARGET CUDA::cudart)
        add_library(CUDA::cudart IMPORTED INTERFACE)
        target_include_directories(CUDA::cudart SYSTEM INTERFACE "${CUDA_INCLUDE_DIRS}")
        target_link_libraries(CUDA::cudart INTERFACE "${CUDA_LIBRARIES}")
      endif()
    endif()

    if(CUDA_cudart_static_LIBRARY)
      if(NOT TARGET CUDA::cudart_static)
        add_library(CUDA::cudart_static IMPORTED INTERFACE)
        target_include_directories(CUDA::cudart_static SYSTEM INTERFACE "${CUDA_INCLUDE_DIRS}")
        target_link_libraries(CUDA::cudart_static INTERFACE "${CUDA_cudart_static_LIBRARY}" Threads::Threads)
      endif()
    endif()

    find_library(
      CUDA_driver_LIBRARY
      NAMES cuda_driver cuda
      HINTS ${CUDA_TOOLKIT_ROOT_DIR} ENV CUDA_PATH
      PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
    if(NOT CUDA_driver_LIBRARY)
      # Don't try any stub directories until we have exhausted all other search locations.
      find_library(
        CUDA_driver_LIBRARY
        NAMES cuda_driver cuda
        HINTS ${CUDA_TOOLKIT_ROOT_DIR} ENV CUDA_PATH
        PATH_SUFFIXES lib64/stubs lib/x64/stubs lib/stubs stubs)
    endif()
    mark_as_advanced(CUDA_driver_LIBRARY)
    if(CUDA_driver_LIBRARY)
      add_library(CUDA::cuda_driver IMPORTED INTERFACE)
      target_include_directories(CUDA::cuda_driver SYSTEM INTERFACE "${CUDA_INCLUDE_DIRS}")
      target_link_libraries(CUDA::cuda_driver INTERFACE "${CUDA_driver_LIBRARY}")
    endif()
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
