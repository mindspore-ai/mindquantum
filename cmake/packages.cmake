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

# OpenMP

set(PARALLEL_LIBS)
if(USE_OPENMP)
  if(APPLE)
    find_program(BREW_CMD brew PATHS /usr/local/bin)
    if(BREW_CMD)
      # Homebrew installs libomp in ${LIBOMP_PREFIX}/lib and the headers in ${LIBOMP_PREFIX}/include
      execute_process(COMMAND ${BREW_CMD} --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX)
      string(STRIP ${LIBOMP_PREFIX} LIBOMP_PREFIX)
      debug_print(STATUS "LIBOMP_PREFIX = ${LIBOMP_PREFIX}")

      find_library(
        LIBOMP_LIB omp gomp libomp
        HINTS ${LIBOMP_PREFIX}
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH)
      debug_print(STATUS "LIBOMP_LIB = ${LIBOMP_LIB}")
      if(LIBOMP_LIB)
        get_filename_component(LIBOMP_DIR ${LIBOMP_LIB} DIRECTORY)
        list(APPEND CMAKE_LIBRARY_PATH ${LIBOMP_DIR})
      endif()

      find_path(
        LIBOMP_INC omp.h
        HINTS ${LIBOMP_PREFIX}
        PATH_SUFFIXES include
        NO_DEFAULT_PATH)
      debug_print(STATUS "LIBOMP_INC = ${LIBOMP_INC}")
      if(LIBOMP_INC)
        list(APPEND CMAKE_INCLUDE_PATH ${LIBOMP_INC})
      else()
        message(WARNING "Unable to locate omp.h, the code might not compile properly.\n"
                        "You might want to try installing the `libomp` Homebrew formula: brew install libomp")
      endif()
    else()
      # MacPorts install libomp in /opt/local/lib/libomp and the headers in /opt/local/include/libomp
      find_library(
        LIBOMP_LIB omp gomp libomp
        PATHS /opt/local/lib
        PATH_SUFFIXES libomp
        NO_DEFAULT_PATH)
      debug_print(STATUS "LIBOMP_LIB = ${LIBOMP_LIB}")
      if(LIBOMP_LIB)
        get_filename_component(LIBOMP_DIR ${LIBOMP_LIB} DIRECTORY)
        list(APPEND CMAKE_LIBRARY_PATH ${LIBOMP_DIR})
      endif()

      find_path(
        LIBOMP_INC omp.h
        PATHS /opt/local/include
        PATH_SUFFIXES libomp
        NO_DEFAULT_PATH)
      debug_print(STATUS "LIBOMP_INC = ${LIBOMP_INC}")
      if(LIBOMP_INC)
        list(APPEND CMAKE_INCLUDE_PATH ${LIBOMP_INC})
      else()
        message(WARNING "Unable to locate omp.h, the code might not compile properly.\n"
                        "You might want to try installing the `libomp` MacPorts port: sudo port install libomp")
      endif()
    endif()
    debug_print(STATUS "CMAKE_INCLUDE_PATH = ${CMAKE_INCLUDE_PATH}")
    debug_print(STATUS "CMAKE_LIBRARY_PATH = ${CMAKE_LIBRARY_PATH}")
  endif()

  # ----------------------------------------------------------------------------

  if(APPLE)
    list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/Modules/apple)
  endif()
  find_package(OpenMP)
  if(OpenMP_FOUND)
    set(MQ_OPENMP_TARGET OpenMP::OpenMP_CXX)
    list(APPEND PARALLEL_LIBS ${OpenMP_target})
  else()
    set(MQ_OPENMP_TARGET)
    # cmake-lint: disable=C0103
    set(USE_OPENMP
        FALSE
        CACHE INTERNAL "Disabled OpenMP support")
  endif()

  if(APPLE)
    list(POP_FRONT CMAKE_MODULE_PATH)
  endif()
endif()

if(USE_PARALLEL_STL)
  find_package(TBB COMPONENTS tbb)
  if(TBB_FOUND)
    target_compile_options(TBB::tbb INTERFACE "$<$<COMPILE_LANGUAGE:DPCXX>:-tbb>")
    list(APPEND PARALLEL_LIBS TBB::tbb)
  else()
    set(USE_PARALLEL_STL
        FALSE
        CACHE BOOL "Use the parallel STL libraries (TBB)")
  endif()
endif()

# ==============================================================================

find_package(Threads REQUIRED)
list(APPEND PARALLEL_LIBS Threads::Threads)

if("${CMAKE_PROJECT_NAME}" STREQUAL "MindQuantum")
  find_package(Patch REQUIRED)
endif()

# ==============================================================================
# CUDA

if(ENABLE_CUDA)
  find_package(CUDAToolkit ${MQ_CUDA_VERSION})
  if(CUDAToolkit_FOUND)
    set(MQ_CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
  else()
    message(STATUS "Disabling CUDA since unable to locate CUDAToolkit")
    # cmake-lint: disable=C0103
    set(ENABLE_CUDA
        OFF
        CACHE INTERNAL "Enable building of CUDA/NVHPC libraries")
  endif()
endif()

# ==============================================================================

# Only helps set the Python executable for CMake >= 3.16
if(DEFINED PYTHON_EXECUTABLE)
  set(Python_EXECUTABLE ${PYTHON_EXECUTABLE}) # cmake-lint: disable=C0103
endif()

set(_python_find_args Python 3.6.0 COMPONENTS Interpreter)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  list(APPEND _python_find_args Development.Module)
else()
  list(APPEND _python_find_args Development)
endif()
find_package(${_python_find_args})

# ------------------------------------------------------------------------------

# Check if we are being used directly or via add_subdirectory()
if("${CMAKE_PROJECT_NAME}" STREQUAL "MindQuantum")
  if(NOT MQ_PYTHON_PACKAGE_NAME)
    execute_process(
      COMMAND
        "${Python_EXECUTABLE}" -c [=[
import sys
try:
    from setuptools.config.setupcfg import read_configuration
except ImportError:
    from setuptools.config import read_configuration

sys.stdout.write(read_configuration("setup.cfg")["metadata"]["name"])
]=]
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      RESULT_VARIABLE result
      OUTPUT_VARIABLE MQ_PYTHON_PACKAGE_NAME)

    if(NOT result EQUAL 0)
      message(FATAL_ERROR "Unable to determine MindQuantum's Python package name")
    endif()

    set(MQ_PYTHON_PACKAGE_NAME
        "${MQ_PYTHON_PACKAGE_NAME}"
        CACHE STRING "MindQuantum's Python package name")
    mark_as_advanced(MQ_PYTHON_PACKAGE_NAME)
  endif()

  # ----------------------------------------------------------------------------

  if(NOT MQ_INSTALL_PYTHONDIR)
    execute_process(
      COMMAND
        "${Python_EXECUTABLE}" -c [=[
import sys
from pathlib import Path

try:
    from distutils import sysconfig

    platlib = Path(sysconfig.get_python_lib(plat_specific=True, standard_lib=False))
    platbase = Path(sysconfig.EXEC_PREFIX)
except Exception:
    import sysconfig

    platlib = Path(sysconfig.get_path("platlib"))
    platbase = Path(sysconfig.get_config_var("base"))

sys.stdout.write(str(platlib.relative_to(platbase)))
]=]
      RESULT_VARIABLE result
      OUTPUT_VARIABLE MQ_INSTALL_PYTHONDIR)

    if(NOT result EQUAL 0)
      message(FATAL_ERROR "Unable to determine Python path to site-packages sub-directory")
    endif()

    set(MQ_INSTALL_PYTHONDIR
        "${MQ_INSTALL_PYTHONDIR}"
        CACHE FILEPATH "Python path to site-packages sub-directory")
    mark_as_advanced(MQ_INSTALL_PYTHONDIR)
  endif()

  GNUInstallDirs_get_absolute_install_dir(MQ_INSTALL_FULL_PYTHONDIR MQ_INSTALL_PYTHONDIR PYTHONDIR)
endif()

# ==============================================================================
# For Huawei internal security assessment

if("${CMAKE_PROJECT_NAME}" STREQUAL "MindQuantum")
  if(BINSCOPE)
    get_filename_component(_binscope_path ${BINSCOPE} DIRECTORY)
    get_filename_component(_binscope_name ${BINSCOPE} NAME)
  endif()

  find_program(
    binscope_exec
    NAMES binscope ${_binscope_name}
    HINTS ${_binscope_path})
  include(binscope)
endif()

# ==============================================================================
