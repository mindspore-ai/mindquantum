# ==============================================================================
#
# Copyright 2021 <Huawei Technologies Co., Ltd>
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

cmake_minimum_required(VERSION 3.20) # mainly because of NVHPC support

# ==============================================================================
# Create the MindQuantum project
project(MindQuantum LANGUAGES C CXX)

message(STATUS "Running using CMake ${CMAKE_VERSION}")

cmake_policy(VERSION 3.0...3.23)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/commands)

# ==============================================================================
# Macro definitions

include(macros)

# NB: we enable C since some of the third-party libraries require C to be compiled properly
setup_language(C)
setup_language(CXX)

include(cmake_configuration)

# ------------------------------------------------------------------------------

include(GNUInstallDirs)
set(CMAKE_INSTALL_LIBDIR "lib")
set(MQ_INSTALL_BINDIR "${CMAKE_INSTALL_BINDIR}")
set(MQ_INSTALL_SBINDIR "${CMAKE_INSTALL_SBINDIR}")
set(MQ_INSTALL_SYSCONFDIR "${CMAKE_INSTALL_SYSCONFDIR}")
set(MQ_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/mindquantum")
set(MQ_INSTALL_DATADIR "${CMAKE_INSTALL_DATADIR}/mindquantum")
set(MQ_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}/mindquantum")
set(MQ_INSTALL_DOCDIR "${CMAKE_INSTALL_DATADIR}/doc/mindquantum")
set(MQ_INSTALL_CMAKEDIR "${MQ_INSTALL_DATADIR}/cmake")
set(MQ_INSTALL_3RDPARTYDIR "${MQ_INSTALL_LIBDIR}/third_party")

# add_compile_options(-Wall -Wextra -Wfloat-equal)

foreach(
  _type
  BINDIR
  SBINDIR
  SYSCONFDIR
  DATADIR
  LIBDIR
  DOCDIR
  CMAKEDIR
  3RDPARTYDIR)
  GNUInstallDirs_get_absolute_install_dir(MQ_INSTALL_FULL_${_type} MQ_INSTALL_${_type} ${_type})
endforeach()

# ==============================================================================
# OS-detection

include(os_detection)

# ==============================================================================
# Options

include(options)

# ==============================================================================
# Package dependencies

include(packages)

# ==============================================================================
# Setup compiler flags

include(compiler_flags)
include(linker_flags)

is_language_enabled(C, _c_enabled)
add_library(mindquantum_setup INTERFACE)
target_link_libraries(mindquantum_setup INTERFACE CXX_mindquantum $<$<BOOL:${_c_enabled}>:C_mindquantum>)
append_to_property(mq_install_targets GLOBAL mindquantum_setup)

if(ENABLE_CMAKE_DEBUG)
  message(STATUS "Content of language specific targets:")
  list(APPEND CMAKE_MESSAGE_INDENT "  ")

  set(_targets CXX_mindquantum)
  if(_c_enabled)
    list(APPEND _targets C_mindquantum)
  endif()
  if(ENABLE_CUDA)
    list(APPEND _targets CUDA_mindquantum)
  endif()

  include(CMakePrintHelpers)
  cmake_print_properties(
    TARGETS ${_targets} PROPERTIES INTERFACE_COMPILE_OPTIONS INTERFACE_COMPILE_DEFINITIONS
                                   INTERFACE_INCLUDE_DIRECTORIES INTERFACE_LINK_OPTIONS INTERFACE_LINK_LIBRARIES)
  list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

# ==============================================================================

# First add third-party libraries
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party)

# Include workarounds for mindspore CI if necessary
if(MINDSPORE_CI)
  include(mindspore_ci)
endif()

# Then re-define some CMake macros/functions
include(macros_more)

# Then add our libraries
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/ccsrc)

# ==============================================================================
# Convenience target to automatically build all pybind11 C++ modules

get_property(_targets GLOBAL PROPERTY _python_targets)
add_custom_target(
  python
  DEPENDS ${_targets}
  COMMENT "Build python C++ modules")

# ==============================================================================

if(binscope_exec)
  gen_binscope_target(${_targets})
endif()

# ==============================================================================
# Unit testing

if(BUILD_TESTING AND NOT IS_PYTHON_BUILD)
  enable_testing()
  add_subdirectory(tests)
else()
  message(STATUS "Disabling building of tests")
  message(STATUS "  (BUILD_TESTING=${BUILD_TESTING}, IS_PYTHON_BUILD=${IS_PYTHON_BUILD})")
  # Essentially only for CI builds with testing disabled
  add_custom_target(build_all_test COMMENT "Dummy target for CI builds")
  add_custom_target(test COMMENT "Dummy target for CI builds")
endif()

# ------------------------------------------------------------------------------
# Convenience target to automatically build all C++ test executables

get_property(_targets GLOBAL PROPERTY _test_exec_targets)
add_custom_target(
  test-exec
  DEPENDS ${_targets}
  COMMENT "Build all test C++ modules")

# ==============================================================================

if(NOT IS_PYTHON_BUILD AND ENABLE_DOCUMENTATION)
  find_package(Doxygen QUIET)
  if(Doxygen_FOUND)
    add_subdirectory(docs)
  endif()
endif()

# ==============================================================================

include(install)

# ==============================================================================

list(POP_FRONT CMAKE_MODULE_PATH)
list(POP_FRONT CMAKE_MODULE_PATH)
list(POP_FRONT CMAKE_MODULE_PATH)

if(_mq_added_nvcxx_module_path)
  list(POP_FRONT CMAKE_MODULE_PATH)
endif()
