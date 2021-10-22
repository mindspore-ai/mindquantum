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

cmake_minimum_required(VERSION 3.15)

set(_policy_list
    CMP0012
    CMP0015
    CMP0022
    CMP0023
    CMP0028
    CMP0042
    CMP0048
    CMP0051
    CMP0054
    CMP0056
    CMP0057
    CMP0066
    CMP0067
    CMP0068
    CMP0074
    CMP0076
    CMP0077
    CMP0079
    CMP0094
    CMP0104)
foreach(_policy ${_policy_list})
  if(POLICY ${_policy})
    cmake_policy(SET ${_policy} NEW)
  endif()
  # cmake-format: off
  # CMP0012: if() recognizes numbers and booleans
  # CMP0015: paths relative to source dir for link_directories
  # CMP0028: :: in target names
  # CMP0042: MACOS_RPATH
  # CMP0048: allow VERSION in project()
  # CMP0051: list TARGET_OBJECTS in SOURCES property
  # CMP0054: no more de-referencing of "expr" in if() statements
  # CMP0056: try_compile(): link flags
  # CMP0057: if IN_LIST
  # CMP0066: try_compile(): use per-config flags, like CMAKE_CXX_FLAGS_RELEASE
  # CMP0067: try_compile(): honor language standard variables (like C++11)
  # CMP0068: RPATH on Mac OS does not affect install_name
  # CMP0074: XXX_ROOT variables for find_package(XXX)
  # CMP0076: target_sources relative paths
  # CMP0077: option() honors normal variables
  # CMP0079: target_link_libraries allows use with targets in other directories
  #          (CMake 3.13 minimum)
  # CMP0094: FindPython* use LOCATION strategy (stop at first valid version)
  # CMP0104: Empty CUDA_ARCHITECTURES target property is an error
  # cmake-format: on
endforeach()

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)

# ==============================================================================
# Macro definitions

include(${CMAKE_CURRENT_LIST_DIR}/cmake/macros.cmake)

# ==============================================================================
# Create the MindQuantum project
project(MindQuantum LANGUAGES C CXX)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/mindquantum)
endif()

# ==============================================================================
# OS-detection

include(${CMAKE_CURRENT_LIST_DIR}/cmake/os_detection.cmake)

# ==============================================================================
# Options

include(${CMAKE_CURRENT_LIST_DIR}/cmake/options.cmake)

# ==============================================================================
# Package dependencies

include(${CMAKE_CURRENT_LIST_DIR}/cmake/packages.cmake)

# ==============================================================================
# Setup compiler flags

include(${CMAKE_CURRENT_LIST_DIR}/cmake/compiler_flags.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/linker_flags.cmake)

# ==============================================================================
# Add submodule dependencies (now rather than later so that the relevant macros/variables are defined)

add_subdirectory(mindquantum/src)

# ==============================================================================
# Some more macro definitions

include(${CMAKE_CURRENT_LIST_DIR}/cmake/macros_more.cmake)

# ==============================================================================
