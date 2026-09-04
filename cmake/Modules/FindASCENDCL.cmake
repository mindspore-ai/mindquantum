# ==============================================================================
#
# Copyright 2026 <Huawei Technologies Co., Ltd>
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

# Find the Ascend Compute Language host runtime used by mqvector_npu.
#
# Result variables:
#   ASCENDCL_FOUND
#   ASCENDCL_INCLUDE_DIR
#   ASCENDCL_LIBRARY
#   ASCENDCL_CCEC
#   ASCENDCL_ASCENDC_INCLUDE_DIR
#   ASCENDCL_ASCENDC_ROOT
#   ASCENDCL_DEVKIT_ROOT
#   ASCENDCL_PLATFORM
#
# Imported target:
#   ASCENDCL::ascendcl
#
# Hints:
#   ASCENDCL_ROOT CMake cache variable
#   ASCEND_HOME_PATH, ASCEND_TOOLKIT_HOME, ASCEND_TOOLKIT_HOME_PATH environment variables

include(FindPackageHandleStandardArgs)

set(ASCENDCL_ROOT
    ""
    CACHE PATH "Root directory of the Ascend CANN installation")

set(_ascendcl_roots "${ASCENDCL_ROOT}")
foreach(_ascendcl_env_var ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_TOOLKIT_HOME_PATH)
  if(DEFINED ENV{${_ascendcl_env_var}} AND NOT "$ENV{${_ascendcl_env_var}}" STREQUAL "")
    list(APPEND _ascendcl_roots "$ENV{${_ascendcl_env_var}}")
  endif()
endforeach()

list(APPEND _ascendcl_roots /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/latest)
file(GLOB _ascendcl_cann_roots LIST_DIRECTORIES TRUE /usr/local/Ascend/cann-*)
list(APPEND _ascendcl_roots ${_ascendcl_cann_roots})
list(REMOVE_DUPLICATES _ascendcl_roots)

find_path(
  ASCENDCL_INCLUDE_DIR
  NAMES acl/acl.h
  HINTS ${_ascendcl_roots}
  PATH_SUFFIXES include x86_64-linux/include aarch64-linux/include)

find_library(
  ASCENDCL_LIBRARY
  NAMES ascendcl
  HINTS ${_ascendcl_roots}
  PATH_SUFFIXES lib64 x86_64-linux/lib64 aarch64-linux/lib64)

find_program(
  ASCENDCL_CCEC
  NAMES ccec bisheng
  HINTS ${_ascendcl_roots}
  PATH_SUFFIXES bin x86_64-linux/ccec_compiler/bin aarch64-linux/ccec_compiler/bin compiler/ccec_compiler/bin)

find_path(
  ASCENDCL_ASCENDC_INCLUDE_DIR
  NAMES kernel_operator.h
  HINTS ${_ascendcl_roots}
  PATH_SUFFIXES x86_64-linux/asc/include aarch64-linux/asc/include asc/include)

find_package_handle_standard_args(
  ASCENDCL REQUIRED_VARS ASCENDCL_INCLUDE_DIR ASCENDCL_LIBRARY ASCENDCL_CCEC ASCENDCL_ASCENDC_INCLUDE_DIR)

if(ASCENDCL_FOUND AND NOT TARGET ASCENDCL::ascendcl)
  get_filename_component(ASCENDCL_ASCENDC_ROOT "${ASCENDCL_ASCENDC_INCLUDE_DIR}" DIRECTORY)
  get_filename_component(ASCENDCL_DEVKIT_ROOT "${ASCENDCL_ASCENDC_ROOT}" DIRECTORY)
  get_filename_component(ASCENDCL_PLATFORM "${ASCENDCL_DEVKIT_ROOT}" NAME)
  add_library(ASCENDCL::ascendcl UNKNOWN IMPORTED)
  set_target_properties(
    ASCENDCL::ascendcl
    PROPERTIES IMPORTED_LOCATION "${ASCENDCL_LIBRARY}" INTERFACE_INCLUDE_DIRECTORIES "${ASCENDCL_INCLUDE_DIR}")
endif()

mark_as_advanced(
  ASCENDCL_INCLUDE_DIR
  ASCENDCL_LIBRARY
  ASCENDCL_CCEC
  ASCENDCL_ASCENDC_INCLUDE_DIR
  ASCENDCL_ASCENDC_ROOT
  ASCENDCL_DEVKIT_ROOT
  ASCENDCL_PLATFORM)
