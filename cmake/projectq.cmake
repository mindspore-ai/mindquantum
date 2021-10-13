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

set(PKG_NAME projectq)
set(PKG_ROOT ${DEP_DIR}/${PKG_NAME}-src)

set(URL "https://gitee.com/mirrors/ProjectQ/repository/archive/v0.5.1.tar.gz")

file(GLOB _patch_files "${PATCH_DIR}/projectq/*.patch*")

# ==============================================================================
# Fetch the source code

FetchContent_Declare(${PKG_NAME} URL ${URL})
FetchContent_Populate(${PKG_NAME})

# ==============================================================================
# Patch the source code

# cmake-lint: disable=C0103
set(${PKG_NAME}_PATCHED
    FALSE
    CACHE STRING INTERNAL)

if(NOT ${PKG_NAME}_PATCHED)
  message("Patching for ${PKG_NAME}")
  foreach(_file ${_patch_files})
    execute_process(
      COMMAND ${Patch_EXECUTABLE} -p1
      INPUT_FILE ${_file}
      WORKING_DIRECTORY ${PKG_ROOT}
      RESULT_VARIABLE _result)

    if(NOT _result EQUAL 0)
      message(FATAL_ERROR "Failed patch: ${_file}")
    endif()
  endforeach()

  set(${PKG_NAME}_PATCHED
      TRUE
      CACHE STRING INTERNAL FORCE)
endif()

# ==============================================================================

add_library(projectq INTERFACE)
target_compile_features(projectq INTERFACE cxx_std_14)
target_include_directories(projectq INTERFACE ${PKG_ROOT})
