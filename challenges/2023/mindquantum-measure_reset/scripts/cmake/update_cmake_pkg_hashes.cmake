# ==============================================================================
#
# Copyright 2022 <Huawei Technologies Co., Ltd>
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

# This script automatically downloads new archives and prints the new MD5 hashes

set(THIRD_PARTY_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../third_party)

# ------------------------------------------------------------------------------

find_package(Git REQUIRED)

if(NOT DEFINED FORCE_REDOWNLOAD)
  set(FORCE_REDOWNLOAD OFF)
endif()
message(STATUS "FORCE_REDOWNLOAD = ${FORCE_REDOWNLOAD}")

# ------------------------------------------------------------------------------

# Parse third-party library directory
function(parse_third_party_lib path)
  cmake_path(GET path FILENAME basename)
  set(cmake_file "${path}/${basename}.cmake")
  if(ENABLE_GITEE)
    set(_type GITEE)
  else()
    set(_type GITHUB)
  endif()

  if(EXISTS "${cmake_file}")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} diff --exit-code "${cmake_file}"
      RESULT_VARIABLE _result
      OUTPUT_VARIABLE _output
      WORKING_DIRECTORY ${THIRD_PARTY_ROOT})

    if(NOT _result EQUAL 0 AND "${_output}" MATCHES ".*VER.*")
      cmake_path(GET lib FILENAME basename)
      message(STATUS "Detected some version change for ${basename}")
      list(APPEND CMAKE_MESSAGE_INDENT "    ")
      include("${cmake_file}")
    else()
      message(STATUS "Nothing to do for ${basename} (${_type})")
    endif()
  endif()
endfunction()

# ==============================================================================

# Monkeypatch the `mindquantum_add_pkg` macro to extract version information
function(mindquantum_add_pkg pkg_name)
  set(options)
  set(oneValueArgs GIT_REPOSITORY GIT_TAG URL VER)
  set(multiValueArgs)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(PKG_URL)
    if(ENABLE_GITEE)
      set(_type GITEE)
      set(output_file "${pkg_name}_${PKG_VER}_gitee")
    else()
      set(_type GITHUB)
      set(output_file "${pkg_name}_${PKG_VER}_github")
    endif()

    if(NOT EXISTS "${output_file}" OR FORCE_REDOWNLOAD)
      message(STATUS "Downloading ${PKG_URL}")
      file(DOWNLOAD ${PKG_URL} "${output_file}" STATUS _downloaded)
      if(NOT _downloaded EQUAL 0)
        message(FATAL_ERROR "Failed to download using ${PKG_URL}")
      endif()
    else()
      message(STATUS "Already downloaded ${output_file}")
    endif()
    file(MD5 "${output_file}" hash)
    if("${_output}" MATCHES ".*${hash}.*")
      message(STATUS "New hash already in place (${_type})")
    else()
      message(STATUS "New MD5 hash (${_type}): ${hash}")
    endif()
  endif()
endfunction()

# ==============================================================================

file(GLOB third_party_files LIST_DIRECTORIES true ${THIRD_PARTY_ROOT}/*)
set(THIRD_PARTY_LIBS)
foreach(element ${third_party_files})
  if(IS_DIRECTORY "${element}")
    cmake_path(GET element FILENAME basename)

    if(NOT basename STREQUAL "cmake")
      list(APPEND THIRD_PARTY_LIBS "${element}")
    endif()
  endif()
endforeach()

# ==============================================================================

set(ENABLE_GITEE OFF)
foreach(lib ${THIRD_PARTY_LIBS})
  parse_third_party_lib("${lib}")
endforeach()

# ------------------------------------------------------------------------------

set(ENABLE_GITEE ON)
foreach(lib ${THIRD_PARTY_LIBS})
  parse_third_party_lib("${lib}")
endforeach()

# ==============================================================================
