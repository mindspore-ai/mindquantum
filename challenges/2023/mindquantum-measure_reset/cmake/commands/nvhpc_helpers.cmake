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

include_guard()

# Convenience function to extract all compute capabilities from command line like strings
#
# nvhpc_extract_cc(<out-var> <element> [<element> ...])
#
# Extract and store the numeric compute capabilities from a list of input arguments. This function supports both
# space-separated values or lists. Compute capabilities are stored using their numeric values (ie. -gpu=cc60 -> 60)
function(nvhpc_extract_cc out_var)
  cmake_parse_arguments(PARSE_ARGV 1 ECC "" "OTHER_ARGS" "")

  set(_args "${ECC_UNPARSED_ARGUMENTS}")
  separate_arguments(_args) # cmake-lint: disable=E1120

  set(_cc_args)
  set(_other_args)
  foreach(_el ${_args})
    if(_el MATCHES "-gpu=cc([0-9]+)")
      list(APPEND _cc_args ${CMAKE_MATCH_1})
    else()
      list(APPEND _other_args ${_el})
    endif()
  endforeach()

  set(${out_var}
      ${_cc_args}
      PARENT_SCOPE)

  if(ECC_OTHER_ARGS)
    set(${ECC_OTHER_ARGS}
        ${_other_args}
        PARENT_SCOPE)
  endif()
endfunction()

# ==============================================================================

# Sanitize compute capabilities arguments
#
# nvhpc_sanitize_cc(<out-var> [LIST|STRING] [SINGLE] <element> [<element> ...])
#
# Sanitize compute capability arguments (-gpu=ccXX) for a NVHPC command line. For NVHPC version < 21.3, only a single CC
# args may be specified on any command line.
#
# If the LIST or STRING keyword is specified, the result will be returned as a list or a whitespace separated string
# respectively. Specifying the SINGLE keyword is present, force the use of a single CC (the lowest).
function(nvhpc_sanitize_cc variable)
  cmake_parse_arguments(PARSE_ARGV 1 SCCA "LIST;STRING;SINGLE" "" "")

  if(NOT SCCA_LIST AND NOT SCCA_STRING)
    set(SCCA_LIST TRUE) # cmake-lint: disable=C0103
  elseif(SCCA_LIST AND SCCA_STRING)
    message(FATAL_ERROR "Cannot specify *both* LIST and STRING!")
  endif()

  set(_variable "${SCCA_UNPARSED_ARGUMENTS}")
  separate_arguments(_variable) # cmake-lint: disable=E1120
  nvhpc_extract_cc(_cc_args OTHER_ARGS _args ${_variable})

  # nvc++-Fatal-The -stdpar option does not currently support compilation for multiple compute capabilities
  if(CMAKE_NVCXX_COMPILER_VERSION VERSION_LESS 21.3 OR SCCA_SINGLE)
    list(SORT _cc_args COMPARE NATURAL) # NB: COMPARE NATURAL requires CMake >= 3.18
    list(GET _cc_args 0 _cc_args)

  endif()

  if(_cc_args)
    foreach(_cc ${_cc_args})
      list(APPEND _args "-gpu=cc${_cc}")
    endforeach()
  endif()

  if(SCCA_STRING)
    string(REPLACE ";" " " _args "${_args}")
  endif()

  file(APPEND ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log "nvhpc_sanitize_cc(${variable} ${ARGN})\n"
       "    result: ${_args}\n\n")

  set(${variable}
      ${_args}
      PARENT_SCOPE)
endfunction()
