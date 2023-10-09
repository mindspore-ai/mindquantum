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

include_guard()

include(FindPackageHandleStandardArgs)

# Find a Python module in the current (potential virtual) environment
#
# find_python_module(<module> [REQUIRED|EXACT|QUIET] [VERSION <version>])
#
# Usage is similar to the builtin find_package(...)
function(find_python_module module)
  # cmake-lint: disable=C0103
  cmake_parse_arguments(PARSE_ARGV 1 PYMOD "REQUIRED;EXACT;QUIET" "VERSION" "")

  string(REPLACE "-" "_" module_name ${module})
  string(TOUPPER ${module_name} MODULE)
  if(NOT PYMOD_${MODULE})
    if(PYMOD_REQUIRED)
      set(PYMOD_${module}_FIND_REQUIRED TRUE)
      set(PYMOD_${MODULE}_FIND_REQUIRED TRUE)
    endif()
    if(PYMOD_QUIET)
      set(PYMOD_${module}_FIND_QUIETLY TRUE)
      set(PYMOD_${MODULE}_FIND_QUIETLY TRUE)
    endif()
    if(PYMOD_EXACT)
      set(PYMOD_${module}_FIND_VERSION_EXACT TRUE)
      set(PYMOD_${MODULE}_FIND_VERSION_EXACT TRUE)
    endif()
    if(PYMOD_VERSION)
      set(PYMOD_${module}_FIND_VERSION ${PYMOD_VERSION})
      set(PYMOD_${MODULE}_FIND_VERSION ${PYMOD_VERSION})
    endif()

    execute_process(
      COMMAND "${Python_EXECUTABLE}" "-c" "import os, ${module_name}; print(os.path.dirname(${module_name}.__file__))"
      RESULT_VARIABLE _${MODULE}_status
      OUTPUT_VARIABLE _${MODULE}_location
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(NOT _${MODULE}_status)
      set(PYMOD_${MODULE}_PATH
          ${_${MODULE}_location}
          CACHE STRING "Location of Python module ${module}")

      if(PYMOD_VERSION)
        execute_process(
          COMMAND "${Python_EXECUTABLE}" "-c" "import ${module_name}; print(${module_name}.__version__)"
          RESULT_VARIABLE _${MODULE}_status
          OUTPUT_VARIABLE _${MODULE}_version
          ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(NOT _${MODULE}_status)
          set(PYMOD_${MODULE}_VERSION
              ${_${MODULE}_version}
              CACHE STRING "Version of Python module ${module}")
          set(PYMOD_${module}_VERSION
              ${PYMOD_${MODULE}_VERSION}
              CACHE STRING "Version of Python module ${module}")
        endif()
      endif()
    endif()
  endif()

  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19 AND CMAKE_VERSION VERSION_LESS 3.20)
    set(CMAKE_FIND_PACKAGE_NAME PYMOD_${module})
  endif()

  # NB: NAME_MISMATCHED is a CMake 3.17 addition
  find_package_handle_standard_args(
    PYMOD_${module_name}
    REQUIRED_VARS PYMOD_${MODULE}_PATH
    VERSION_VAR PYMOD_${MODULE}_VERSION NAME_MISMATCHED)

  set(PYMOD_${MODULE}_FOUND
      ${PYMOD_${MODULE}_FOUND}
      CACHE INTERNAL "")

  mark_as_advanced(PYMOD_${MODULE}_FOUND PYMOD_${MODULE}_PATH PYMOD_${MODULE}_VERSION)
endfunction()
