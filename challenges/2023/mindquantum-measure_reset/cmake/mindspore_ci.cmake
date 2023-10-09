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
#
# This file essentially contains some workarounds for building MindQuantum on the MindSpore CI which may have issues
# such as:
#   - old system pybind11 version (2.5.0)
#
# ==============================================================================

# lint_cmake: -whitespace/indent

include(debug_print)

# ==============================================================================

# ~~~
# Same as CMake try_compile but does not attempt to link the program
#
# try_compile_cxx_no_link(<var>
#                         <source>
#                         [QUIET]
#                         [OUTPUT <output>]
#                         [INCLUDE_DIRECTORIES <include_directories> [... <include_directories>]]
#                         [LINK_LIBRARIES <link_libraries> [... <link_libraries>]])
# ~~~
function(try_compile_cxx_no_link var source)
  cmake_parse_arguments(PARSE_ARGV 1 TCCNL "QUIET" "OUTPUT" "INCLUDE_DIRECTORIES;LINK_LIBRARIES")

  set(_bindir "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp")
  file(MAKE_DIRECTORY "${_bindir}")
  set(_filename "${_bindir}/src.cpp")
  file(WRITE "${_filename}" "${source}")

  if(NOT DEFINED ${var})
    if(NOT TCCNL_QUIET)
      message(CHECK_START "Performing Test ${var}")
    endif()
    try_compile(
      ${var} "${CMAKE_BINARY_DIR}"
      "${_filename}"
      OUTPUT_VARIABLE _output
      LINK_LIBRARIES ${TCCNL_LINK_LIBRARIES}
      CMAKE_FLAGS
        "-DINCLUDE_DIRECTORIES:STRING=${TCCNL_INCLUDE_DIRECTORIES}"
        "-DCMAKE_CXX_LINK_EXECUTABLE=${CMAKE_COMMAND} -E echo \"Not linking\"" CXX_STANDARD ${CMAKE_CXX_STANDARD})
    if(${var})
      if(NOT TCCNL_QUIET)
        message(CHECK_PASS "Success")
      endif()
      set(${var}
          1
          CACHE INTERNAL "Test ${var}")
      file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
           "Performing C++ SOURCE FILE Test ${var} succeeded with the following output:\n" "${_output}\n"
           "Source file was:\n${_source}\n")
    else()
      if(NOT TCCNL_QUIET)
        message(CHECK_FAIL "Failed")
      endif()
      set(${var}
          ""
          CACHE INTERNAL "Test ${var}")
      file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
           "Performing C++ SOURCE FILE Test ${var} failed with the following output:\n" "${_output}\n"
           "Source file was:\n${_source}\n")
    endif()
  endif()
  if(TCCNL_OUTPUT)
    set(${TCCNL_OUTPUT}
        "${_output}"
        PARENT_SCOPE)
  endif()
endfunction()

# ==============================================================================
# Pybind11 workaround
#
# ~~~
# Old pybind11 version installed in system. During compilation, we might have the system headers for python included
# *before* the ones from pybind11.
#
# For example:
# -isystem /usr/local/python/python375/include/python3.7m \
#     -isystem /.../venv/lib/python3.7/site-packages/pybind11/include
#
# The solution here is to first try to compile some code that does not work on pybind11 2.5.0
# ~~~
# ==============================================================================

set(_pybind11_dependency_targets Python::Module pybind11::module)

# First try to compile a simple pybind11 2.9.0+ code (pybind11::detail::const_name) using the same target list as would
# happen if we were to use pybind11_add_module()
if(NOT DEFINED compile_pybind11_normally_works)
  try_compile_cxx_no_link(
    compile_pybind11_normally_works
    [[
#include <pybind11/pybind11.h>
int main() {
     const auto name = pybind11::detail::const_name("Hello world").text;
     return 0;
}
]]
    LINK_LIBRARIES ${_pybind11_dependency_targets})
endif()

# If the above failed, we have a conflict between system pybind11 and the one located inside the virtual environment.
# Try prepending the pybind11::module target to link libraries.
if(NOT compile_pybind11_normally_works AND NOT DEFINED system_prepend_pybind11_include_works)
  try_compile_cxx_no_link(
    system_prepend_pybind11_include_works
    [[
#include <pybind11/pybind11.h>
int main() {
     const auto name = pybind11::detail::const_name("Hello world").text;
     return 0;
}
]]
    LINK_LIBRARIES pybind11::headers ${_pybind11_dependency_targets})
endif()

if(NOT compile_pybind11_normally_works AND system_prepend_pybind11_include_works)
  set(_mq_pybind11_prepend_to_link_libraries
      pybind11::headers
      CACHE INTERNAL "")
elseif(NOT compile_pybind11_normally_works)
  message(FATAL_ERROR "Unable to compile a recent pybind11 (2.9.0+) example and the workaround failed.")
endif()
