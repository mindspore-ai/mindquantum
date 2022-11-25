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

# Add a Python library (overload of the original python_add_library())
#
# python_add_library(<target>)
#
# Override the original python_add_library() to keep track of all python libraries and properly set some target
# properties depending on the current CMake version.
macro(python_add_library target)
  set(_args ${ARGN})
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    # Position 0 is the library type
    list(GET _args 0 _lib_type)
    if("${_lib_type}" STREQUAL "MODULE")
      list(INSERT _args 1 WITH_SOABI)
    endif()
  endif()

  _python_add_library(${target} ${_args})
  force_at_least_cxx17_workaround(${target})
  append_to_property(_doc_targets GLOBAL ${target})
  append_to_property(_python_targets GLOBAL ${target})
endmacro()

# ~~~
# Add a Pybind11 library (overload of the original pybind11_add_module())
#
# pybind11_add_module(<target>
#                     [OUTPUT_HINT <output_hint>])
#
# Override the original python_add_module() to keep track of all python libraries and properly set some target
# properties depending on the current CMake version.
#
# The <output-hint> argument is passed onto set_output_directory_auto() if present
# ~~~
function(pybind11_add_module target)
  cmake_parse_arguments(PARSE_ARGV 1 PAM "" "OUTPUT_HINT" "")

  _pybind11_add_module(${target} ${PAM_UNPARSED_ARGUMENTS})

  # Do we need to apply a workaround to compile using the correct pybind11? (see mindspore_ci.cmake for more
  # information)
  if(_mq_pybind11_prepend_to_link_libraries)
    get_target_property(_link_libraries ${target} LINK_LIBRARIES)
    list(PREPEND _link_libraries "${_mq_pybind11_prepend_to_link_libraries}")
    list(REMOVE_DUPLICATES _link_libraries)
    set_target_properties(${target} PROPERTIES LINK_LIBRARIES "${_link_libraries}")
  endif()

  append_to_property(_doc_targets GLOBAL ${target})
  append_to_property(_python_targets GLOBAL ${target})

  set(_install_lib_dir "${MQ_INSTALL_PYTHONDIR}")
  if(PAM_OUTPUT_HINT)
    set_output_directory_auto(${target} "${PAM_OUTPUT_HINT}")
    set(_install_lib_dir "${_install_lib_dir}/${PAM_OUTPUT_HINT}")
  endif()

  if(NOT IS_PYTHON_BUILD)
    install(
      TARGETS ${target}
      EXPORT mindquantumPythonTargets
      ARCHIVE DESTINATION ${_install_lib_dir}
      LIBRARY DESTINATION ${_install_lib_dir}
      RUNTIME DESTINATION ${MQ_INSTALL_BINDIR})
  endif()
endfunction()

# ==============================================================================

# ~~~
# Add a (Catch2) C++ test executable.
#
# add_test_executable(<target>
#                     [LIBS <libs> [... <libs>]]
#                     [DEFINES <defines> [... <defines>]])
#
# The <libs> and <defines> add link libraries and compile definitions to the generated target.
# ~~~
function(add_test_executable target)
  cmake_parse_arguments(PARSE_ARGV 1 "${target}" "" "" "LIBS;DEFINES")

  add_executable(${target} ${target}.cpp ${${target}_UNPARSED_ARGUMENTS})
  force_at_least_cxx17_workaround(${target})
  # NB: these will never be installed so need RPATH if we want to run them
  set_target_properties(${target} PROPERTIES BUILD_WITH_INSTALL_RPATH FALSE SKIP_BUILD_RPATH FALSE)
  target_include_directories(${target} PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tests>
                                               $<INSTALL_INTERFACE:${MQ_INSTALL_INCLUDEDIR}/include/tests>)
  target_link_libraries(${target} PRIVATE Catch2::Catch2 mindquantum_catch2_main mindquantum_catch2_utils
                                          ${${target}_LIBS})
  catch_discover_tests(${target})
  target_compile_definitions(${target} PRIVATE ${${target}_DEFINES})
  append_to_property(_test_exec_targets GLOBAL ${target})
endfunction()

# ==============================================================================
