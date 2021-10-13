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

include(CheckCompilerFlag OPTIONAL RESULT_VARIABLE _check_compiler_flag)
if(NOT _check_compiler_flag)
  include(Internal/CMakeCheckCompilerFlag)
endif()

# Check if a language has been enabled without attempting to enable it
#
# is_language_enabled(<lang> <resultvar>)
#
# If the language <lang> has already been enabled, <resultvar> is set to TRUE. Otherwise it is set to FALSE.
function(is_language_enabled _lang _var)
  get_property(_supported_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if(NOT _lang IN_LIST _supported_languages)
    set(${_var}
        FALSE
        PARENT_SCOPE)
  else()
    set(${_var}
        TRUE
        PARENT_SCOPE)
  endif()
endfunction()

# ==============================================================================

# ~~~
# Convenience function to test for the existence of some compiler flags for a a particular language
#
# check_compiler_flag(<lang> <var_prefix> <flags1> [<flags2>...])
#
# Check whether a compiler option is valid for the <lang> compiler. For each set of compiler options provided in the
# lists <flagsN>, it will test whether one of the element can be used by the corresponding compiler. If a flag is valid,
# it will be added to the GLOBAL property named <prefix>_<lang> as well as to a variable with the same name. If the
# property already exists, any valid flag is appended to the current value.
#
# Each call to this function also sets the _added_count variable to the number of flags added automatically (if any).
# ~~~
function(check_compiler_flags lang var_prefix)
  # cmake-lint: disable=C0103,E1120
  set(_${lang}_opts)

  foreach(_flag_list ${ARGN})
    separate_arguments(_flag_list)

    foreach(_flag ${_flag_list})
      # Drop the first character (most likely either '-' or '/')
      string(SUBSTRING ${_flag} 1 -1 _flag_name)
      string(REGEX REPLACE "^-+" "" _flag_name ${_flag_name})
      string(REGEX REPLACE "[-:/,=]" "_" _flag_name ${_flag_name})

      cmake_check_compiler_flag(${lang} ${_flag} ${lang}_compiler_has_${_flag_name})
      if(${lang}_compiler_has_${_flag_name})
        list(APPEND _${lang}_opts ${_flag})
        break()
      endif()
    endforeach()
  endforeach()

  # Is there a property that already corresponds to this?
  get_property(_opts GLOBAL PROPERTY ${var_prefix}_${lang})
  if(_opts)
    list(APPEND _opts ${_${lang}_opts})
  else()
    define_property(
      GLOBAL
      PROPERTY ${var_prefix}_${lang}
      BRIEF_DOCS "Compiler flags for ${var_prefix}"
      FULL_DOCS "Compiler flags for ${var_prefix}")
    set(_opts ${_${lang}_opts})
  endif()

  # Set GLOBAL property so that other parts of the code can have access to it
  set_property(GLOBAL PROPERTY ${var_prefix}_${lang} ${_opts})

  list(LENGTH _${lang}_opts _added_count)
  set(_added_count
      ${_added_count}
      PARENT_SCOPE)

  # Also set a variable for convenience
  set(${var_prefix}_${lang}
      ${_opts}
      PARENT_SCOPE)
endfunction()

# ~~~
# Convenience function to test for the existence of some compiler flags for a set of languages.
#
# test_compile_option(<prefix>
#                     LANGS <lang1> [<lang2>...]
#                     FLAGS <flags1> [<flags2>...]
#                     [AUTO_ADD_CO]
#                     [GENEX <genex>])
#
# Check that a compiler option can be applied to each of the specified languages <lang>. For each set of compiler
# options provided in the lists <flagsN>, it will test whether one of the element can be used by the corresponding
# compiler. If a flag is valid, it will be added to the GLOBAL property named <prefix>_<lang> as well as to a variable
# with the same name.
# If AUTO_ADD_CO is specified, the compiler option will be automatically added globally using
# add_compile_option(...). By default, the generator expression used in that function call restricts the compile option
# to the current language (ie. $<$<COMPILE_LANGUAGE:@lang@>:${_flag}>). This can be changed by using the <genex>
# argument (which defaults to "$<COMPILE_LANGUAGE:@lang@>").
#
# NB: This function calls check_compiler_flags() internally.
#
# ~~~
function(test_compile_option prefix)
  cmake_parse_arguments(PARSE_ARGV 1 TEST_CO "AUTO_ADD_CO" "GENEX" "LANGS;FLAGS")

  if(NOT TEST_CO_LANGS)
    message(FATAL_ERROR "Missing LANGS argument")
  endif()
  if(NOT TEST_CO_FLAGS)
    message(FATAL_ERROR "Missing FLAGS argument")
  endif()

  if(NOT TEST_CO_GENEX)
    set(TEST_CO_GENEX "$<COMPILE_LANGUAGE:@lang@>")
  endif()

  # cmake-lint: disable=C0103
  foreach(lang ${TEST_CO_LANGS})
    is_language_enabled(${lang} _enabled)
    if(_enabled)
      check_compiler_flags(${lang} ${prefix} ${TEST_CO_FLAGS})

      set(${prefix}_${lang}
          ${${prefix}_${lang}}
          PARENT_SCOPE)

      if(TEST_CO_AUTO_ADD_CO)
        string(CONFIGURE "${TEST_CO_GENEX}" _genex @ONLY)
        list(LENGTH ${prefix}_${lang} _L)
        math(EXPR _start_idx "${_L} - ${_added_count}")
        list(SUBLIST ${prefix}_${lang} ${_start_idx} -1 _added_flags)
        foreach(_flag ${_added_flags})
          add_compile_options("$<${_genex}:${_flag}>")
        endforeach()
      endif()
    else()
      set(${prefix}_${lang} PARENT_SCOPE)
    endif()
  endforeach()
endfunction()

# ==============================================================================

# ~~~
# Convenience function to test for the existence of some compiler flags for a a particular language
#
# check_link_flag(<lang> <var_prefix> [VERBATIM] <flags1> [<flags2>...])
#
# Check whether a linker option is valid for the <lang> linker. For each set of linker options provided in the lists
# <flagsN>, it will test whether one of the element can be used by the corresponding compiler. If a flag is valid, it
# will be added to the GLOBAL property named <prefix>_<lang> as well as to a variable with the same name. If the
# property already exists, any valid flag is appended to the current value.
#
# If VERBATIM is passed as argument, the flag is passed onto the linker without prepending the 'LINKER:' prefix.
#
# Each call to this function also sets the _added_count variable to the number of flags added automatically (if any).
# ~~~
function(check_link_flags lang var_prefix)
  # cmake-lint: disable=R0915,C0103,E1120

  cmake_parse_arguments(PARSE_ARGV 2 CHECK_LF "VERBATIM" "" "")
  set(_${lang}_link_opts)
  # This is a CMake 3.18 addition
  include(CheckLinkerFlag OPTIONAL RESULT_VARIABLE _check_linker_flags)

  set(_wrapper_flag ${CMAKE_${lang}_LINKER_WRAPPER_FLAG})
  list(GET _wrapper_flag -1 _last)
  set(_separate_options FALSE)
  if(_last STREQUAL " ")
    set(_separate_options TRUE)
    list(REMOVE_AT _wrapper_flag -1)
  endif()
  string(REPLACE ";" " " _wrapper_flag ${_wrapper_flag})

  foreach(_flag_list ${CHECK_LF_UNPARSED_ARGUMENTS})
    separate_arguments(_flag_list)

    foreach(_flag ${_flag_list})
      # Drop the first character (most likely either '-' or '/')
      string(SUBSTRING ${_flag} 1 -1 _flag_name)
      string(REGEX REPLACE "^-+" "" _flag_name ${_flag_name})
      string(REGEX REPLACE "[-:/,=]" "_" _flag_name ${_flag_name})
      if(CHECK_LF_VERBATIM)
        set(_prefix)
      else()
        set(_prefix "LINKER:")
      endif()

      if(_check_linker_flags)
        check_linker_flag(${lang} "${_prefix}${_flag}" ${lang}_linker_has_${_flag_name})
      else()
        if(NOT CHECK_LF_VERBATIM)
          if(_separate_options)
            string(REPLACE "," ";" _flags ${_flag})
            set(_expanded_flag)
            foreach(subflag ${_flags})
              set(_expanded_flag "${_expanded_flag} ${_wrapper_flag} ${subflag}")
            endforeach()
          else()
            set(_expanded_flag "${_wrapper_flag}${_flag}")
          endif()
        else()
          set(_expanded_flag "${_flag}")
        endif()

        set(CMAKE_REQUIRED_LINK_OPTIONS ${_expanded_flag})
        check_compiler_flag(${lang} "" ${lang}_linker_has_${_flag_name})
      endif()

      if(${lang}_linker_has_${_flag_name})
        list(APPEND _${lang}_link_opts ${_flag})
        break()
      endif()
    endforeach()
  endforeach()

  # Is there a property that already corresponds to this?
  get_property(_opts GLOBAL PROPERTY ${var_prefix}_${lang})
  if(_opts)
    list(APPEND _opts ${_${lang}_link_opts})
  else()
    define_property(
      GLOBAL
      PROPERTY ${var_prefix}_${lang}
      BRIEF_DOCS "Linker flags ${var_prefix}"
      FULL_DOCS "Linker flags ${var_prefix}")
    set(_opts ${_${lang}_link_opts})
  endif()

  # Set GLOBAL property so that other parts of the code can have access to it
  set_property(GLOBAL PROPERTY ${var_prefix}_${lang} ${_opts})

  list(LENGTH _${lang}_link_opts _added_count)
  set(_added_count
      ${_added_count}
      PARENT_SCOPE)

  # Also set a variable for convenience
  set(${var_prefix}_${lang}
      ${_opts}
      PARENT_SCOPE)
endfunction()

# ~~~
# Convenience function to test for the existence of some linker flags for a set of languages.
#
# test_link_option(<prefix>
#                  LANGS <lang1> [<lang2>...]
#                  FLAGS <flags1> [<flags2>...]
#                  [AUTO_ADD_LO]
#                  [VERBATIM]
#                  [GENEX <genex>])
#
# Check that a linker option can be applied to each of the specified languages <lang>. For each set of linker
# options provided in the lists <flagsN>, it will test whether one of the element can be used by the corresponding
# linker. If a flag is valid, it will be added to the GLOBAL property named <prefix>_<lang> as well as to a variable
# with the same name.
# If VERBATIM is passed as argument, the flag is passed onto the linker without prepending the 'LINKER:' prefix.
# If AUTO_ADD_LO is specified, the linker option will be automatically added globally using
# add_compile_option(...). By default, the generator expression used in that function call restricts the compile option
# to the current language (ie. $<$<LINK_LANGUAGE:@lang@>:LINKER:${_flag}>). This can be changed by using the <genex>
# argument (which defaults to "$<LINK_LANGUAGE:@lang@>").
#
# NB: This function calls check_link_flags() internally.
# ~~~
function(test_link_option prefix)
  cmake_parse_arguments(PARSE_ARGV 1 TEST_LO "AUTO_ADD_LO;VERBATIM" "GENEX" "LANGS;FLAGS")

  if(NOT TEST_LO_LANGS)
    message(FATAL_ERROR "Missing LANGS argument")
  endif()
  if(NOT TEST_LO_FLAGS)
    message(FATAL_ERROR "Missing FLAGS argument")
  endif()

  if(NOT TEST_LO_GENEX)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
      set(TEST_LO_GENEX "$<LINK_LANGUAGE:@lang@>")
    else()
      set(TEST_LO_GENEX "1")
    endif()
  endif()

  # cmake-lint: disable=C0103
  foreach(lang ${TEST_LO_LANGS})
    is_language_enabled(${lang} _enabled)
    if(_enabled)
      set(_args ${TEST_LO_FLAGS})
      if(TEST_LO_VERBATIM)
        set(_args "VERBATIM;${_args}")
      endif()
      check_link_flags(${lang} ${prefix} ${_args})

      set(${prefix}_${lang}
          ${${prefix}_${lang}}
          PARENT_SCOPE)

      if(TEST_LO_AUTO_ADD_LO)
        string(CONFIGURE "${TEST_LO_GENEX}" _genex @ONLY)
        list(LENGTH ${prefix}_${lang} _L)
        math(EXPR _start_idx "${_L} - ${_added_count}")
        list(SUBLIST ${prefix}_${lang} ${_start_idx} -1 _added_flags)
        foreach(_flag ${_added_flags})
          if(TEST_LO_VERBATIM)
            add_link_options("$<${_genex}:${_flag}>")
          else()
            add_link_options("$<${_genex}:LINKER:${_flag}>")
          endif()
        endforeach()
      endif()
    else()
      set(${prefix}_${lang} PARENT_SCOPE)
    endif()
  endforeach()
endfunction()

# ==============================================================================

# ~~~
# Append a value to a property (creating the latter if necessary)
#
# append_to_property(<property_name>
#                    <GLOBAL             |
#                     DIRECTORY [<dir>]  |
#                     TARGET    <target> |
#                     SOURCE    <source> |
#                               [DIRECTORY <dir> | TARGET_DIRECTORY <target>] |
#                     INSTALL   <file>   |
#                     TEST      <test>   |
#                     CACHE     <entry>  |
#                     VARIABLE           >
#                    <value>)
# ~~~
macro(append_to_property name scope value)
  get_property(_prop ${scope} PROPERTY ${name})
  if(_prop)
    list(APPEND _prop ${value})
  else()
    define_property(
      ${scope}
      PROPERTY ${name}
      BRIEF_DOCS "${scope} property for ${name}"
      FULL_DOCS "${scope} property for ${name}")
    set(_prop ${value})
  endif()

  set_property(${scope} PROPERTY ${name} ${_prop})
endmacro()

# ==============================================================================

# Automatically set the output directory for a particular target with a potential hint
#
# set_output_directory_auto(<target> <hint>)
#
# Automatically set the output directory for <target>. <hint> must be an existing path in the current CMake project
# directory. It is only used if the CMake variable IN_PLACE_BUILD is set to a truthful value. Otherwise, the macro will
# look at ${target}_OUTPUT_DIR CMake variable (if it exists) to set the output directory (it will create the directory
# if it does not already exist on the filesystem).
#
macro(set_output_directory_auto target hint)
  string(TOUPPER ${target} _TARGET)

  if(IN_PLACE_BUILD)
    # Automatically calculate the output directory
    set(${_TARGET}_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/${hint})
  endif()

  # Normalize variable name
  if(${target}_OUTPUT_DIR)
    set(${_TARGET}_OUTPUT_DIR ${${target}_OUTPUT_DIR})
  endif()

  # Create output directory if it does not exist already
  if(${_TARGET}_OUTPUT_DIR AND NOT EXISTS ${${_TARGET}_OUTPUT_DIR})
    file(MAKE_DIRECTORY ${${_TARGET}_OUTPUT_DIR})
  endif()

  # Properly set output directory for a target so that during an installation using either 'pip install' or 'python3
  # setup.py install' the libraries get built in the proper directory
  if(${_TARGET}_OUTPUT_DIR)
    set_target_properties(
      ${target}
      PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${${_TARGET}_OUTPUT_DIR}
                 LIBRARY_OUTPUT_DIRECTORY_DEBUG ${${_TARGET}_OUTPUT_DIR}
                 LIBRARY_OUTPUT_DIRECTORY_RELEASE ${${_TARGET}_OUTPUT_DIR}
                 LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${${_TARGET}_OUTPUT_DIR}
                 LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${${_TARGET}_OUTPUT_DIR})
  elseif(IS_PYTHON_BUILD)
    message(
      WARNING "IS_PYTHON_BUILD=ON but ${_TARGET}_OUTPUT_DIR "
              "was not defined! The shared library for target ${target} "
              "will probably not be copied to the correct directory. "
              "Did you forget to add a CMakeExtension in setup.py?")
  elseif(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(
      ${target}
      PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
                 LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
                 LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
                 LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
                 LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  endif()
endmacro()

# ==============================================================================

# Set RPATH of target only if building for Python (ie. IS_PYTHON_BUILD=ON) or if building in-place (IN_PLACE_BUILD=ON)
macro(python_install_set_rpath target path)
  if((IS_PYTHON_BUILD OR IN_PLACE_BUILD) AND LINKER_RPATH)
    if(APPLE)
      set_target_properties(${target} PROPERTIES INSTALL_RPATH "@loader_path/${path}")
    elseif(UNIX)
      set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/${path}")
    endif()
  endif()
endmacro()

# ==============================================================================

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

  find_package_handle_standard_args(
    PYMOD_${module_name}
    REQUIRED_VARS PYMOD_${MODULE}_PATH
    VERSION_VAR PYMOD_${MODULE}_VERSION)

  set(PYMOD_${MODULE}_FOUND
      ${PYMOD_${MODULE}_FOUND}
      CACHE INTERNAL "")

  mark_as_advanced(PYMOD_${MODULE}_FOUND PYMOD_${MODULE}_PATH PYMOD_${MODULE}_VERSION)
endfunction()

# ==============================================================================
