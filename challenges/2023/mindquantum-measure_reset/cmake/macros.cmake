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

# This is a CMake 3.19 addition
include(CheckCompilerFlag OPTIONAL RESULT_VARIABLE _check_compiler_flag)
if(NOT _check_compiler_flag)
  include(Internal/CMakeCheckCompilerFlag)
endif()
# This is a CMake 3.18 addition
include(CheckLinkerFlag)

# ==============================================================================

# Wrapper over execute_process()
function(__exec_cmd)
  set(options)
  set(oneValueArgs WORKING_DIRECTORY)
  set(multiValueArgs COMMAND)

  cmake_parse_arguments(EXEC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # cmake-format: off
  debug_print(
    STATUS "  execute_process("
    "      COMMAND ${EXEC_COMMAND}"
    "      WORKING_DIRECTORY ${EXEC_WORKING_DIRECTORY}"
    "      ...)"
  )
  if(APPLE)
    debug_print(STATUS "   MACOSX_DEPLOYMENT_TARGET = $ENV{MACOSX_DEPLOYMENT_TARGET}")
  endif()
  execute_process(
    COMMAND ${EXEC_COMMAND}
    WORKING_DIRECTORY ${EXEC_WORKING_DIRECTORY}
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
    RESULTS_VARIABLE _results_out
    RESULT_VARIABLE RESULT
  )
  # cmake-format: on
  if(NOT RESULT EQUAL "0")
    debug_print(SEND_ERROR "STDOUT:\n${_stdout}" "STDERR:\n${_stderr}" "RESULTS OUT:\n${_results_out}")
    message(FATAL_ERROR "error! when ${EXEC_COMMAND} in ${EXEC_WORKING_DIRECTORY}")
  endif()
endfunction()

# ------------------------------------------------------------------------------

include(is_language_enabled)

# ------------------------------------------------------------------------------

# ~~~
# Setup a language for use with MindQuantum
#
# setup_language(<lang>)
#
# This function creates 3 targets that are used throughout MindQuantum in order to store compiler and linker flags:
#   - <lang>_try_compile
#   - <lang>_try_compile_flagcheck
#   - <lang>_mindquantum
#
# The first two are used in try_compile() calls while the last one should contain the definite list of compiler and
# linker options that MindQuantum requires for all targets.
# ~~~
function(setup_language lang)
  add_library(${lang}_try_compile_flagcheck INTERFACE)
  add_library(${lang}_try_compile INTERFACE)
  add_library(${lang}_mindquantum INTERFACE)

  append_to_property(mq_install_targets GLOBAL ${lang}_mindquantum)
endfunction()

# ------------------------------------------------------------------------------

# ~~~
# Append a list of compile definitions to some of the language specific targets (see setup_language())
#
# mq_add_compile_definitions([TRYCOMPILE, TRYCOMPILE_FLAGCHECK]
#                           <definition>, [<definitions>, ...])
#
# Always modify the <LANG>_mindquantum target. If any of TRYCOMPILE, TRYCOMPILE_FLAGCHECK is also specified, then modify
# the corresponding target.
# ~~~
function(mq_add_compile_definitions)
  cmake_parse_arguments(PARSE_ARGV 0 MACD "TRYCOMPILE;TRYCOMPILE_FLAGCHECK" "" "")

  foreach(lang C CXX CUDA NVCXX DPCXX)
    is_language_enabled(${lang} _enabled)
    if(_enabled)
      target_compile_definitions(${lang}_mindquantum
                                 INTERFACE "$<$<COMPILE_LANGUAGE:${lang}>:${MACD_UNPARSED_ARGUMENTS}>")
      if(MACD_TRYCOMPILE)
        target_compile_definitions(${lang}_try_compile
                                   INTERFACE "$<$<COMPILE_LANGUAGE:${lang}>:${MACD_UNPARSED_ARGUMENTS}>")
      endif()
      if(MACD_TRYCOMPILE_FLAGCHECK)
        target_compile_definitions(${lang}_try_compile_flagcheck
                                   INTERFACE "$<$<COMPILE_LANGUAGE:${lang}>:${MACD_UNPARSED_ARGUMENTS}>")
      endif()
    endif()
  endforeach()
endfunction()

# ------------------------------------------------------------------------------

# ~~~
# Return the human name of a compiler language
#
# get_language_name(<lang> <out-var>)
# ~~~
function(get_language_name lang var)
  if("${lang}" STREQUAL "C")
    set(_lang_textual "C")
  elseif("${lang}" STREQUAL "CXX")
    set(_lang_textual "C++")
  elseif("${lang}" STREQUAL "CUDA")
    set(_lang_textual "CUDA")
  elseif("${lang}" STREQUAL "Fortran")
    set(_lang_textual "Fortran")
  elseif("${lang}" STREQUAL "HIP")
    set(_lang_textual "HIP")
  elseif("${lang}" STREQUAL "ISPC")
    set(_lang_textual "ISPC")
  elseif("${lang}" STREQUAL "NVCXX")
    set(_lang_textual "NVHPC-C++")
  elseif("${lang}" STREQUAL "OBJC")
    set(_lang_textual "Objective-C")
  elseif("${lang}" STREQUAL "OBJCXX")
    set(_lang_textual "Objective-C++")
  else()
    message(SEND_ERROR "check_source_compiles: ${lang}: unknown language.")
  endif()
  set(${var}
      ${_lang_textual}
      PARENT_SCOPE)
endfunction()

# ==============================================================================

include(to_cmake_path)

# ------------------------------------------------------------------------------

# ~~~
# (helper function) Compute the absolute path to an existing file or directory with symlinks resolved.
#
# real_path(<path> <out-var>)
# ~~~
function(real_path path var)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
    file(REAL_PATH "${path}" _resolved_path)
  else()
    get_filename_component(_resolved_path "${path}" REALPATH)
  endif()

  set(${var}
      "${_resolved_path}"
      PARENT_SCOPE)
endfunction()

# ==============================================================================

# ~~~
# Macro used to disable CUDA support in MindQuantum
#
# disable_cuda([<msg>])
# ~~~
macro(disable_cuda)
  if(${ARGC} GREATER 0)
    set(_msg "${ARGV0}")
  else()
    set(_msg "inexistent CUDA/NVHPC compiler, NVHPC < 20.11")
  endif()
  message(WARNING "Disabling CUDA due to ${_msg} or error during compiler setup")
  set(ENABLE_CUDA
      OFF
      CACHE INTERNAL "Enable building of CUDA/NVHPC libraries")
endmacro()

# ==============================================================================

# ~~~
# Force at least C++17 for some older compilers:
#   - GCC 7.0
#
# force_at_least_cxx17_workaround(<target>)
# ~~~
function(force_at_least_cxx17_workaround target)
  if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
    set_target_properties(${target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED TRUE)
  endif()

  # NVCC (up to 11.8) only supports C++17
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set_target_properties(${target} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED TRUE)
  endif()
endfunction()

# ==============================================================================

# ~~~
# Apply a patch file
#
# _apply_patch_file(ERROR_VARIABLE <var-name>
#                   INPUT_FILE <patch-file>
#                   OUTPUT_VARIABLE <var-name>
#                   RESULT_VARIABLE <var-name>
#                   RESULTS_VARIABLE <var-name>
#                   WORKING_DIRECTORY <dir>
#                   PATCH_ARGS <arg1> [... <argN>])
#
# This function will first run the patch command in dry-run mode in order to test the application of the patch. If that
# is successful, the patch will be applied using the same arguments. Otherwise, nothing will happen.
#
# The caller is responsible for testing if the patch was applied successfully or not based on the value of the
# RESULT_VARIABLE variable.
# ~~~
function(_apply_patch_file)
  cmake_parse_arguments(
    APF "" "WORKING_DIRECTORY;INPUT_FILE;OUTPUT_VARIABLE;ERROR_VARIABLE;RESULT_VARIABLE;RESULTS_VARIABLE" "PATCH_ARGS"
    ${ARGN})

  debug_print(STATUS "  -> calling patch command (dry-run) with: ${Patch_EXECUTABLE} ${APF_PATCH_ARGS}")
  execute_process(
    COMMAND "${Patch_EXECUTABLE}" ${APF_PATCH_ARGS} --dry-run
    INPUT_FILE "${APF_INPUT_FILE}"
    WORKING_DIRECTORY "${APF_WORKING_DIRECTORY}"
    OUTPUT_VARIABLE _stdout
    COMMAND_ECHO NONE
    ERROR_VARIABLE _stderr
    RESULTS_VARIABLE _results
    RESULT_VARIABLE _result)

  if(_result EQUAL "0")
    string(REGEX REPLACE "\n" ";" _stdout "${_stdout}")
    foreach(_line ${_stdout})
      message(STATUS "     :: ${_line}")
    endforeach()
    debug_print(STATUS "  -> dry-run patch successful, applying patch for real")
    execute_process(
      COMMAND "${Patch_EXECUTABLE}" ${APF_PATCH_ARGS}
      INPUT_FILE "${APF_INPUT_FILE}"
      WORKING_DIRECTORY "${APF_WORKING_DIRECTORY}"
      OUTPUT_VARIABLE _stdout
      COMMAND_ECHO NONE
      ERROR_VARIABLE _stderr
      RESULTS_VARIABLE _results
      RESULT_VARIABLE _result)
  else()
    debug_print(STATUS "  -> application of patch failed (dry-run)")
  endif()

  set(${APF_OUTPUT_VARIABLE}
      ${_stdout}
      PARENT_SCOPE)
  set(${APF_ERROR_VARIABLE}
      ${_stderr}
      PARENT_SCOPE)
  set(${APF_RESULT_VARIABLE}
      ${_result}
      PARENT_SCOPE)
  set(${APF_RESULTS_VARIABLE}
      ${_results}
      PARENT_SCOPE)
endfunction()

# ==============================================================================

# ~~~
# Apply a list of patches by calling `patch -p1 <patch-file>`
#
# apply_patches(<working_directory> [<patch-file> [... <patch-file>]])
# ~~~
function(apply_patches)
  # cmake-lint: disable=R0912,R0915
  cmake_parse_arguments(PARSE_ARGV 0 AP "TRY_GIT_RESET" "WORKING_DIRECTORY" "")

  if(NOT AP_WORKING_DIRECTORY)
    message(FATAL_ERROR "Missing <working-directory> argument!")
  endif()

  if(AP_TRY_GIT_RESET)
    find_package(Git REQUIRED)
  endif()

  # ------------------------------------

  set(_try_git_reset FALSE)
  # cmake-lint: disable=R0915
  foreach(_patch_file ${AP_UNPARSED_ARGUMENTS})
    # NB: All these shenanigans with file(CONFIGURE ...) are just to make sure that we get a file with LF line
    # endings...
    get_filename_component(_patch_file_name ${_patch_file} NAME)
    set(_lf_patch_file ${PROJECT_BINARY_DIR}/_mq_patch/lf_${_patch_file_name})
    set(_crlf_patch_file ${PROJECT_BINARY_DIR}/_mq_patch/crlf_${_patch_file_name})
    file(READ "${_patch_file}" _content)
    # NB: escape patches that have @XXX@ since those would be replaced by the call to file(CONFIGURE)
    set(_at @)
    string(REGEX REPLACE [[\@([a-zA-Z_]+)\@]] [[@_at@\1@_at@]] _content "${_content}")
    if(CMAKE_VERSION VERSION_LESS 3.20)
      set(_less <)
      set(_greater >)
      # NB: some of the patches contain email addresses <name.surname@email.com>
      string(REGEX REPLACE [[<([a-zA-Z0-9\.]+)\@([a-zA-Z0-9\.]+)>]] [[@_less@\1@_at@\2@_greater@]] _content
                           "${_content}")
      string(REPLACE "<" "@_less@" _content "${_content}")
      string(REPLACE ">" "@_greater@" _content "${_content}")
    endif()

    file(
      CONFIGURE
      OUTPUT "${_lf_patch_file}"
      CONTENT "${_content}"
      @ONLY
      NEWLINE_STYLE LF)
    file(
      CONFIGURE
      OUTPUT "${_crlf_patch_file}"
      CONTENT "${_content}"
      @ONLY
      NEWLINE_STYLE CRLF)

    file(MD5 "${_lf_patch_file}" _lf_md5)
    file(MD5 "${_crlf_patch_file}" _crlf_md5)
    set(_lf_patch_lock_file "${AP_WORKING_DIRECTORY}/mq_applied_patch_${_lf_md5}")
    set(_crlf_patch_lock_file "${AP_WORKING_DIRECTORY}/mq_applied_patch_${_crlf_md5}")

    if(NOT EXISTS "${_lf_patch_lock_file}" AND NOT EXISTS "${_crlf_patch_lock_file}")
      message(STATUS "Applying patch ${_patch_file}")
      set(_patch_args -p1)
      if(WIN32)
        list(APPEND _patch_args --binary)
      endif()

      # First try LF patch
      _apply_patch_file(
        PATCH_ARGS ${_patch_args}
        INPUT_FILE "${_lf_patch_file}"
        WORKING_DIRECTORY "${AP_WORKING_DIRECTORY}"
        OUTPUT_VARIABLE _lf_stdout
        ERROR_VARIABLE _lf_stderr
        RESULTS_VARIABLE _lf_results_out
        RESULT_VARIABLE _result)

      if(NOT _result EQUAL "0" AND WIN32)
        # If not successful, try LF patch again without --binary
        debug_print(STATUS "  -> trying LF patch again without --binary on Windows")
        _apply_patch_file(
          PATCH_ARGS -p1
          INPUT_FILE "${_lf_patch_file}"
          WORKING_DIRECTORY "${AP_WORKING_DIRECTORY}"
          OUTPUT_VARIABLE _lf_stdout
          ERROR_VARIABLE _lf_stderr
          RESULTS_VARIABLE _lf_results_out
          RESULT_VARIABLE _result)
      endif()

      if(NOT _result EQUAL "0")
        # If not successful, try CRLF patch
        debug_print(STATUS "  -> trying CRLF patch since LF patch failed")
        _apply_patch_file(
          PATCH_ARGS -p1
          INPUT_FILE "${_crlf_patch_file}"
          WORKING_DIRECTORY "${AP_WORKING_DIRECTORY}"
          OUTPUT_VARIABLE _crlf_stdout
          ERROR_VARIABLE _crlf_stderr
          RESULTS_VARIABLE _crlf_results_out
          RESULT_VARIABLE _result)

        if(NOT _result EQUAL "0")
          if(AP_TRY_GIT_RESET AND EXISTS "${AP_WORKING_DIRECTORY}/.git")
            set(_try_git_reset TRUE)
            break()
          endif()
          debug_print(SEND_ERROR "STDOUT(LF):\n${_lf_stdout}" "STDERR(LF):\n${_lf_stderr}"
                      "RESULTS OUT (LF):\n${_lf_results_out}")
          debug_print(SEND_ERROR "STDOUT(CRLF):\n${_crlf_stdout}" "STDERR(CRLF):\n${_crlf_stderr}"
                      "RESULTS OUT (CRLF):\n${_crlf_results_out}")
          message(
            FATAL_ERROR
              "\
All attempts at applying ${_patch_file} have failed. This includes the following generated files:
  - ${_lf_patch_file}
  - ${_crlf_patch_file}

Did you by any change make a modification to the patch file and forgot to revert the old version of the patch?
You might want to remove the ${AP_WORKING_DIRECTORY} directory completely and re-run CMake.
")
        else()
          file(TOUCH ${_crlf_patch_lock_file})
        endif()
      else()
        file(TOUCH ${_lf_patch_lock_file})
      endif()
    else()
      message(STATUS "Skipping patch ${_patch_file} since already applied.")
    endif()
  endforeach()

  if(_try_git_reset)
    message(STATUS "Some patches failed. Attempting `git stash -all` and re-applying all patches.")
    __exec_cmd(COMMAND ${GIT_EXECUTABLE} stash --all WORKING_DIRECTORY ${AP_WORKING_DIRECTORY})
    __exec_cmd(COMMAND ${GIT_EXECUTABLE} stash drop WORKING_DIRECTORY ${AP_WORKING_DIRECTORY})

    set(_args ${ARGN})
    list(REMOVE_ITEM _args TRY_GIT_RESET)
    debug_print(STATUS "Calling apply_patches(${_args})")
    apply_patches(${_args})
  endif()
endfunction()

# ==============================================================================

# ~~~
# Convenience function to test for the existence of some compiler flags for a a particular language
#
# check_compiler_flag(<lang> <out-var> [FLAGCHECK] <flags1> [<flags2>...])
#
# Check whether a compiler option is valid for the <lang> compiler. For each set of compiler options provided in the
# lists <flagsN>, it will test whether one of the element can be used by the corresponding compiler. If a flag is valid,
# it will be added to the variable returned by this function.
#
# If FLAGCHECK is specified, call the compiler directly using execute_process() instead of using
# cmake_check_compiler_flag() as we expect that no output file will be produced.
#
# Note that this function will use the flags contained in either <LANG>_try_compile or <LANG>_try_compile_flagcheck (if
# FLAGCHECK is passed as an argument)
# ~~~
function(check_compiler_flags lang out_var)
  cmake_parse_arguments(PARSE_ARGV 2 CCF "FLAGCHECK" "LINKER_FLAGS" "")

  # cmake-lint: disable=C0103,E1120
  set(_${lang}_opts)

  if(CCF_FLAGCHECK)
    set(_target ${lang}_try_compile_flagcheck)
  else()
    set(_target ${lang}_try_compile)
  endif()

  get_target_property(_required_flags ${_target} INTERFACE_COMPILE_OPTIONS)
  if(NOT _required_flags)
    unset(_required_flags)
  endif()

  get_language_name(${lang} _lang_textual)

  foreach(_flag_list ${CCF_UNPARSED_ARGUMENTS})
    separate_arguments(_flag_list)

    foreach(_flag ${_flag_list})
      # Drop the first character (most likely either '-' or '/')
      string(SUBSTRING ${_flag} 1 -1 _flag_name)
      string(REGEX REPLACE "^-+" "" _flag_name ${_flag_name})
      string(REGEX REPLACE "[-:/,=]" "_" _flag_name ${_flag_name})

      if(NOT CCF_FLAGCHECK)
        set(CMAKE_REQUIRED_FLAGS ${_required_flags})
        if(CCF_LINKER_FLAGS)
          set(CMAKE_REQUIRED_LINK_OPTIONS "${CCF_LINKER_FLAGS}")
        endif()
        cmake_check_compiler_flag(${lang} ${_flag} ${lang}_compiler_has_${_flag_name})
      else()
        set(_var ${lang}_compiler_has_${_flag_name})

        if(NOT DEFINED "${_var}")
          message(CHECK_START "Performing Test ${_var}")
          set(_cmd ${CMAKE_${lang}_COMPILER} ${_flag} ${_required_flags})
          execute_process(
            COMMAND ${_cmd}
            RESULT_VARIABLE _result
            OUTPUT_VARIABLE _output
            ERROR_VARIABLE _error)

          if(_result EQUAL 0)
            set(${_var}
                1
                CACHE INTERNAL "Test ${_var}")
            message(CHECK_PASS "Success")
            file(APPEND ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
                 "Performing ${_lang_textual} FLAG CHECK Test ${_var} with the following command line:\n${_cmd}\n"
                 "has succeeded\n\n")
          else()
            message(CHECK_FAIL "Failed")
            set(${_var}
                ""
                CACHE INTERNAL "Test ${_var}")
            file(
              APPEND
              ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
              "Performing ${_lang_textual} FLAG CHECK Test ${_var} with the following command line:\n${_cmd}\n"
              "has failed with the following output:\n"
              "${_output}\n"
              "And the following error output:\n"
              "${_error}\n\n")
          endif()
        endif()
      endif()

      if(${lang}_compiler_has_${_flag_name})
        list(APPEND _${lang}_opts ${_flag})
        break()
      endif()
    endforeach()
  endforeach()

  set(${out_var}
      ${_${lang}_opts}
      PARENT_SCOPE)
endfunction()

# ~~~
# Convenience function to test for the existence of some compiler flags for a set of languages.
#
# test_compile_option(<name>
#                     LANGS <lang1> [<lang2>...]
#                     FLAGS <flags1> [<flags2>...]
#                     [FLAGCHECK, NO_MQ_TARGET, NO_TRYCOMPILE_TARGET, NO_TRYCOMPILE_FLAGCHECK_TARGET]
#                     [GENEX <genex>]
#                     [LINKER_FLAGS <flag>]
#                     [CMAKE_OPTION <option>])
#
# Check that a compiler option can be applied to each of the specified languages <langN>. For each set of compiler
# options provided in the lists <flagsN>, it will test whether one of the element can be used by the corresponding
# compiler. If a flag is valid, it will be added to the language-specific targets (unless a corresponding negating flag
# was specified):
#  - <name>_<LANG> (created if does not exist already)
#  - <LANG>_mindquantum
#  - <LANG>_try_compile
#  - <LANG>_try_compile_flagcheck (only if FLACHECK is passed as argument
#
# If CMAKE_OPTION is specified, the check will only be performed if <option> is set to a truthful value. Also, the
# compiler option will only be enabled if <option> is true.
#
# In addition, regardless of whether a language is enabled or not, this function will set a variable in the caller's
# scope named <name>_<LANG> to TRUE/FALSE depending on whether one or more flags were found.
#
# NB: This function calls check_compiler_flags() internally.
# ~~~
function(test_compile_option name)
  # cmake-lint: disable=R0912,R0915,C0103
  cmake_parse_arguments(PARSE_ARGV 1 TCO "FLAGCHECK;NO_MQ_TARGET;NO_TRYCOMPILE_TARGET;NO_TRYCOMPILE_FLAGCHECK_TARGET"
                        "CMAKE_OPTION;GENEX;LINKER_FLAGS" "LANGS;FLAGS")

  if(NOT TCO_LANGS)
    message(FATAL_ERROR "Missing LANGS argument")
  endif()
  if(NOT TCO_FLAGS)
    message(FATAL_ERROR "Missing FLAGS argument")
  endif()

  if(NOT TCO_GENEX)
    set(TCO_GENEX "$<COMPILE_LANGUAGE:@lang@>")
  else()
    set(TCO_GENEX "$<AND:$<COMPILE_LANGUAGE:@lang@>,${TCO_GENEX}>")
  endif()

  if(${TCO_CMAKE_OPTION})
    set(TCO_GENEX "$<AND:$<BOOL:${TCO_CMAKE_OPTION}>,${TCO_GENEX}>")
  endif()

  # ------------------------------------

  set(_ccf_args)
  if(TCO_FLAGCHECK)
    list(APPEND _ccf_args FLAGCHECK)
  endif()

  if(TCO_LINKER_FLAGS)
    list(APPEND _ccf_args LINKER_FLAGS "${TCO_LINKER_FLAGS}")
  endif()

  # ------------------------------------

  foreach(lang ${TCO_LANGS})
    set(_has_flags FALSE)
    is_language_enabled(${lang} _enabled)
    if(_enabled)
      if(NOT TARGET ${name}_${lang})
        add_library(${name}_${lang} INTERFACE)
      endif()

      set(_do_check ON)
      if(NOT "${TCO_CMAKE_OPTION}" STREQUAL "" AND NOT ${${TCO_CMAKE_OPTION}})
        set(_do_check OFF)
      endif()
      if(_do_check)
        check_compiler_flags(${lang} _${lang}_flags ${_ccf_args} ${TCO_FLAGS})
      endif()
      string(CONFIGURE "${TCO_GENEX}" _genex @ONLY)

      if(NOT "${_${lang}_flags}" STREQUAL "")
        set(_has_flags TRUE)

        target_compile_options(${name}_${lang} INTERFACE "$<${_genex}:${_${lang}_flags}>")
        if(NOT TCO_NO_MQ_TARGET)
          target_compile_options(${lang}_mindquantum INTERFACE "$<${_genex}:${_${lang}_flags}>")
        endif()
        if(NOT TCO_NO_TRYCOMPILE_TARGET)
          target_compile_options(${lang}_try_compile INTERFACE ${_${lang}_flags})
        endif()
        if(TCO_FLAGCHECK AND NOT TCO_NO_TRYCOMPILE_FLAGCHECK_TARGET)
          target_compile_options(${lang}_try_compile_flagcheck INTERFACE ${_${lang}_flags})
        endif()
      endif()
    endif()

    set(${name}_${lang}
        ${_has_flags}
        PARENT_SCOPE)
  endforeach()
endfunction()

# ==============================================================================

# ~~~
# Convenience function to test for the existence of some compiler flags for a a particular language
#
# check_link_flag(<lang> <out-var> [VERBATIM] <flags1> [<flags2>...])
#
# Check whether a linker option is valid for the <lang> linker. For each set of linker options provided in the lists
# <flagsN>, it will test whether one of the element can be used by the corresponding compiler. If a flag is valid, it
# will be added to the variable returned by this function.
#
# If VERBATIM is passed as argument, the flag is passed onto the linker without prepending the 'LINKER:' prefix.
#
# Note that this function will use the compiler flags contained <LANG>_try_compile
# ~~~
function(check_link_flags lang out_var)
  cmake_parse_arguments(PARSE_ARGV 2 CLF "VERBATIM" "" "")

  # cmake-lint: disable=R0915,C0103,E1120
  set(_${lang}_link_opts)

  get_target_property(_required_flags ${lang}_try_compile INTERFACE_COMPILE_OPTIONS)
  if(NOT _required_flags)
    unset(_required_flags)
  endif()

  # NB: speed up compilation when using CUDA
  if("${lang}" STREQUAL "CUDA")
    list(GET CMAKE_CUDA_ARCHITECTURES 0 _cuda_arch)
    set(CMAKE_CUDA_ARCHITECTURES ${_cuda_arch})
  endif()

  foreach(_flag_list ${CLF_UNPARSED_ARGUMENTS})
    separate_arguments(_flag_list)

    foreach(_flag ${_flag_list})
      # Drop the first character (most likely either '-' or '/')
      string(SUBSTRING ${_flag} 1 -1 _flag_name)
      string(REGEX REPLACE "^-+" "" _flag_name ${_flag_name})
      string(REGEX REPLACE "[-:/,=]" "_" _flag_name ${_flag_name})
      if(CLF_VERBATIM)
        set(_prefix)
      else()
        set(_prefix "LINKER:")
      endif()

      set(CMAKE_REQUIRED_FLAGS ${_required_flags})
      check_linker_flag(${lang} "${_prefix}${_flag}" ${lang}_linker_has_${_flag_name})
      if(${lang}_linker_has_${_flag_name})
        list(APPEND _${lang}_link_opts "${_prefix}${_flag}")
        break()
      endif()
    endforeach()
  endforeach()

  set(${out_var}
      ${_${lang}_link_opts}
      PARENT_SCOPE)
endfunction()

# ~~~
# Convenience function to test for the existence of some linker flags for a set of languages.
#
# test_linker_option(<name>
#                    LANGS <lang1> [<lang2>...]
#                    FLAGS <flags1> [<flags2>...]
#                    [VERBATIM, NO_MQ_TARGET]
#                    [GENEX <genex>]
#                    [CMAKE_OPTION <option>])
#
# Check that a linker option can be applied to each of the specified languages <langN>. For each set of linker
# options provided in the lists <flagsN>, it will test whether one of the element can be used by the corresponding
# linker. If a flag is valid, it will be added to the language-specific targets (unless a corresponding negating flag
# was specified):
#  - <name>_<LANG> (created if does not exist already)
#  - <LANG>_mindquantum
#  - <LANG>_try_compile
#  - <LANG>_try_compile_flagcheck (only if FLACHECK is passed as argument
#
# If VERBATIM is passed as argument, the flag is passed onto the linker without prepending the 'LINKER:' prefix.
#
# If CMAKE_OPTION is specified, the check will only be performed if <option> is set to a truthful value. Also, the
# linker option will only be enabled if <option> is true.
#
# In addition, regardless of whether a language is enabled or not, this function will set a 3variable in the caller's
# scope named <name>_<LANG> to TRUE/FALSE depending on whether one or more flags were found.
#
# NB: This function calls check_link_flags() internally.
# ~~~
function(test_linker_option name)
  cmake_parse_arguments(PARSE_ARGV 1 TLO "VERBATIM;NO_MQ_TARGET" "CMAKE_OPTION;GENEX" "LANGS;FLAGS")

  if(NOT TLO_LANGS)
    message(FATAL_ERROR "Missing LANGS argument")
  endif()
  if(NOT TLO_FLAGS)
    message(FATAL_ERROR "Missing FLAGS argument")
  endif()

  if(NOT TLO_GENEX)
    set(TLO_GENEX "$<LINK_LANGUAGE:@lang@>")
  else()
    set(TLO_GENEX "$<AND:$<LINK_LANGUAGE:@lang@>,${TLO_GENEX}>")
  endif()

  if(TLO_CMAKE_OPTION)
    set(TLO_GENEX "$<AND:$<BOOL:${TLO_CMAKE_OPTION}>,${TLO_GENEX}>")
  endif()

  # ------------------------------------

  # cmake-lint: disable=C0103
  foreach(lang ${TLO_LANGS})
    set(_has_flags FALSE)
    is_language_enabled(${lang} _enabled)
    if(_enabled)
      if(NOT TARGET ${name}_${lang})
        add_library(${name}_${lang} INTERFACE)
      endif()

      set(_args)
      if(TLO_VERBATIM)
        set(_args "VERBATIM;${_args}")
      endif()
      set(_do_check ON)
      if(NOT "${TLO_CMAKE_OPTION}" STREQUAL "" AND NOT ${${TLO_CMAKE_OPTION}})
        set(_do_check OFF)
      endif()
      if(_do_check)
        check_link_flags(${lang} _${lang}_flags ${_args} ${TLO_FLAGS})
      endif()

      string(CONFIGURE "${TLO_GENEX}" _genex @ONLY)

      if(NOT "${_${lang}_flags}" STREQUAL "")
        set(_has_flags TRUE)

        target_link_options(${name}_${lang} INTERFACE "$<${_genex}:${_${lang}_flags}>")
        if(NOT TCO_NO_MQ_TARGET)
          target_link_options(${lang}_mindquantum INTERFACE "$<${_genex}:${_${lang}_flags}>")
        endif()
      endif()
    endif()

    set(${name}_${lang}
        ${_has_flags}
        PARENT_SCOPE)
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
function(set_output_directory_auto target hint)
  string(TOUPPER ${target} _TARGET)

  if(IN_PLACE_BUILD)
    # Automatically calculate the output directory
    set(${_TARGET}_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/${hint})
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
endfunction()

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

include(find_python_module)

# ==============================================================================
