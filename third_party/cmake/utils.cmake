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
# NB: Some of the functions in this file have been adapted from version found in the MindSpore package.
# ==============================================================================

# lint_cmake: -whitespace/indent,-whitespace/extra

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

include(debug_print)
include(external_targets)

# ==============================================================================
# Setup helper variables

# If the CMake generator is not `make`, then look for it NB: this is potentially used when installing packages in the
# local prefix path

if(CMAKE_GENERATOR MATCHES ".*Makefiles")
  set(_make_exec ${CMAKE_MAKE_PROGRAM})
elseif(NOT WIN32)
  find_program(
    _make_exec
    NAMES make gmake mingw32-make
    DOC "Path to make command" REQUIRED)
endif()

find_package(Git REQUIRED)

# ------------------------------------------------------------------------------

if(NOT JOBS)
  include(ProcessorCount)
  ProcessorCount(JOBS)
endif()
message(STATUS "Calling make using ${JOBS} jobs")

# ------------------------------------------------------------------------------

if(MINDSPORE_CI)
  set(_libs_path_file_dest "${PROJECT_SOURCE_DIR}")
else()
  set(_libs_path_file_dest "${PROJECT_BINARY_DIR}")
endif()
set(_local_libs_path_file "${_libs_path_file_dest}/ld_library_paths.txt")
file(WRITE "${_local_libs_path_file}" "")

# ------------------------------------------------------------------------------
# Allow the user to specify some packages to always built from source locally

if(MQ_FORCE_LOCAL_PKGS)
  string(TOLOWER ${MQ_FORCE_LOCAL_PKGS} _val)
  if("${_val}" STREQUAL "all")
    set(MQ_FORCE_LOCAL_PKGS ON)
  elseif(
    NOT
    ("${_val}" STREQUAL "on"
     OR "${_val}" STREQUAL "true"
     OR "${_val}" STREQUAL "1"
     OR "${_val}" STREQUAL "off"
     OR "${_val}" STREQUAL "false"
     OR "${_val}" STREQUAL "0"))
    string(REPLACE "," ";" MQ_FORCE_LOCAL_PKGS ${MQ_FORCE_LOCAL_PKGS})
    foreach(_name ${MQ_FORCE_LOCAL_PKGS})
      string(TOUPPER ${_name} _name)
      set(MQ_${_name}_FORCE_LOCAL ON)
      message(STATUS "MQ_${_name}_FORCE_LOCAL = ${MQ_${_name}_FORCE_LOCAL}")
    endforeach()
    set(MQ_FORCE_LOCAL_PKGS "")
  endif()
endif()

# ------------------------------------------------------------------------------
# Local prefix path for installing packages from source if we cannot find any suitable versions on the system

if(DEFINED ENV{MQLIBS_CACHE_PATH})
  debug_print(STATUS "Using cache path from MQLIBS_CACHE_PATH environment variable")
  set(_mq_local_prefix "$ENV{MQLIBS_CACHE_PATH}")
elseif(DEFINED ENV{MSLIBS_CACHE_PATH})
  debug_print(STATUS "Using cache path from MSLIBS_CACHE_PATH environment variable")
  set(_mq_local_prefix "$ENV{MSLIBS_CACHE_PATH}") # compatibility with MindSpore CI
elseif(DEFINED ENV{MQLIBS_LOCAL_PREFIX_PATH})
  debug_print(STATUS "Using cache path from MQLIBS_LOCAL_PREFIX_PATH environment variable")
  set(_mq_local_prefix "$ENV{MQLIBS_LOCAL_PREFIX_PATH}")
elseif(DEFINED ENV{MSLIBS_LOCAL_PREFIX_PATH})
  debug_print(STATUS "Using cache path from MSLIBS_LOCAL_PREFIX_PATH environment variable")
  set(_mq_local_prefix "$ENV{MSLIBS_LOCAL_PREFIX_PATH}") # compatibility with MindSpore CI
else()
  debug_print(STATUS "Using default cache path")
  set(_mq_local_prefix ${PROJECT_BINARY_DIR}/.mqlibs)
endif()
message(STATUS "MQ local prefix:  ${_mq_local_prefix}")

if(NOT EXISTS ${_mq_local_prefix})
  file(MAKE_DIRECTORY ${_mq_local_prefix})
endif()

string(FIND ${_mq_local_prefix} " " _whitespace)
if(NOT _whitespace EQUAL -1)
  message(
    WARNING
      [[
    There appear to be some whitespace in the path to the local prefix path. This could lead to some of the
    third-party libraries not installing properly.
  ]])
endif()

# ------------------------------------------------------------------------------
# If desired (e.g. like on CIs), a local server can be used to download the source of packages that need to be built
# locally.

debug_print(STATUS "ENV{MQLIBS_SERVER} = $ENV{MQLIBS_SERVER}")
debug_print(STATUS "ENV{MSLIBS_SERVER} = $ENV{MSLIBS_SERVER}")

if(DEFINED ENV{MQLIBS_SERVER} AND NOT ENABLE_GITEE)
  set(_local_server "$ENV{MQLIBS_SERVER}")
elseif(DEFINED ENV{MSLIBS_SERVER} AND NOT ENABLE_GITEE)
  set(_local_server "$ENV{MSLIBS_SERVER}")
elseif(LOCAL_LIBS_SERVER)
  set(_local_server "${LOCAL_LIBS_SERVER}")
endif()

if(_local_server)
  debug_print(STATUS "Raw local server URL: ${_local_server}")
  if(_local_server MATCHES "(http|https|ssh|ftp)://(.*)")
    set(_local_server_protocol ${CMAKE_MATCH_1})
    set(_local_server ${CMAKE_MATCH_2})
  else()
    set(_local_server_protocol "http")
  endif()

  if(_local_server MATCHES "[ \t\r\n]*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+):([0-9]+)")
    set(_local_server_host ${CMAKE_MATCH_1})
    set(_local_server_port ${CMAKE_MATCH_2})
  elseif(_local_server MATCHES "[ \t\r\n]*([^:]+):([0-9]+)")
    set(_local_server_host ${CMAKE_MATCH_1})
    set(_local_server_port ${CMAKE_MATCH_2})
  else()
    set(_local_server_host ${_local_server})
    set(_local_server_port 8081)
  endif()

  set(_local_server "${_local_server_protocol}://${_local_server_host}:${_local_server_port}")

  message(STATUS "Local server for third-party libraries: ${_local_server}")

  if(NOT ENV{no_proxy})
    set(ENV{no_proxy} "${_local_server_host}")
  else()
    string(FIND $ENV{no_proxy} ${_local_server_host} IP_POS)
    if(${IP_POS} EQUAL -1)
      set(ENV{no_proxy} "$ENV{no_proxy},${_local_server_host}")
    endif()
  endif()
endif()

# ==============================================================================
# Helper functions for argument parsing and checking

# ~~~
# Check that only one of the values passed as argument evaluates to TRUE
#  __check_only_one_of(<prefix> <var> [... <var>])
# ~~~
function(__check_only_one_of prefix)
  set(_value 0)
  set(_value_list)
  set(_err_msg "")
  foreach(_arg ${ARGN})
    list(APPEND _value_list "${_arg_for_print}")
    string(REPLACE "${prefix}_" "" _arg_for_print "${_arg}")
    if(${_arg})
      set(_err_msg "${_err_msg}\n  - ${_arg_for_print} = ${${_arg}}")
      math(EXPR _value "${_value} + 1" OUTPUT_FORMAT DECIMAL)
    endif()
  endforeach()

  string(JOIN "," _value_list ${_value_list})
  if(NOT _value LESS_EQUAL 1)
    message(FATAL_ERROR "Cannot specify more than one of ${_value_list}.\n" "Reason for error:${_err_msg}")
  endif()
endfunction()

# ==============================================================================
# Functions to download a package from a URL or a GIT repository

# Wrapper around the FetchContent_XXX macros
macro(__do_fetch_content pkg_name pkg_url)
  FetchContent_GetProperties(${pkg_name})
  message(STATUS "Fetching content for ${pkg_name} using ${pkg_url} in ${${pkg_name}_SOURCE_DIR}")

  if(NOT ${pkg_name}_POPULATED)
    FetchContent_Populate(${pkg_name})
    string(TOLOWER ${pkg_name} _pkg_name)
    set(${pkg_name}_SOURCE_DIR
        ${${_pkg_name}_SOURCE_DIR}
        PARENT_SCOPE)
  endif()
endmacro()

# Fetch some content by downloading an archive
function(__download_pkg pkg_name pkg_url pkg_md5)
  debug_print(STATUS "URL = ${pkg_url}")
  if(_local_server)
    get_filename_component(_url_file_name ${pkg_url} NAME)
    set(pkg_url "${_local_server}/libs/${pkg_name}/${_url_file_name}")
    debug_print(STATUS "Using local server URL: ${pkg_url}")
  endif()

  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(_opts DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  endif()

  set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
  FetchContent_Declare(
    ${pkg_name}
    ${_opts}
    URL ${pkg_url}
    URL_HASH MD5=${pkg_md5})

  __do_fetch_content(${pkg_name} ${pkg_url})
endfunction()

# Fetch some content by downloading a Git repository (or from an archive on the local server with a specific commit)
function(__download_pkg_with_git pkg_name pkg_url pkg_git_commit pkg_md5)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(_opts DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  endif()
  set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)

  debug_print(STATUS "GIT_REPOSITORY = ${pkg_url}")
  debug_print(STATUS "GIT_TAG = ${pkg_git_commit}")
  if(_local_server)
    set(pkg_url "${_local_server}/libs/${pkg_name}/${pkg_git_commit}")
    debug_print(STATUS "Using local server URL: ${pkg_url}")
    FetchContent_Declare(
      ${pkg_name}
      ${_opts}
      URL ${pkg_url}
      URL_HASH MD5=${pkg_md5})
  else()
    FetchContent_Declare(
      ${pkg_name}
      GIT_REPOSITORY ${pkg_url}
      GIT_TAG ${pkg_git_commit})
  endif()

  __do_fetch_content(${pkg_name} ${pkg_url})
endfunction()

# ==============================================================================
# Some helper functions

# Function to check that patches are valid
function(__check_patches pkg_patches)
  # check patches
  if(pkg_patches)
    file(TOUCH ${_mq_local_prefix}/${pkg_name}_patch.md5)
    file(READ ${_mq_local_prefix}/${pkg_name}_patch.md5 ${pkg_name}_PATCHES_MD5)

    message(STATUS "patches MD5:${${pkg_name}_PATCHES_MD5}")

    set(${pkg_name}_PATCHES_NEW_MD5)
    foreach(_patch ${pkg_patches})
      file(MD5 ${_patch} _patch_md5)
      set(${pkg_name}_PATCHES_NEW_MD5 "${${pkg_name}_PATCHES_NEW_MD5},${_patch_md5}")
    endforeach()

    if(NOT "${${pkg_name}_PATCHES_MD5}" STREQUAL "${${pkg_name}_PATCHES_NEW_MD5}")
      set(${pkg_name}_PATCHES ${pkg_patches})
      file(REMOVE_RECURSE "${_mq_local_prefix}/${pkg_name}-subbuild")
      file(WRITE ${_mq_local_prefix}/${pkg_name}_patch.md5 ${${pkg_name}_PATCHES_NEW_MD5})
      message(STATUS "patches changed : ${${pkg_name}_PATCHES_NEW_MD5}")
    endif()
  endif()
endfunction()

# ------------------------------------------------------------------------------

# ~~~
# Extract a number of elements from a list
#
# __extract_n_args(<start_idx> <stop_idx> <var_name>)
# ~~~
function(__extract_n_args start stop var)
  # cmake-lint: disable=E1120
  set(num_range)
  foreach(idx RANGE ${start} ${stop})
    list(APPEND num_range ${idx})
  endforeach()
  list(GET ARGN ${num_range} _value)
  set(${var}
      ${_value}
      PARENT_SCOPE)
endfunction()

# ~~~
# Create target aliases based on a list of either
#  1. tuple of key-value pairs where:
#      + key: target alias name
#      + value: target to alias
#  2. triplets of the form (number, values) where:
#      + number: size of values list
#      + values: list of values [alias, targets]
#         * alias: target alias name
#         * targets: list of potential targets to alias
#
# Using 1. syntax:
# __create_target_aliases(<pkg_name> <skip_in_install_for_config> [[<tgt_alias>, <tgt_name>]...])
# Using 2. syntax:
# __create_target_aliases(<pkg_name> <skip_in_install_for_config> [[<N> <tgt_alias>, <tgt_name_1>...<tgt_name_N>]...])
#
# Examples:
# __create_target_aliases(FALSE A B 3 D E F)
#   -> create alias A -> B, and either D -> E or D -> F
#
# NB: the <skip_in_install_for_config> argument only prevents this function from creating the alias targets into the
#     installation configuration file but has otherwise no effect.
# ~~~
function(__create_target_aliases pkg_name skip_in_install_config)
  # cmake-lint: disable=R0915,R0912
  list(LENGTH ARGN n_args)
  if(NOT n_args)
    return()
  endif()

  message(STATUS "Creating mindspore target aliases")
  list(APPEND CMAKE_MESSAGE_INDENT "  ")

  set(idx 0)
  while(idx LESS n_args)
    set(incr 2)
    list(GET ARGN ${idx} _arg)
    if(_arg MATCHES "^[0-9]+$")
      set(incr ${_arg})
      math(EXPR incr "${incr} + 1") # NB: we need to count _arg itself for shifting
      math(EXPR start "${idx} + 1")
    else()
      set(start ${idx})
    endif()
    math(EXPR stop "${idx} + ${incr} -1")
    __extract_n_args(${start} ${stop} _args ${ARGN})
    math(EXPR idx "${idx} + ${incr}")

    # ----------------------------------

    list(POP_FRONT _args tgt_alias)
    set(tgt_list ${_args})

    set(tgt_name)
    foreach(_tgt ${tgt_list})
      if(NOT TARGET ${_tgt})
        message(STATUS "${_tgt} does not exist... -> skipping")
      else()
        message(STATUS "${_tgt} exists -> using it")
        set(tgt_name ${_tgt})
        break()
      endif()
    endforeach()
    if("${tgt_name}" STREQUAL "")
      message(FATAL_ERROR "None of ${tgt_list} targets can be found, not defining ${tgt_alias} alias")
    endif()

    get_target_property(_aliased ${tgt_name} ALIASED_TARGET)
    if(_aliased)
      set(tgt_name ${_aliased})
    endif()

    get_target_property(_imported ${tgt_name} IMPORTED)
    get_target_property(_imported_global ${tgt_name} IMPORTED_GLOBAL)
    if(_imported AND NOT _imported_global)
      set_property(TARGET ${tgt_name} PROPERTY IMPORTED_GLOBAL TRUE)
    endif()

    if(_imported)
      __get_library_imported_location(${tgt_name} _imported_location)
      if(_imported_location)
        foreach(_config RELEASE DEBUG MINSIZEREL RELWITHDEBINFO)
          get_target_property(_config_location ${tgt_name} IMPORTED_LOCATION_${_config})
          if(NOT _config_location)
            debug_print(STATUS "Propagating ${_imported_location} to IMPORTED_LOCATION_${_config}")
            set_target_properties(${tgt_name} PROPERTIES IMPORTED_LOCATION_${_config} ${_imported_location})
          endif()
        endforeach()
      endif()
    endif()

    get_target_property(_aliased ${tgt_name} ALIASED_TARGET)
    if(_aliased)
      set(tgt_name ${_aliased})
    endif()

    get_target_property(_type ${tgt_name} TYPE)
    list(APPEND _target_aliases "if(TARGET ${tgt_name} AND NOT TARGET ${tgt_alias})")
    if("${_type}" STREQUAL "EXECUTABLE")
      add_executable(${tgt_alias} ALIAS ${tgt_name})
      list(APPEND _target_aliases "  add_executable(${tgt_alias} ALIAS ${tgt_name})")
    else()
      add_library(${tgt_alias} ALIAS ${tgt_name})
      list(APPEND _target_aliases "  add_library(${tgt_alias} ALIAS ${tgt_name})")
    endif()
    list(APPEND _target_aliases "endif()")
    message(STATUS "Creating alias target: ${tgt_alias} -> ${tgt_name}")
  endwhile()

  if(NOT skip_in_install_config)
    load_from_cache(_${pkg_name}_find_pkg_str _tmp)
    list(APPEND _tmp ${_target_aliases})
    store_in_cache(_${pkg_name}_find_pkg_str "${_tmp}")
  endif()

  list(POP_BACK CMAKE_MESSAGE_INDENT)
endfunction()

# ------------------------------------------------------------------------------

# ~~~
# Append some target properties to an (imported) target.
#
# <type> must be one of:
#   - COMPILE_DEFINITIONS
#   - COMPILE_OPTIONS
#   - INCLUDE_DIRECTORIES
#   - LINK_LIBRARIES
#   - LINK_OPTIONS
#
# __append_target_properties(<type> [TARGET <tgt_name> [<properties>, ...]] ...)
#
# Examples:
# __append_target_properties(COMPILE_DEFINITIONS TARGET mylib::mylib HAS_CXX20)
#   -> effectively call target_compile_definitions(mylib::mylib INTERFACE HAS_CXX20)
# ~~~
function(__append_target_properties type)
  list(LENGTH ARGN n_args)
  if(NOT n_args)
    return()
  endif()

  # Minimum is 3, because at least `TARGET` `<tgt_name>` and one `<property>`
  if(n_args LESS 3)
    message(FATAL_ERROR "Number of arguments to __append_target_property() must be at least 3!")
  endif()

  set(idx 0)
  while(idx LESS n_args)
    list(GET ARGN ${idx} _arg)
    if(NOT "${_arg}" STREQUAL "TARGET")
      message(
        FATAL_ERROR "Argument ${idx} of __append_target_property needs to be `TARGET` followed by a valid target name."
                    " Got: ${_arg}")
    elseif(NOT idx LESS n_args)
      message(FATAL_ERROR "`TARGET` needs to be followed by a target name!")
    endif()
    math(EXPR idx "${idx} + 1")
    list(GET ARGN ${idx} tgt_name)
    math(EXPR idx "${idx} + 1")
    if(NOT idx LESS n_args)
      message(FATAL_ERROR "Need to have at least one property!")
    elseif(NOT TARGET "${tgt_name}")
      message(FATAL_ERROR "${tgt_name} is not a valid target name!")
    endif()
    get_target_property(_is_aliased "${tgt_name}" ALIASED_TARGET)
    if(_is_aliased)
      message(FATAL_ERROR "${tgt_name} is an alias target! Please use ${_is_aliased} instead.")
    endif()

    set(_properties)
    while(idx LESS n_args)
      list(GET ARGN ${idx} _arg)
      if("${_arg}" STREQUAL "TARGET")
        break()
      endif()
      list(APPEND _properties "${_arg}")
      math(EXPR idx "${idx} + 1")
    endwhile()

    message(STATUS "Modifying INTERFACE_${type} property of ${tgt_name}")

    get_target_property(_data ${tgt_name} INTERFACE_${type})
    if(_data)
      debug_print(STATUS "  read ${_data}")
      list(APPEND _data ${_properties})
      list(REMOVE_DUPLICATES _data)
    else()
      debug_print(STATUS "  property is originally empty!")
      set(_data "${_properties}")
    endif()
    debug_print(STATUS "  writing ${_data}")
    set_target_properties(${tgt_name} PROPERTIES INTERFACE_${type} "${_data}")
  endwhile()
endfunction()

# ------------------------------------------------------------------------------

# ~~~
# Find the largest common prefix of two strings
#
# __largest_common_prefix(<a> <b> <var>)
# ~~~
function(__largest_common_prefix str_a str_b var)
  # cmake-lint: disable=E1120
  string(LENGTH "${str_a}" _len_a)
  string(LENGTH "${str_b}" _len_b)

  if(${_len_a} LESS ${_len_b})
    set(_len ${_len_a})
  else()
    set(_len ${_len_b})
  endif()

  # iterate over the length
  foreach(end RANGE 1 ${_len})
    # get substrings
    string(SUBSTRING "${str_a}" 0 ${end} sub_a)
    string(SUBSTRING "${str_b}" 0 ${end} sub_b)

    if("${sub_a}" STREQUAL "${sub_b}")
      set(${var}
          ${sub_a}
          PARENT_SCOPE)
    else()
      break()
    endif()
  endforeach()
endfunction()

# ==============================================================================

# ~~~
# Extract imported location from an interface library
#
# __get_imported_location_from_interface_library(<target> <out-var>)
# ~~~
function(__get_imported_location_from_interface_library tgt var)
  # cmake-lint: disable=E1126
  set(_basename ${tgt})
  if(tgt MATCHES "([a-zA-Z0-9_]+)::([a-zA-Z0-9_]+)")
    set(_basename ${CMAKE_MATCH_2})
    string(TOLOWER ${CMAKE_MATCH_2} _basename_lower)
  endif()

  set(_imported_location _imported_location-NOTFOUND)
  get_target_property(_libs ${tgt} INTERFACE_LINK_LIBRARIES)
  if(_libs)
    foreach(_lib ${_libs})
      if(_lib MATCHES ".*${_basename}.*" OR _lib MATCHES ".*${_basename_lower}.*")
        set(_imported_location "${_lib}")
        real_path("${_imported_location}" _imported_location)
        break()
      endif()
    endforeach()
  endif()
  set(${var}
      "${_imported_location}"
      PARENT_SCOPE)
endfunction()

# ------------------------------------------------------------------------------

# ~~~
# Get the imported location of a library
#
# __get_library_imported_location(<target> <out-var>)
# ~~~
function(__get_library_imported_location tgt var)
  # cmake-lint: disable=E1126
  foreach(_prop IMPORTED_LOCATION IMPORTED_LOCATION_DEBUG IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION_NONE
                IMPORTED_LOCATION_NOCONFIG)
    get_target_property(_location ${tgt} ${_prop})
    if(_location)
      real_path("${_location}" _location)
      set(${var}
          ${_location}
          PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${var}
      ""
      PARENT_SCOPE)
endfunction()

# ------------------------------------------------------------------------------

# ~~~
# Get the real type of a library
#
# __get_library_real_type(<filename> <out-var>)
# ~~~
function(__get_library_real_type filename var)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
    cmake_path(GET filename EXTENSION _ext)
  else()
    get_filename_component(_ext "${filename}" EXT)
  endif()

  if(_ext MATCHES "${CMAKE_SHARED_LIBRARY_SUFFIX}(\\.[0-9]+)*")
    set(${var}
        "SHARED_LIBRARY"
        PARENT_SCOPE)
  elseif(_ext MATCHES "${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(${var}
        "STATIC_LIBRARY"
        PARENT_SCOPE)
  else()
    set(${var}
        "UNKNOWN_LIBRARY"
        PARENT_SCOPE)
  endif()
endfunction()

# ------------------------------------------------------------------------------

# ~~~
# Generate a "fake" CMake config file for packages not built using CMake
#
# __generate_pseudo_cmake_package_config(<dest_dir> <pkg_name> <pkg_namespace> <components>)
# ~~~
function(__generate_pseudo_cmake_package_config root_dir pkg_name pkg_namespace comp_list)
  # cmake-lint: disable=R0912,R0915,W0106,E1126
  set(dest_dir "${root_dir}/${pkg_name}/share/${pkg_name}")
  set(tgt_list)
  foreach(_comp ${comp_list})
    if(TARGET ${pkg_namespace}::${_comp})
      list(APPEND tgt_list ${pkg_namespace}::${_comp})
    endif()
    if(TARGET ${pkg_namespace}::${PKG_NS_NAME}_${_comp})
      list(APPEND tgt_list ${pkg_namespace}::${PKG_NS_NAME}_${_comp})
    endif()
  endforeach()

  # ----------------------------------------------------------------------------

  file(MAKE_DIRECTORY ${dest_dir})
  string(REPLACE ";" "\;" tgt_list_escaped "${tgt_list}")

  # ----------------------------------------------------------------------------

  set(_config_version_content
      "set(PACKAGE_VERSION \"${PKG_VER}\")"
      [[

if(PACKAGE_FIND_VERSION_RANGE)
  # Package version must be in the requested version range
  if(("${PACKAGE_FIND_VERSION_RANGE_MIN}" STREQUAL "INCLUDE" AND PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION_MIN)
      OR (("${PACKAGE_FIND_VERSION_RANGE_MAX}" STREQUAL "INCLUDE"
             AND PACKAGE_VERSION VERSION_GREATER PACKAGE_FIND_VERSION_MAX)
           OR ("${PACKAGE_FIND_VERSION_RANGE_MAX}" STREQUAL "EXCLUDE"
                 AND PACKAGE_VERSION VERSION_GREATER_EQUAL PACKAGE_FIND_VERSION_MAX)))
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
  else()
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
  endif()
else()
  if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
  else()
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
    if("${PACKAGE_FIND_VERSION}" STREQUAL "${PACKAGE_VERSION}")
      set(PACKAGE_VERSION_EXACT TRUE)
    endif()
  endif()
endif()
]])

  string(REPLACE ";" "\n" _config_version_content ${_config_version_content})
  file(WRITE ${dest_dir}/${pkg_name}ConfigVersion.cmake ${_config_version_content})

  # ----------------------------------------------------------------------------

  set(_targets_prefix_content
      [[
cmake_policy(PUSH)
cmake_policy(VERSION 2.6...3.20)

set(CMAKE_IMPORT_FILE_VERSION 1)

]])

  set(_targets_suffix_content
      [[

if(CMAKE_VERSION VERSION_LESS 3.0.0)
  message(FATAL_ERROR "This file relies on consumers using CMake 3.0.0 or greater.")
endif()

set(CMAKE_IMPORT_FILE_VERSION)
cmake_policy(POP)
]])

  foreach(_tgt ${tgt_list})
    if(NOT TARGET ${_tgt})
      continue()
    endif()

    get_target_property(_libs ${_tgt} INTERFACE_LINK_LIBRARIES)
    if(_libs)
      foreach(_lib ${_libs})
        if(_lib MATCHES "([a-zA-Z0-9_]+)::([a-zA-Z0-9_]+)")
          set(_ns ${CMAKE_MATCH_1})
          set(_tgt ${CMAKE_MATCH_2})

          if("${_ns}" STREQUAL "${ns}" AND TARGET ${_lib})
            list(PREPEND tgt_list ${_lib})
          endif()
        endif()
      endforeach()
    endif()
  endforeach()

  set(_tgt_props
      COMPILE_DEFINITIONS
      COMPILE_FEATURES
      COMPILE_OPTIONS
      INCLUDE_DIRECTORIES
      LINK_OPTIONS
      LINK_LIBRARIES
      INTERFACE_COMPILE_DEFINITIONS
      INTERFACE_COMPILE_FEATURES
      INTERFACE_COMPILE_OPTIONS
      INTERFACE_INCLUDE_DIRECTORIES
      INTERFACE_LINK_OPTIONS
      INTERFACE_LINK_LIBRARIES
      IMPORTED_CONFIGURATIONS
      IMPORTED_SONAME
      IMPORTED_SONAME_DEBUG
      IMPORTED_SONAME_RELEASE
      IMPORTED_IMPLIB
      IMPORTED_IMPLIB_DEBUG
      IMPORTED_IMPLIB_RELEASE
      IMPORTED_IMPLIB_NONE
      IMPORTED_IMPLIB_NOCONFIG
      IMPORTED_LOCATION
      IMPORTED_LOCATION_DEBUG
      IMPORTED_LOCATION_RELEASE
      IMPORTED_LOCATION_NONE
      IMPORTED_LOCATION_NOCONFIG)

  real_path("${root_dir}" root_dir_absolute)
  set(_library_content)

  set(_include_dirs)
  set(_libraries)
  set(_libraries_debug)
  set(_imported_targets)
  foreach(_tgt ${tgt_list})
    if(NOT TARGET ${_tgt})
      continue()
    endif()

    list(APPEND _imported_targets ${_tgt})
    list(APPEND _library_content "if(NOT TARGET ${_tgt})\n")

    get_target_property(_type ${_tgt} TYPE)

    # If the CMake library type is UNKNOWN, try to guess it
    if("${_type}" STREQUAL "UNKNOWN_LIBRARY")
      __get_library_imported_location(${_tgt} _location)
      __get_library_real_type("${_location}" _type)
    elseif("${_type}" STREQUAL "INTERFACE_LIBRARY")
      __get_imported_location_from_interface_library(${_tgt} _imported_location)
      if(_imported_location)
        __get_library_real_type("${_imported_location}" _type)
      endif()
    endif()

    if("${_type}" STREQUAL "EXECUTABLE")
      list(APPEND _library_content "  add_executable(${_tgt} IMPORTED)\n")
    elseif("${_type}" STREQUAL "UNKNOWN_LIBRARY")
      list(APPEND _library_content "  add_library(${_tgt} UNKNOWN IMPORTED)\n")
    elseif("${_type}" STREQUAL "INTERFACE_LIBRARY")
      list(APPEND _library_content "  add_library(${_tgt} INTERFACE IMPORTED)\n")
    elseif("${_type}" STREQUAL "STATIC_LIBRARY")
      list(APPEND _library_content "  add_library(${_tgt} STATIC IMPORTED)\n")
    elseif("${_type}" STREQUAL "SHARED_LIBRARY")
      list(APPEND _library_content "  add_library(${_tgt} SHARED IMPORTED)\n")
    endif()

    if(_imported_location)
      list(APPEND _libraries "${_imported_location}")
      list(APPEND _library_content "\n  set_target_properties(${_tgt} PROPERTIES\n")
      list(APPEND _library_content "      IMPORTED_LOCATION \"${_imported_location}\"")
      list(APPEND _library_content "  )\n")
    endif()

    list(APPEND _library_content "\n  set_target_properties(${_tgt} PROPERTIES\n")
    foreach(_prop ${_tgt_props})
      get_target_property(_val ${_tgt} ${_prop})
      if(_val)
        set(_resolved_var)
        foreach(_value ${_val})
          if(_value MATCHES [[\$<BUILD_INTERFACE:[^>]*>]])
            continue()
          endif()
          if(_value MATCHES [[\$<INSTALL_INTERFACE:([^>]*)>]])
            set(_value "\${_IMPORT_PREFIX}/${CMAKE_MATCH_1}")
          endif()
          if(EXISTS "${_value}")
            real_path("${_value}" _value)
            string(REPLACE "${root_dir_absolute}" "\${_IMPORT_PREFIX}" _value "${_value}")
          endif()
          list(APPEND _resolved_var "${_value}")
        endforeach()
        set(_val ${_resolved_var})

        if(_imported_location)
          list(REMOVE_ITEM _val "${_imported_location}")
        endif()

        if(NOT "${_val}" STREQUAL "")
          string(REPLACE ";" "\;" _val "${_val}")
          list(APPEND _library_content "       ${_prop} \"${_val}\"\n")
          if(_prop STREQUAL "INCLUDE_DIRECTORIES" OR _prop STREQUAL "INTERFACE_INCLUDE_DIRECTORIES")
            list(APPEND _include_dirs "${_val}")
          endif()
          if(_prop STREQUAL "IMPORTED_LOCATION"
             OR _prop STREQUAL "IMPORTED_LOCATION_RELEASE"
             OR _prop STREQUAL "IMPORTED_LOCATION_NOCONFIG"
             OR _prop STREQUAL "IMPORTED_LOCATION_NONE")
            list(APPEND _libraries "${_val}")
          endif()
          if(_prop STREQUAL "IMPORTED_LOCATION_DEBUG")
            list(APPEND _libraries_debug "${_val}")
          endif()
        endif()
      endif()
    endforeach()
    list(APPEND _library_content "  )\n\n\n")

    list(APPEND _library_content "endif()\n")
  endforeach()

  if(_${pkg}_COMPONENTS_SEARCHED)
    string(REPLACE ";" "\;" _val "${_${pkg}_COMPONENTS_SEARCHED}")
    list(APPEND _library_content "set(_${pkg}_COMPONENTS_SEARCHED ${_val})\n\n\n")
  endif()

  file(WRITE ${dest_dir}/${pkg_name}Targets.cmake ${_targets_prefix_content} ${_library_content}
       ${_targets_suffix_content})

  # ----------------------------------------------------------------------------

  list(REMOVE_DUPLICATES _libraries_debug)
  if(NOT _libraries)
    list(APPEND _libraries ${_libraries_debug})
  endif()
  list(REMOVE_DUPLICATES _include_dirs)
  list(REMOVE_DUPLICATES _libraries)
  string(REPLACE ";" "\;" _imported_targets "${_imported_targets}")
  string(REPLACE ";" "\;" _include_dirs "${_include_dirs}")
  string(REPLACE ";" "\;" _libraries "${_libraries}")
  set(_config_content
      [[
get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
if("${_IMPORT_PREFIX}" STREQUAL "/")
  set(_IMPORT_PREFIX "")
endif()
]]
      "include(\"\${CMAKE_CURRENT_LIST_DIR}/${pkg_name}Targets.cmake\")\n"
      "set(${pkg_name}_INCLUDE_DIRS \"${_include_dirs}\")\n"
      "set(${pkg_name}_LIBRARIES \"${_libraries}\")\n"
      "set(${pkg_name}_IMPORTED_TARGETS \"${_imported_targets}\")\n")

  file(WRITE ${dest_dir}/${pkg_name}Config.cmake ${_config_content})
endfunction()

# ==============================================================================

# ~~~
# Find a package.
#
# In practice this simply calls find_package() and prints out some additional information to the console
#
# __find_package(<pkg_name>
#                [SEARCH_NAME <search_name>]
#                [<arg1> <arg2> ... <argn>])
#
# <SEARCH_NAME> is used to print an informational message to distinguish between different find_package() calls.
# ~~~
macro(__find_package pkg_name) # cmake-lint: disable=R0913
  cmake_parse_arguments(FP "" "SEARCH_NAME" "" ${ARGN})

  message(CHECK_START "Looking ${pkg_name} using CMake find_package(): ${FP_SEARCH_NAME}")

  list(APPEND CMAKE_MESSAGE_INDENT "  ")

  string(TOUPPER ${pkg_name} PKG_NAME)

  # Prefer system installed libraries instead of compiling everything from source
  debug_print(STATUS "find_package(${pkg_name} ${FP_UNPARSED_ARGUMENTS})")

  find_package(${pkg_name} ${FP_UNPARSED_ARGUMENTS})
  if(${pkg_name}_FOUND)
    list(POP_BACK CMAKE_MESSAGE_INDENT)
    message(CHECK_PASS "Done")
  else()
    list(POP_BACK CMAKE_MESSAGE_INDENT)
    message(CHECK_FAIL "Failed")
  endif()
endmacro()

# ------------------------------------------------------------------------------

# ~~~
# Make sure that a package was located in a specific root directory
#
# __check_package_location(<base_dir> <pkg_name> <pkg_namespace> [<component>, ...])
#
# This function will look at the IMPORTED_LOCATION* properties of each of the <pkg_namespace>::<component> targets and
# make sure that it is located within the local installation prefix directory.
#
# <pkg_name> is used to locate the local installation prefix by looking at the value of <pkg_name>_BASE_DIR CMake
# variable.
# ~~~
function(__check_package_location pkg_name pkg_namespace)
  file(TO_CMAKE_PATH "${${pkg_name}_BASE_DIR}" _base_dir)
  set(_tgt_list)
  foreach(_comp ${ARGN})
    set(_tgt ${pkg_namespace}::${_comp})
    if(TARGET ${_tgt})
      foreach(_prop IMPORTED_LOCATION IMPORETD_LOCATION_DEBUG IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION_NONE
                    IMPORTED_LOCATION_NOCONFIG)
        get_target_property(_imported_location ${_tgt} ${_prop})
        if(_imported_location)
          file(TO_CMAKE_PATH "${_imported_location}" _imported_location)
          break()
        endif()
      endforeach()
      if(_imported_location)
        string(FIND "${_imported_location}" "${_base_dir}" _idx)
        if(_idx LESS 0)
          message(
            FATAL_ERROR
              "Imported location of ${_tgt} not found in ${_base_dir}!
- ${_imported_location}
Please clear up CMake cache (${PROJECT_BINARY_DIR}/CMakeCache.txt and re-run CMake.")
        endif()
      endif()
    endif()
  endforeach()
endfunction()

# ------------------------------------------------------------------------------

# ~~~
# Make imported targets global (also make alias global if possible)
#
# __make_target_global(<pkg_namespace> [<target>, ...])
#
# Iterate through each of the <pkg_namespace>::<target> targets and make them either ALIAS_GLOBAL or IMPORTED_GLOBAL
# depending on their target type.
# ~~~
function(__make_target_global pkg_namespace)
  foreach(_tgt ${ARGN})
    set(_tgt ${pkg_namespace}::${_tgt})

    get_target_property(_aliased ${_tgt} ALIASED_TARGET)
    if(_aliased)
      if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        set_property(TARGET ${_tgt} PROPERTY ALIAS_GLOBAL TRUE)
      endif()
      set(_tgt ${_aliased})
    endif()

    set_property(TARGET ${_tgt} PROPERTY IMPORTED_GLOBAL TRUE)
  endforeach()
endfunction()

# ==============================================================================

# ~~~
# Setup the 3rd-party library for the CMake install target
#
# __setup_install_target(<pkg_name>)
#
# This function appends some pre-formatted strings to the mq_external_find_packages global CMake property.
#
# Those strings may contain references to MQ_3RDPARTY_PREFIX_PATH in the form of @MQ_3RDPARTY_PREFIX_PATH@ references
# that must be substituted by calling string(CONFIGURE) at a later time.
#
# This function also saves the strings content into the CMake cache at _<pkg_name>_find_pkg_str.
# ~~~
function(__setup_install_target pkg_name)
  set(_generate_data TRUE)
  if(_${pkg_name}_find_pkg_str)
    string(MD5 _hash "${ARGN}")
    if("${_hash}" STREQUAL "${_${pkg_name}_find_pkg_str_control_hash}")
      set(_generate_data FALSE)
    endif()
  endif()

  if(_generate_data)
    set(_find_pkg_str)
    set(_find_pkg_args ${ARGN} REQUIRED)
    if(_${pkg_name}_SYSTEM)
      list(APPEND _find_pkg_str "# ${pkg_name} (system)")
    else()
      list(APPEND _find_pkg_str "# ${pkg_name} (local)")
      set(_args_patched)
      foreach(_value ${_find_pkg_args})
        if(_value MATCHES "^${_mq_local_prefix}/(.*)")
          cmake_path(RELATIVE_PATH _value BASE_DIRECTORY "${_mq_local_prefix}")
          list(APPEND _args_patched "\"@MQ_3RDPARTY_PREFIX_PATH@/${_value}\"")
        else()
          list(APPEND _args_patched "${_value}")
        endif()
      endforeach()
      set(_find_pkg_args ${_args_patched})
      # NB: to filter static libraries, use  (... REGEX [[.*\.(a|lib)$]] EXCLUDE) below
      install(DIRECTORY ${${pkg_name}_BASE_DIR} DESTINATION ${MQ_INSTALL_3RDPARTYDIR})
    endif()
    set(_tmp)
    string(REPLACE ";" ";             " _tmp "${_find_pkg_args}")
    list(APPEND _find_pkg_str "find_package(${_tmp})")

    store_in_cache(_${pkg_name}_find_pkg_str "${_find_pkg_str}")
  endif()

  append_to_property(mq_external_packages GLOBAL ${pkg_name})
endfunction()

# ==============================================================================
# ==============================================================================

# ~~~
# Add an external dependency
#
# mindquantum_add_pkg(<pkg_name>
#                     # Mandatory options
#                     [VER <version-num>]
#                     [MD5 <archive-md5>]
#                     [[GIT_REPOSITORY <git-url>] [GIT_TAG <tag>]  |
#                      [URL <archive-url>]]
#                     [LIBS <lib-names> [... <lib-names>]]
#                     [EXE <exec-name>]
#
#                     # General flags/options
#                     [CMAKE_PKG_NO_COMPONENTS, FORCE_CONFIG_SEARCH, FORCE_EXACT_VERSION,
#                        FORCE_LOCAL_PKG, SKIP_IN_INSTALL_CONFIG]
#                     [BUILD_DEPENDENCIES <package> [... <package>]]
#                     [LANGS <lang> [... <lang>]]
#                     [LOCAL_EXTRA_DEFINES [TARGET <target> <defines> [... <defines>]]...]
#                     [PATCHES <patch-file> [... <patch-file>]]
#                     [SYSTEM_EXTRA_DEFINES [TARGET <target> <defines> [... <defines>]]...]
#                     [TARGET_ALIAS <alias-name> <target-library>|
#                        TARGET_ALIAS <num> <alias-name> <target-library> [...<target-library>]]
#
#                     <other-options>)
#
# The purpose of this function is to manage the download, configuration and build of third-party libraries while
# executing the CMake configure step for MindQuantum. This function will always attempt to locate the third-party
# library on the system (unless otherwise specified) before downloading and compiling the third-party locally. If
# compiled locally, the third-party will be installed within the local installation prefix.
#
# The actual location of the installation prefix defaults to `.mqlibs` inside the build directory, but may be influenced
# by specifying one of the following environment variables:
#   - `MQLIBS_CACHE_PATH`
#   - `MSLIBS_CACHE_PATH`
#   - `MQLIBS_LOCAL_PREFIX_PATH`
#   - `MSLIBS_LOCAL_PREFIX_PATH`
#
# The minimum amount of arguments you need to specify are either:
#   - <pkg_name>, <VER>, <URL>, <MD5>
#   - <pkg_name>, <VER>, <GIT_REPOSITORY>, <GIT_TAG> *
#
# * You might need to specify <MD5> as well for the Git case. More information about that below or on the wiki on Gitee.
#
# and at least one element for either <EXE> or <LIBS> to specify which compiled executable or library you are looking
# for within the build third-party library. The function will then checkout the code, either by downloading an archive
# or checking out a Git repository, and then start building the third-party library given the instructions specified.
#
# By specifying <TARGET_ALIAS> argument you may define some alias targets. There are two valid syntax for this argument:
#   - TARGET_ALIAS <alias-name> <target-library>
#   - TARGET_ALIAS <num> <alias-name> <target-library> [... <target-library>]
# In the second case, `<num>` needs to be the number arguments to consume. The function will then try them one by one
# until it finds a valid target.
#
# <BUILD_DEPENDENCIES> is a list of lists of arguments to pass onto to `find_package()` prior to start building the
# package.
# e.g. (... BUILD_DEPENDENCIES "Git REQUIRED" "Boost COMPONENTS system" ...)
#
# Use <CMAKE_PKG_NO_COMPONENTS>, <FORCE_CONFIG_SEARCH> and <FORCE_EXACT_VERSION> to customize how the function looks for
# the third-party library using find_package(). The former makes sure that no `COMPONENTS` argument gets added to the
# call while the latter two add either `CONFIG` or `EXACT` to the list of arguments to find_package().
#
# You can force a particular third-party library to *always* be compiled locally by specifying <FORCE_LOCAL_PKG>.
#
# <LANG> is a list of languages to activate by default for a third-party library. This defaults to C++ (and C if
# enabled).
#
# <LOCAL_EXTRA_DEFINES> can be used to set some additional COMPILE_DEFINITIONS in the case the specified target is built
# locally (as opposed to the case where the package is found as a system library)
#
# <PATCHES> can be used to specify a list of patches to apply to the unpacked source code before any other build steps
# is run. Once a patch has been applied once, a checksum of the patch is stored inside of the source directory and
# subsequent rebuilds of the third-party library (which may include re-configuration, etc.) will not be run if the patch
# checksum has not changed. If you change a patch the patch application might fail and you will get a CMake error
# message that will let you know how to solve that issue.
#
# The <SKIP_IN_INSTALL_CONFIG> option can be used to prevent certain alias targets of being defined inside MindQuantum's
# installation configuration file.
#
# <SYSTEM_EXTRA_DEFINES> can be used to set some additional COMPILE_DEFINITIONS in the case the specified target is
# found in the system (as opposed to the case where the package was built locally)
#
# ========== CMake projects ==========
#
# mindquantum_add_pkg(...
#                     [BUILD_USING_CMAKE, USE_STATIC_LIBS]
#                     [CMAKE_OPTION <cmake_option> [... <cmake_option>]]
#                     [CMAKE_PATH <path-to-cmakefiles-txt>])
#
# If your third-party library is a CMake project, either specify <BUILD_USING_CMAKE> or a non-empty value for
# <CMAKE_OPTION>. The function will then run CMake in order to build and install the third-party library.
#
# The function will automatically pass on the following variables to the sub-CMake process:
#   - CMAKE_<lang>_COMPILER (for all activated languages or languages passed in <LANGS>)
#   - CMAKE_BUILD_TYPE
#   - CMAKE_GENERATOR
#   - CMAKE_MODULE_PATH
#   - CMAKE_VERBOSE_MAKEFILE
#
# You may use the <CMAKE_PATH> option to specify a path to the CMakeLists.txt file (relative to the unpacked source
# directory) you wish to use. By default it assumes that a `CMakeLists.txt` file is located at the root of the unpacked
# source directory.
#
# By specifying <USE_STATIC_LIBS> you will force the build of static libraries (by setting BUILD_SHARED_LIBS=OFF)
#
# In addition to passing flags or options to this function, you may also define the following CMake variables in order
# to further customize the build of the third-party dependency:
#   - <pkg_name>_CFLAGS
#   - <pkg_name>_CUDAFLAGS
#   - <pkg_name>_CXXFLAGS
#   - <pkg_name>_LDFLAGS
# These will be passed as their corresponding CMake variable equivalents.
#
# ======== Non-CMake projects ========
#
# mindquantum_add_pkg(...
#                     ONLY_COPY_DIRS <directory> [... <directory>])
#
# In this mode, the function will only unpack the third-party source code and then attempt to copy some directories
# specified by the <ONLY_COPY_DIRS> CMake list into the installation prefix. The paths for each element of
# <ONLY_COPY_DIRS> is taken relative to the unpacked source directory.
#
# Note that in this mode <GEN_CMAKE_CONFIG> always implied and a pseudo-CMake installation configuration file will
# always be generated.
#
#
# mindquantum_add_pkg(...
#                     ONLY_MAKE
#                     [GEN_CMAKE_CONFIG]
#                     [ONLY_MAKE_INCS <directory> [... <directory>]]
#                     [ONLY_MAKE_LIBS <directory> [... <directory>]])
#
# When using this form, the function will only attempt to call `make all` from within the unpacked source directory.
#
# The folders stored within <ONLY_MAKE_INCS> and/or <ONLY_MAKE_LIBS> will be copied to the local installation prefix
# under `<prefix>/include` and `<prefix>/lib` respectively. The paths are taken relative to the source directory.
#
# By specifying <GEN_CMAKE_CONFIG>, you are instructing the function to generate some pseudo CMake installation
# configuration files (like mindquantumConfig.cmake, mindquantumConfigVersion.cmake, etc.). Use this for third-party
# libraries that do not use CMake for building.
#
# mindquantum_add_pkg(...
#                     [GEN_CMAKE_CONFIG, SKIP_BUILD_STEP, SKIP_INSTALL_STEP]
#                     [PRE_CONFIGURE_COMMAND <command> [... <args>]]
#                     [CONFIGURE_COMMAND  <command> [... <args>]]
#                     [[BUILD_COMMAND  <command> [... <args>]]
#                         | [BUILD_OPTION <option> [... <option>]>
#                     [[INSTALL_INCS <directory> [... <directory>]]
#                      [INSTALL_LIBS <directory> [... <directory>]]
#                         | [INSTALL_COMMAND  <command> [... <args>]]])
#
# <PRE_CONFIGURE_COMMAND>, <CONFIGURE_COMMAND>, <BUILD_COMMAND> and <INSTALL_COMMAND> can be used for non-CMake projects
# to specify the commands to run in order to configure, build and install the third-party library.
#
# If <PRE_CONFIGURE_COMMAND> or <CONFIGURE_COMMAND> are not specified, these steps are simply skipped. If
# <BUILD_COMMAND> and/or <INSTALL_COMMAND> are not present, they default to `make all` and `make install` respectively
# unless <SKIP_BUILD_STEP> and/or <SKIP_INSTALL_STEP> are present.
#
# During the configure step, the function will append the following `key=value` pairs to the call if the corresponding
# values are not empty strings:
#   - CC=<compiler>, CXX=<compiler>, CUDACXX=<compiler> (depending on active languages or languages passed in <LANGS>)
#   - CFLAGS=<pkg_name>_CFLAGS, CXXFLAGS=<pkg_name>_CXXFLAGS, CUDAFLAGS=<pkg_name>_CUDAFLAGS
#   - LDFLAGS=<pkg_name>_LDFLAGS
#
# If no <CONFIGURE_COMMAND> is present, the above variables are added to the <BUILD_COMMAND> list of arguments.
#
# <BUILD_OPTION> can be used if no <BUILD_COMMAND> is specified in order to pass some more argument to the build
# command. These options will be passed on *before* the variable described above.
#
# If either of <INSTALL_INCS> or <INSTALL_LIBS> are present, instead of calling the <INSTALL_COMMAND>, the function will
# simply copy the folders listed in those variables (taken relative to the unpacked source directory) into the local
# installation prefix under `<prefix>/include` and `<prefix>/lib` respectively.
#
# By specifying <GEN_CMAKE_CONFIG>, you are instructing the function to generate some pseudo CMake installation
# configuration files (like mindquantumConfig.cmake, mindquantumConfigVersion.cmake, etc.). Use this for third-party
# libraries that do not use CMake for building.
#
# ========== Advanced topic ==========
#
# This function is also able to use a local server to download the third-party library archives. Note that this is also
# valid for the cases where <GIT_REPOSITORY> is specified. In order to use this functionality, either specify
#   - `MQLIBS_SERVER` environment variable (only if `ENABLE_GITEE=OFF`)
#   - `MSLIBS_SERVER` environment variable (only if `ENABLE_GITEE=OFF`)
#   - `LOCAL_LIBS_SERVER` CMake variable
#
# In each case, the variable should contain the address of a web server that will be used to download the archives
# instead of using either <URL> or <GIT_REPOSITORY>. The value must be a valid URL with port (port defaults to 8081 if
# left unspecified).
#
# The function expects the following paths to be present on the server:
#   - <server-url>/libs/<pkg_name>/<pkg_archive_name>
#   - <server-url>/libs/<pkg_name>/<pkg_git_tag>
# NB: <pkg_archive_name> is the name of file taken from the <URL>.
#
# If you are using the local server, you *must* specify a <MD5> value, even for <GIT_REPOSITORY>, <GIT_TAG> case, so
# that the function may check the downloaded archive MD5 checksum.
#
# ~~~
function(mindquantum_add_pkg pkg_name)
  # cmake-lint: disable=R0912,R0915,C0103,E1126
  set(options
      BUILD_USING_CMAKE
      CMAKE_PKG_NO_COMPONENTS
      FORCE_CONFIG_SEARCH
      FORCE_EXACT_VERSION
      FORCE_LOCAL_PKG
      GEN_CMAKE_CONFIG
      ONLY_MAKE
      SKIP_BUILD_STEP
      SKIP_INSTALL_STEP
      SKIP_IN_INSTALL_CONFIG)
  set(oneValueArgs
      CMAKE_PATH
      EXE
      GIT_REPOSITORY
      GIT_TAG
      MD5
      URL
      VER)
  set(multiValueArgs
      BUILD_COMMAND
      BUILD_DEPENDENCIES
      BUILD_OPTION
      CMAKE_OPTION
      CONFIGURE_COMMAND
      INSTALL_COMMAND
      INSTALL_INCS
      INSTALL_LIBS
      LANGS
      LIBS
      LOCAL_EXTRA_DEFINES
      ONLY_COPY_DIRS
      ONLY_MAKE_INCS
      ONLY_MAKE_LIBS
      PATCHES
      PRE_CONFIGURE_COMMAND
      SYSTEM_EXTRA_DEFINES
      TARGET_ALIAS)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(PKG_NS_NAME ${pkg_name})

  if(NOT PKG_LANGS)
    set(PKG_LANGS C CXX)
  endif()

  if(NOT PKG_MD5)
    set(PKG_MD5 XXXX)
  endif()

  set(_components ${PKG_LIBS} ${PKG_EXE})

  string(TOUPPER ${pkg_name} PKG_NAME)

  message(CHECK_START "Adding external dependency: ${pkg_name}")
  list(APPEND CMAKE_MESSAGE_INDENT "  ")

  set(_find_package_args)
  if(PKG_FORCE_EXACT_VERSION)
    list(APPEND _find_package_args EXACT)
  endif()

  if(NOT PKG_CMAKE_PKG_NO_COMPONENTS AND NOT "${_components}" STREQUAL "")
    list(APPEND _find_package_args COMPONENTS ${_components})
  endif()

  # ------------------------------------------------------------------------------

  __check_only_one_of(PKG PKG_URL PKG_GIT_REPOSITORY)
  __check_only_one_of(PKG PKG_URL PKG_GIT_TAG)
  __check_only_one_of(PKG PKG_ONLY_COPY_DIRS PKG_ONLY_MAKE)
  __check_only_one_of(PKG PKG_BUILD_COMMAND PKG_BUILD_OPTION PKG_SKIP_BUILD_STEP)
  __check_only_one_of(PKG PKG_INSTALL_COMMAND PKG_SKIP_INSTALL_STEP)

  if((PKG_INSTALL_INCS OR PKG_INSTALL_LIBS) AND (PKG_INSTALL_COMMAND OR PKG_SKIP_INSTALL_STEP))
    message(
      FATAL_ERROR
        [[
Cannot specify either of <INSTALL_INCS> or <INSTALL_LIBS> with either of <INSTALL_COMMAND> or <SKIP_INSTALL_STEP>
]])
  endif()

  # ----------------------------------------------------------------------------

  set(${pkg_name}_PATCHES_HASH)
  foreach(_patch ${PKG_PATCHES})
    file(MD5 ${_patch} _patch_md5)
    set(${pkg_name}_PATCHES_HASH "${${pkg_name}_PATCHES_HASH},${_patch_md5}")
  endforeach()

  # check options
  string(REPLACE "${PROJECT_BINARY_DIR}" "<binary-dir>" _purged_ARGN "${ARGN}")
  string(REPLACE "${PROJECT_SOURCE_DIR}" "<source-dir>" _purged_ARGN "${_purged_ARGN}")
  set(${pkg_name}_CONFIG_TXT
      "${CMAKE_CXX_COMPILER_VERSION}-${CMAKE_C_COMPILER_VERSION}-${CMAKE_CUDA_COMPILER_VERSION}
            ${_purged_ARGN} - ${${pkg_name}_USE_STATIC_LIBS}- ${${pkg_name}_PATCHES_HASH}
            ${${pkg_name}_CXXFLAGS}--${${pkg_name}_CFLAGS}--${${pkg_name}_LDFLAGS}")
  string(REPLACE ";" "-" ${pkg_name}_CONFIG_TXT ${${pkg_name}_CONFIG_TXT})
  string(MD5 ${pkg_name}_CONFIG_HASH "${${pkg_name}_CONFIG_TXT}")

  if(NOT _${pkg_name}_SYSTEM AND NOT "${pkg_name}_BASE_DIR" STREQUAL "")
    # Package is not from the system and from a previous CMake run -> check if the config hash has changed
    if(EXISTS "${${pkg_name}_BASE_DIR}/options.txt")
      file(READ "${${pkg_name}_BASE_DIR}/options.txt" _old_config_txt)
      string(MD5 _old_config_hash "${_old_config_txt}")
      if(NOT _old_config_hash STREQUAL ${pkg_name}_CONFIG_HASH)
        # Config hash has changed -> remove all relevant directories:
        #
        # * local install prefix (BASE_DIR)
        # * unpacked source
        # * unpacked source subbuild directory
        # * unpacked source build directory
        #
        # If the third-party has a Git repository, then simply run git stash --all within the repository

        message(STATUS "Old config hash does not match new config hash")
        set(_directories_to_clean "${${pkg_name}_BASE_DIR}" "${_mq_local_prefix}/../_deps/${pkg_name}-src")
        if(NOT PKG_GIT_REPOSITORY)
          list(APPEND _directories_to_clean "${_mq_local_prefix}/../_deps/${pkg_name}-subbuild"
               "${_mq_local_prefix}/../_deps/${pkg_name}-build")
        endif()
        foreach(_dir ${_directories_to_clean})
          if(NOT "${_dir}" STREQUAL "")
            if(EXISTS "${_dir}/.git")
              message(STATUS "  - resetting git repository: ${_dir}")
              __exec_cmd(COMMAND ${GIT_EXECUTABLE} stash --all WORKING_DIRECTORY ${_dir})
              __exec_cmd(COMMAND ${GIT_EXECUTABLE} stash drop WORKING_DIRECTORY ${_dir})
            elseif(EXISTS "${_dir}")
              message(STATUS "  - deleting ${_dir}")
              file(REMOVE_RECURSE "${_dir}")
            endif()
          endif()
        endforeach()

        unset(${pkg_name}_DIR)
        unset(${pkg_name}_DIR CACHE)
      endif()
    endif()
  endif()

  # ============================================================================

  # NB: this branch will only be taken if not the first CMake configure call (or if manually set)
  if(${pkg_name}_DIR)
    set(_args
        "${pkg_name}"
        "${PKG_VER}"
        CONFIG
        ${_find_package_args}
        NO_DEFAULT_PATH
        PATHS
        ${${pkg_name}_DIR})
    __find_package(${_args} SEARCH_NAME "config")

    if(${pkg_name}_FOUND)
      if(CLEAN_3RDPARTY_INSTALL_DIR)
        file(GLOB _installations ${_mq_local_prefix}/${pkg_name}_${PKG_VER}_*)
        message(STATUS "Deleting old installation directories (if any):")
        foreach(_dir ${_installations})
          cmake_path(CONVERT "${_dir}" TO_CMAKE_PATH_LIST _dir NORMALIZE)
          cmake_path(CONVERT "${${pkg_name}_DIRPATH}" TO_CMAKE_PATH_LIST _dirpath NORMALIZE)
          if(NOT "${_dir}" STREQUAL "${_dirpath}")
            message(STATUS "  - ${_dir}")
            file(REMOVE_RECURSE "${_dir}")
          endif()
        endforeach()
      endif()

      if(${pkg_name}_DIR)
        message(STATUS "Package CMake config dir: ${${pkg_name}_DIR}")
      endif()
      if(_${pkg_name}_SYSTEM)
        set(_defines_type SYSTEM)
      else()
        set(_defines_type LOCAL)
        debug_print(STATUS "Adding ${${pkg_name}_DIRPATH} to ${_local_libs_path_file}")
        file(APPEND "${_local_libs_path_file}" "${${pkg_name}_DIRPATH}\n")
      endif()
      if(NOT "${PKG_${_defines_type}_EXTRA_DEFINES}" STREQUAL "")
        __append_target_properties(COMPILE_DEFINITIONS ${PKG_${_defines_type}_EXTRA_DEFINES})
      endif()
      if(NOT PKG_SKIP_IN_INSTALL_CONFIG)
        __setup_install_target(${pkg_name} ${_args})
      endif()
      __make_target_global(${PKG_NS_NAME} ${PKG_LIBS} ${PKG_EXE})
      __create_target_aliases(${pkg_name} ${PKG_SKIP_IN_INSTALL_CONFIG} ${PKG_TARGET_ALIAS})

      list(POP_BACK CMAKE_MESSAGE_INDENT)
      message(CHECK_PASS "Done")
      return()
    endif()
  endif()

  if(NOT MQ_FORCE_LOCAL_PKGS
     AND NOT MQ_${PKG_NAME}_FORCE_LOCAL
     AND NOT PKG_FORCE_LOCAL_PKG)
    set(_args "${pkg_name}" "${PKG_VER}")
    if(PKG_FORCE_CONFIG_SEARCH)
      list(APPEND _args CONFIG)
    endif()
    list(APPEND _args ${_find_package_args})

    __find_package(${_args} SEARCH_NAME "system packages")

    if(${pkg_name}_FOUND)
      set(_${pkg_name}_SYSTEM
          TRUE
          CACHE BOOL "Found ${pkg_name} in the system folders")
      if(${pkg_name}_DIR)
        message(STATUS "Package CMake config dir: ${${pkg_name}_DIR}")
      endif()

      if(NOT "${PKG_SYSTEM_EXTRA_DEFINES}" STREQUAL "")
        __append_target_properties(COMPILE_DEFINITIONS ${PKG_SYSTEM_EXTRA_DEFINES})
      endif()

      if(NOT PKG_SKIP_IN_INSTALL_CONFIG)
        __setup_install_target(${pkg_name} ${_args})
      endif()
      __make_target_global(${PKG_NS_NAME} ${PKG_LIBS} ${PKG_EXE})
      __create_target_aliases(${pkg_name} ${PKG_SKIP_IN_INSTALL_CONFIG} ${PKG_TARGET_ALIAS})

      list(POP_BACK CMAKE_MESSAGE_INDENT)
      message(CHECK_PASS "Done")
      return()
    endif()

    # Otherwise we try to compile from source
  endif()

  # ==============================================================================
  # Ignore system installed libraries and compile a local version instead

  set(_${pkg_name}_SYSTEM
      FALSE
      CACHE BOOL "Found ${pkg_name} in the system folders")

  message(STATUS "${pkg_name} config hash: ${${pkg_name}_CONFIG_HASH}")

  # NB: If the package is not found on the system, this is where we will be looking for it
  set(${pkg_name}_BASE_DIR
      ${_mq_local_prefix}/${pkg_name}_${PKG_VER}_${${pkg_name}_CONFIG_HASH}
      CACHE FILEPATH INTERNAL)

  set(${pkg_name}_DIRPATH
      ${${pkg_name}_BASE_DIR}
      CACHE FILEPATH INTERNAL)
  debug_print(STATUS "Adding ${${pkg_name}_DIRPATH} to ${_local_libs_path_file}")
  file(APPEND "${_local_libs_path_file}" "${${pkg_name}_DIRPATH}\n")

  if(CLEAN_3RDPARTY_INSTALL_DIR)
    file(GLOB _installations ${_mq_local_prefix}/${pkg_name}_${PKG_VER}_*)
    message(STATUS "Deleting old installation directories (if any):")
    foreach(_dir ${_installations})
      cmake_path(CONVERT "${_dir}" TO_CMAKE_PATH_LIST _dir NORMALIZE)
      cmake_path(CONVERT "${${pkg_name}_BASE_DIR}" TO_CMAKE_PATH_LIST _basedir NORMALIZE)
      if(NOT "${_dir}" STREQUAL "${_basedir}")
        message(STATUS "  - ${_dir}")
        file(REMOVE_RECURSE "${_dir}")
      endif()
    endforeach()
  endif()

  if(EXISTS "${${pkg_name}_BASE_DIR}")
    set(_args
        "${pkg_name}"
        "${PKG_VER}"
        CONFIG
        NO_DEFAULT_PATH
        PATHS
        "${${pkg_name}_BASE_DIR}"
        ${_find_package_args})

    __find_package(${_args} SEARCH_NAME "MindQuantum build dir")
    if(${pkg_name}_FOUND)
      if(${pkg_name}_DIR)
        message(STATUS "Package CMake config dir: ${${pkg_name}_DIR}")
      endif()
      if(NOT "${PKG_LOCAL_EXTRA_DEFINES}" STREQUAL "")
        __append_target_properties(COMPILE_DEFINITIONS ${PKG_LOCAL_EXTRA_DEFINES})
      endif()
      if(NOT PKG_SKIP_IN_INSTALL_CONFIG)
        __setup_install_target(${pkg_name} ${_args})
      endif()
      __check_package_location(${pkg_name} ${PKG_NS_NAME} ${PKG_LIBS} ${PKG_EXE})
      __make_target_global(${PKG_NS_NAME} ${PKG_LIBS} ${PKG_EXE})
      __create_target_aliases(${pkg_name} ${PKG_SKIP_IN_INSTALL_CONFIG} ${PKG_TARGET_ALIAS})

      list(POP_BACK CMAKE_MESSAGE_INDENT)
      message(CHECK_PASS "Done")
      return()
    endif()
  endif()

  if(PKG_GIT_REPOSITORY)
    message(STATUS "PKG_GIT_REPOSITORY = ${PKG_GIT_REPOSITORY}")
    __download_pkg_with_git(${pkg_name} ${PKG_GIT_REPOSITORY} ${PKG_GIT_TAG} ${PKG_MD5})
  else()
    message(STATUS "PKG_URL = ${PKG_URL}")
    __download_pkg(${pkg_name} ${PKG_URL} ${PKG_MD5})
  endif()
  file(WRITE ${${pkg_name}_BASE_DIR}/options.txt ${${pkg_name}_CONFIG_TXT})
  message(STATUS "${pkg_name}_SOURCE_DIR : ${${pkg_name}_SOURCE_DIR}")

  set(${pkg_name}_SOURCE_DIR
      ${${pkg_name}_SOURCE_DIR}
      CACHE FILEPATH INTERNAL)

  apply_patches(WORKING_DIRECTORY "${${pkg_name}_SOURCE_DIR}" TRY_GIT_RESET ${PKG_PATCHES})

  file(
    LOCK
    ${${pkg_name}_BASE_DIR}
    DIRECTORY
    GUARD
    FUNCTION
    RESULT_VARIABLE
    ${pkg_name}_LOCK_RET
    TIMEOUT
    600)
  if(NOT ${pkg_name}_LOCK_RET EQUAL "0")
    message(FATAL_ERROR "error! when try lock ${${pkg_name}_BASE_DIR} : ${${pkg_name}_LOCK_RET}")
  endif()

  if(${pkg_name}_SOURCE_DIR)
    # Look for any dependencies (if any)
    foreach(_pkg_dep ${PKG_BUILD_DEPENDENCIES})
      debug_print(STATUS "Looking for build dependency for ${pkg_name}: find_package(${_pkg_dep} REQUIRED)")
      find_package(${_pkg_dep} REQUIRED)
    endforeach()

    if(PKG_ONLY_COPY_DIRS)
      cmake_path(GET ${pkg_name}_BASE_DIR FILENAME _basename)

      add_library(${PKG_NS_NAME}::${pkg_name} INTERFACE IMPORTED)
      foreach(_dir ${PKG_ONLY_COPY_DIRS})
        message(STATUS "Copying ${${pkg_name}_SOURCE_DIR}/${_dir} -> ${${pkg_name}_BASE_DIR}")
        cmake_path(GET _dir PARENT_PATH _parent)
        file(COPY ${${pkg_name}_SOURCE_DIR}/${_dir} DESTINATION ${${pkg_name}_BASE_DIR}/${_parent})
        target_include_directories(${PKG_NS_NAME}::${pkg_name} INTERFACE $<BUILD_INTERFACE:${${pkg_name}_BASE_DIR}>
                                                                         $<INSTALL_INTERFACE:.>)
      endforeach()

      message(STATUS "Generating fake CMake config file...")
      __generate_pseudo_cmake_package_config("${${pkg_name}_BASE_DIR}" ${pkg_name} ${PKG_NS_NAME} "${pkg_name}")
    elseif(PKG_ONLY_MAKE)
      __exec_cmd(COMMAND ${_make_exec} ${${pkg_name}_CXXFLAGS} -j${JOBS} WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR})
      set(PKG_INSTALL_INCS ${PKG_ONLY_MAKE_INCS})
      set(PKG_INSTALL_LIBS ${PKG_ONLY_MAKE_LIBS})
      file(GLOB ${pkg_name}_INSTALL_INCS ${${pkg_name}_SOURCE_DIR}/${PKG_INSTALL_INCS})
      file(GLOB ${pkg_name}_INSTALL_LIBS ${${pkg_name}_SOURCE_DIR}/${PKG_INSTALL_LIBS})
      file(COPY ${${pkg_name}_INSTALL_INCS} DESTINATION ${${pkg_name}_BASE_DIR}/include)
      file(COPY ${${pkg_name}_INSTALL_LIBS} DESTINATION ${${pkg_name}_BASE_DIR}/lib)

    elseif(NOT "${PKG_CMAKE_OPTION}" STREQUAL "" OR PKG_BUILD_USING_CMAKE)
      set(${pkg_name}_CMAKE_COMPILERS)
      foreach(_lang ${PKG_LANGS})
        if(CMAKE_${_lang}_COMPILER)
          list(APPEND ${pkg_name}_CMAKE_COMPILERS -DCMAKE_${_lang}_COMPILER=${CMAKE_${_lang}_COMPILER})
        endif()
      endforeach()

      get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

      set(_cmake_original_build_types Debug Release RelWithDebInfo MinSizeRel)
      set(_cmake_build_type "${CMAKE_BUILD_TYPE}")
      if(NOT "${CMAKE_BUILD_TYPE}" IN_LIST _cmake_original_build_types)
        set(_cmake_build_type RelWithDebInfo)
        debug_print(WARNING "${CMAKE_BUILD_TYPE} not in known build types, using ${_cmake_build_type} instead")
      endif()

      if("${_cmake_build_type}" STREQUAL "Release" OR _is_multi_config)
        set(_cmake_build_dir "${${pkg_name}_SOURCE_DIR}/_build")
      else()
        string(TOLOWER "${_cmake_build_type}" _build_type)
        set(_cmake_build_dir "${${pkg_name}_SOURCE_DIR}/_build_${_build_type}")
      endif()

      file(MAKE_DIRECTORY ${_cmake_build_dir})
      if(${pkg_name}_CFLAGS)
        set(${pkg_name}_CMAKE_CFLAGS "-DCMAKE_C_FLAGS=${${pkg_name}_CFLAGS}")
      endif()

      if(${pkg_name}_CXXFLAGS)
        set(${pkg_name}_CMAKE_CXXFLAGS "-DCMAKE_CXX_FLAGS=${${pkg_name}_CXXFLAGS}")
      endif()

      if(${pkg_name}_CUDAFLAGS)
        set(${pkg_name}_CMAKE_CUDAFLAGS "-DCMAKE_CUDA_FLAGS=${${pkg_name}_CUDAFLAGS}")
      endif()

      if(MSVC)
        if(ENABLE_MT)
          set(${pkg_name}_CL_RT_FLAG "-DCMAKE_CXX_FLAGS_DEBUG=/MTd" "-DCMAKE_CXX_FLAGS_RELEASE=/MT")
        elseif(ENABLE_MD)
          set(${pkg_name}_CL_RT_FLAG "-DCMAKE_CXX_FLAGS_DEBUG=/MDd" "-DCMAKE_CXX_FLAGS_RELEASE=/MD")
        endif()
      endif()

      if(${pkg_name}_LDFLAGS)
        if(${pkg_name}_USE_STATIC_LIBS)
          list(APPEND _cmake_args "-DBUILD_SHARED_LIBS=OFF")
          set(${pkg_name}_CMAKE_LDFLAGS "-DCMAKE_STATIC_LINKER_FLAGS=${${pkg_name}_LDFLAGS}")
        else()
          set(${pkg_name}_CMAKE_LDFLAGS "-DCMAKE_SHARED_LINKER_FLAGS=${${pkg_name}_LDFLAGS}")
        endif()
      endif()

      string(REPLACE ";" "\\\\\;" _cmake_module_path "${CMAKE_MODULE_PATH}")

      set(_cmake_args)
      if(NOT _is_multi_config)
        list(APPEND _cmake_args -DCMAKE_BUILD_TYPE=${_cmake_build_type})
      endif()

      message(STATUS "Calling CMake configure for ${pkg_name}")
      __exec_cmd(
        COMMAND
          ${CMAKE_COMMAND} ${${pkg_name}_CMAKE_COMPILERS} "-DCMAKE_MODULE_PATH=${_cmake_module_path}"
          -DCMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE} -G${CMAKE_GENERATOR} ${PKG_CMAKE_OPTION}
          ${${pkg_name}_CMAKE_CFLAGS} ${${pkg_name}_CMAKE_CXXFLAGS} ${${pkg_name}_CUDAFLAGS} ${${pkg_name}_CL_RT_FLAG}
          ${${pkg_name}_CMAKE_LDFLAGS} -DCMAKE_INSTALL_PREFIX=${${pkg_name}_BASE_DIR}
          ${${pkg_name}_SOURCE_DIR}/${PKG_CMAKE_PATH} ${_cmake_args}
        WORKING_DIRECTORY ${_cmake_build_dir})

      message(STATUS "Building CMake targets for ${pkg_name}")
      if(_is_multi_config)
        __exec_cmd(COMMAND ${CMAKE_COMMAND} --build . --target install -j${JOBS} --config Debug
                   WORKING_DIRECTORY ${_cmake_build_dir})
        __exec_cmd(COMMAND ${CMAKE_COMMAND} --build . --target install -j${JOBS} --config Release
                   WORKING_DIRECTORY ${_cmake_build_dir})
      else()
        __exec_cmd(COMMAND ${CMAKE_COMMAND} --build . --target install -j${JOBS} WORKING_DIRECTORY ${_cmake_build_dir})
      endif()
    else()
      set(PREFIX ${${pkg_name}_BASE_DIR})
      set(MAKE ${_make_exec})

      set(${pkg_name}_COMPILERS)
      foreach(_lang ${PKG_LANGS})
        if(CMAKE_${_lang}_COMPILER)
          if("${_lang}" STREQUAL "C")
            set(_var C)
          elseif("${_lang}" STREQUAL "CXX")
            set(_var CC)
          elseif("${_lang}" STREQUAL "CUDA")
            set(_var CUDACXX)
          else()
            message(WARNING "Unsupported language: ${_lang} -> skipping setting compiler environment variable")
            continue()
          endif()
          list(APPEND ${pkg_name}_COMPILERS ${_var}=${CMAKE_${_lang}_COMPILER})
        endif()
      endforeach()

      if(${pkg_name}_CFLAGS)
        set(${pkg_name}_MAKE_CFLAGS "CFLAGS=${${pkg_name}_CFLAGS}")
      endif()
      if(${pkg_name}_CXXFLAGS)
        set(${pkg_name}_MAKE_CXXFLAGS "CXXFLAGS=${${pkg_name}_CXXFLAGS}")
      endif()
      if(${pkg_name}_CUDAFLAGS)
        if(NOT CUDA IN_LIST ${PKG_LANGS})
          message(WARNING "Set ${pkg_name}_CUDAFLAGS but CUDA was not passed in the <LANGS> argument!")
        endif()
        set(${pkg_name}_MAKE_CUDAFLAGS "CUDAFLAGS=${${pkg_name}_CUDAFLAGS}")
      endif()
      if(${pkg_name}_LDFLAGS)
        set(${pkg_name}_MAKE_LDFLAGS "LDFLAGS=${${pkg_name}_LDFLAGS}")
      endif()
      # in configure && make
      if(PKG_PRE_CONFIGURE_COMMAND)
        message(STATUS "Calling pre-configure script for ${pkg_name}")
        __exec_cmd(COMMAND ${PKG_PRE_CONFIGURE_COMMAND} WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR})
      endif()

      if(PKG_CONFIGURE_COMMAND)
        message(STATUS "Calling configure script for ${pkg_name}")
        __exec_cmd(
          COMMAND
            ${PKG_CONFIGURE_COMMAND} ${${pkg_name}_COMPILERS} ${${pkg_name}_MAKE_CFLAGS} ${${pkg_name}_MAKE_CXXFLAGS}
            ${${pkg_name}_MAKE_CUDAFLAGS} ${${pkg_name}_MAKE_LDFLAGS} --prefix=${${pkg_name}_BASE_DIR}
          WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR})
      endif()

      if(PKG_BUILD_COMMAND)
        message(STATUS "Calling build command for ${pkg_name}")
        string(CONFIGURE "${PKG_BUILD_COMMAND}" PKG_BUILD_COMMAND @ONLY ESCAPE_QUOTES)
        __exec_cmd(COMMAND ${PKG_BUILD_COMMAND} -j${JOBS} WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR})
      elseif(NOT PKG_SKIP_BUILD_STEP)
        message(STATUS "Calling build command for ${pkg_name}")
        string(CONFIGURE "${PKG_BUILD_OPTION}" PKG_BUILD_OPTION @ONLY ESCAPE_QUOTES)
        set(${pkg_name}_BUILD_OPTION ${PKG_BUILD_OPTION})
        if(NOT PKG_CONFIGURE_COMMAND)
          set(${pkg_name}_BUILD_OPTION ${${pkg_name}_BUILD_OPTION} ${${pkg_name}_MAKE_CFLAGS}
                                       ${${pkg_name}_MAKE_CXXFLAGS} ${${pkg_name}_MAKE_LDFLAGS})
        endif()

        __exec_cmd(COMMAND ${_make_exec} ${${pkg_name}_BUILD_OPTION} -j${JOBS}
                   WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR})
      endif()

      if(PKG_INSTALL_INCS OR PKG_INSTALL_LIBS)
        file(GLOB ${pkg_name}_INSTALL_INCS ${${pkg_name}_SOURCE_DIR}/${PKG_INSTALL_INCS})
        file(GLOB ${pkg_name}_INSTALL_LIBS ${${pkg_name}_SOURCE_DIR}/${PKG_INSTALL_LIBS})
        file(COPY ${${pkg_name}_INSTALL_INCS} DESTINATION ${${pkg_name}_BASE_DIR}/include)
        file(COPY ${${pkg_name}_INSTALL_LIBS} DESTINATION ${${pkg_name}_BASE_DIR}/lib)
      elseif(NOT PKG_SKIP_INSTALL_STEP)
        message(STATUS "Calling install command for ${pkg_name}")
        if(PKG_INSTALL_COMMAND)
          string(CONFIGURE "${PKG_INSTALL_COMMAND}" PKG_INSTALL_COMMAND @ONLY ESCAPE_QUOTES)
          __exec_cmd(COMMAND ${PKG_INSTALL_COMMAND} WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR})
        else()
          __exec_cmd(COMMAND ${_make_exec} install WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR})
        endif()
      endif()
      unset(PREFIX)
      unset(MAKE)
    endif()
  endif()

  if(PKG_GEN_CMAKE_CONFIG AND NOT PKG_ONLY_COPY_DIRS)
    message(STATUS "Generating fake CMake config file...")

    list(PREPEND CMAKE_PREFIX_PATH "${${pkg_name}_BASE_DIR}")
    set(${PKG_NAME}_ROOT "${${pkg_name}_BASE_DIR}")
    set(${pkg_name}_ROOT "${${pkg_name}_BASE_DIR}")

    set(_comps ${PKG_LIBS})
    if(PKG_EXE)
      list(APPEND _comps ${PKG_EXE})
    endif()

    # Not using PkgConfig here so that we get the libraries type right (ie. SHARED or STATIC instead of INTERFACE)
    set(${pkg_name}_NO_PKGCONFIG ON)
    find_package(
      ${pkg_name} QUIET
      COMPONENTS ${_comps}
      REQUIRED)
    __generate_pseudo_cmake_package_config("${${pkg_name}_BASE_DIR}" ${pkg_name} ${PKG_NS_NAME} "${_comps}")
    list(POP_FRONT CMAKE_PREFIX_PATH)
  endif()

  set(_args
      "${pkg_name}"
      "${PKG_VER}"
      REQUIRED
      CONFIG
      NO_DEFAULT_PATH
      HINTS
      "${${pkg_name}_BASE_DIR}"
      ${_find_package_args})
  __find_package(${_args} SEARCH_NAME "MindQuantum build dir")
  if(NOT "${PKG_LOCAL_EXTRA_DEFINES}" STREQUAL "")
    __append_target_properties(COMPILE_DEFINITIONS ${PKG_LOCAL_EXTRA_DEFINES})
  endif()
  if(NOT PKG_SKIP_IN_INSTALL_CONFIG)
    __setup_install_target(${pkg_name} ${_args})
  endif()
  __check_package_location(${pkg_name} ${PKG_NS_NAME} ${PKG_LIBS} ${PKG_EXE})
  __make_target_global(${PKG_NS_NAME} ${PKG_LIBS} ${PKG_EXE})
  __create_target_aliases(${pkg_name} ${PKG_SKIP_IN_INSTALL_CONFIG} ${PKG_TARGET_ALIAS})

  list(POP_BACK CMAKE_MESSAGE_INDENT)
  message(CHECK_PASS "Done")
endfunction()

# ==============================================================================
