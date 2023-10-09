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
# Largely inspired by the FindBoost.cmake module:
#
# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.
# ==============================================================================

# lint_cmake: -whitespace/indent,-package/stdargs,-convention/filename,-linelength

# cmake-lint: disable=C0103,E1126,E1120

# ~~~
# Variables that may be defined:
# ${_pkg}_TARGET_DEFINITIONS - Compile definitions added to all targets
# ${_pkg}_NAMESPACE - Specify the namespace for manually found imported targets (defaults to ${_pkg})
#                     NB: has no effects if the package is found using CMake config mode
# ${_pkg}_DEFINE_PREFIX - PREFIX for pre-processor defines (defaults to ${_pkg} in uppercase); mostly used to
#                         automatically determine the package's version
# ${_pkg}_${component}_TYPE - Type of component
#                             Can be one of: LIBRARY (default), EXECUTABLE
# ${_pkg}_${component}_TARGET_DEFINITIONS - Compile definitions added to a particular component (in uppercase)
# ${_pkg}_CMAKE_CONFIG_NO_COMPONENTS - When looking for the package using CMake CONFIG mode, do not use components
#                                      (TRUE/FALSE, defaults to FALSE)
# ${_pkg}_NO_CMAKE - No search for CMake packages
# ${_pkg}_NO_PKGCONFIG - No search for package using PkgConfig
# ${_pkg}_USE_STATIC_LIBS - If TRUE, only look for static libs (.a on UNIX or .lib .a on Windows)
#                           The extension list may be customized using ${_pkg}_FIND_LIBRARY_SUFFIXES for a more fine
#                           grained control.
# ${_pkg}_INCLUDE_FILE - Include file name to look for include directory
# ${_pkg}_INCLUDE_DIR_UP_INDEX - Number of sub-directory levels the include file is located into (defaults to 0)
#                                e.g. /usr/include/mylib/dir/mylib.h & 0 => inc_dir = /usr/include/mylib/dir)
#                                e.g. /usr/include/mylib/dir/mylib.h & 1  => inc_dir = /usr/include/mylib)
# ${_pkg}_INCLUDE_PATH_SUFFIXES - List of path suffixes for include file search
#                                 Defaults to [] (ie. an empty list)
# ${_pkg}_EXECUTABLE_PATH_SUFFIXES - List of path suffixes for executable file search
#                                 Defaults to ["bin", "sbin"]
# ${_pkg}_DEFAULT_COMPONENTS - Default list of components
# ${_pkg}_${component}_NAMES or ${_pkg}_${component}_RELEASE_NAMES and/or ${_pkg}_${component}_DEBUG_NAMES
# ${_pkg}_${component}_DEPENDENCIES - List of dependencies between components of the package
# ${_pkg}_${component}_EXTERNAL_DEPENDENCIES - List of libraries/CMake targets
# ${_pkg}_${component}_HEADER_ONLY - TRUE/FALSE if component is header-only
# ${_pkg}_INC_SYSTEM_PATHS - List of system paths to consider when looking for header files
# ${_pkg}_LIB_SYSTEM_PATHS - List of system paths to consider when looking for libraries
#                            Will set ${_pkg}_LIB_SYSTEM_PATHS_[DEBUG,RELEASE] if those are empty
# ${_pkg}_LIB_SYSTEM_PATHS_DEBUG - List of system paths to consider when looking for DEBUG libraries
# ${_pkg}_LIB_SYSTEM_PATHS_RELEASE - List of system paths to consider when looking for RELEASE libraries
# ${_pkg}_FIND_LIBRARY_PREFIXES - List of library prefixes (in lieu of CMAKE_FIND_LIBRARY_PREFIXES)
# ${_pkg}_FIND_LIBRARY_SUFFIXES - List of library suffixes (in lieu of CMAKE_FIND_LIBRARY_SUFFIXES)
#
# Functions
# ${_pkg}_version_function function with signature:
#    func(include_dir)
#    -> need to assign in parent scope ${_pkg}_VERSION, ${_pkg}_VERSION_MAJOR, ${_pkg}_VERSION_MINOR,
#       ${_pkg}_VERSION_PATCH
# ${_pkg}_include_dir_post_process function with signature:
#    func(var include_dir)
#    -> `var` is the name of the include dir variable and `include_dir` its original value
# ${_pkg}_update_library_search_dirs_with_prebuilt_paths function with signature:
#    func(componentlibvar basedir)
#    -> append to ${componentlibvar} pre-built paths based on ${basedir}
# ${_pgk}_windows_set_import_library_path with signature:
#    func(target basename) or func(target basename config) where config is RELEASE, DEBUG, etc.
#    -> typical call: ${_pkg}_windows_set_import_library_path(some_target IMPORTED_LOCATION)
# ${_pkg}_post_process function with signature:
#    func(pkg)
# ~~~

# The FPHSA helper provides standard way of reporting final search results to the user including the version and
# component checks.
include(FindPackageHandleStandardArgs)
include(CMakePrintHelpers)

# Save project's policies
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW) # if IN_LIST
cmake_policy(SET CMP0102 NEW) # if mark_as_advanced(non_cache_var)

# ==============================================================================
# FindPkg functions & macros
#

# ~~~
# Print debug text if ${_pkg}_DEBUG is set. Call example:
# _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "debug message")
# ~~~
function(_debug_print file line text)
  if(${_pkg}_DEBUG)
    message(STATUS "[ ${file}:${line} ] ${text}")
  endif()
endfunction()

# ~~~
# _debug_print_var(file line variable_name [ENVIRONMENT] [SOURCE "short explanation of origin of var value"])
#
# ENVIRONMENT - look up environment variable instead of CMake variable
#
# Print variable name and its value if ${_pkg}_DEBUG is set. Call example:
# _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" ${_pkg}_ROOT)
# ~~~
function(_debug_print_var file line name)
  if(${_pkg}_DEBUG)
    cmake_parse_arguments(_args "ENVIRONMENT" "SOURCE" "" ${ARGN})

    unset(source)
    if(_args_SOURCE)
      set(source " (${_args_SOURCE})")
    endif()

    if(_args_ENVIRONMENT)
      if(DEFINED ENV{${name}})
        set(value "\"$ENV{${name}}\"")
      else()
        set(value "<unset>")
      endif()
      set(_name "ENV{${name}}")
    else()
      if(DEFINED "${name}")
        set(value "\"${${name}}\"")
      else()
        set(value "<unset>")
      endif()
      set(_name "${name}")
    endif()

    _debug_print("${file}" "${line}" "${_name} = ${value}${source}")
  endif()
endfunction()

# ######################################################################################################################
#
# Check the existence of the libraries.
#
# ######################################################################################################################
# This macro was taken directly from the FindQt4.cmake file that is included with the CMake distribution. This is NOT my
# work. All work was done by the original authors of the FindQt4.cmake file. Only minor modifications were made to
# remove references to Qt and make this file more generally applicable And ELSE/ENDIF pairs were removed for
# readability.
# ######################################################################################################################
macro(_adjust_lib_vars basename)
  if(${_pkg}_INCLUDE_DIR)
    if(${_pkg}_${basename}_HEADER_ONLY)
      string(TOUPPER ${basename} BASENAME)
      set(${_pkg}_${BASENAME}_FOUND ON)
    else()
      if(${_pkg}_${basename}_LIBRARY_DEBUG AND ${_pkg}_${basename}_LIBRARY_RELEASE)
        # if the generator is multi-config or if CMAKE_BUILD_TYPE is set for single-config generators, set optimized and
        # debug libraries
        get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
        if(_isMultiConfig OR CMAKE_BUILD_TYPE)
          set(${_pkg}_${basename}_LIBRARY optimized ${${_pkg}_${basename}_LIBRARY_RELEASE} debug
                                          ${${_pkg}_${basename}_LIBRARY_DEBUG})
        else()
          # For single-config generators where CMAKE_BUILD_TYPE has no value, just use the release libraries
          set(${_pkg}_${basename}_LIBRARY ${${_pkg}_${basename}_LIBRARY_RELEASE})
        endif()
        # FIXME: This probably should be set for both cases
        set(${_pkg}_${basename}_LIBRARIES optimized ${${_pkg}_${basename}_LIBRARY_RELEASE} debug
                                          ${${_pkg}_${basename}_LIBRARY_DEBUG})
      endif()

      # if only the release version was found, set the debug variable also to the release version
      if(${_pkg}_${basename}_LIBRARY_RELEASE AND NOT ${_pkg}_${basename}_LIBRARY_DEBUG)
        set(${_pkg}_${basename}_LIBRARY_DEBUG ${${_pkg}_${basename}_LIBRARY_RELEASE})
        set(${_pkg}_${basename}_LIBRARY ${${_pkg}_${basename}_LIBRARY_RELEASE})
        set(${_pkg}_${basename}_LIBRARIES ${${_pkg}_${basename}_LIBRARY_RELEASE})
      endif()

      # if only the debug version was found, set the release variable also to the debug version
      if(${_pkg}_${basename}_LIBRARY_DEBUG AND NOT ${_pkg}_${basename}_LIBRARY_RELEASE)
        set(${_pkg}_${basename}_LIBRARY_RELEASE ${${_pkg}_${basename}_LIBRARY_DEBUG})
        set(${_pkg}_${basename}_LIBRARY ${${_pkg}_${basename}_LIBRARY_DEBUG})
        set(${_pkg}_${basename}_LIBRARIES ${${_pkg}_${basename}_LIBRARY_DEBUG})
      endif()

      # If the debug & release library ends up being the same, omit the keywords
      if("${${_pkg}_${basename}_LIBRARY_RELEASE}" STREQUAL "${${_pkg}_${basename}_LIBRARY_DEBUG}")
        set(${_pkg}_${basename}_LIBRARY ${${_pkg}_${basename}_LIBRARY_RELEASE})
        set(${_pkg}_${basename}_LIBRARIES ${${_pkg}_${basename}_LIBRARY_RELEASE})
      endif()

      if(${_pkg}_${basename}_LIBRARY)
        set(${_pkg}_${basename}_FOUND ON)
      endif()
    endif()
  endif()

  # Make variables changeable to the advanced user
  mark_as_advanced(${_pkg}_${basename}_LIBRARY_RELEASE ${_pkg}_${basename}_LIBRARY_DEBUG)
endmacro()

# ~~~
# Detect changes in used variables.
# Compares the current variable value with the last one.
# In short form:
# v != v_LAST                      -> CHANGED = 1
# v is defined, v_LAST not         -> CHANGED = 1
# v is not defined, but v_LAST is  -> CHANGED = 1
# otherwise                        -> CHANGED = 0
# CHANGED is returned in variable named ${changed_var}
# ~~~
macro(_change_detect changed_var)
  set(${changed_var} 0)
  foreach(var ${ARGN})
    if(DEFINED _${_pkg}_COMPONENTS_SEARCHED)
      if(${var})
        if(_${var}_LAST)
          string(COMPARE NOTEQUAL "${${var}}" "${_${var}_LAST}" _${var}_CHANGED)
        else()
          set(_${var}_CHANGED 1)
        endif()
      elseif(_${var}_LAST)
        set(_${var}_CHANGED 1)
      endif()
      if(_${var}_CHANGED)
        set(${changed_var} 1)
      endif()
    else()
      set(_${var}_CHANGED 0)
    endif()
  endforeach()
endmacro()

#
# Find the given library (var). Use 'build_type' to support different lib paths for RELEASE or DEBUG builds
#
macro(_find_library var build_type)
  # cmake-lint: disable=C0103

  find_library(${var} ${ARGN})

  if(${var})
    # If this is the first library found then save ${_pkg}_LIBRARY_DIR_[RELEASE,DEBUG].
    if(NOT ${_pkg}_LIBRARY_DIR_${build_type})
      get_filename_component(_dir "${${var}}" PATH)
      set(${_pkg}_LIBRARY_DIR_${build_type}
          "${_dir}"
          CACHE PATH "${_pkg} library directory ${build_type}" FORCE)
    endif()
  elseif(_${_pkg}_FIND_LIBRARY_HINTS_FOR_COMPONENT)
    # Try component-specific hints but do not save ${_pkg}_LIBRARY_DIR_[RELEASE,DEBUG].
    find_library(${var} HINTS ${_${_pkg}_FIND_LIBRARY_HINTS_FOR_COMPONENT} ${ARGN})
  endif()

  # If ${_pkg}_LIBRARY_DIR_[RELEASE,DEBUG] is known then search only there.
  if(${_pkg}_LIBRARY_DIR_${build_type})
    set(_${_pkg}_LIBRARY_SEARCH_DIRS_${build_type} ${${_pkg}_LIBRARY_DIR_${build_type}} NO_DEFAULT_PATH
                                                   NO_CMAKE_FIND_ROOT_PATH)
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_LIBRARY_DIR_${build_type}")
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                     "_${_pkg}_LIBRARY_SEARCH_DIRS_${build_type}")
  endif()
endmacro()

#
# Find the given program (var).
#
macro(_find_program var)
  # cmake-lint: disable=C0103

  find_program(${var} ${ARGN})

  if(${var})
    # If this is the first executable found then save ${_pkg}_EXECUTABLE_DIR.
    if(NOT ${_pkg}_EXECUTABLE_DIR)
      get_filename_component(_dir "${${var}}" DIRECTORY)
      set(${_pkg}_EXECUTABLE_DIR
          "${_dir}"
          CACHE PATH "${_pkg} executable directory" FORCE)
    endif()
  elseif(_${_pkg}_FIND_EXECUTABLE_HINTS_FOR_COMPONENT)
    # Try component-specific hints but do not save ${_pkg}_EXECUTABLE_DIR.
    find_program(${var} HINTS ${_${_pkg}_FIND_EXECUTABLE_HINTS_FOR_COMPONENT} ${ARGN})
  endif()

  # If ${_pkg}_EXECUTABLE_DIR is known then search only there.
  if(${_pkg}_EXECUTABLE_DIR)
    set(_${_pkg}_EXECUTABLE_SEARCH_DIRS ${${_pkg}_EXECUTABLE_DIR} NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_EXECUTABLE_DIR")
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_${_pkg}_EXECUTABLE_SEARCH_DIRS")
  endif()
endmacro()

# ------------------------------------------------------------------------------
# Determine if any missing dependencies require adding to the component list.
#
# Sets ${_pkg}_${COMPONENT}_DEPENDENCIES for each required component
#
# * componentvar - the component list variable name
# * extravar - the indirect dependency list variable name
#
function(_missing_dependencies componentvar extravar)
  # * _${_pkg}_unprocessed_components - list of components requiring processing
  # * _${_pkg}_processed_components - components already processed (or currently being processed)
  # * _${_pkg}_new_components - new components discovered for future processing

  list(APPEND _${_pkg}_unprocessed_components ${${componentvar}})

  while(_${_pkg}_unprocessed_components)
    list(APPEND _${_pkg}_processed_components ${_${_pkg}_unprocessed_components})
    foreach(component ${_${_pkg}_unprocessed_components})
      string(TOUPPER ${component} uppercomponent)
      # set(${_ret} ${${_pkg}_${component}_DEPENDENCIES} PARENT_SCOPE)

      foreach(componentdep ${${_pkg}_${component}_DEPENDENCIES})
        if(NOT ("${componentdep}" IN_LIST _${_pkg}_processed_components OR "${componentdep}" IN_LIST
                                                                           _${_pkg}_new_components))
          list(APPEND _${_pkg}_new_components ${componentdep})
        endif()
      endforeach()
    endforeach()
    set(_${_pkg}_unprocessed_components ${_${_pkg}_new_components})
    unset(_${_pkg}_new_components)
  endwhile()
  set(_${_pkg}_extra_components ${_${_pkg}_processed_components})
  if(_${_pkg}_extra_components AND ${componentvar})
    list(REMOVE_ITEM _${_pkg}_extra_components ${${componentvar}})
  endif()
  set(${componentvar}
      ${_${_pkg}_processed_components}
      PARENT_SCOPE)
  set(${extravar}
      ${_${_pkg}_extra_components}
      PARENT_SCOPE)
endfunction()

# Update library search directories using some pre-built paths
function(_update_library_search_dirs_with_prebuilt_paths componentlibvar basedir)
  if(COMMAND ${_pkg}_update_library_search_dirs_with_prebuilt_paths)
    cmake_language(CALL ${_pkg}_update_library_search_dirs_with_prebuilt_paths ${componentlibvar} ${basedir})
  endif()

  set(${componentlibvar}
      ${${componentlibvar}}
      PARENT_SCOPE)
endfunction()

# Update executable search directories using some pre-built paths
function(_update_exec_search_dirs_with_prebuilt_paths componentlibvar basedir)
  if(COMMAND ${_pkg}_update_exec_search_dirs_with_prebuilt_paths)
    cmake_language(CALL ${_pkg}_update_exec_search_dirs_with_prebuilt_paths ${componentlibvar} ${basedir})
  endif()

  set(${componentlibvar}
      ${${componentlibvar}}
      PARENT_SCOPE)
endfunction()

# Copy IMPORTED_LOCATION* to IMPORTED_IMPLIB*
function(_copy_imported_location_to_implib target imported implib)
  get_target_property(_imported_location ${target} ${imported})
  if(_imported_location)
    set_target_properties(${target} PROPERTIES ${implib} "${_imported_location}")
  endif()
endfunction()

# Set import library paths on Windows
function(_windows_set_import_library_path target basename)
  if(COMMAND ${_pkg}_windows_set_import_library_path)
    cmake_language(CALL ${_pkg}_windows_set_import_library_path ${target} ${basename})
  elseif(
    MSYS
    OR MINGW
    OR CYGWIN)
    get_target_property(_type ${target} TYPE)
    if(_type MATCHES ".*_LIBRARY")
      _copy_imported_location_to_implib(${target} IMPORTED_LOCATION IMPORTED_IMPLIB)
      foreach(_name DEBUG RELEASE NONE NOCONFIG)
        _copy_imported_location_to_implib(${target} IMPORTED_LOCATION_${_name} IMPORTED_IMPLIB_${_name})
      endforeach()
    endif()
  endif()
endfunction()

# Extract version from the library file extension
function(_extract_version_from_lib_name pkg_name lib_path)
  if(CMAKE_VERSION VERSION_LESS 3.19)
    return()
  endif()

  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
    file(REAL_PATH "${lib_path}" lib_path)
  else()
    get_filename_component(lib_path "${lib_path}" REALPATH)
  endif()

  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
    cmake_path(GET lib_path EXTENSION _ext)
  else()
    get_filename_component(_ext "${lib_path}" EXT)
  endif()

  if(_ext MATCHES "([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    set(${pkg_name}_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(${pkg_name}_VERSION_MINOR ${CMAKE_MATCH_2})
    set(${pkg_name}_VERSION_PATCH ${CMAKE_MATCH_3})
  elseif(_ext MATCHES "([0-9]+)\\.([0-9]+)")

    set(${pkg_name}_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(${pkg_name}_VERSION_MINOR ${CMAKE_MATCH_2})
    set(${pkg_name}_VERSION_PATCH 0)
  endif()

  if((NOT "${${pkg_name}_VERSION_MAJOR}" STREQUAL "") AND (NOT "${${pkg_name}_VERSION_MINOR}" STREQUAL ""))
    _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                 "Extracting version information from filename: ${lib_path}")

    set(${pkg_name}_VERSION_MAJOR
        ${${pkg_name}_VERSION_MAJOR}
        PARENT_SCOPE)
    set(${pkg_name}_VERSION_MINOR
        ${${pkg_name}_VERSION_MINOR}
        PARENT_SCOPE)
    set(${pkg_name}_VERSION_PATCH
        ${${pkg_name}_VERSION_PATCH}
        PARENT_SCOPE)
    set(${pkg_name}_VERSION
        ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}
        PARENT_SCOPE)
    set(${pkg_name}_HAS_VERSION
        TRUE
        CACHE BOOL "If ${pkg_name} version was found" FORCE)
    set(${pkg_name}_HAS_VERSION
        TRUE
        PARENT_SCOPE)
  endif()
endfunction()

# Initialize some component-related variables
macro(_init_component_vars components)
  set(_${_pkg}_find_types)
  foreach(_comp ${${components}})
    set(${_pkg}_FIND_REQUIRED_${_comp} ${${_pkg}_FIND_REQUIRED})
    if(NOT ${_pkg}_${_comp}_TYPE)
      set(${_pkg}_${_comp}_TYPE "LIBRARY")
    elseif((NOT "${${_pkg}_${_comp}_TYPE}" STREQUAL "LIBRARY") AND (NOT "${${_pkg}_${_comp}_TYPE}" STREQUAL "EXECUTABLE"
                                                                   ))
      message(FATAL_ERROR "Invalid value for ${_pkg}_${_comp}_TYPE: ${${_pkg}_${_comp}_TYPE}")
    endif()
    list(APPEND _${_pkg}_find_types ${${_pkg}_${_comp}_TYPE})
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_FIND_REQUIRED_${_comp}")
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_${_comp}_TYPE")
  endforeach()
endmacro()

# ==============================================================================
# Initialize some helper variables

if("${${_pkg}_NAMESPACE}" STREQUAL "")
  set(${_pkg}_NAMESPACE ${_pkg})
endif()

if("${${_pkg}_DEFINE_PREFIX}" STREQUAL "")
  string(TOUPPER ${_pkg} ${_pkg}_DEFINE_PREFIX)
endif()

# ==============================================================================
# Before we go searching, check whether a ${_pkg} cmake package is available, unless the user specifically asked NOT to
# search for one.
#
# If ${_pkg}_DIR is set, this behaves as any find_package call would. If not, it looks at ${_pkg}_ROOT and ${_pkg}ROOT
# to find ${_pkg}.
#
if(NOT ${_pkg}_NO_CMAKE)
  # If ${_pkg}_DIR is not set, look for ${_pkg}ROOT and ${_pkg}_ROOT as alternatives, since these are more conventional
  # for ${_pkg}.
  if("$ENV{${_pkg}_DIR}" STREQUAL "")
    if(NOT "$ENV{${_pkg}_ROOT}" STREQUAL "")
      set(ENV{${_pkg}_DIR} $ENV{${_pkg}_ROOT})
    elseif(NOT "$ENV{${_pkg}ROOT}" STREQUAL "")
      set(ENV{${_pkg}_DIR} $ENV{${_pkg}ROOT})
    endif()
  endif()

  set(_${_pkg}_FIND_PACKAGE_ARGS "")
  if(${_pkg}_NO_SYSTEM_PATHS)
    list(APPEND _${_pkg}_FIND_PACKAGE_ARGS NO_CMAKE_SYSTEM_PATH NO_SYSTEM_ENVIRONMENT_PATH)
  endif()

  # Additional components may be required via component dependencies. Add any missing components to the list.
  _missing_dependencies(${_pkg}_FIND_COMPONENTS _${_pkg}_EXTRA_FIND_COMPONENTS)
  _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_${_pkg}_EXTRA_FIND_COMPONENTS")
  _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_FIND_COMPONENTS")
  _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_CMAKE_CONFIG_NO_COMPONENTS")

  if(${_pkg}_CMAKE_CONFIG_NO_COMPONENTS)
    set(_${_pkg}_FIND_COMPONENTS_ORIG ${${_pkg}_FIND_COMPONENTS})
    set(${_pkg}_FIND_COMPONENTS)
  endif()

  # Do the same find_package call but look specifically for the CMake version. Note that args are passed in the
  # ${_pkg}_FIND_xxxxx variables, so there is no need to delegate them to this find_package call.
  _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
               "find_package(${_pkg} QUIET CONFIG ${_${_pkg}_FIND_PACKAGE_ARGS})")
  find_package(${_pkg} QUIET CONFIG ${_${_pkg}_FIND_PACKAGE_ARGS})

  if(_${_pkg}_FIND_COMPONENTS_ORIG)
    set(${_pkg}_FIND_COMPONENTS ${_${_pkg}_FIND_COMPONENTS_ORIG})
  endif()
  _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_FIND_COMPONENTS")

  if(DEFINED ${_pkg}_DIR)
    mark_as_advanced(${_pkg}_DIR)
  endif()

  # If we found a ${_pkg} cmake package, then we're done. Print out what we found. Otherwise let the rest of the module
  # try to find it.
  if(${_pkg}_FOUND)
    if(COMMAND ${_pkg}_post_process)
      cmake_language(CALL ${_pkg}_post_process ${_pkg})
    endif()

    # Convert component found variables to standard variables if required
    if(${_pkg}_FIND_COMPONENTS)
      foreach(_comp IN LISTS ${_pkg}_FIND_COMPONENTS)
        if(DEFINED ${_pkg}_${_comp}_FOUND)
          # Nothing to do
        elseif(TARGET ${${_pkg}_NAMESPACE}::${_comp})
          set(${_pkg}_${_comp}_FOUND TRUE)
        endif()
        string(TOUPPER ${_comp} _uppercomp)
        if(DEFINED ${_pkg}_${_uppercomp}_FOUND)
          set(${_pkg}_${_comp}_FOUND ${${_pkg}_${_uppercomp}_FOUND})
        endif()
      endforeach()
    endif()

    find_package_handle_standard_args(${_pkg} HANDLE_COMPONENTS CONFIG_MODE)

    # Restore project's policies
    cmake_policy(POP)
    return()
  endif()
endif()

# ==============================================================================
# Default initialization of components to search for

if(NOT ${_pkg}_FIND_COMPONENTS)
  set(${_pkg}_FIND_COMPONENTS ${${_pkg}_DEFAULT_COMPONENTS})
endif()

_init_component_vars(${_pkg}_FIND_COMPONENTS)
if("EXECUTABLE" IN_LIST _${_pkg}_find_types)
  set(${_pkg}_NO_PKGCONFIG ON)
  _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
               "Disabling PkgConfig search since searching for at least one executable")
endif()

# ==============================================================================
# Before we go searching by hand, check whether we can find PkgConfig and look for the package that wayunless the user
# specifically asked NOT to for one.
#
if(NOT ${_pkg}_NO_PKGCONFIG)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                 "Trying to find package ${_pkg} using PkgConfig: ${PKG_CONFIG_EXECUTABLE}")

    if(${_pkg}_FIND_COMPONENTS)
      set(_${_pkg}_pkgconfig_found)
      set(_${_pkg}_pkgconfig_args)
      if(NOT ${_pkg}_DEBUG)
        list(APPEND _${_pkg}_pkgconfig_args QUIET)
      endif()

      foreach(_comp ${${_pkg}_FIND_COMPONENTS})
        _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "Looking for component: ${_comp}")
        set(_prefix ${_pkg}_${_comp})

        list(APPEND ${_prefix}_PKGCONFIG_NAMES ${${_pkg}_${_comp}_NAMES})
        foreach(_name ${${_pkg}_${_comp}_NAMES})
          list(APPEND ${_prefix}_PKGCONFIG_NAMES "lib${_name}")
        endforeach()
        list(REMOVE_DUPLICATES ${_pkg}_${_comp}_PKGCONFIG_NAMES)

        _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_${_comp}_PKGCONFIG_NAMES")

        if(NOT ${_pkg}_${_comp}_PKGCONFIG_NAMES)
          _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                       "No names for component ${_comp} of ${_pkg}, skipping PkgConfig search for it")
          continue()
        endif()

        _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                     "pkg_search_module(${_prefix} ${_${_pkg}_pkgconfig_args} ${${_pkg}_${_comp}_PKGCONFIG_NAMES})")

        list(APPEND CMAKE_MESSAGE_INDENT "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] ")
        pkg_search_module(${_prefix} ${_${_pkg}_pkgconfig_args} ${${_pkg}_${_comp}_PKGCONFIG_NAMES})
        list(POP_BACK CMAKE_MESSAGE_INDENT)
        if(${_prefix}_FOUND)
          list(APPEND _${_pkg}_pkgconfig_found ${_comp})
          _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_prefix}_INCLUDE_DIRS")
          _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_prefix}_LINK_LIBRARIES")
          _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_prefix}_LDFLAGS_OTHER")
          _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_prefix}_CFLAGS_OTHER")
        endif()
      endforeach()

      # If we found a ${_pkg} PkgConfig package, then we're done. Print out what we found. Otherwise let the rest of the
      # module try to find it.
      if(_${_pkg}_pkgconfig_found)
        set(${_pkg}_FOUND TRUE)
        list(GET _${_pkg}_pkgconfig_found 0 _comp)
        set(${_pkg}_VERSION ${${_pkg}_${_comp}_VERSION})
        if(${_pkg}_VERSION VERSION_GREATER 0)
          set(${_pkg}_HAS_VERSION
              TRUE
              CACHE BOOL "If ${_pkg} version was found" FORCE)
        else()
          set(${_pkg}_HAS_VERSION
              FALSE
              CACHE BOOL "If ${_pkg} version was found" FORCE)
        endif()

        # Create the imported targets
        foreach(_comp ${_${_pkg}_pkgconfig_found})
          set(_tgt_name ${${_pkg}_NAMESPACE}::${_comp})
          set(_prefix ${_pkg}_${_comp})

          set(_imported_location)
          string(REPLACE ";" "|" _regex "${${_pkg}_${_comp}_NAMES}")
          foreach(_lib ${${_prefix}_LINK_LIBRARIES})
            if(EXISTS "${_lib}" AND _lib MATCHES ".*(${_regex}).*")
              set(_imported_location "${_lib}")
              break()
            endif()
          endforeach()

          set(_lib_type UNKNOWN)
          if(_imported_location)
            _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_imported_location")
            # NB: not perfect but should work in most cases..
            if(_imported_location MATCHES [[\.(so|dylib|dll|.dll.a)]])
              set(_lib_type SHARED)
            endif()
            list(REMOVE_ITEM ${_prefix}_LINK_LIBRARIES ${_imported_location})
            if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
              file(REAL_PATH "${_imported_location}" _imported_location)
            else()
              get_filename_component(_imported_location "${_imported_location}" REALPATH)
            endif()
          endif()

          if(NOT TARGET ${_tgt_name})
            add_library(${_tgt_name} ${_lib_type} IMPORTED)
          endif()
          if(${_prefix}_INCLUDE_DIRS)
            set_target_properties(${_tgt_name} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${${_prefix}_INCLUDE_DIRS}")
          endif()
          if(${_prefix}_LINK_LIBRARIES)
            set_target_properties(${_tgt_name} PROPERTIES INTERFACE_LINK_LIBRARIES "${${_prefix}_LINK_LIBRARIES}")
          endif()
          if(_imported_location)
            set_target_properties(${_tgt_name} PROPERTIES IMPORTED_LOCATION "${_imported_location}")
            if(MSYS
               OR MINGW
               OR CYGWIN)
              set_target_properties(${_tgt_name} PROPERTIES IMPORTED_IMPLIB "${_imported_location}")
            endif()
          endif()
          if(${_prefix}_LDFLAGS_OTHER)
            set_target_properties(${_tgt_name} PROPERTIES INTERFACE_LINK_OPTIONS "${${_prefix}_LDFLAGS_OTHER}")
          endif()
          if(${_prefix}_CFLAGS_OTHER)
            set_target_properties(${_tgt_name} PROPERTIES INTERFACE_COMPILE_OPTIONS "${${_prefix}_CFLAGS_OTHER}")
          endif()
          list(APPEND ${_pkg}_IMPORTED_TARGETS ${_tgt_name})
        endforeach()
        list(REMOVE_DUPLICATES ${_pkg}_IMPORTED_TARGETS)

        # NB: no adjustments with ${_pkg}_${_uppercomp}_DEPENDENCIES here since PkgConfig takes cares of this for us.
        # This has the downside that the dependencies are included into one single CMake targets rather than using
        # CMake's target dependency mechanisms.

        if(${_pkg}_${_comp}_EXTERNAL_DEPENDENCIES)
          target_link_libraries(${${_pkg}_NAMESPACE}::${_comp} INTERFACE "${${_pkg}_${_comp}_EXTERNAL_DEPENDENCIES}")
        endif()

        if(COMMAND ${_pkg}_post_process)
          cmake_language(CALL ${_pkg}_post_process ${_pkg})
        endif()

        if(${_pkg}_HAS_VERSION)
          find_package_handle_standard_args(
            ${_pkg}
            VERSION_VAR ${_pkg}_VERSION
            HANDLE_COMPONENTS REASON_FAILURE_MESSAGE "PkgConfig search failed")
        else()
          find_package_handle_standard_args(${_pkg} HANDLE_COMPONENTS REASON_FAILURE_MESSAGE "PkgConfig search failed")
        endif()

        # ======================================================================

        if(${_pkg}_DEBUG)
          list(APPEND CMAKE_MESSAGE_INDENT "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] ")
          cmake_print_properties(
            TARGETS ${${_pkg}_IMPORTED_TARGETS}
            PROPERTIES INTERFACE_COMPILE_OPTIONS INTERFACE_INCLUDE_DIRECTORIES INTERFACE_LINK_OPTIONS
                       INTERFACE_LINK_LIBRARIES IMPORTED_LOCATION IMPORTED_IMPLIB)
          list(POP_BACK CMAKE_MESSAGE_INDENT)
          list(POP_BACK CMAKE_MESSAGE_INDENT)
        endif()

        # ======================================================================
        # Finalize

        # Report ${_pkg}_LIBRARIES and ${_pkg}_INCLUDE_DIRS
        set(${_pkg}_INCLUDE_DIRS "")
        set(${_pkg}_LIBRARIES "")
        foreach(_target IN LISTS ${_pkg}_IMPORTED_TARGETS)
          get_target_property(_include_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
          if(_include_dirs)
            list(APPEND ${_pkg}_INCLUDE_DIRS ${_include_dirs})
          endif()
          get_target_property(_imported_location ${_target} IMPORTED_LOCATION)
          if(_imported_location)
            list(APPEND ${_pkg}_LIBRARIES ${_imported_location})
          endif()
        endforeach()
        list(REMOVE_DUPLICATES ${_pkg}_INCLUDE_DIRS)
        list(REMOVE_DUPLICATES ${_pkg}_LIBRARIES)

        # Configure display of cache entries in GUI.
        foreach(var ${_pkg}ROOT ${_pkg}_ROOT ${_${_pkg}_VARS_INC} ${_${_pkg}_VARS_LIB})
          get_property(
            _type
            CACHE ${var}
            PROPERTY TYPE)
          if(_type)
            set_property(CACHE ${var} PROPERTY ADVANCED 1)
            if("x${_type}" STREQUAL "xUNINITIALIZED")
              set_property(CACHE ${var} PROPERTY TYPE PATH)
            endif()
          endif()
        endforeach()

        # Record last used values of input variables so we can detect on the next run if the user changed them.
        foreach(var ${_${_pkg}_VARS_INC} ${_${_pkg}_VARS_LIB} ${_${_pkg}_VARS_DIR} ${_${_pkg}_VARS_NAME})
          if(DEFINED ${var})
            set(_${var}_LAST
                "${${var}}"
                CACHE INTERNAL "Last used ${var} value.")
          else()
            unset(_${var}_LAST CACHE)
          endif()
        endforeach()

        # Maintain a persistent list of components requested anywhere since the last flush.
        set(_${_pkg}_COMPONENTS_SEARCHED "${_${_pkg}_COMPONENTS_SEARCHED}")
        list(APPEND _${_pkg}_COMPONENTS_SEARCHED ${${_pkg}_FIND_COMPONENTS})
        list(REMOVE_DUPLICATES _${_pkg}_COMPONENTS_SEARCHED)
        list(SORT _${_pkg}_COMPONENTS_SEARCHED)
        set(_${_pkg}_COMPONENTS_SEARCHED
            "${_${_pkg}_COMPONENTS_SEARCHED}"
            CACHE INTERNAL "Components requested for this build tree.")

        # Restore project's policies
        cmake_policy(POP)
        return()
      endif()
      _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                   "PkgConfig search failed for at least one component. Proceeding with normal search...")
    else()
      _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                   "Skipping PkgConfig since no COMPONENTS have been specified (not yet supported)")
    endif()
  else()
    _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                 "Skipping PkgConfig since no pkg-config executable could not be found.")
  endif()
endif()

# ==============================================================================
# Start of main.
# ==============================================================================

# cmake-lint: disable=C0103

# If the user sets ${_pkg}_LIBRARY_DIR, use it as the default for both configurations.
if(NOT ${_pkg}_LIBRARY_DIR_RELEASE AND ${_pkg}_LIBRARY_DIR)
  set(${_pkg}_LIBRARY_DIR_RELEASE "${${_pkg}_LIBRARY_DIR}")
endif()
if(NOT ${_pkg}_LIBRARY_DIR_DEBUG AND ${_pkg}_LIBRARY_DIR)
  set(${_pkg}_LIBRARY_DIR_DEBUG "${${_pkg}_LIBRARY_DIR}")
endif()

if(NOT DEFINED ${_pkg}_FIND_RELEASE_ONLY)
  set(${_pkg}_FIND_RELEASE_ONLY FALSE)
endif()
if(NOT DEFINED ${_pkg}_USE_DEBUG_LIBS AND NOT ${_pkg}_FIND_RELEASE_ONLY)
  set(${_pkg}_USE_DEBUG_LIBS TRUE)
endif()
if(NOT DEFINED ${_pkg}_USE_RELEASE_LIBS)
  set(${_pkg}_USE_RELEASE_LIBS TRUE)
endif()

_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_FIND_RELEASE_ONLY")
_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_NO_SYSTEM_PATHS")

# ------------------------------------------------------------------------------

# Collect environment variable inputs as hints.  Do not consider changes.
foreach(var ${_pkg}ROOT ${_pkg}_ROOT ${_pkg}_INCLUDEDIR ${_pkg}_LIBRARYDIR ${_pkg}_EXECDIR)
  set(_env $ENV{${var}})
  if(_env)
    file(TO_CMAKE_PATH "${_env}" _ENV_${var})
  else()
    set(_ENV_${var} "")
  endif()
endforeach()
if(NOT _ENV_${_pkg}_ROOT AND _ENV_${_pkg}ROOT)
  set(_ENV_${_pkg}_ROOT "${_ENV_${_pkg}ROOT}")
endif()

# Collect inputs and cached results. Detect changes since the last run.
if(NOT ${_pkg}_ROOT AND ${_pkg}ROOT)
  set(${_pkg}_ROOT "${${_pkg}ROOT}")
endif()
set(_${_pkg}_VARS_DIR ${_pkg}_ROOT ${_pkg}_NO_SYSTEM_PATHS)

_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_ROOT")
_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_ROOT" ENVIRONMENT)
_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_INCLUDEDIR")
_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_INCLUDEDIR" ENVIRONMENT)
_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_LIBRARYDIR")
_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_LIBRARYDIR" ENVIRONMENT)
_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_EXECDIR")
_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_EXECDIR" ENVIRONMENT)

# ==============================================================================
# Search for ${_pkg} include DIR

set(_${_pkg}_VARS_INC ${_pkg}_INCLUDEDIR ${_pkg}_INCLUDE_DIR)
_change_detect(_${_pkg}_CHANGE_INCDIR ${_${_pkg}_VARS_DIR} ${_${_pkg}_VARS_INC})
# Clear ${_pkg}_INCLUDE_DIR if it did not change but other input affecting the location did.  We will find a new one
# based on the new inputs.
if(_${_pkg}_CHANGE_INCDIR AND NOT _${_pkg}_INCLUDE_DIR_CHANGED)
  unset(${_pkg}_INCLUDE_DIR CACHE)
endif()

if(NOT ${_pkg}_INCLUDE_DIR)
  set(_${_pkg}_INCLUDE_SEARCH_DIRS "")
  if(${_pkg}_INCLUDEDIR)
    list(APPEND _${_pkg}_INCLUDE_SEARCH_DIRS ${${_pkg}_INCLUDEDIR})
  elseif(_ENV_${_pkg}_INCLUDEDIR)
    list(APPEND _${_pkg}_INCLUDE_SEARCH_DIRS ${_ENV_${_pkg}_INCLUDEDIR})
  endif()

  if(${_pkg}_ROOT)
    list(APPEND _${_pkg}_INCLUDE_SEARCH_DIRS ${${_pkg}_ROOT})
  elseif(_ENV_${_pkg}_ROOT)
    list(APPEND _${_pkg}_INCLUDE_SEARCH_DIRS ${_ENV_${_pkg}_ROOT})
  endif()

  if(${_pkg}_NO_SYSTEM_PATHS)
    list(APPEND _${_pkg}_INCLUDE_SEARCH_DIRS NO_CMAKE_SYSTEM_PATH NO_SYSTEM_ENVIRONMENT_PATH)
  else()
    list(
      APPEND
      _${_pkg}_INCLUDE_SEARCH_DIRS
      PATHS
      /usr
      /usr/local/
      /usr/local/homebrew # Mac OS X
      /opt
      /opt/local
      /opt/local/var/macports/software # Mac OS X.
      /sw/local)

    if(WIN32)
      list(
        APPEND
        _${_pkg}_INCLUDE_SEARCH_DIRS
        "$ENV{LOCALAPPDATA}/${_pkg}"
        "$ENV{ALLUSERSPROFILE}/${_pkg}"
        "C:/Program Files/${_pkg}"
        "C:/Program Files (x86)/${_pkg}"
        C:/${_pkg})
    endif()

    if(${_pkg}_INC_SYSTEM_PATHS)
      list(APPEND _${_pkg}_INCLUDE_SEARCH_DIRS ${${_pkg}_INC_SYSTEM_PATHS})
    endif()
  endif()

  _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_${_pkg}_INCLUDE_SEARCH_DIRS")
  _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_INCLUDE_PATH_SUFFIXES")

  # Look for a standard ${_pkg} header file.
  find_path(
    ${_pkg}_INCLUDE_DIR
    NAMES ${${_pkg}_INCLUDE_FILE}
    HINTS ${_${_pkg}_INCLUDE_SEARCH_DIRS}
    PATH_SUFFIXES ${${_pkg}_INCLUDE_PATH_SUFFIXES})

  find_file(
    _${_pkg}_include_file
    NAMES ${${_pkg}_INCLUDE_FILE}
    HINTS ${${_pkg}_INCLUDE_DIR}
    PATH_SUFFIXES ${${_pkg}_INCLUDE_PATH_SUFFIXES}
    NO_DEFAULT_PATH)

  if(COMMAND ${_pkg}_include_dir_post_process)
    cmake_language(CALL ${_pkg}_include_dir_post_process ${_pkg}_INCLUDE_DIR ${${_pkg}_INCLUDE_DIR})
  elseif(${_pkg}_INCLUDE_DIR_UP_INDEX)
    # cmake-lint: disable=E1120
    foreach(_ RANGE 1 ${${_pkg}_INCLUDE_DIR_UP_INDEX})
      if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
        cmake_path(GET ${_pkg}_INCLUDE_DIR PARENT_PATH ${_pkg}_INCLUDE_DIR)
      else()
        get_filename_component(${_pkg}_INCLUDE_DIR ${${_pkg}_INCLUDE_DIR} DIRECTORY)
      endif()
    endforeach()
  endif()
endif()

_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_INCLUDE_DIR")

# ------------------------------------------------------------------------
# Extract version information from version.hpp
# ------------------------------------------------------------------------

if(${_pkg}_INCLUDE_DIR)
  set(${_pkg}_HAS_VERSION
      FALSE
      CACHE BOOL "If ${_pkg} version was found" FORCE)
  if(COMMAND ${_pkg}_version_function)
    cmake_language(CALL ${_pkg}_version_function ${${_pkg}_INCLUDE_DIR})
    set(${_pkg}_HAS_VERSION
        TRUE
        CACHE BOOL "If ${_pkg} version was found" FORCE)
  else()
    set(_${_pkg}_version_file_list "${_${_pkg}_include_file}")

    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20.0)
      cmake_path(GET ${_pkg}_INCLUDE_FILE STEM _${_pkg}_include_basename)
    else()
      get_filename_component(_${_pkg}_include_basename "${${_pkg}_INCLUDE_FILE}" NAME_WE)
    endif()

    string(TOLOWER ${_pkg} _${_pkg}_lowercase)
    set(_${_pkg}_inc_ver_ext h hh hpp hxx h++)
    set(_${_pkg}_inc_ver_suffixes -version _version -config _config)
    foreach(_${_pkg}_suffix ${_${pkg}_inc_ver_suffixes})
      foreach(_${_pkg}_ext ${_${_pkg}_inc_ver_ext})
        foreach(_${_pkg}_dir ${${_pkg}_INCLUDE_DIR} ${${_pkg}_INCLUDE_DIR}/${_${_pkg}_lowercase})
          set(_${_pkg}_fname "${_${_pkg}_dir}/${_${_pkg}_include_basename}${_${_pkg}_suffix}.${_${_pkg}_ext}")
          if(EXISTS "${_${_pkg}_fname}")
            list(APPEND _${_pkg}_version_file_list "${_${_pkg}_fname}")
          endif()
        endforeach()
      endforeach()
    endforeach()

    foreach(_${_pkg}_file ${_${_pkg}_version_file_list})
      _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                   "Attempting to read library version from: ${_${_pkg}_file}")

      file(READ ${_${_pkg}_file} _${_pkg}_include_content)
      string(REGEX MATCHALL "#define[ \t]+[a-zA-Z0-9_]+VERSION[a-zA-Z0-9_ ]+" _${_pkg}_version_defines
                   "${_${_pkg}_include_content}")
      list(APPEND _${_pkg}_version_defines "")

      # Try to determine the version number using one of:
      #
      # * `#define PKG_VERSION X.Y.Z`
      # * `#define PKG_VERSION X.Y`
      # * `#define PKG_VERSION_{MAJOR,MINOR,PATCH,PATCHLEVEL} X`
      # * `#define PKG_{MAJOR,MINOR,PATCH,PATCHLEVEL}_VERSION X`
      #
      # Note the special case where `#define PKG_VERSION X` is interpreted as MAJOR version number

      if("${_${_pkg}_version_defines}" MATCHES
         ".*#define[ \t]+${${_pkg}_DEFINE_PREFIX}_VERSION[ \t]+([0-9]+)\\.([0-9]+)\\.*([0-9]+).*")
        _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                     "  version read from ${${_pkg}_DEFINE_PREFIX}_VERSION")
        set(${_pkg}_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(${_pkg}_VERSION_MINOR ${CMAKE_MATCH_2})
        set(${_pkg}_VERSION_PATCH ${CMAKE_MATCH_3})
        set(${_pkg}_VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3})
        break()
      elseif("${_${_pkg}_version_defines}" MATCHES
             ".*#define[ \t]+${${_pkg}_DEFINE_PREFIX}_VERSION[ \t]+([0-9]+)\\.([0-9]+).*")
        _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                     "  version read from ${${_pkg}_DEFINE_PREFIX}_VERSION")
        set(${_pkg}_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(${_pkg}_VERSION_MINOR ${CMAKE_MATCH_2})
        set(${_pkg}_VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2})
        break()
      else()
        # NB: for MAJOR versions we also look at the XXX_VERSION variable if its value consists of a single integer
        set(_${_pkg}_MAJOR_NAMES ${${_pkg}_DEFINE_PREFIX}_MAJOR_VERSION ${${_pkg}_DEFINE_PREFIX}_VERSION_MAJOR
                                 ${${_pkg}_DEFINE_PREFIX}_MAJOR ${${_pkg}_DEFINE_PREFIX}_VERSION)
        set(_${_pkg}_MINOR_NAMES ${${_pkg}_DEFINE_PREFIX}_MINOR_VERSION ${${_pkg}_DEFINE_PREFIX}_VERSION_MINOR
                                 ${${_pkg}_DEFINE_PREFIX}_MINOR)
        set(_${_pkg}_PATCH_NAMES
            ${${_pkg}_DEFINE_PREFIX}_PATCH_VERSION ${${_pkg}_DEFINE_PREFIX}_VERSION_PATCH
            ${${_pkg}_DEFINE_PREFIX}_PATCH ${${_pkg}_DEFINE_PREFIX}_PATCHLEVEL_VERSION
            ${${_pkg}_DEFINE_PREFIX}_VERSION_PATCHLEVEL)

        foreach(_${_pkg}_type MAJOR MINOR PATCH)
          foreach(_var ${_${_pkg}_${_${_pkg}_type}_NAMES})
            if("${_${_pkg}_version_defines}" MATCHES "#define[ \t]+${_var}[ \t]+([0-9]+)[^\\.]")
              _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                           "  ${_${_pkg}_type} version read from ${_var}")
              set(${_pkg}_VERSION_${_${_pkg}_type} ${CMAKE_MATCH_1})
              break()
            endif()
          endforeach()
        endforeach()
        if((NOT "${${_pkg}_VERSION_MAJOR}" STREQUAL "") AND (NOT "${${_pkg}_VERSION_MINOR}" STREQUAL ""))
          break()
        endif()
      endif()
    endforeach()

    if((NOT "${${_pkg}_VERSION_MAJOR}" STREQUAL "") AND (NOT "${${_pkg}_VERSION_MINOR}" STREQUAL ""))
      set(${_pkg}_VERSION "${${_pkg}_VERSION_MAJOR}.${${_pkg}_VERSION_MINOR}")
      if(NOT "${${_pkg}_VERSION_PATCH}" STREQUAL "")
        set(${_pkg}_VERSION "${${_pkg}_VERSION}.${${_pkg}_VERSION_PATCH}")
      endif()
      set(${_pkg}_HAS_VERSION
          TRUE
          CACHE BOOL "If ${_pkg} version was found" FORCE)
    endif()
  endif()

  if(${_pkg}_HAS_VERSION)
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_VERSION")
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_VERSION_MAJOR")
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_VERSION_MINOR")
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_VERSION_PATCH")
  endif()
endif()

# ==============================================================================
# Begin finding ${_pkg} libraries

if(NOT "${${_pkg}_LIB_SYSTEM_PATHS}" STREQUAL "" AND "${${_pkg}_LIB_SYSTEM_PATHS_DEBUG}" STREQUAL "")
  set(${_pkg}_LIB_SYSTEM_PATHS_DEBUG ${${_pkg}_LIB_SYSTEM_PATHS})
endif()
if(NOT "${${_pkg}_LIB_SYSTEM_PATHS}" STREQUAL "" AND "${${_pkg}_LIB_SYSTEM_PATHS_RELEASE}" STREQUAL "")
  set(${_pkg}_LIB_SYSTEM_PATHS_RELEASE ${${_pkg}_LIB_SYSTEM_PATHS})
endif()

set(_${_pkg}_VARS_LIB "")
foreach(config DEBUG RELEASE)
  set(_${_pkg}_VARS_LIB_${config} ${_pkg}_LIBRARYDIR ${_pkg}_LIBRARY_DIR_${config})
  list(APPEND _${_pkg}_VARS_LIB ${_${_pkg}_VARS_LIB_${config}})
  _change_detect(_${_pkg}_CHANGE_LIBDIR_${config} ${_${_pkg}_VARS_DIR} ${_${_pkg}_VARS_LIB_${config}}
                 ${_pkg}_INCLUDE_DIR)
  # Clear ${_pkg}_LIBRARY_DIR_${config} if it did not change but other input affecting the location did.  We will find a
  # new one based on the new inputs.
  if(_${_pkg}_CHANGE_LIBDIR_${config} AND NOT _${_pkg}_LIBRARY_DIR_${config}_CHANGED)
    unset(${_pkg}_LIBRARY_DIR_${config} CACHE)
  endif()

  # If ${_pkg}_LIBRARY_DIR_[RELEASE,DEBUG] is set, prefer its value.
  if(${_pkg}_LIBRARY_DIR_${config})
    set(_${_pkg}_LIBRARY_SEARCH_DIRS_${config} ${${_pkg}_LIBRARY_DIR_${config}} NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  else()
    set(_${_pkg}_LIBRARY_SEARCH_DIRS_${config} "")
    if(${_pkg}_LIBRARYDIR)
      list(APPEND _${_pkg}_LIBRARY_SEARCH_DIRS_${config} ${${_pkg}_LIBRARYDIR})
    elseif(_ENV_${_pkg}_LIBRARYDIR)
      list(APPEND _${_pkg}_LIBRARY_SEARCH_DIRS_${config} ${_ENV_${_pkg}_LIBRARYDIR})
    endif()

    if(${_pkg}_ROOT)
      list(APPEND _${_pkg}_LIBRARY_SEARCH_DIRS_${config} ${${_pkg}_ROOT})
      _update_library_search_dirs_with_prebuilt_paths(_${_pkg}_LIBRARY_SEARCH_DIRS_${config} "${${_pkg}_ROOT}")
    elseif(_ENV_${_pkg}_ROOT)
      list(APPEND _${_pkg}_LIBRARY_SEARCH_DIRS_${config} ${_ENV_${_pkg}_ROOT})
      _update_library_search_dirs_with_prebuilt_paths(_${_pkg}_LIBRARY_SEARCH_DIRS_${config} "${_ENV_${_pkg}_ROOT}")
    endif()

    if(COMMAND ${_pkg}_library_search_dirs_from_include_path)
      cmake_language(CALL ${_pkg}_library_search_dirs_from_include_path ${pkg}_INCLUDE_BASED_SEARCH_PATHS
                     ${${_pkg}_INCLUDE_DIR})
    else()
      # Add some common cases
      set(${_pkg}_INCLUDE_BASED_SEARCH_PATHS ${${_pkg}_INCLUDE_DIR} ${${_pkg}_INCLUDE_DIR}/..
                                             ${${_pkg}_INCLUDE_DIR}/../..)
    endif()
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_INCLUDE_BASED_SEARCH_PATHS")

    foreach(_dir ${${_pkg}_INCLUDE_BASED_SEARCH_PATHS})
      list(APPEND _${_pkg}_LIBRARY_SEARCH_DIRS_${config} ${_dir})
      _update_library_search_dirs_with_prebuilt_paths(_${_pkg}_LIBRARY_SEARCH_DIRS_${config} ${_dir})
    endforeach()

    if(${_pkg}_NO_SYSTEM_PATHS)
      list(APPEND _${_pkg}_LIBRARY_SEARCH_DIRS_${config} NO_CMAKE_SYSTEM_PATH NO_SYSTEM_ENVIRONMENT_PATH)
    elseif(${_pkg}_LIB_SYSTEM_PATHS_${config})
      list(APPEND _${_pkg}_LIBRARY_SEARCH_DIRS_${config} PATHS ${${_pkg}_LIB_SYSTEM_PATHS_${config}})
      foreach(_dir ${${_pkg}_LIB_SYSTEM_PATHS_${config}})
        _update_library_search_dirs_with_prebuilt_paths(_${_pkg}_LIBRARY_SEARCH_DIRS_${config} ${_dir})
      endforeach()
    endif()
  endif()
endforeach()

_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_${_pkg}_LIBRARY_SEARCH_DIRS_RELEASE")
_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_${_pkg}_LIBRARY_SEARCH_DIRS_DEBUG")

# ------------------------------------------------------------------------------
# Begin finding ${_pkg} executables

if("${${_pkg}_EXECUTABLE_PATH_SUFFIXES}" STREQUAL "")
  set(${_pkg}_EXECUTABLE_PATH_SUFFIXES "bin" "sbin")
endif()

set(_${_pkg}_VARS_EXEC ${_pkg}_EXECDIR ${_pkg}_EXECUTABLE_DIR)
_change_detect(_${_pkg}_CHANGE_EXECDIR ${_${_pkg}_VARS_DIR} ${_${_pkg}_VARS_EXEC})
# Clear ${_pkg}_EXECUTABLE_DIR if it did not change but other input affecting the location did.  We will find a new one
# based on the new inputs.
if(_${_pkg}_CHANGE_EXECDIR AND NOT _${_pkg}_EXECUTABLE_DIR_CHANGED)
  unset(${_pkg}_EXECUTABLE_DIR CACHE)
endif()

# If ${_pkg}_EXECUTABLE_DIR is set, prefer its value.
if(${_pkg}_EXECUTABLE_DIR)
  set(_${_pkg}_EXECUTABLE_SEARCH_DIRS ${${_pkg}_EXECUTABLE_DIR} NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
else()
  set(_${_pkg}_EXECUTABLE_SEARCH_DIRS "")
  if(${_pkg}_EXECDIR)
    list(APPEND _${_pkg}_EXECUTABLE_SEARCH_DIRS ${${_pkg}_EXECDIR})
  elseif(_ENV_${_pkg}_EXECDIR)
    list(APPEND _${_pkg}_EXECUTABLE_SEARCH_DIRS ${_ENV_${_pkg}_EXECDIR})
  endif()

  if(${_pkg}_ROOT)
    list(APPEND _${_pkg}_EXECUTABLE_SEARCH_DIRS ${${_pkg}_ROOT})
    _update_exec_search_dirs_with_prebuilt_paths(_${_pkg}_EXECUTABLE_SEARCH_DIRS "${${_pkg}_ROOT}")
  elseif(_ENV_${_pkg}_ROOT)
    list(APPEND _${_pkg}_EXECUTABLE_SEARCH_DIRS ${_ENV_${_pkg}_ROOT})
    _update_exec_search_dirs_with_prebuilt_paths(_${_pkg}_EXECUTABLE_SEARCH_DIRS "${_ENV_${_pkg}_ROOT}")
  endif()

  if(COMMAND ${_pkg}_exec_search_dirs_from_include_path)
    cmake_language(CALL ${_pkg}_exec_search_dirs_from_include_path ${pkg}_INCLUDE_BASED_SEARCH_PATHS
                   ${${_pkg}_INCLUDE_DIR})
  else()
    # Add some common cases
    set(${_pkg}_INCLUDE_BASED_SEARCH_PATHS ${${_pkg}_INCLUDE_DIR} ${${_pkg}_INCLUDE_DIR}/..
                                           ${${_pkg}_INCLUDE_DIR}/../..)
  endif()
  _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_INCLUDE_BASED_SEARCH_PATHS")

  foreach(_dir ${${_pkg}_INCLUDE_BASED_SEARCH_PATHS})
    list(APPEND _${_pkg}_EXECUTABLE_SEARCH_DIRS ${_dir})
    _update_exec_search_dirs_with_prebuilt_paths(_${_pkg}_EXECUTABLE_SEARCH_DIRS ${_dir})
  endforeach()

  if(${_pkg}_NO_SYSTEM_PATHS)
    list(APPEND _${_pkg}_EXECUTABLE_SEARCH_DIRS NO_CMAKE_SYSTEM_PATH NO_SYSTEM_ENVIRONMENT_PATH)
  elseif(${_pkg}_LIB_SYSTEM_PATHS)
    list(APPEND _${_pkg}_EXECUTABLE_SEARCH_DIRS PATHS ${${_pkg}_LIB_SYSTEM_PATHS})
    foreach(_dir ${${_pkg}_LIB_SYSTEM_PATHS})
      _update_exec_search_dirs_with_prebuilt_paths(_${_pkg}_EXECUTABLE_SEARCH_DIRS ${_dir})
    endforeach()
  endif()
endif()

_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_${_pkg}_EXECUTABLE_SEARCH_DIRS")

# ------------------------------------------------------------------------------

if(${_pkg}_FIND_LIBRARY_SUFFIXES)
  set(_${_pkg}_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${${_pkg}_FIND_LIBRARY_SUFFIXES})
elseif(${_pkg}_USE_STATIC_LIBS)
  set(_${_pkg}_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  if(WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
  endif()
elseif("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
  # Properly search for shared libraries on Windows with Visual Studio
  set(_${_pkg}_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .dll)
endif()

if(${_pkg}_FIND_LIBRARY_PREFIXES)
  set(_${_pkg}_ORIG_CMAKE_FIND_LIBRARY_PREFIXES ${CMAKE_FIND_LIBRARY_PREFIXES})
  set(CMAKE_FIND_LIBRARY_PREFIXES ${${_pkg}_FIND_LIBRARY_PREFIXES})
endif()

_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "CMAKE_FIND_LIBRARY_PREFIXES")
_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "CMAKE_FIND_LIBRARY_SUFFIXES")

# Additional components may be required via component dependencies. Add any missing components to the list.
_missing_dependencies(${_pkg}_FIND_COMPONENTS _${_pkg}_EXTRA_FIND_COMPONENTS)
_debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_${_pkg}_EXTRA_FIND_COMPONENTS")
_init_component_vars(${_pkg}_FIND_COMPONENTS)
_init_component_vars(_${_pkg}_EXTRA_FIND_COMPONENTS)

# If the user changed any of our control inputs flush previous results.
if(_${_pkg}_CHANGE_LIBDIR_DEBUG OR _${_pkg}_CHANGE_LIBDIR_RELEASE)
  foreach(COMPONENT ${_${_pkg}_COMPONENTS_SEARCHED})
    string(TOUPPER ${COMPONENT} UPPERCOMPONENT)
    foreach(config DEBUG RELEASE)
      set(_var ${_pkg}_${UPPERCOMPONENT}_LIBRARY_${config})
      unset(${_var} CACHE)
      set(${_var} "${_var}-NOTFOUND")
    endforeach()
  endforeach()
  set(_${_pkg}_COMPONENTS_SEARCHED "")
endif()

if(_${_pkg}_CHANGE_EXECDIR)
  foreach(COMPONENT ${_${_pkg}_COMPONENTS_SEARCHED})
    string(TOUPPER ${COMPONENT} UPPERCOMPONENT)
    set(_var ${_pkg}_${UPPERCOMPONENT}_EXECUTABLE)
    unset(${_var} CACHE)
    set(${_var} "${_var}-NOTFOUND")
  endforeach()
  set(_${_pkg}_COMPONENTS_SEARCHED "")
endif()

foreach(COMPONENT ${${_pkg}_FIND_COMPONENTS})
  string(TOUPPER ${COMPONENT} UPPERCOMPONENT)

  set(_${_pkg}_docstring_release "${_pkg} ${COMPONENT} library (release)")
  set(_${_pkg}_docstring_debug "${_pkg} ${COMPONENT} library (debug)")
  set(_${_pkg}_docstring_executable "${_pkg} ${COMPONENT} executable")

  if("${${_pkg}_${COMPONENT}_TYPE}" STREQUAL "LIBRARY")
    if(${_pkg}_${COMPONENT}_NAMES)
      set(${_pkg}_${COMPONENT}_RELEASE_NAMES ${${_pkg}_${COMPONENT}_NAMES})
      set(${_pkg}_${COMPONENT}_DEBUG_NAMES ${${_pkg}_${COMPONENT}_NAMES})
    endif()

    # Skip finding libraries if the component is header-only
    if(${_pkg}_${COMPONENT}_HEADER_ONLY)
      _adjust_lib_vars(${COMPONENT})
      continue()
    endif()

    #
    # Find RELEASE libraries
    #
    if(NOT ${_pkg}_${COMPONENT}_RELEASE_NAMES)
      message(SEND_ERROR "Required CMake variable ${_pkg}_${COMPONENT}_RELEASE_NAMES not defined!")
    endif()
    set(_${_pkg}_RELEASE_NAMES ${${_pkg}_${COMPONENT}_RELEASE_NAMES})

    _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                 "Searching for ${UPPERCOMPONENT}_LIBRARY_RELEASE: ${_${_pkg}_RELEASE_NAMES}")

    # if ${_pkg}_LIBRARY_DIR_RELEASE is not defined, but ${_pkg}_LIBRARY_DIR_DEBUG is, look there first for RELEASE libs
    if(NOT ${_pkg}_LIBRARY_DIR_RELEASE AND ${_pkg}_LIBRARY_DIR_DEBUG)
      list(INSERT _${_pkg}_LIBRARY_SEARCH_DIRS_RELEASE 0 ${${_pkg}_LIBRARY_DIR_DEBUG})
    endif()

    if(${_pkg}_USE_RELEASE_LIBS)
      _find_library(
        ${_pkg}_${UPPERCOMPONENT}_LIBRARY_RELEASE
        RELEASE
        NAMES
        ${_${_pkg}_RELEASE_NAMES}
        HINTS
        ${_${_pkg}_LIBRARY_SEARCH_DIRS_RELEASE}
        NAMES_PER_DIR
        DOC
        "${_${_pkg}_docstring_release}")
    endif()
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                     "${_pkg}_${UPPERCOMPONENT}_LIBRARY_RELEASE")

    #
    # Find DEBUG libraries
    #
    if(NOT ${_pkg}_${COMPONENT}_DEBUG_NAMES)
      if(${_pkg}_${COMPONENT}_RELEASE_NAMES)
        set(${_pkg}_${COMPONENT}_DEBUG_NAMES ${${_pkg}_${COMPONENT}_RELEASE_NAMES})
      else()
        message(SEND_ERROR "Required CMake variable ${_pkg}_${COMPONENT}_DEBUG_NAMES not defined!")
      endif()
    endif()
    set(_${_pkg}_DEBUG_NAMES ${${_pkg}_${COMPONENT}_DEBUG_NAMES})

    _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                 "Searching for ${UPPERCOMPONENT}_LIBRARY_DEBUG: ${_${_pkg}_DEBUG_NAMES}")

    # if ${_pkg}_LIBRARY_DIR_DEBUG is not defined, but ${_pkg}_LIBRARY_DIR_RELEASE is, look there first for DEBUG libs
    if(NOT ${_pkg}_LIBRARY_DIR_DEBUG AND ${_pkg}_LIBRARY_DIR_RELEASE)
      list(INSERT _${_pkg}_LIBRARY_SEARCH_DIRS_DEBUG 0 ${${_pkg}_LIBRARY_DIR_RELEASE})
    endif()

    if(${_pkg}_USE_DEBUG_LIBS)
      _find_library(
        ${_pkg}_${UPPERCOMPONENT}_LIBRARY_DEBUG
        DEBUG
        NAMES
        ${_${_pkg}_DEBUG_NAMES}
        HINTS
        ${_${_pkg}_LIBRARY_SEARCH_DIRS_DEBUG}
        NAMES_PER_DIR
        DOC
        "${_${_pkg}_docstring_debug}")
    endif()
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                     "${_pkg}_${UPPERCOMPONENT}_LIBRARY_DEBUG")
    _adjust_lib_vars(${UPPERCOMPONENT})
  else()
    #
    # Find executable
    #
    if(NOT ${_pkg}_${COMPONENT}_NAMES)
      message(SEND_ERROR "Required CMake variable ${_pkg}_${COMPONENT}_NAMES not defined!")
    endif()
    set(_${_pkg}_NAMES ${${_pkg}_${COMPONENT}_NAMES})

    _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                 "Searching for ${UPPERCOMPONENT}_EXECUTABLE: ${_${_pkg}_NAMES}")

    _find_program(
      ${_pkg}_${UPPERCOMPONENT}_EXECUTABLE
      NAMES
      ${_${_pkg}_NAMES}
      HINTS
      ${_${_pkg}_EXECUTABLE_SEARCH_DIRS}
      NAMES_PER_DIR
      PATH_SUFFIXES
      ${${_pkg}_EXECUTABLE_PATH_SUFFIXES}
      DOC
      "${_${_pkg}_docstring_executable}")
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_${UPPERCOMPONENT}_EXECUTABLE")

    if(${_pkg}_${UPPERCOMPONENT}_EXECUTABLE)
      set(${_pkg}_${UPPERCOMPONENT}_FOUND ON)
    endif()

    # Make variables changeable to the advanced user
    mark_as_advanced(${_pkg}_${UPPERCOMPONENT}_EXECUTABLE)
  endif()
endforeach()

# Restore the original find library ordering
if(_${_pkg}_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_${_pkg}_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()
if(_${_pkg}_ORIG_CMAKE_FIND_LIBRARY_PREFIXES)
  set(CMAKE_FIND_LIBRARY_PREFIXES ${_${_pkg}_ORIG_CMAKE_FIND_LIBRARY_PREFIXES})
endif()

# ------------------------------------------------------------------------
# End finding ${_pkg} libraries/executable

set(${_pkg}_INCLUDE_DIRS ${${_pkg}_INCLUDE_DIR})
set(${_pkg}_LIBRARY_DIRS)
if(${_pkg}_LIBRARY_DIR_RELEASE)
  list(APPEND ${_pkg}_LIBRARY_DIRS ${${_pkg}_LIBRARY_DIR_RELEASE})
endif()
if(${_pkg}_LIBRARY_DIR_DEBUG)
  list(APPEND ${_pkg}_LIBRARY_DIRS ${${_pkg}_LIBRARY_DIR_DEBUG})
endif()
if(${_pkg}_LIBRARY_DIRS)
  list(REMOVE_DUPLICATES ${_pkg}_LIBRARY_DIRS)
endif()
set(${_pkg}_EXECUTABLE_DIRS ${${_pkg}_EXECUTABLE_DIR})

# ==============================================================================
# Call FPHSA helper, see https://cmake.org/cmake/help/latest/module/FindPackageHandleStandardArgs.html

# Define aliases as needed by the component handler in the FPHSA helper below
foreach(_comp IN LISTS ${_pkg}_FIND_COMPONENTS)
  string(TOUPPER ${_comp} _uppercomp)
  if(DEFINED ${_pkg}_${_uppercomp}_FOUND)
    set(${_pkg}_${_comp}_FOUND ${${_pkg}_${_uppercomp}_FOUND})
  endif()
  _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_${_comp}_FOUND")
endforeach()

if(NOT ${_pkg}_HAS_VERSION)
  set(_${_pkg}_version_list)

  # Try to get version number from shared library extension on UNIX systems
  foreach(_comp IN LISTS ${_pkg}_FIND_COMPONENTS)
    string(TOUPPER ${_comp} _uppercomp)
    if(${_pkg}_${_comp}_FOUND)
      _extract_version_from_lib_name(${_pkg} "${${_pkg}_${_uppercomp}_LIBRARY}")
      list(APPEND _${_pkg}_version_list ${${_pkg}_VERSION})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _${_pkg}_version_list)
  list(LENGTH _${_pkg}_version_list _${_pkg}_L)
  if(NOT _${_pkg}_L EQUAL 1)
    if(_${_pkg}_L EQUAL 0)
      _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "No version number found.")
    else()
      _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                   "Inconsistent version numbers: ${_${_pkg}_version_list}, ignoring all of them.")
    endif()
    set(${_pkg}_VERSION)
    set(${_pkg}_VERSION_MAJOR)
    set(${_pkg}_VERSION_MINOR)
    set(${_pkg}_VERSION_PATCH)
    set(${_pkg}_HAS_VERSION
        FALSE
        CACHE BOOL "If ${pkg_name} version was found" FORCE)
    set(${_pkg}_HAS_VERSION FALSE)
  endif()

  if(${_pkg}_HAS_VERSION)
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_VERSION")
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_VERSION_MAJOR")
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_VERSION_MINOR")
    _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "${_pkg}_VERSION_PATCH")
  endif()
endif()

if(${_pkg}_HAS_VERSION)
  find_package_handle_standard_args(
    ${_pkg}
    REQUIRED_VARS ${_pkg}_INCLUDE_DIR
    VERSION_VAR ${_pkg}_VERSION
    HANDLE_COMPONENTS)
else()
  find_package_handle_standard_args(
    ${_pkg}
    REQUIRED_VARS ${_pkg}_INCLUDE_DIR
    HANDLE_COMPONENTS)
endif()

if(NOT ${_pkg}_FOUND)
  # ${_pkg} headers were not found so no components were found.
  foreach(COMPONENT ${${_pkg}_FIND_COMPONENTS})
    string(TOUPPER ${COMPONENT} UPPERCOMPONENT)
    set(${_pkg}_${UPPERCOMPONENT}_FOUND FALSE)
  endforeach()
endif()

# ==============================================================================
# Add imported targets

if(${_pkg}_FOUND)
  # Look for external dependencies
  foreach(_${_pkg}_dep ${${_pkg}_EXTERNAL_DEPENDENCIES})
    set(_${_pkg}_dep_found FALSE)
    foreach(_found_var ${${_pkg}_EXTERNAL_DEPENDENCY_${_${_pkg}_dep}_FOUND_VARS})
      if(${_found_var})
        set(_${_pkg}_dep_found TRUE)
      endif()
    endforeach()

    if(NOT _${_pkg}_dep_found)
      _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                   "Looking for external dependency: ${_${_pkg}_dep}")
      set(_${_pkg}_find_dep_args)
      if(${_pkg}_EXTERNAL_DEPENDENCY_${_${_pkg}_dep}_COMPONENTS)
        list(APPEND _${_pkg}_find_dep_args COMPONENTS ${${_pkg}_EXTERNAL_DEPENDENCY_${_${_pkg}_dep}_COMPONENTS})
      endif()

      _debug_print(
        "${CMAKE_CURRENT_LIST_FILE}"
        "${CMAKE_CURRENT_LIST_LINE}"
        "find_package(${_${_pkg}_dep} ${${_pkg}_EXTERNAL_DEPENDENCY_${_${_pkg}_dep}_FIND_ARGS} ${_${_pkg}_find_dep_args})"
      )
      find_package(${_${_pkg}_dep} ${${_pkg}_EXTERNAL_DEPENDENCY_${_${_pkg}_dep}_FIND_ARGS} ${_${_pkg}_find_dep_args})
    else()
      _debug_print("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                   "External dependency: ${_${_pkg}_dep} already found!")
    endif()
  endforeach()

  foreach(_comp ${${_pkg}_FIND_COMPONENTS})
    set(_${_pkg}_tgt ${${_pkg}_NAMESPACE}::${_comp})
    string(TOUPPER ${_comp} _uppercomp)
    if(${_pkg}_${_uppercomp}_FOUND)
      if("${${_pkg}_${_comp}_TYPE}" STREQUAL "LIBRARY")
        if(NOT TARGET ${_${_pkg}_tgt})
          if(${_pkg}_${_comp}_HEADER_ONLY)
            add_library(${_${_pkg}_tgt} INTERFACE IMPORTED)
          else()
            set(_${_pkg}_libtype SHARED)
            foreach(_${_pkg}_lib "${${_pkg}_${_uppercomp}_LIBRARY_RELEASE}" "${${_pkg}_${_uppercomp}_LIBRARY_DEBUG}")
              if(EXISTS "${_${_pkg}_lib}")
                set(_${_pkg}_fname "${_${_pkg}_lib}")
                break()
              endif()
            endforeach()
            if(_${_pkg}_fname)
              if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
                cmake_path(GET _${_pkg}_fname EXTENSION _${_pkg}_ext)
              else()
                get_filename_component(_${_pkg}_ext "${_${_pkg}_fname}" EXT)
              endif()
              if("${_${_pkg}_ext}" STREQUAL "${CMAKE_STATIC_LIBRARY_SUFFIX}")
                set(_${_pkg}_libtype STATIC)
              endif()
            endif()

            add_library(${_${_pkg}_tgt} ${_${_pkg}_libtype} IMPORTED)
          endif()
        endif()

        # Always reset the target properties (in case we are called multiple times)
        set_target_properties(
          ${_${_pkg}_tgt}
          PROPERTIES INTERFACE_COMPILE_DEFINITIONS ""
                     INTERFACE_INCLUDE_DIRECTORIES ""
                     INTERFACE_LINK_LIBRARIES "")

        if(${_pkg}_TARGET_DEFINITIONS)
          target_compile_definitions(${_${_pkg}_tgt} INTERFACE ${${_pkg}_TARGET_DEFINITIONS})
        endif()
        if(${_pkg}_${_comp}_TARGET_DEFINITIONS)
          target_compile_definitions(${_${_pkg}_tgt} INTERFACE ${${_pkg}_${_comp}_TARGET_DEFINITIONS})
        endif()
        if(${_pkg}_INCLUDE_DIRS)
          target_include_directories(${_${_pkg}_tgt} INTERFACE "${${_pkg}_INCLUDE_DIRS}")
        endif()
        if(EXISTS "${${_pkg}_${_uppercomp}_LIBRARY}")
          set_target_properties(${_${_pkg}_tgt} PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                                                           IMPORTED_LOCATION "${${_pkg}_${_uppercomp}_LIBRARY}")
          _windows_set_import_library_path(${_${_pkg}_tgt} IMPORTED_LOCATION)
        endif()
        if(EXISTS "${${_pkg}_${_uppercomp}_LIBRARY_RELEASE}")
          set_property(
            TARGET ${_${_pkg}_tgt}
            APPEND
            PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
          set_target_properties(
            ${${_pkg}_NAMESPACE}::${_comp}
            PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX" IMPORTED_LOCATION_RELEASE
                                                                       "${${_pkg}_${_uppercomp}_LIBRARY_RELEASE}")
          _windows_set_import_library_path(${${_pkg}_NAMESPACE}::${_comp} IMPORTED_LOCATION RELEASE)
        endif()
        if(EXISTS "${${_pkg}_${_uppercomp}_LIBRARY_DEBUG}")
          set_property(
            TARGET ${${_pkg}_NAMESPACE}::${_comp}
            APPEND
            PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
          set_target_properties(
            ${${_pkg}_NAMESPACE}::${_comp} PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
                                                      IMPORTED_LOCATION_DEBUG "${${_pkg}_${_uppercomp}_LIBRARY_DEBUG}")
          _windows_set_import_library_path(${${_pkg}_NAMESPACE}::${_comp} IMPORTED_LOCATION DEBUG)
        endif()

        if(${_pkg}_${_uppercomp}_DEPENDENCIES)
          unset(_${_pkg}_${_uppercomp}_TARGET_DEPENDENCIES)
          foreach(dep ${${_pkg}_${_comp}_DEPENDENCIES})
            list(APPEND _${_pkg}_${_uppercomp}_TARGET_DEPENDENCIES ${${_pkg}_NAMESPACE}::${dep})
          endforeach()
          target_link_libraries(${${_pkg}_NAMESPACE}::${_comp}
                                INTERFACE "${_${_pkg}_${_uppercomp}_TARGET_DEPENDENCIES}")
        endif()

        if(${_pkg}_${_comp}_EXTERNAL_DEPENDENCIES)
          target_link_libraries(${${_pkg}_NAMESPACE}::${_comp} INTERFACE "${${_pkg}_${_comp}_EXTERNAL_DEPENDENCIES}")
        endif()
      else()
        if(NOT TARGET ${_${_pkg}_tgt})
          add_executable(${_${_pkg}_tgt} IMPORTED)
        endif()
      endif()

      set_target_properties(${${_pkg}_NAMESPACE}::${_comp} PROPERTIES IMPORTED_LOCATION
                                                                      "${${_pkg}_${_uppercomp}_EXECUTABLE}")
    endif()

    list(APPEND ${_pkg}_IMPORTED_TARGETS ${${_pkg}_NAMESPACE}::${_comp})
  endforeach()
endif()
list(REMOVE_DUPLICATES ${_pkg}_IMPORTED_TARGETS)

if(COMMAND ${_pkg}_post_process)
  cmake_language(CALL ${_pkg}_post_process ${_pkg})
endif()

# ==============================================================================

if(${_pkg}_DEBUG)
  list(APPEND CMAKE_MESSAGE_INDENT "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] ")
  cmake_print_properties(
    TARGETS ${${_pkg}_IMPORTED_TARGETS}
    PROPERTIES INTERFACE_COMPILE_DEFINITIONS
               INTERFACE_INCLUDE_DIRECTORIES
               INTERFACE_LINK_LIBRARIES
               IMPORTED_LOCATION
               IMPORTED_LOCATION_DEBUG
               IMPORTED_LOCATION_RELEASE
               IMPORTED_IMPLIB
               IMPORTED_IMPLIB_DEBUG
               IMPORTED_IMPLIB_RELEASE)
  list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

# ==============================================================================
# Finalize

# Report ${_pkg}_LIBRARIES
set(${_pkg}_LIBRARIES "")
foreach(_comp IN LISTS ${_pkg}_FIND_COMPONENTS)
  string(TOUPPER ${_comp} _uppercomp)
  if(${_pkg}_${_uppercomp}_FOUND)
    list(APPEND ${_pkg}_LIBRARIES ${${_pkg}_${_uppercomp}_LIBRARY})
  endif()
endforeach()
list(REMOVE_DUPLICATES ${_pkg}_LIBRARIES)

# Configure display of cache entries in GUI.
foreach(var ${_pkg}ROOT ${_pkg}_ROOT ${_${_pkg}_VARS_INC} ${_${_pkg}_VARS_LIB})
  get_property(
    _type
    CACHE ${var}
    PROPERTY TYPE)
  if(_type)
    set_property(CACHE ${var} PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE ${var} PROPERTY TYPE PATH)
    endif()
  endif()
endforeach()

# Record last used values of input variables so we can detect on the next run if the user changed them.
foreach(var ${_${_pkg}_VARS_INC} ${_${_pkg}_VARS_LIB} ${_${_pkg}_VARS_DIR} ${_${_pkg}_VARS_NAME})
  if(DEFINED ${var})
    set(_${var}_LAST
        "${${var}}"
        CACHE INTERNAL "Last used ${var} value.")
  else()
    unset(_${var}_LAST CACHE)
  endif()
endforeach()

# Maintain a persistent list of components requested anywhere since the last flush.
set(_${_pkg}_COMPONENTS_SEARCHED "${_${_pkg}_COMPONENTS_SEARCHED}")
list(APPEND _${_pkg}_COMPONENTS_SEARCHED ${${_pkg}_FIND_COMPONENTS})
list(REMOVE_DUPLICATES _${_pkg}_COMPONENTS_SEARCHED)
list(SORT _${_pkg}_COMPONENTS_SEARCHED)
set(_${_pkg}_COMPONENTS_SEARCHED
    "${_${_pkg}_COMPONENTS_SEARCHED}"
    CACHE INTERNAL "Components requested for this build tree.")

# Restore project's policies
cmake_policy(POP)

# ==============================================================================
