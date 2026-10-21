# ==============================================================================
#
# Copyright 2022 <Huawei Technologies Co., Ltd>
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
#
# Locate the CMake package shipped by a Python-installed pybind11. Two
# distributions exist: `pybind11` (config under the module directory, reported
# by `python -m pybind11 --cmakedir`) and `pybind11-global` (config under
# `<prefix>/share/cmake/pybind11`).
#
# Search order:
# 1. Inside the active virtualenv: pybind11-global, then the pybind11 module.
# 2. Otherwise (or if nothing suitable was found there): pybind11-global in the
#    user and global sites, then the pybind11 module in the user and global
#    sites, preferring whichever is more recent.
#
# This module adds no version policy of its own: the minimum acceptable version
# is the one requested by the caller (`find_package(pybind11 <version>)`), which
# the third-party wrapper sets to the version it would otherwise build.
#
# ==============================================================================

# lint_cmake: -convention/filename,-package/stdargs,-whitespace/indent

# ~~~
# Make sure that pybind11 uses the interpreter already selected by FindPython: its classic Python discovery mode reads
# PYTHON_EXECUTABLE, while its FindPython mode reads Python_EXECUTABLE (already set).
# ~~~
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
set(PYBIND11_PYTHON_VERSION ${Python_VERSION}) # maybe not strictly required

include(${CMAKE_CURRENT_LIST_DIR}/../commands/to_cmake_path.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../commands/find_python_module.cmake)

# Message printing function
macro(_pybind11_message)
  if(NOT pybind11_FIND_QUIETLY)
    message(${ARGN})
  endif()
endmacro()

function(_mq_reset_pybind11_cache)
  foreach(
    _var
    _pybind11_already_found
    _pybind11_cached_python_executable
    FIND_PACKAGE_MESSAGE_DETAILS_PYMOD_pybind11
    FIND_PACKAGE_MESSAGE_DETAILS_pybind11
    PYMOD_PYBIND11_FOUND
    PYMOD_PYBIND11_PATH
    PYMOD_PYBIND11_VERSION
    PYMOD_pybind11_VERSION
    pybind11_DIR
    pybind11_INCLUDE_DIR
    pybind11_INCLUDE_DIRS)
    unset(${_var} CACHE)
    unset(${_var})
  endforeach()
endfunction()

# Version requested by the caller, forwarded verbatim to every lookup below.
set(_pybind11_version_args)
if(pybind11_FIND_VERSION)
  set(_pybind11_version_args VERSION ${pybind11_FIND_VERSION})
endif()

set(_pybind11_cache_is_stale FALSE)
if(DEFINED _pybind11_cached_python_executable AND NOT "${_pybind11_cached_python_executable}" STREQUAL
                                                  "${Python_EXECUTABLE}")
  set(_pybind11_cache_is_stale TRUE)
endif()
if(DEFINED PYMOD_PYBIND11_PATH AND NOT EXISTS "${PYMOD_PYBIND11_PATH}")
  set(_pybind11_cache_is_stale TRUE)
endif()
if(DEFINED pybind11_DIR
   AND NOT "${pybind11_DIR}" MATCHES "-NOTFOUND$"
   AND NOT EXISTS "${pybind11_DIR}")
  set(_pybind11_cache_is_stale TRUE)
endif()
if(_pybind11_cache_is_stale)
  _pybind11_message(STATUS "Invalidating stale pybind11 cache entries")
  _mq_reset_pybind11_cache()
endif()
set(_pybind11_cached_python_executable
    "${Python_EXECUTABLE}"
    CACHE INTERNAL "Python executable used to locate pybind11" FORCE)

_pybind11_message(CHECK_START "Looking for pybind11")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

if(NOT _pybind11_already_found)
  # ==============================================================================
  # First detect whether we are in a virtualenv or not

  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sys; print(int(sys.prefix != sys.base_prefix))"
    OUTPUT_VARIABLE _is_virtualenv
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(_is_virtualenv)
    # Candidates inside the virtualenv, each tried in turn: pybind11-global, then the pybind11 module. A candidate that
    # exists but does not satisfy the requested version simply yields to the next one.
    _pybind11_message(CHECK_START "Looking for pybind11-global in virtualenv")

    execute_process(
      COMMAND ${Python_EXECUTABLE} -c "import sys, os; print(os.path.join(sys.prefix, 'share', 'cmake', 'pybind11'))"
      OUTPUT_VARIABLE pybind11_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    to_cmake_path(pybind11_DIR)

    if(EXISTS "${pybind11_DIR}")
      find_package(pybind11 ${pybind11_FIND_VERSION} CONFIG QUIET NO_DEFAULT_PATH)
    endif()
    if(pybind11_FOUND)
      _pybind11_message(CHECK_PASS "Found")
    else()
      _pybind11_message(CHECK_FAIL "Not-found")

      _pybind11_message(CHECK_START "Looking for the pybind11 Python module in virtualenv")
      find_python_module(pybind11 ${_pybind11_version_args})
      if(PYMOD_PYBIND11_FOUND)
        _pybind11_message(CHECK_PASS "Found")
        execute_process(
          COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
          OUTPUT_VARIABLE pybind11_DIR
          OUTPUT_STRIP_TRAILING_WHITESPACE)
        to_cmake_path(pybind11_DIR)
        find_package(pybind11 ${pybind11_FIND_VERSION} CONFIG QUIET NO_DEFAULT_PATH)
      else()
        _pybind11_message(CHECK_FAIL "Not-found")
      endif()
    endif()
  endif()

  # ------------------------------------------------------------------------------

  if(NOT pybind11_FOUND)
    _pybind11_message(CHECK_START "Looking for pybind11-global in global and user sites")

    if(NOT _is_virtualenv)
      # Try pybind11-global in user site
      execute_process(
        COMMAND ${Python_EXECUTABLE} -m site --user-base
        RESULT_VARIABLE _status
        OUTPUT_VARIABLE _user_base
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      to_cmake_path(_user_base)

      # NB: _status == 0 means user site enabled. Anything else and we really should not consider it
      if(_status EQUAL 0 AND EXISTS "${_user_base}/share/cmake/pybind11/")
        # cmake-lint: disable=C0103
        set(pybind11_DIR "${_user_base}/share/cmake/pybind11/")
      endif()
    endif()

    # Try to find pybind11-global either in user-site or global-site (even in the case of a virtualenv since we did not
    # find anything useful in it previously)
    find_package(pybind11 ${pybind11_FIND_VERSION} CONFIG QUIET)

    if(pybind11_FOUND)
      _pybind11_message(CHECK_PASS "Found")
    else()
      _pybind11_message(CHECK_PASS "Not-found")
    endif()

    if(NOT _is_virtualenv)
      _pybind11_message(CHECK_START "Looking for pybind11 Python module")

      # Try to find pybind11 either in user-site or global-site
      if(pybind11_FIND_QUIETLY)
        set(_args QUIET)
      endif()
      find_python_module(pybind11 ${_pybind11_version_args} ${_args})
      unset(_args)

      if(PYMOD_PYBIND11_FOUND)
        _pybind11_message(CHECK_PASS "Found")
      else()
        _pybind11_message(CHECK_PASS "Not-found")
      endif()

      if(PYMOD_PYBIND11_FOUND AND PYMOD_PYBIND11_VERSION VERSION_GREATER pybind11_VERSION)
        # We prefer pybind11 over pybind11 global if its version is more recent. This could typically be the case if a
        # user installs a more recent version of pybind11 in its user site.
        execute_process(
          COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
          OUTPUT_VARIABLE pybind11_DIR
          OUTPUT_STRIP_TRAILING_WHITESPACE)
        to_cmake_path(pybind11_DIR)
      endif()

      _pybind11_message(CHECK_START "Looking for pybind11 (final)")

      # Look for pybind11 again in case we have pybind11 more recent than pybind11-global
      find_package(pybind11 ${pybind11_FIND_VERSION} CONFIG QUIET NO_DEFAULT_PATH)

      if(pybind11_FOUND)
        _pybind11_message(CHECK_PASS "Found")
      else()
        _pybind11_message(CHECK_PASS "Not-found")
      endif()
    endif()
  endif()
endif()

# ------------------------------------------------------------------------------
# Load the package configuration from the location established above. On later configure runs this is the only lookup
# (pybind11_DIR is cached), so it is what actually defines the imported targets.

list(POP_BACK CMAKE_MESSAGE_INDENT)
if(pybind11_DIR)
  _pybind11_message(CHECK_PASS "Done")
else()
  _pybind11_message(CHECK_FAIL "Failed")
endif()
find_package(pybind11 ${pybind11_FIND_VERSION} CONFIG NO_DEFAULT_PATH)

# ==============================================================================

if(pybind11_FOUND)
  if(NOT TARGET pybind11::headers)
    _pybind11_message(SEND_ERROR
                      "Target pybind11::headers was not defined! Perhaps try updating pybind11 on your system?")
  endif()
  if(NOT TARGET pybind11::pybind11)
    _pybind11_message(SEND_ERROR
                      "Target pybind11::pybind11 was not defined! Perhaps try updating pybind11 on your system?")
  endif()
  if(NOT TARGET pybind11::module)
    _pybind11_message(SEND_ERROR
                      "Target pybind11::module was not defined! Perhaps try updating pybind11 on your system?")
  endif()

  set(_tgt_name pybind11::pybind11_headers)

  # NB: workardound for NVHPC compiler that requires -isystem (e.g. for /usr/include)
  get_target_property(_include_dir ${_tgt_name} INTERFACE_INCLUDE_DIRECTORIES)

  if(_include_dir)
    get_target_property(_aliased ${_tgt_name} ALIASED_TARGET)
    if(_aliased)
      set(_tgt_name ${_aliased})
    endif()

    target_include_directories(${_tgt_name} SYSTEM INTERFACE ${_include_dir})
  endif()
endif()

# ------------------------------------------------------------------------------

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  pybind11
  REQUIRED_VARS pybind11_INCLUDE_DIR
  VERSION_VAR pybind11_VERSION)

# ------------------------------------------------------------------------------

# For debugging
if(pybind11_FOUND AND NOT _pybind11_already_found)
  _pybind11_message(STATUS "Found pybind11 using the CONFIG method in ${pybind11_DIR}")
  _pybind11_message(STATUS "Found pybind11 and defined the pybind11::pybind11 imported target:")
  _pybind11_message(STATUS "  - include:      ${pybind11_INCLUDE_DIR}")
  _pybind11_message(STATUS "  - version:      ${pybind11_VERSION}")
endif()

# ==============================================================================

# Remember a successful lookup only: after a failure every configure run must probe again, since the environment may
# have gained a pybind11 in the meantime.
if(pybind11_FOUND)
  set(_pybind11_already_found
      ON
      CACHE INTERNAL "Findpybind11 located pybind11 in a previous configure run" FORCE)
endif()
set(pybind11_DIR
    "${pybind11_DIR}"
    CACHE INTERNAL "" FORCE)
mark_as_advanced(pybind11_DIR _pybind11_already_found)

unset(_pybind11_version_args)

# ==============================================================================
