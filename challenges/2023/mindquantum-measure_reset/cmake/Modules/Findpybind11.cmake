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
# Looking for pybind11 can be somewhat complicated now that there are two
# possible packages: - pybind11 - pybind11-global
#
# Here we try our best to look for pybind11 in all possible locations and in a
# meaningful way. In practice, this means that we look for a valid package in
# the following order: 1. First we look in a virtualenv (if one is active) for
# a) pybind11-global b) pybind11 >= 2.6.0 c) pybind11 < 2.6.0 && pybind11-cmake
# 2. pybind11-gobal in user site (if not in a virtualenv) 3. pybind11-gobal in
# global site package (also done if within a virtualenv) 4. pybind11 >= 2.6.0 in
# user and global sites NB: if the version of pybind11 is more recent than
# pybind11-global, we choose pybind11 5. pybind11 < 2.6.0 && pybind11-cmake in
# user and global sites
#
# ==============================================================================

# lint_cmake: -convention/filename,-package/stdargs

# ~~~
# Make sure that pybind11 finds the correct python version
#  - pybind11 < 2.6.0 uses find_package(PythonInterp ...)
#  - pybind11 >= 2.6.0 looks for the `python` command or PYTHON_EXECUTABLE
# => specify PYTHON_EXECUTABLE manually to guarantee we find the same
#    interpreter and libraries as for other python packages
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
    # ~~~
    # Try to find in order within the virtualenv:
    # - pybind11-global
    # - pybind11 >= 2.6.0
    # - pybind11 < 2.6.0 with pybind11-cmake
    # - if all of that fails, revert to user and global sites (as far as is possible)
    # ~~~

    _pybind11_message(CHECK_START "Looking for pybind11-global in virtualenv")

    # Look for pybind11-global
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c "import sys, os; print(os.path.join(sys.prefix, 'share', 'cmake', 'pybind11'))"
      OUTPUT_VARIABLE pybind11_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    to_cmake_path(pybind11_DIR)

    if(NOT EXISTS pybind11_DIR)
      _pybind11_message(CHECK_FAIL "Not-found")

      # Could not find pybind11-global, try pybind11 >= 2.6.0
      _pybind11_message(CHECK_START "Looking for pybind11 >= 2.6.0 in virtualenv")

      find_python_module(pybind11 VERSION 2.6.0)
      if(PYMOD_PYBIND11_FOUND)
        _pybind11_message(CHECK_PASS "Found")
        execute_process(
          COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
          OUTPUT_VARIABLE pybind11_DIR
          OUTPUT_STRIP_TRAILING_WHITESPACE)
        to_cmake_path(pybind11_DIR)
      else()
        _pybind11_message(CHECK_FAIL "Not-found")
        # Now try pybind11 < 2.6.0 && pybind11-cmake
        find_python_module(pybind11-cmake)
        if(PYMOD_PYBIND11_CMAKE_FOUND)
          execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import pybind11_cmake; print(pybind11_cmake.__path__[0])"
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE)
          to_cmake_path(pybind11_DIR)
        endif()
      endif()
    else()
      _pybind11_message(CHECK_PASS "Found")
    endif()

    # Try to find some valid pybind11 config within the virtualenv
    find_package(pybind11 2.6.0 CONFIG QUIET NO_DEFAULT_PATH)
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
    find_package(pybind11 2.6.0 CONFIG QUIET)

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
      find_python_module(pybind11 VERSION 2.6.0 ${_args})
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
      find_package(pybind11 2.6.0 CONFIG QUIET NO_DEFAULT_PATH)

      # If we are not in a virtualenv, we should still try to look for pybind11 < 2.6.0 && pybind11-cmake
      if(NOT pybind11_FOUND)
        _pybind11_message(CHECK_PASS "Not-found")

        _pybind11_message(CHECK_START "Looking for pybind11-cmake")
        # If everything else fails, rely on the pybind11_cmake package
        find_python_module(pybind11-cmake)
        if(PYMOD_PYBIND11_CMAKE_FOUND)
          _pybind11_message(CHECK_PASS "Done")
          execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import pybind11_cmake; print(pybind11_cmake.__path__[0])"
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE)
          to_cmake_path(pybind11_DIR)
        else()
          _pybind11_message(CHECK_FAIL "Failed")
        endif()
      else()
        _pybind11_message(CHECK_PASS "Found")
      endif()
    endif()
  endif()
endif()

# ------------------------------------------------------------------------------
# Now look for pybind11 using the CONFIG method one last time. This would typically only be useful in the case of
# pybind11 < 2.6.0 located in either user or global sites.

list(POP_BACK CMAKE_MESSAGE_INDENT)
if(pybind11_DIR)
  _pybind11_message(CHECK_PASS "Done")
else()
  _pybind11_message(CHECK_FAIL "Failed")
endif()
find_package(pybind11 2.6.0 CONFIG NO_DEFAULT_PATH)

# ==============================================================================

# With pybind11-cmake 1.0.0 we might need to fix the include path
if(PYMOD_PYBIND11_CMAKE_FOUND)
  include(CheckCXXSourceCompiles)
  check_cxx_source_compiles("#include <pybind11/pybind11.h>
int main() {return 0;}" pybind11_compiles)

  if(NOT pybind11_compiles)
    # This could happen with pybind11_cmake == 1.0.0 because there is a typo in the pybind11Config.cmake
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c "import pybind11
print(pybind11.get_include(False) + ';' + pybind11.get_include(True))"
      OUTPUT_VARIABLE _pybind11_inc_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    to_cmake_path(_pybind11_inc_dir)
    get_directory_property(_inc_dirs INCLUDE_DIRECTORIES)
    list(FILTER _inc_dirs EXCLUDE REGEX .*pybind11_INCLUDE_DIR$)
    list(APPEND _inc_dirs ${_pybind11_inc_dir})
    set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "${_inc_dirs}")
  endif()
endif()

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

set(_pybind11_already_found
    ON
    CACHE INTERNAL "Findpybind11 was already run once")
set(pybind11_DIR
    "${pybind11_DIR}"
    CACHE INTERNAL "")
mark_as_advanced(pybind11_DIR _pybind11_already_found)

# ==============================================================================
