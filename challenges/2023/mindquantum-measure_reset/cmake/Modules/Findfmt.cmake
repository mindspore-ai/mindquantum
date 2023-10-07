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

# lint_cmake: -package/consistency,-convention/filename

#[=======================================================================[.rst:
Findfmt
---------

Find fmt include dirs and libraries

Use this module by invoking :command:`find_package` with the form:

.. code-block:: cmake

  find_package(fmt
    [version] [EXACT]      # Minimum or EXACT version e.g. 2020.03
    [REQUIRED]             # Fail with error if fmt is not found
    [COMPONENTS <libs>...] # fmt libraries by their canonical name
    )

This module finds headers and requested component libraries OR a CMake package configuration file provided by a "fmt
CMake" build.  For the latter case skip to the :ref:`fmt CMake` section below.

Available components: fmt, fmt-header-only

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``fmt_FOUND``
  True if headers and requested libraries were found.

``fmt_INCLUDE_DIRS``
  fmt include directories.

``fmt_LIBRARY_DIRS``
  Link directories for fmt libraries.

``fmt_LIBRARIES``
  fmt component libraries to be linked.

``fmt_<COMPONENT>_FOUND``
  True if component ``<COMPONENT>`` was found (``<COMPONENT>`` name is upper-case).

``fmt_<COMPONENT>_LIBRARY``
  Libraries to link for component ``<COMPONENT>`` (may include :command:`target_link_libraries` debug/optimized
  keywords).

``fmt_VERSION``
  fmt version number in ``X.Y`` format.

``fmt_VERSION_MAJOR``
  fmt major version number (``X`` in ``X.Y``).

``fmt_VERSION_MINOR``
  fmt minor version number (``Y`` in ``X.Y``).

Cache variables
^^^^^^^^^^^^^^^

Search results are saved persistently in CMake cache entries:

``fmt_INCLUDE_DIR``
  Directory containing fmt headers.

``fmt_LIBRARY_DIR_RELEASE``
  Directory containing release fmt libraries.

``fmt_LIBRARY_DIR_DEBUG``
  Directory containing debug fmt libraries.

``fmt_<COMPONENT>_LIBRARY_DEBUG``
  Component ``<COMPONENT>`` library debug variant.

``fmt_<COMPONENT>_LIBRARY_RELEASE``
  Component ``<COMPONENT>`` library release variant.

Hints
^^^^^

This module reads hints about search locations from variables:

``fmt_ROOT``, ``fmtROOT``
  Preferred installation prefix.

``fmt_INCLUDEDIR``
  Preferred include directory e.g. ``<prefix>/include``.

``fmt_LIBRARYDIR``
  Preferred library directory e.g. ``<prefix>/lib``.

``fmt_NO_SYSTEM_PATHS``
  Set to ``ON`` to disable searching in locations not specified by these hint variables. Default is ``OFF``.

Users may set these hints or results as ``CACHE`` entries.  Projects should not read these entries directly but
instead use the above result variables.  Note that some hint names start in upper-case ``fmt``.  One may specify these
as environment variables if they are not specified as CMake variables or cache entries.

This module first searches for the fmt header files using the above hint variables (excluding ``fmt_LIBRARYDIR``) and
saves the result in ``fmt_INCLUDE_DIR``.  Then it searches for requested component libraries using the above hints
(excluding ``fmt_INCLUDEDIR``), "lib" directories near ``fmt_INCLUDE_DIR``, and the library name configuration
settings below.  It saves the library directories in ``fmt_LIBRARY_DIR_DEBUG`` and ``fmt_LIBRARY_DIR_RELEASE`` and
individual library locations in ``fmt_<COMPONENT>_LIBRARY_DEBUG`` and ``fmt_<COMPONENT>_LIBRARY_RELEASE``.  When one
changes settings used by previous searches in the same build tree (excluding environment variables) this module
discards previous search results affected by the changes and searches again.

Imported Targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` targets:

``fmt::<component>``
  Target for specific component dependency (shared or static library); ``<component>`` name is lower-case.

It is important to note that the imported targets behave differently than variables created by this module: multiple
calls to :command:`find_package(fmt)` in the same directory or sub-directories with different options (e.g. static or
shared) will not override the values of the targets created by the first call.

Other Variables
^^^^^^^^^^^^^^^

fmt libraries come in many variants encoded in their file name.  Users or projects may tell this module which variant
to find by setting variables:

``fmt_FIND_RELEASE_ONLY``
  Set to ``ON`` or ``OFF`` to specify whether to restrict the search to release libraries only.  Default is ``OFF``.

``fmt_USE_DEBUG_LIBS``
  Set to ``ON`` or ``OFF`` to specify whether to search and use the debug libraries.  Default is ``ON`` (except when
  fmt_FIND_RELEASE_ONLY is ``ON``).

``fmt_USE_RELEASE_LIBS``
  Set to ``ON`` or ``OFF`` to specify whether to search and use the release libraries.  Default is ``ON``.

Other variables one may set to control this module are:

``fmt_DEBUG``
  Set to ``ON`` to enable debug output from ``Findfmt``.  Please enable this before filing any bug report.

``fmt_LIBRARY_DIR``
  Default value for ``fmt_LIBRARY_DIR_RELEASE`` and ``fmt_LIBRARY_DIR_DEBUG``.


Examples
^^^^^^^^

Find fmt headers only:

.. code-block:: cmake

  find_package(fmt 6.2.1)
  if(fmt_FOUND)
    include_directories(${fmt_INCLUDE_DIRS})
    add_executable(foo foo.cc)
    target_link_libraries(foo PUBLIC fmt::fmt)
  endif()

Find fmt libraries and use imported targets:

.. code-block:: cmake

  find_package(fmt 6.2.1 REQUIRED COMPONENTS fmt)
  add_executable(foo foo.cc)
  target_link_libraries(foo PUBLIC fmt::fmt)

Find fmt headers and some *static* (release only) libraries:

.. code-block:: cmake

  set(fmt_USE_DEBUG_LIBS        OFF)  # ignore debug libs and
  set(fmt_USE_RELEASE_LIBS       ON)  # only find release libs
  find_package(fmt 6.2.1 COMPONENTS fmt)
  if(fmt_FOUND)
    add_executable(foo foo.cc)
    target_link_libraries(foo PUBLIC fmt::fmt)
  endif()

.. _`fmt CMake`:

fmt CMake
^^^^^^^^^^^

If fmt was built using CMake, it provides a package configuration file for use with find_package's config mode.
This module looks for the package configuration file called ``fmtConfig.cmake`` and stores the result in ``CACHE``
entry ``fmt_DIR``.  If found, the package configuration file is loaded and this module returns with no further action.
See documentation of the fmt CMake package configuration for details on what it provides.

Set ``fmt_NO_CMAKE`` to ``ON``, to disable the search for tbb-cmake.

#]=======================================================================]

# cmake-lint: disable=C0103

include(_find_utils_begin)

set(_pkg fmt)
set(${_pkg}_INCLUDE_DIR_UP_INDEX 1)
set(${_pkg}_INCLUDE_FILE format.h)
set(${_pkg}_INCLUDE_PATH_SUFFIXES include/${_pkg} ${_pkg})
set(${_pkg}_CMAKE_CONFIG_NO_COMPONENTS TRUE)
set(${_pkg}_DEFAULT_COMPONENTS fmt)
set(${_pkg}_fmt_NAMES fmt)
set(${_pkg}_fmt_TARGET_DEFINITIONS FMT_LOCALE FMT_SHARED)
set(${_pkg}_fmt-header-only_HEADER_ONLY TRUE)
set(${_pkg}_fmt-header-only_TARGET_DEFINITIONS FMT_HEADER_ONLY=1)

if(WIN32)

elseif(APPLE)
  list(APPEND system_specific_paths "/usr/local/opt/fmt")
endif()

if(system_specific_paths)
  set(${_pkg}_INC_SYSTEM_PATHS)
  set(${_pkg}_LIB_SYSTEM_PATHS)
  foreach(_path ${system_specific_paths})
    to_cmake_path(_path)
    list(APPEND ${_pkg}_INC_SYSTEM_PATHS "${_path}")
    list(APPEND ${_pkg}_LIB_SYSTEM_PATHS "${_path}")
  endforeach()
endif()

function(fmt_version_function include_dir)
  find_file(
    _fmt_core_h core.h
    PATHS ${include_dir}/fmt
    NO_DEFAULT_PATH)

  if(_fmt_core_h)
    file(STRINGS ${_fmt_core_h} _fmt_core_h_contents REGEX "#define[ \t]+FMT_VERSION[ \t]+")
    if(_fmt_core_h_contents MATCHES "#define[ \t]+FMT_VERSION[ \t]+([0-9]+)")
      set(fmt_LIB_VERSION "${CMAKE_MATCH_1}")
    endif()
    math(EXPR ${_pkg}_VERSION_MAJOR "${fmt_LIB_VERSION} / 10000")
    math(EXPR ${_pkg}_VERSION_MINOR "${fmt_LIB_VERSION} / 100 % 100")
    math(EXPR ${_pkg}_VERSION_PATCH "${fmt_LIB_VERSION} % 100")
    set(${_pkg}_VERSION "${${_pkg}_VERSION_MAJOR}.${${_pkg}_VERSION_MINOR}.${${_pkg}_VERSION_PATCH}")
  else()
    message(WARNING "Unable to determine fmt's version since fmt/core.h file cannot be found!")
    set(${_pkg}_VERSION_MAJOR 99)
    set(${_pkg}_VERSION_MINOR 99)
    set(${_pkg}_VERSION_PATCH 99)
  endif()

  set(${_pkg}_VERSION_MAJOR
      ${${_pkg}_VERSION_MAJOR}
      PARENT_SCOPE)
  set(${_pkg}_VERSION_MINOR
      ${${_pkg}_VERSION_MINOR}
      PARENT_SCOPE)
  set(${_pkg}_VERSION_PATCH
      ${${_pkg}_VERSION_PATCH}
      PARENT_SCOPE)
  set(${_pkg}_VERSION
      ${${_pkg}_VERSION}
      PARENT_SCOPE)
endfunction()

# Update FMT library search directories with pre-built paths
function(fmt_update_library_search_dirs_with_prebuilt_paths componentlibvar basedir)
  if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")

  elseif(UNIX)
    list(APPEND ${componentlibvar} ${basedir}/lib64)
    list(APPEND ${componentlibvar} ${basedir}/lib)
    list(APPEND ${componentlibvar} ${basedir}/lib/x86_64-linux-gnu)
  endif()

  set(${componentlibvar}
      ${${componentlibvar}}
      PARENT_SCOPE)
endfunction()

include(_find_utils_end)

# ==============================================================================
