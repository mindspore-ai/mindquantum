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

return()

#[=======================================================================[.rst:
Findabsl
---------

Find absl include dirs and libraries

Use this module by invoking :command:`find_package` with the form:

.. code-block:: cmake

  find_package(absl
    [version] [EXACT]      # Minimum or EXACT version e.g. 2020.03
    [REQUIRED]             # Fail with error if absl is not found
    [COMPONENTS <libs>...] # absl libraries by their canonical name
    )

This module finds headers and requested component libraries OR a CMake package configuration file provided by a "absl
CMake" build.  For the latter case skip to the :ref:`absl CMake` section below.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``absl_FOUND``
  True if headers and requested libraries were found.

``absl_INCLUDE_DIRS``
  absl include directories.

``absl_LIBRARY_DIRS``
  Link directories for absl libraries.

``absl_LIBRARIES``
  absl component libraries to be linked.

``absl_<COMPONENT>_FOUND``
  True if component ``<COMPONENT>`` was found (``<COMPONENT>`` name is upper-case).

``absl_<COMPONENT>_LIBRARY``
  Libraries to link for component ``<COMPONENT>`` (may include :command:`target_link_libraries` debug/optimized
  keywords).

``absl_VERSION``
  absl version number in ``X.Y`` format.

``absl_VERSION_MAJOR``
  absl major version number (``X`` in ``X.Y``).

``absl_VERSION_MINOR``
  absl minor version number (``Y`` in ``X.Y``).

Cache variables
^^^^^^^^^^^^^^^

Search results are saved persistently in CMake cache entries:

``absl_INCLUDE_DIR``
  Directory containing absl headers.

``absl_LIBRARY_DIR_RELEASE``
  Directory containing release absl libraries.

``absl_LIBRARY_DIR_DEBUG``
  Directory containing debug absl libraries.

``absl_<COMPONENT>_LIBRARY_DEBUG``
  Component ``<COMPONENT>`` library debug variant.

``absl_<COMPONENT>_LIBRARY_RELEASE``
  Component ``<COMPONENT>`` library release variant.

Hints
^^^^^

This module reads hints about search locations from variables:

``absl_ROOT``, ``abslROOT``
  Preferred installation prefix.

``absl_INCLUDEDIR``
  Preferred include directory e.g. ``<prefix>/include``.

``absl_LIBRARYDIR``
  Preferred library directory e.g. ``<prefix>/lib``.

``absl_NO_SYSTEM_PATHS``
  Set to ``ON`` to disable searching in locations not specified by these hint variables. Default is ``OFF``.

Users may set these hints or results as ``CACHE`` entries.  Projects should not read these entries directly but
instead use the above result variables.  Note that some hint names start in upper-case ``absl``.  One may specify these
as environment variables if they are not specified as CMake variables or cache entries.

This module first searches for the absl header files using the above hint variables (excluding ``absl_LIBRARYDIR``) and
saves the result in ``absl_INCLUDE_DIR``.  Then it searches for requested component libraries using the above hints
(excluding ``absl_INCLUDEDIR``), "lib" directories near ``absl_INCLUDE_DIR``, and the library name configuration
settings below.  It saves the library directories in ``absl_LIBRARY_DIR_DEBUG`` and ``absl_LIBRARY_DIR_RELEASE`` and
individual library locations in ``absl_<COMPONENT>_LIBRARY_DEBUG`` and ``absl_<COMPONENT>_LIBRARY_RELEASE``.  When one
changes settings used by previous searches in the same build tree (excluding environment variables) this module
discards previous search results affected by the changes and searches again.

Imported Targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` targets:

``absl::<component>``
  Target for specific component dependency (shared or static library); ``<component>`` name is lower-case.

It is important to note that the imported targets behave differently than variables created by this module: multiple
calls to :command:`find_package(absl)` in the same directory or sub-directories with different options (e.g. static or
shared) will not override the values of the targets created by the first call.

Other Variables
^^^^^^^^^^^^^^^

absl libraries come in many variants encoded in their file name.  Users or projects may tell this module which variant
to find by setting variables:

``absl_FIND_RELEASE_ONLY``
  Set to ``ON`` or ``OFF`` to specify whether to restrict the search to release libraries only.  Default is ``OFF``.

``absl_USE_DEBUG_LIBS``
  Set to ``ON`` or ``OFF`` to specify whether to search and use the debug libraries.  Default is ``ON`` (except when
  absl_FIND_RELEASE_ONLY is ``ON``).

``absl_USE_RELEASE_LIBS``
  Set to ``ON`` or ``OFF`` to specify whether to search and use the release libraries.  Default is ``ON``.

Other variables one may set to control this module are:

``absl_DEBUG``
  Set to ``ON`` to enable debug output from ``Findabsl``.  Please enable this before filing any bug report.

``absl_LIBRARY_DIR``
  Default value for ``absl_LIBRARY_DIR_RELEASE`` and ``absl_LIBRARY_DIR_DEBUG``.


Examples
^^^^^^^^

Find absl headers only:

.. code-block:: cmake

  find_package(absl 2.0.0)
  if(absl_FOUND)
    include_directories(${absl_INCLUDE_DIRS})
    add_executable(foo foo.cc)
    target_link_libraries(foo PUBLIC absl::absl)
  endif()

Find absl libraries and use imported targets:

.. code-block:: cmake

  find_package(absl 2.0.0 REQUIRED COMPONENTS absl)
  add_executable(foo foo.cc)
  target_link_libraries(foo PUBLIC absl::absl)

Find absl headers and some *static* (release only) libraries:

.. code-block:: cmake

  set(absl_USE_DEBUG_LIBS        OFF)  # ignore debug libs and
  set(absl_USE_RELEASE_LIBS       ON)  # only find release libs
  find_package(absl 2.0.0 COMPONENTS absl)
  if(absl_FOUND)
    add_executable(foo foo.cc)
    target_link_libraries(foo PUBLIC absl::absl)
  endif()

.. _`absl CMake`:

absl CMake
^^^^^^^^^^^

If absl was built using CMake, it provides a package configuration file for use with find_package's config mode.
This module looks for the package configuration file called ``abslConfig.cmake`` and stores the result in ``CACHE``
entry ``absl_DIR``.  If found, the package configuration file is loaded and this module returns with no further action.
See documentation of the absl CMake package configuration for details on what it provides.

Set ``absl_NO_CMAKE`` to ``ON``, to disable the search for the package using the CONFIG method.

.. _`absl pkg-config`:

absl CMake
^^^^^^^^^^^

If absl was installed with its pkg-config files, this module may attempt to look for absl by relying on pkg-config.
If the components are found using this method, this module returns with no further action.

Set ``absl_NO_PKGCONFIG`` to ``ON``, to disable the search for the package using the pkg-config method.

#]=======================================================================]

# cmake-lint: disable=C0103

include(_find_utils_begin)

set(_pkg absl)
set(${_pkg}_INCLUDE_FILE config.h)
set(${_pkg}_INCLUDE_PATH_SUFFIXES include/${_pkg}/base ${_pkg}/base)
set(${_pkg}_INCLUDE_DIR_UP_INDEX 1)
set(${_pkg}_DEFAULT_COMPONENTS absl)

if(NOT absl_FIND_VERSION_MAJOR)
  set(absl_FIND_VERSION_MAJOR 20220623)
endif()
include(${CMAKE_CURRENT_LIST_DIR}/absl/${absl_FIND_VERSION_MAJOR}.cmake)

if(WIN32)

elseif(APPLE)
  list(APPEND system_specific_paths "/usr/local/opt/abseil")
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

function(absl_version_function include_dir)
  find_file(
    _absl_config_h config.h
    PATHS ${include_dir}/base
    NO_DEFAULT_PATH)

  if(_absl_config_h)
    file(READ ${_absl_config_h} _absl_content)
    string(REGEX MATCHALL "#define[ \t]+[a-zA-Z0-9_]+LTS_RELEASE[a-zA-Z0-9_ ]+" _absl_content "${_absl_content}")
    list(APPEND _absl_content "")

    if("${_absl_content}" MATCHES "#define[ \t]+ABSL_LTS_RELEASE_VERSION[ \t]+([0-9]+)[^\\.]")
      set(${_pkg}_VERSION_MAJOR ${CMAKE_MATCH_1})
    endif()
    if("${_absl_content}" MATCHES "#define[ \t]+ABSL_LTS_RELEASE_PATCH_LEVEL[ \t]+([0-9]+)[^\\.]")
      set(${_pkg}_VERSION_MINOR ${CMAKE_MATCH_1})
    endif()
    set(${_pkg}_VERSION_PATCH 0)
  endif()

  if("${${_pkg}_VERSION_MAJOR}" STREQUAL "" AND "${${_pkg}_VERSION_MINOR}" STREQUAL "")
    message(WARNING "Unable to determine absl's version since absl/base/config.h file cannot be found!")
    set(${_pkg}_VERSION_MAJOR 99999999)
    set(${_pkg}_VERSION_MINOR 99)
    set(${_pkg}_VERSION_PATCH 0)
  endif()
  set(${_pkg}_VERSION "${${_pkg}_VERSION_MAJOR}.${${_pkg}_VERSION_MINOR}")

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

# Update ABSL library search directories with pre-built paths
function(absl_update_library_search_dirs_with_prebuilt_paths componentlibvar basedir)
  if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
    list(APPEND ${componentlibvar} ${basedir}/lib)
    list(APPEND ${componentlibvar} ${basedir}/bin)
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
