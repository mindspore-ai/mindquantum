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

# lint_cmake: -package/consistency,-convention/filename,-whitespace/extra,-whitespace/indent

#[=======================================================================[.rst:
Findcereal
---------

Find cereal include dirs and libraries

Use this module by invoking :command:`find_package` with the form:

.. code-block:: cmake

  find_package(cereal
    [version] [EXACT]      # Minimum or EXACT version e.g. 2020.03
    [REQUIRED]             # Fail with error if cereal is not found
    [COMPONENTS <libs>...] # cereal libraries by their canonical name
    )

This module finds headers and requested component libraries OR a CMake package configuration file provided by a "cereal
CMake" build.  For the latter case skip to the :ref:`cereal CMake` section below.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``cereal_FOUND``
  True if headers and requested libraries were found.

``cereal_INCLUDE_DIRS``
  cereal include directories.

``cereal_LIBRARY_DIRS``
  Link directories for cereal libraries.

``cereal_LIBRARIES``
  cereal component libraries to be linked.

``cereal_<COMPONENT>_FOUND``
  True if component ``<COMPONENT>`` was found (``<COMPONENT>`` name is upper-case).

``cereal_<COMPONENT>_LIBRARY``
  Libraries to link for component ``<COMPONENT>`` (may include :command:`target_link_libraries` debug/optimized
  keywords).

``cereal_VERSION``
  cereal version number in ``X.Y`` format.

``cereal_VERSION_MAJOR``
  cereal major version number (``X`` in ``X.Y``).

``cereal_VERSION_MINOR``
  cereal minor version number (``Y`` in ``X.Y``).

Cache variables
^^^^^^^^^^^^^^^

Search results are saved persistently in CMake cache entries:

``cereal_INCLUDE_DIR``
  Directory containing cereal headers.

``cereal_LIBRARY_DIR_RELEASE``
  Directory containing release cereal libraries.

``cereal_LIBRARY_DIR_DEBUG``
  Directory containing debug cereal libraries.

``cereal_<COMPONENT>_LIBRARY_DEBUG``
  Component ``<COMPONENT>`` library debug variant.

``cereal_<COMPONENT>_LIBRARY_RELEASE``
  Component ``<COMPONENT>`` library release variant.

Hints
^^^^^

This module reads hints about search locations from variables:

``cereal_ROOT``, ``cerealROOT``
  Preferred installation prefix.

``cereal_INCLUDEDIR``
  Preferred include directory e.g. ``<prefix>/include``.

``cereal_LIBRARYDIR``
  Preferred library directory e.g. ``<prefix>/lib``.

``cereal_NO_SYSTEM_PATHS``
  Set to ``ON`` to disable searching in locations not specified by these hint variables. Default is ``OFF``.

Users may set these hints or results as ``CACHE`` entries.  Projects should not read these entries directly but instead
use the above result variables.  Note that some hint names start in upper-case ``cereal``.  One may specify these as
environment variables if they are not specified as CMake variables or cache entries.

This module first searches for the cereal header files using the above hint variables (excluding ``cereal_LIBRARYDIR``)
and saves the result in ``cereal_INCLUDE_DIR``.  Then it searches for requested component libraries using the above
hints (excluding ``cereal_INCLUDEDIR``), "lib" directories near ``cereal_INCLUDE_DIR``, and the library name
configuration settings below.  It saves the library directories in ``cereal_LIBRARY_DIR_DEBUG`` and
``cereal_LIBRARY_DIR_RELEASE`` and individual library locations in ``cereal_<COMPONENT>_LIBRARY_DEBUG`` and
``cereal_<COMPONENT>_LIBRARY_RELEASE``.  When one changes settings used by previous searches in the same build tree
(excluding environment variables) this module discards previous search results affected by the changes and searches
again.

Imported Targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` targets:

``cereal::<component>``
  Target for specific component dependency (shared or static library); ``<component>`` name is lower-case.

It is important to note that the imported targets behave differently than variables created by this module: multiple
calls to :command:`find_package(cereal)` in the same directory or sub-directories with different options (e.g. static or
shared) will not override the values of the targets created by the first call.

Other Variables
^^^^^^^^^^^^^^^

cereal libraries come in many variants encoded in their file name.  Users or projects may tell this module which variant
to find by setting variables:

``cereal_FIND_RELEASE_ONLY``
  Set to ``ON`` or ``OFF`` to specify whether to restrict the search to release libraries only.  Default is ``OFF``.

``cereal_USE_DEBUG_LIBS``
  Set to ``ON`` or ``OFF`` to specify whether to search and use the debug libraries.  Default is ``ON`` (except when
  cereal_FIND_RELEASE_ONLY is ``ON``).

``cereal_USE_RELEASE_LIBS``
  Set to ``ON`` or ``OFF`` to specify whether to search and use the release libraries.  Default is ``ON``.

Other variables one may set to control this module are:

``cereal_DEBUG``
  Set to ``ON`` to enable debug output from ``Findcereal``.  Please enable this before filing any bug report.

``cereal_LIBRARY_DIR``
  Default value for ``cereal_LIBRARY_DIR_RELEASE`` and ``cereal_LIBRARY_DIR_DEBUG``.


Examples
^^^^^^^^

Find cereal headers only:

.. code-block:: cmake

  find_package(cereal 6.2.1)
  if(cereal_FOUND)
    include_directories(${cereal_INCLUDE_DIRS})
    add_executable(foo foo.cc)
    target_link_libraries(foo PUBLIC cereal::cereal)
  endif()

Find cereal libraries and use imported targets:

.. code-block:: cmake

  find_package(cereal 6.2.1 REQUIRED COMPONENTS cereal)
  add_executable(foo foo.cc)
  target_link_libraries(foo PUBLIC cereal::cereal)

Find cereal headers and some *static* (release only) libraries:

.. code-block:: cmake

  set(cereal_USE_DEBUG_LIBS        OFF)  # ignore debug libs and
  set(cereal_USE_RELEASE_LIBS       ON)  # only find release libs
  find_package(cereal 6.2.1 COMPONENTS cereal)
  if(cereal_FOUND)
    add_executable(foo foo.cc)
    target_link_libraries(foo PUBLIC cereal::cereal)
  endif()

.. _`cereal CMake`:

cereal CMake
^^^^^^^^^^^

If cereal was built using CMake, it provides a package configuration file for use with find_package's config mode.  This
module looks for the package configuration file called ``cerealConfig.cmake`` and stores the result in ``CACHE`` entry
``cereal_DIR``.  If found, the package configuration file is loaded and this module returns with no further action.  See
documentation of the cereal CMake package configuration for details on what it provides.

Set ``cereal_NO_CMAKE`` to ``ON``, to disable the search for tbb-cmake.

#]=======================================================================]

# cmake-lint: disable=C0103

include(_find_utils_begin)

set(_pkg cereal)
set(${_pkg}_INCLUDE_DIR_UP_INDEX 1)
set(${_pkg}_INCLUDE_FILE cereal.h)
set(${_pkg}_INCLUDE_PATH_SUFFIXES include include/${_pkg} ${_pkg})
set(${_pkg}_DEFAULT_COMPONENTS cereal)
set(${_pkg}_cereal_NAMES cereal)
set(${_pkg}_cereal_HEADER_ONLY TRUE)

# Update CEREAL library search directories with pre-built paths
function(cereal_update_library_search_dirs_with_prebuilt_paths componentlibvar basedir)
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

# Extract EIGEN3 version from include directory
function(cereal_version_function include_dir)
  set(_paths ${include_dir})

  _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_paths")

  find_file(
    _cereal_h version.hpp
    PATHS ${_paths}
    NO_DEFAULT_PATH)

  if(_cereal_h)
    file(READ ${_cereal_h} _cereal_version_file)
    string(REGEX REPLACE ".*#define[ \t]+CEREAL_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" cereal_VERSION_MAJOR
                         "${_cereal_version_file}")
    string(REGEX REPLACE ".*#define[ \t]+CEREAL_VERSION_MINOR[ \t]+([0-9]+).*" "\\1" cereal_VERSION_MINOR
                         "${_cereal_version_file}")
    string(REGEX REPLACE ".*#define[ \t]+CEREAL_VERSION_PATCH[ \t]+([0-9]+).*" "\\1" cereal_VERSION_PATCH
                         "${_cereal_version_file}")
  else()
    message(WARNING "Unable to determine cereal's version since cereal/version.hpp file cannot be found!")
    set(cereal_VERSION_MAJOR 99)
    set(cereal_VERSION_MINOR 99)
    set(cereal_VERSION_PATCH 99)
  endif()
  set(cereal_VERSION "${cereal_VERSION_MAJOR}.${cereal_VERSION_MINOR}.${cereal_VERSION_PATCH}")

  set(cereal_VERSION_MAJOR
      ${cereal_VERSION_MAJOR}
      PARENT_SCOPE)
  set(cereal_VERSION_MINOR
      ${cereal_VERSION_MINOR}
      PARENT_SCOPE)
  set(cereal_VERSION_PATCH
      ${cereal_VERSION_PATCH}
      PARENT_SCOPE)
  set(cereal_VERSION
      ${cereal_VERSION}
      PARENT_SCOPE)
endfunction()

include(_find_utils_end)

# ==============================================================================
