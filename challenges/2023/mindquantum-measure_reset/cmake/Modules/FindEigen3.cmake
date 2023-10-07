# ==============================================================================
#
# Copyright 2021 <Huawei Technologies Co., Ltd>
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
#
# ==============================================================================

# lint_cmake: -whitespace/indent,-package/consistency,-convention/filename,-whitespace/extra

#[=======================================================================[.rst:
FindEigen3
---------

Find Eigen3 include dirs and libraries

Use this module by invoking :command:`find_package` with the form:

.. code-block:: cmake

  find_package(Eigen3
    [version] [EXACT]      # Minimum or EXACT version e.g. 2020.03
    [REQUIRED]             # Fail with error if Eigen3 is not found
    )

This module finds headers and requested component libraries OR a CMake package configuration file provided by a "Eigen3
CMake" build.  For the latter case skip to the :ref:`Eigen3 CMake` section below.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Eigen3_FOUND``
  True if headers and requested libraries were found.

``Eigen3_INCLUDE_DIRS``
  Eigen3 include directories.

``Eigen3_LIBRARY_DIRS``
  Link directories for Eigen3 libraries.

``Eigen3_LIBRARIES``
  Eigen3 component libraries to be linked.

``Eigen3_VERSION``
  Eigen3 version number in ``X.Y`` format.

``Eigen3_VERSION_MAJOR``
  Eigen3 major version number (``X`` in ``X.Y``).

``Eigen3_VERSION_MINOR``
  Eigen3 minor version number (``Y`` in ``X.Y``).

Cache variables
^^^^^^^^^^^^^^^

Search results are saved persistently in CMake cache entries:

``Eigen3_INCLUDE_DIR``
  Directory containing Eigen3 headers.

``Eigen3_LIBRARY_DIR_RELEASE``
  Directory containing release Eigen3 libraries.

``Eigen3_LIBRARY_DIR_DEBUG``
  Directory containing debug Eigen3 libraries.

Hints
^^^^^

This module reads hints about search locations from variables:

``Eigen3_ROOT``, ``Eigen3ROOT``
  Preferred installation prefix.

``Eigen3_INCLUDEDIR``
  Preferred include directory e.g. ``<prefix>/include``.

``Eigen3_LIBRARYDIR``
  Preferred library directory e.g. ``<prefix>/lib``.

``Eigen3_NO_CMAKE``
  Set to ``ON`` to disable searching for CMake configuration files.Default is ``OFF``.

``Eigen3_NO_SYSTEM_PATHS``
  Set to ``ON`` to disable searching in locations not specified by these hint variables. Default is ``OFF``.

Users may set these hints or results as ``CACHE`` entries.  Projects should not read these entries directly but instead
use the above result variables.  Note that some hint names start in upper-case ``Eigen3``.  One may specify these as
environment variables if they are not specified as CMake variables or cache entries.

This module first searches for the Eigen3 header files using the above hint variables (excluding ``Eigen3_LIBRARYDIR``)
and saves the result in ``Eigen3_INCLUDE_DIR``.  Then it searches for requested component libraries using the above
hints (excluding ``Eigen3_INCLUDEDIR``), "lib" directories near ``Eigen3_INCLUDE_DIR``, and the library name
configuration settings below.  It saves the library directories in ``Eigen3_LIBRARY_DIR_DEBUG`` and
``Eigen3_LIBRARY_DIR_RELEASE`` and individual library locations in ``Eigen3_<COMPONENT>_LIBRARY_DEBUG`` and
``Eigen3_<COMPONENT>_LIBRARY_RELEASE``.  When one changes settings used by previous searches in the same build tree
(excluding environment variables) this module discards previous search results affected by the changes and searches
again.

Imported Targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` targets:

``Eigen3::<component>``
  Target for specific component dependency (shared or static library); ``<component>`` name is lower-case.

It is important to note that the imported targets behave differently than variables created by this module: multiple
calls to :command:`find_package(Eigen3)` in the same directory or sub-directories with different options (e.g. static or
shared) will not override the values of the targets created by the first call.

Other Variables
^^^^^^^^^^^^^^^

Other variables one may set to control this module are:

``Eigen3_DEBUG``
  Set to ``ON`` to enable debug output from ``FindEigen3``.  Please enable this before filing any bug report.

``Eigen3_LIBRARY_DIR``
  Default value for ``Eigen3_LIBRARY_DIR_RELEASE`` and ``Eigen3_LIBRARY_DIR_DEBUG``.


Examples
^^^^^^^^

Find Eigen3 headers only:

.. code-block:: cmake

  find_package(Eigen3 3.4.0)
  if(Eigen3_FOUND)
    include_directories(${Eigen3_INCLUDE_DIRS})
    add_executable(foo foo.cc)
    target_link_libraries(foo PUBLIC Eigen3::Eigen3)
  endif()

Find Eigen3 libraries and use imported targets:

.. code-block:: cmake

  find_package(Eigen3 6.2.1 REQUIRED)
  add_executable(foo foo.cc)
  target_link_libraries(foo PUBLIC Eigen3::Eigen3)

.. _`Eigen3 CMake`:

Eigen3 CMake
^^^^^^^^^^^^

If Eigen3 was built using CMake, it provides a package configuration file for use with find_package's config mode.  This
module looks for the package configuration file called ``Eigen3Config.cmake`` and stores the result in ``CACHE`` entry
``Eigen3_DIR``.  If found, the package configuration file is loaded and this module returns with no further action.  See
documentation of the Eigen3 CMake package configuration for details on what it provides.

Set ``Eigen3_NO_CMAKE`` to ``ON``, to disable the search for Eigen3-cmake.

Eigen3 PkgConfig
^^^^^^^^^^^^^^^^

If Eigen3 was installed with its pkg-config files, this module may attempt to look for Eigen3 by relying on pkg-config.
If the components are found using this method, this module returns with no further action.

Set ``Eigen3_NO_PKGCONFIG`` to ``ON``, to disable the search for the package using the pkg-config method.

#]=======================================================================]

# cmake-lint: disable=C0103

include(_find_utils_begin)

set(_pkg Eigen3)
set(${_pkg}_INCLUDE_FILE signature_of_eigen3_matrix_library)
set(${_pkg}_INCLUDE_PATH_SUFFIXES include include/${_pkg} ${_pkg} include/eigen3 eigen3)
set(${_pkg}_DEFAULT_COMPONENTS Eigen)
set(${_pkg}_Eigen_NAMES Eigen)
set(${_pkg}_Eigen_HEADER_ONLY TRUE)

if(WIN32)

elseif(APPLE)
  list(APPEND system_specific_paths "/usr/local/opt/eigen")
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

# Extract EIGEN3 version from include directory
function(Eigen3_version_function include_dir)
  set(_paths ${include_dir}/Eigen/src/Core/util/)

  _debug_print_var("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_paths")

  find_file(
    _Eigen3_h Macros.h
    PATHS ${_paths}
    NO_DEFAULT_PATH)

  if(_Eigen3_h)
    file(READ ${_Eigen3_h} _Eigen3_version_file)
    string(REGEX REPLACE ".*#define[ \t]+EIGEN_WORLD_VERSION[ \t]+([0-9]+).*" "\\1" Eigen3_VERSION_MAJOR
                         "${_Eigen3_version_file}")
    string(REGEX REPLACE ".*#define[ \t]+EIGEN_MAJOR_VERSION[ \t]+([0-9]+).*" "\\1" Eigen3_VERSION_MINOR
                         "${_Eigen3_version_file}")
    string(REGEX REPLACE ".*#define[ \t]+EIGEN_MINOR_VERSION[ \t]+([0-9]+).*" "\\1" Eigen3_VERSION_PATCH
                         "${_Eigen3_version_file}")
  else()
    message(WARNING "Unable to determine Eigen3's version since Eigen/src/Core/util/Macros.h file cannot be found!")
    set(Eigen3_VERSION_MAJOR 99)
    set(Eigen3_VERSION_MINOR 99)
    set(Eigen3_VERSION_PATCH 99)
  endif()
  set(Eigen3_VERSION "${Eigen3_VERSION_MAJOR}.${Eigen3_VERSION_MINOR}.${Eigen3_VERSION_PATCH}")

  set(Eigen3_VERSION_MAJOR
      ${Eigen3_VERSION_MAJOR}
      PARENT_SCOPE)
  set(Eigen3_VERSION_MINOR
      ${Eigen3_VERSION_MINOR}
      PARENT_SCOPE)
  set(Eigen3_VERSION_PATCH
      ${Eigen3_VERSION_PATCH}
      PARENT_SCOPE)
  set(Eigen3_VERSION
      ${Eigen3_VERSION}
      PARENT_SCOPE)
endfunction()

include(_find_utils_end)

# ==============================================================================
