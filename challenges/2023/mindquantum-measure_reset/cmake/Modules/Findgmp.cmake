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

# lint_cmake: -package/consistency,-convention/filename

#[=======================================================================[.rst:
Findgmp
---------

Find gmp include dirs and libraries

Use this module by invoking :command:`find_package` with the form:

.. code-block:: cmake

  find_package(gmp
    [version] [EXACT]      # Minimum or EXACT version e.g. 2020.03
    [REQUIRED]             # Fail with error if gmp is not found
    [COMPONENTS <libs>...] # gmp libraries by their canonical name
    )

This module finds headers and requested component libraries OR a CMake package configuration file provided by a "gmp
CMake" build.  For the latter case skip to the :ref:`gmp CMake` section below.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``gmp_FOUND``
  True if headers and requested libraries were found.

``gmp_INCLUDE_DIRS``
  gmp include directories.

``gmp_LIBRARY_DIRS``
  Link directories for gmp libraries.

``gmp_LIBRARIES``
  gmp component libraries to be linked.

``gmp_<COMPONENT>_FOUND``
  True if component ``<COMPONENT>`` was found (``<COMPONENT>`` name is upper-case).

``gmp_<COMPONENT>_LIBRARY``
  Libraries to link for component ``<COMPONENT>`` (may include :command:`target_link_libraries` debug/optimized
  keywords).

``gmp_VERSION``
  gmp version number in ``X.Y`` format.

``gmp_VERSION_MAJOR``
  gmp major version number (``X`` in ``X.Y``).

``gmp_VERSION_MINOR``
  gmp minor version number (``Y`` in ``X.Y``).

Cache variables
^^^^^^^^^^^^^^^

Search results are saved persistently in CMake cache entries:

``gmp_INCLUDE_DIR``
  Directory containing gmp headers.

``gmp_LIBRARY_DIR_RELEASE``
  Directory containing release gmp libraries.

``gmp_LIBRARY_DIR_DEBUG``
  Directory containing debug gmp libraries.

``gmp_<COMPONENT>_LIBRARY_DEBUG``
  Component ``<COMPONENT>`` library debug variant.

``gmp_<COMPONENT>_LIBRARY_RELEASE``
  Component ``<COMPONENT>`` library release variant.

Hints
^^^^^

This module reads hints about search locations from variables:

``gmp_ROOT``, ``gmpROOT``
  Preferred installation prefix.

``gmp_INCLUDEDIR``
  Preferred include directory e.g. ``<prefix>/include``.

``gmp_LIBRARYDIR``
  Preferred library directory e.g. ``<prefix>/lib``.

``gmp_NO_SYSTEM_PATHS``
  Set to ``ON`` to disable searching in locations not specified by these hint variables. Default is ``OFF``.

Users may set these hints or results as ``CACHE`` entries.  Projects should not read these entries directly but
instead use the above result variables.  Note that some hint names start in upper-case ``gmp``.  One may specify these
as environment variables if they are not specified as CMake variables or cache entries.

This module first searches for the gmp header files using the above hint variables (excluding ``gmp_LIBRARYDIR``) and
saves the result in ``gmp_INCLUDE_DIR``.  Then it searches for requested component libraries using the above hints
(excluding ``gmp_INCLUDEDIR``), "lib" directories near ``gmp_INCLUDE_DIR``, and the library name configuration
settings below.  It saves the library directories in ``gmp_LIBRARY_DIR_DEBUG`` and ``gmp_LIBRARY_DIR_RELEASE`` and
individual library locations in ``gmp_<COMPONENT>_LIBRARY_DEBUG`` and ``gmp_<COMPONENT>_LIBRARY_RELEASE``.  When one
changes settings used by previous searches in the same build tree (excluding environment variables) this module
discards previous search results affected by the changes and searches again.

Imported Targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` targets:

``gmp::<component>``
  Target for specific component dependency (shared or static library); ``<component>`` name is lower-case.

It is important to note that the imported targets behave differently than variables created by this module: multiple
calls to :command:`find_package(gmp)` in the same directory or sub-directories with different options (e.g. static or
shared) will not override the values of the targets created by the first call.

Other Variables
^^^^^^^^^^^^^^^

gmp libraries come in many variants encoded in their file name.  Users or projects may tell this module which variant
to find by setting variables:

``gmp_FIND_RELEASE_ONLY``
  Set to ``ON`` or ``OFF`` to specify whether to restrict the search to release libraries only.  Default is ``OFF``.

``gmp_USE_DEBUG_LIBS``
  Set to ``ON`` or ``OFF`` to specify whether to search and use the debug libraries.  Default is ``ON`` (except when
  gmp_FIND_RELEASE_ONLY is ``ON``).

``gmp_USE_RELEASE_LIBS``
  Set to ``ON`` or ``OFF`` to specify whether to search and use the release libraries.  Default is ``ON``.

Other variables one may set to control this module are:

``gmp_DEBUG``
  Set to ``ON`` to enable debug output from ``Findgmp``.  Please enable this before filing any bug report.

``gmp_LIBRARY_DIR``
  Default value for ``gmp_LIBRARY_DIR_RELEASE`` and ``gmp_LIBRARY_DIR_DEBUG``.


Examples
^^^^^^^^

Find gmp headers only:

.. code-block:: cmake

  find_package(gmp 6.2.1)
  if(gmp_FOUND)
    include_directories(${gmp_INCLUDE_DIRS})
    add_executable(foo foo.cc)
    target_link_libraries(foo PUBLIC gmp::gmp)
  endif()

Find gmp libraries and use imported targets:

.. code-block:: cmake

  find_package(gmp 6.2.1 REQUIRED COMPONENTS gmp)
  add_executable(foo foo.cc)
  target_link_libraries(foo PUBLIC gmp::gmp)

Find gmp headers and some *static* (release only) libraries:

.. code-block:: cmake

  set(gmp_USE_DEBUG_LIBS        OFF)  # ignore debug libs and
  set(gmp_USE_RELEASE_LIBS       ON)  # only find release libs
  find_package(gmp 6.2.1 COMPONENTS gmp)
  if(gmp_FOUND)
    add_executable(foo foo.cc)
    target_link_libraries(foo PUBLIC gmp::gmp)
  endif()

.. _`gmp CMake`:

gmp CMake
^^^^^^^^^^^

If gmp was built using CMake, it provides a package configuration file for use with find_package's config mode.
This module looks for the package configuration file called ``gmpConfig.cmake`` and stores the result in ``CACHE``
entry ``gmp_DIR``.  If found, the package configuration file is loaded and this module returns with no further action.
See documentation of the gmp CMake package configuration for details on what it provides.

Set ``gmp_NO_CMAKE`` to ``ON``, to disable the search for tbb-cmake.

#]=======================================================================]

# cmake-lint: disable=C0103

include(_find_utils_begin)

set(_pkg gmp)
set(${_pkg}_DEFINE_PREFIX __GNU_MP)
set(${_pkg}_INCLUDE_FILE gmp.h)
set(${_pkg}_INCLUDE_PATH_SUFFIXES include include/${_pkg} ${_pkg})
set(${_pkg}_DEFAULT_COMPONENTS gmp)
set(${_pkg}_gmp_NAMES gmp)
set(${_pkg}_gmpxx_NAMES gmpxx)
set(${_pkg}_gmpxx_DEPENDENCIES gmp)

# Update GMP library search directories with pre-built paths
function(gmp_update_library_search_dirs_with_prebuilt_paths componentlibvar basedir)
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
