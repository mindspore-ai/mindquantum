#.rst:
# Findtweedledum
# -------------
#
# Find the Google tweedledum logging library and its include directories
#
# Use this module by invoking find_package with the form::
#
#   find_package(tweedledum
#     [REQUIRED]             # Fail with error if tweedledum is not found
#     [QUIET]                # Find library quietly
#     )
#
# This module defines::
#
#   tweedledum_FOUND            - True if headers and requested libraries were found
#   tweedledum_INCLUDE_DIRS     - tweedledum include directories
#   tweedledum_LIBRARIES        - tweedledum libraries to be linked
#
#
# This module reads hints about search locations from variables
# (either CMake variables or environment variables)::
#
#   tweedledum_ROOT             - Preferred installation prefix for tweedledum
#   tweedledum_DIR              - Preferred installation prefix for tweedledum
#
#
# The following :prop_tgt:`IMPORTED` targets are also defined::
#
#   tweedledum::tweedledum         - Imported target for the tweedledum library
#
#
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
Findtweedledum
---------

Find tweedledum include dirs and libraries

Use this module by invoking :command:`find_package` with the form:

.. code-block:: cmake

  find_package(tweedledum
    [version] [EXACT]      # Minimum or EXACT version e.g. 2020.03
    [REQUIRED]             # Fail with error if tweedledum is not found
    [COMPONENTS <libs>...] # tweedledum libraries by their canonical name
    )

This module finds headers and requested component libraries OR a CMake package configuration file provided by a
"tweedledum CMake" build.  For the latter case skip to the :ref:`tweedledum CMake` section below.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``tweedledum_FOUND``
  True if headers and requested libraries were found.

``tweedledum_INCLUDE_DIRS``
  tweedledum include directories.

``tweedledum_LIBRARY_DIRS``
  Link directories for tweedledum libraries.

``tweedledum_LIBRARIES``
  tweedledum component libraries to be linked.

``tweedledum_<COMPONENT>_FOUND``
  True if component ``<COMPONENT>`` was found (``<COMPONENT>`` name is upper-case).

``tweedledum_<COMPONENT>_LIBRARY``
  Libraries to link for component ``<COMPONENT>`` (may include :command:`target_link_libraries` debug/optimized
  keywords).

``tweedledum_VERSION``
  tweedledum version number in ``X.Y`` format.

``tweedledum_VERSION_MAJOR``
  tweedledum major version number (``X`` in ``X.Y``).

``tweedledum_VERSION_MINOR``
  tweedledum minor version number (``Y`` in ``X.Y``).

Cache variables
^^^^^^^^^^^^^^^

Search results are saved persistently in CMake cache entries:

``tweedledum_INCLUDE_DIR``
  Directory containing tweedledum headers.

``tweedledum_LIBRARY_DIR_RELEASE``
  Directory containing release tweedledum libraries.

``tweedledum_LIBRARY_DIR_DEBUG``
  Directory containing debug tweedledum libraries.

``tweedledum_<COMPONENT>_LIBRARY_DEBUG``
  Component ``<COMPONENT>`` library debug variant.

``tweedledum_<COMPONENT>_LIBRARY_RELEASE``
  Component ``<COMPONENT>`` library release variant.

Hints
^^^^^

This module reads hints about search locations from variables:

``tweedledum_ROOT``, ``tweedledumROOT``
  Preferred installation prefix.

``tweedledum_INCLUDEDIR``
  Preferred include directory e.g. ``<prefix>/include``.

``tweedledum_LIBRARYDIR``
  Preferred library directory e.g. ``<prefix>/lib``.

``tweedledum_NO_SYSTEM_PATHS``
  Set to ``ON`` to disable searching in locations not specified by these hint variables. Default is ``OFF``.

Users may set these hints or results as ``CACHE`` entries.  Projects should not read these entries directly but
instead use the above result variables.  Note that some hint names start in upper-case ``tweedledum``.  One may
specify these as environment variables if they are not specified as CMake variables or cache entries.

This module first searches for the tweedledum header files using the above hint variables (excluding
``tweedledum_LIBRARYDIR``) and saves the result in ``tweedledum_INCLUDE_DIR``.  Then it searches for requested
component libraries using the above hints (excluding ``tweedledum_INCLUDEDIR``), "lib" directories near
``tweedledum_INCLUDE_DIR``, and the library name configuration settings below.  It saves the library directories in
``tweedledum_LIBRARY_DIR_DEBUG`` and ``tweedledum_LIBRARY_DIR_RELEASE`` and individual library locations in
``tweedledum_<COMPONENT>_LIBRARY_DEBUG`` and ``tweedledum_<COMPONENT>_LIBRARY_RELEASE``.  When one changes settings
used by previous searches in the same build tree (excluding environment variables) this module discards previous
search results affected by the changes and searches again.

Imported Targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` targets:

``tweedledum::<component>``
  Target for specific component dependency (shared or static library); ``<component>`` name is lower-case.

It is important to note that the imported targets behave differently than variables created by this module: multiple
calls to :command:`find_package(tweedledum)` in the same directory or sub-directories with different options
(e.g. static or shared) will not override the values of the targets created by the first call.

Other Variables
^^^^^^^^^^^^^^^

tweedledum libraries come in many variants encoded in their file name.  Users or projects may tell this module which
variant to find by setting variables:

``tweedledum_FIND_RELEASE_ONLY``
  Set to ``ON`` or ``OFF`` to specify whether to restrict the search to release libraries only.  Default is ``OFF``.

``tweedledum_USE_DEBUG_LIBS``
  Set to ``ON`` or ``OFF`` to specify whether to search and use the debug libraries.  Default is ``ON`` (except when
  tweedledum_FIND_RELEASE_ONLY is ``ON``).

``tweedledum_USE_RELEASE_LIBS``
  Set to ``ON`` or ``OFF`` to specify whether to search and use the release libraries.  Default is ``ON``.

Other variables one may set to control this module are:

``tweedledum_DEBUG``
  Set to ``ON`` to enable debug output from ``Findtweedledum``.  Please enable this before filing any bug report.

``tweedledum_LIBRARY_DIR``
  Default value for ``tweedledum_LIBRARY_DIR_RELEASE`` and ``tweedledum_LIBRARY_DIR_DEBUG``.


Examples
^^^^^^^^

Find tweedledum libraries and use imported targets:

.. code-block:: cmake

  find_package(tweedledum 1.0.0 REQUIRED COMPONENTS tweedledum)
  add_executable(foo foo.cc)
  target_link_libraries(foo PUBLIC tweedledum::tweedledum)

Find tweedledum headers and some *static* (release only) libraries:

.. code-block:: cmake

  set(tweedledum_USE_DEBUG_LIBS        OFF)  # ignore debug libs and
  set(tweedledum_USE_RELEASE_LIBS       ON)  # only find release libs
  find_package(tweedledum 1.0.0 COMPONENTS tweedledum)
  if(tweedledum_FOUND)
    add_executable(foo foo.cc)
    target_link_libraries(foo PUBLIC tweedledum::tweedledum)
  endif()

.. _`tweedledum CMake`:

tweedledum CMake
^^^^^^^^^^^

If tweedledum was built using CMake, it provides a package configuration file for use with find_package's config mode.
This module looks for the package configuration file called ``tweedledumConfig.cmake`` and stores the result in
``CACHE`` entry ``tweedledum_DIR``.  If found, the package configuration file is loaded and this module returns with
no further action.  See documentation of the tweedledum CMake package configuration for details on what it provides.

Set ``tweedledum_NO_CMAKE`` to ``ON``, to disable the search for tbb-cmake.

#]=======================================================================]

# cmake-lint: disable=C0103

include(_find_utils_begin)

set(_pkg tweedledum)
set(${_pkg}_INCLUDE_FILE Operator.h)
set(${_pkg}_INCLUDE_PATH_SUFFIXES include/IR include/${_pkg}/IR ${_pkg}/IR)
set(${_pkg}_INCLUDE_DIR_UP_INDEX 2)
set(${_pkg}_DEFAULT_COMPONENTS tweedledum)
set(${_pkg}_tweedledum_NAMES tweedledum)

include(_find_utils_end)

# ==============================================================================
