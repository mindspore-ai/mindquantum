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
Findnlohmann_json
---------

Find nlohmann_json include dirs and libraries

Use this module by invoking :command:`find_package` with the form:

.. code-block:: cmake

  find_package(nlohmann_json
    [version] [EXACT]      # Minimum or EXACT version e.g. 2020.03
    [REQUIRED]             # Fail with error if nlohmann_json is not found
    [COMPONENTS <libs>...] # nlohmann_json libraries by their canonical name
    )

This module finds headers and requested component libraries OR a CMake package configuration file provided by a
"nlohmann_json CMake" build.  For the latter case skip to the :ref:`nlohmann_json CMake` section below.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``nlohmann_json_FOUND``
  True if headers and requested libraries were found.

``nlohmann_json_INCLUDE_DIRS``
  nlohmann_json include directories.

``nlohmann_json_LIBRARY_DIRS``
  Link directories for nlohmann_json libraries.

``nlohmann_json_LIBRARIES``
  nlohmann_json component libraries to be linked.

``nlohmann_json_<COMPONENT>_FOUND``
  True if component ``<COMPONENT>`` was found (``<COMPONENT>`` name is upper-case).

``nlohmann_json_<COMPONENT>_LIBRARY``
  Libraries to link for component ``<COMPONENT>`` (may include :command:`target_link_libraries` debug/optimized
  keywords).

``nlohmann_json_VERSION``
  nlohmann_json version number in ``X.Y`` format.

``nlohmann_json_VERSION_MAJOR``
  nlohmann_json major version number (``X`` in ``X.Y``).

``nlohmann_json_VERSION_MINOR``
  nlohmann_json minor version number (``Y`` in ``X.Y``).

Cache variables
^^^^^^^^^^^^^^^

Search results are saved persistently in CMake cache entries:

``nlohmann_json_INCLUDE_DIR``
  Directory containing nlohmann_json headers.

``nlohmann_json_LIBRARY_DIR_RELEASE``
  Directory containing release nlohmann_json libraries.

``nlohmann_json_LIBRARY_DIR_DEBUG``
  Directory containing debug nlohmann_json libraries.

``nlohmann_json_<COMPONENT>_LIBRARY_DEBUG``
  Component ``<COMPONENT>`` library debug variant.

``nlohmann_json_<COMPONENT>_LIBRARY_RELEASE``
  Component ``<COMPONENT>`` library release variant.

Hints
^^^^^

This module reads hints about search locations from variables:

``nlohmann_json_ROOT``, ``nlohmann_jsonROOT``
  Preferred installation prefix.

``nlohmann_json_INCLUDEDIR``
  Preferred include directory e.g. ``<prefix>/include``.

``nlohmann_json_LIBRARYDIR``
  Preferred library directory e.g. ``<prefix>/lib``.

``nlohmann_json_NO_SYSTEM_PATHS``
  Set to ``ON`` to disable searching in locations not specified by these hint variables. Default is ``OFF``.

Users may set these hints or results as ``CACHE`` entries.  Projects should not read these entries directly but instead
use the above result variables.  Note that some hint names start in upper-case ``nlohmann_json``.  One may specify these
as environment variables if they are not specified as CMake variables or cache entries.

This module first searches for the nlohmann_json header files using the above hint variables (excluding
``nlohmann_json_LIBRARYDIR``) and saves the result in ``nlohmann_json_INCLUDE_DIR``.  Then it searches for requested
component libraries using the above hints (excluding ``nlohmann_json_INCLUDEDIR``), "lib" directories near
``nlohmann_json_INCLUDE_DIR``, and the library name configuration settings below.  It saves the library directories in
``nlohmann_json_LIBRARY_DIR_DEBUG`` and ``nlohmann_json_LIBRARY_DIR_RELEASE`` and individual library locations in
``nlohmann_json_<COMPONENT>_LIBRARY_DEBUG`` and ``nlohmann_json_<COMPONENT>_LIBRARY_RELEASE``.  When one changes
settings used by previous searches in the same build tree (excluding environment variables) this module discards
previous search results affected by the changes and searches again.

Imported Targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` targets:

``nlohmann_json::<component>``
  Target for specific component dependency (shared or static library); ``<component>`` name is lower-case.

It is important to note that the imported targets behave differently than variables created by this module: multiple
calls to :command:`find_package(nlohmann_json)` in the same directory or sub-directories with different options
(e.g. static or shared) will not override the values of the targets created by the first call.

Other Variables
^^^^^^^^^^^^^^^

nlohmann_json libraries come in many variants encoded in their file name.  Users or projects may tell this module which
variant to find by setting variables:

``nlohmann_json_FIND_RELEASE_ONLY``
  Set to ``ON`` or ``OFF`` to specify whether to restrict the search to release libraries only.  Default is ``OFF``.

``nlohmann_json_USE_DEBUG_LIBS``
  Set to ``ON`` or ``OFF`` to specify whether to search and use the debug libraries.  Default is ``ON`` (except when
  nlohmann_json_FIND_RELEASE_ONLY is ``ON``).

``nlohmann_json_USE_RELEASE_LIBS``
  Set to ``ON`` or ``OFF`` to specify whether to search and use the release libraries.  Default is ``ON``.

Other variables one may set to control this module are:

``nlohmann_json_DEBUG``
  Set to ``ON`` to enable debug output from ``Findnlohmann_json``.  Please enable this before filing any bug report.

``nlohmann_json_LIBRARY_DIR``
  Default value for ``nlohmann_json_LIBRARY_DIR_RELEASE`` and ``nlohmann_json_LIBRARY_DIR_DEBUG``.


Examples
^^^^^^^^

Find nlohmann_json headers only:

.. code-block:: cmake

  find_package(nlohmann_json 2.0.0)
  if(nlohmann_json_FOUND)
    include_directories(${nlohmann_json_INCLUDE_DIRS})
    add_executable(foo foo.cc)
    target_link_libraries(foo PUBLIC nlohmann_json::nlohmann_json)
  endif()

Find nlohmann_json libraries and use imported targets:

.. code-block:: cmake

  find_package(nlohmann_json 2.0.0 REQUIRED COMPONENTS nlohmann_json)
  add_executable(foo foo.cc)
  target_link_libraries(foo PUBLIC nlohmann_json::nlohmann_json)

Find nlohmann_json headers and some *static* (release only) libraries:

.. code-block:: cmake

  set(nlohmann_json_USE_DEBUG_LIBS        OFF)  # ignore debug libs and
  set(nlohmann_json_USE_RELEASE_LIBS       ON)  # only find release libs
  find_package(nlohmann_json 2.0.0 COMPONENTS nlohmann_json)
  if(nlohmann_json_FOUND)
    add_executable(foo foo.cc)
    target_link_libraries(foo PUBLIC nlohmann_json::nlohmann_json)
  endif()

.. _`nlohmann_json CMake`:

nlohmann_json CMake
^^^^^^^^^^^^^^^^^^^

If nlohmann_json was built using CMake, it provides a package configuration file for use with find_package's config
mode.  This module looks for the package configuration file called ``nlohmann_jsonConfig.cmake`` and stores the result
in ``CACHE`` entry ``nlohmann_json_DIR``.  If found, the package configuration file is loaded and this module returns
with no further action.  See documentation of the nlohmann_json CMake package configuration for details on what it
provides.

Set ``nlohmann_json_NO_CMAKE`` to ``ON``, to disable the search for the package using the CONFIG method.

.. _`nlohmann_json pkg-config`:

nlohmann_json CMake
^^^^^^^^^^^^^^^^^^^

If nlohmann_json was installed with its pkg-config files, this module may attempt to look for nlohmann_json by relying
on pkg-config.  If the components are found using this method, this module returns with no further action.

Set ``nlohmann_json_NO_PKGCONFIG`` to ``ON``, to disable the search for the package using the pkg-config method.

#]=======================================================================]

# cmake-lint: disable=C0103

include(_find_utils_begin)

set(_pkg nlohmann_json)
set(${_pkg}_INCLUDE_FILE json.hpp)
set(${_pkg}_INCLUDE_PATH_SUFFIXES include include/nlohmann nlohmann)
set(${_pkg}_DEFAULT_COMPONENTS nlohmann_json)
set(${_pkg}_nlohmann_json_NAMES nlohmann_json)
set(${_pkg}_nlohmann_json_HEADER_ONLY TRUE)

if(WIN32)

elseif(APPLE)
  list(APPEND system_specific_paths "/usr/local/opt/nlohmann-json")
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

include(_find_utils_end)

# ==============================================================================
