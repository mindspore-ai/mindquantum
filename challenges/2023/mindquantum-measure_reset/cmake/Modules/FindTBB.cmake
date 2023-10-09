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

# lint_cmake: -package/consistency,-whitespace/indent

# cmake-lint: disable=C0103,C0111,C0307

#[=======================================================================[.rst:
FindTBB
---------

Find TBB include dirs and libraries

Use this module by invoking :command:`find_package` with the form:

.. code-block:: cmake

  find_package(TBB
    [version] [EXACT]      # Minimum or EXACT version e.g. 2020.03
    [REQUIRED]             # Fail with error if TBB is not found
    [COMPONENTS <libs>...] # TBB libraries by their canonical name
    )

This module finds headers and requested component libraries OR a CMake package configuration file provided by a "TBB
CMake" build.  For the latter case skip to the :ref:`TBB CMake` section below.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``TBB_FOUND``
  True if headers and requested libraries were found.

``TBB_INCLUDE_DIRS``
  TBB include directories.

``TBB_LIBRARY_DIRS``
  Link directories for TBB libraries.

``TBB_LIBRARIES``
  TBB component libraries to be linked.

``TBB_<COMPONENT>_FOUND``
  True if component ``<COMPONENT>`` was found (``<COMPONENT>`` name is upper-case).

``TBB_<COMPONENT>_LIBRARY``
  Libraries to link for component ``<COMPONENT>`` (may include :command:`target_link_libraries` debug/optimized
  keywords).

``TBB_VERSION``
  TBB version number in ``X.Y`` format.

``TBB_VERSION_MAJOR``
  TBB major version number (``X`` in ``X.Y``).

``TBB_VERSION_MINOR``
  TBB minor version number (``Y`` in ``X.Y``).

``TBB_INTERFACE_VERSION``
  TBB engineering-focused interface version

``TBB_INTERFACE_VERSION_MAJOR``
  TBB engineering-focused interface major version

``TBB_BINARY_VERSION`` or ``TBB_COMPATIBLE_INTERFACE_VERSION`` (legacy)
  TBB binary compatibility version

Cache variables
^^^^^^^^^^^^^^^

Search results are saved persistently in CMake cache entries:

``TBB_INCLUDE_DIR``
  Directory containing TBB headers.

``TBB_LIBRARY_DIR_RELEASE``
  Directory containing release TBB libraries.

``TBB_LIBRARY_DIR_DEBUG``
  Directory containing debug TBB libraries.

``TBB_<COMPONENT>_LIBRARY_DEBUG``
  Component ``<COMPONENT>`` library debug variant.

``TBB_<COMPONENT>_LIBRARY_RELEASE``
  Component ``<COMPONENT>`` library release variant.

Hints
^^^^^

This module reads hints about search locations from variables:

``TBB_ROOT``, ``TBBROOT``
  Preferred installation prefix.

``TBB_INCLUDEDIR``
  Preferred include directory e.g. ``<prefix>/include``.

``TBB_LIBRARYDIR``
  Preferred library directory e.g. ``<prefix>/lib``.

``TBB_NO_SYSTEM_PATHS``
  Set to ``ON`` to disable searching in locations not specified by these hint variables. Default is ``OFF``.

Users may set these hints or results as ``CACHE`` entries.  Projects should not read these entries directly but
instead use the above result variables.  Note that some hint names start in upper-case ``TBB``.  One may specify these
as environment variables if they are not specified as CMake variables or cache entries.

This module first searches for the TBB header files using the above hint variables (excluding ``TBB_LIBRARYDIR``) and
saves the result in ``TBB_INCLUDE_DIR``.  Then it searches for requested component libraries using the above hints
(excluding ``TBB_INCLUDEDIR``), "lib" directories near ``TBB_INCLUDE_DIR``, and the library name configuration
settings below.  It saves the library directories in ``TBB_LIBRARY_DIR_DEBUG`` and ``TBB_LIBRARY_DIR_RELEASE`` and
individual library locations in ``TBB_<COMPONENT>_LIBRARY_DEBUG`` and ``TBB_<COMPONENT>_LIBRARY_RELEASE``.  When one
changes settings used by previous searches in the same build tree (excluding environment variables) this module
discards previous search results affected by the changes and searches again.

Imported Targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` targets:

``TBB::<component>``
  Target for specific component dependency (shared or static library); ``<component>`` name is lower-case.

Implicit dependencies such as ``TBB::tbbmalloc`` requiring ``TBB::tbb`` will be automatically detected and satisfied,
even if tbb is not specified when using :command:`find_package` and if ``TBB::tbb`` is not added to
:command:`target_link_libraries`.

It is important to note that the imported targets behave differently than variables created by this module: multiple
calls to :command:`find_package(TBB)` in the same directory or sub-directories with different options (e.g. static or
shared) will not override the values of the targets created by the first call.

Other Variables
^^^^^^^^^^^^^^^

TBB libraries come in many variants encoded in their file name.  Users or projects may tell this module which variant
to find by setting variables:

``TBB_FIND_RELEASE_ONLY``
  Set to ``ON`` or ``OFF`` to specify whether to restrict the search to release libraries only.  Default is ``OFF``.

``TBB_USE_DEBUG_LIBS``
  Set to ``ON`` or ``OFF`` to specify whether to search and use the debug libraries.  Default is ``ON`` (except when
  TBB_FIND_RELEASE_ONLY is ``ON``).

``TBB_USE_RELEASE_LIBS``
  Set to ``ON`` or ``OFF`` to specify whether to search and use the release libraries.  Default is ``ON``.

Other variables one may set to control this module are:

``TBB_DEBUG``
  Set to ``ON`` to enable debug output from ``FindTBB``.  Please enable this before filing any bug report.

``TBB_LIBRARY_DIR``
  Default value for ``TBB_LIBRARY_DIR_RELEASE`` and ``TBB_LIBRARY_DIR_DEBUG``.


Examples
^^^^^^^^

Find TBB headers only:

.. code-block:: cmake

  find_package(TBB 2020.03)
  if(TBB_FOUND)
    include_directories(${TBB_INCLUDE_DIRS})
    add_executable(foo foo.cc)
    target_link_libraries(foo PUBLIC TBB::tbb)
  endif()

Find TBB libraries and use imported targets:

.. code-block:: cmake

  find_package(TBB 2020.03 REQUIRED COMPONENTS tbbmalloc tbbmalloc_proxy)
  add_executable(foo foo.cc)
  target_link_libraries(foo PUBLIC TBB::tbbmalloc TBB::tbbmalloc_proxy)

Find TBB headers and some *static* (release only) libraries:

.. code-block:: cmake

  set(TBB_USE_DEBUG_LIBS        OFF)  # ignore debug libs and
  set(TBB_USE_RELEASE_LIBS       ON)  # only find release libs
  find_package(TBB 2020.03 COMPONENTS TBB::tbbmalloc TBB::tbbmalloc_proxy)
  if(TBB_FOUND)
    add_executable(foo foo.cc)
    target_link_libraries(foo PUBLIC TBB::tbbmalloc TBB::tbbmalloc_proxy)
  endif()

.. _`TBB CMake`:

TBB CMake
^^^^^^^^^^^

If TBB was built using CMake, it provides a package configuration file for use with find_package's config mode.
This module looks for the package configuration file called ``TBBConfig.cmake`` and stores the result in ``CACHE``
entry ``TBB_DIR``.  If found, the package configuration file is loaded and this module returns with no further action.
See documentation of the TBB CMake package configuration for details on what it provides.

Set ``TBB_NO_TBB_CMAKE`` to ``ON``, to disable the search for tbb-cmake.

#]=======================================================================]

include(_find_utils_begin)

set(_pkg TBB)
set(${_pkg}_NO_PKGCONFIG ON) # NB: because only tbb can be found using PkgConfig...
set(${_pkg}_INCLUDE_FILE tbb.h)
set(${_pkg}_INCLUDE_PATH_SUFFIXES include/oneapi oneapi include/tbb tbb)
set(${_pkg}_INCLUDE_DIR_UP_INDEX 1)
set(${_pkg}_DEFAULT_COMPONENTS tbb tbbmalloc tbbmalloc_proxy irml)

set(${_pkg}_tbb_NAMES tbb)
set(${_pkg}_tbb_TARGET_DEFINITIONS "\$<\$<CONFIG:DEBUG>:TBB_USE_DEBUG>")

set(${_pkg}_tbbmalloc_NAMES tbbmalloc)
set(${_pkg}_tbbmalloc_PKGCONFIG_NAMES tbb)
set(${_pkg}_tbbmalloc_TARGET_DEFINITIONS "\$<\$<CONFIG:DEBUG>:TBB_USE_DEBUG>")

set(${_pkg}_tbbmalloc_proxy_NAMES tbbmalloc_proxy)
set(${_pkg}_tbbmalloc_proxy_PKGCONFIG_NAMES tbb)
set(${_pkg}_tbbmalloc_proxy_DEPENDENCIES tbbmalloc)
set(${_pkg}_tbbmalloc_proxy_TARGET_DEFINITIONS "\$<\$<CONFIG:DEBUG>:TBB_USE_DEBUG>")

set(${_pkg}_tbbbind_2_5_NAMES tbbbind_2_5)
set(${_pkg}_tbbbind_2_5_PKGCONFIG_NAMES tbb)
set(${_pkg}_tbbbind_2_5_EXTERNAL_DEPENDENCIES PkgConfig::HWLOC)
set(${_pkg}_tbbbind_2_5_TARGET_DEFINITIONS "\$<\$<CONFIG:DEBUG>:TBB_USE_DEBUG>")

set(${_pkg}_irml_NAMES irml)
set(${_pkg}_irml_PKGCONFIG_NAMES tbb)
set(${_pkg}_irml_TARGET_DEFINITIONS "\$<\$<CONFIG:DEBUG>:TBB_USE_DEBUG>")

function(TBB_version_function include_dir)
  find_file(
    _TBB_version_h version.h
    PATHS ${include_dir}/oneapi/tbb ${include_dir}/tbb
    NO_DEFAULT_PATH)

  if(_TBB_version_h)
    file(READ ${_TBB_version_h} _TBB_content)
    string(REGEX MATCHALL "#define[ \t]+TBB_VERSION[a-zA-Z0-9_ ]+" _TBB_content "${_TBB_content}")
    list(APPEND _TBB_content "")

    if("${_TBB_content}" MATCHES "#define[ \t]+TBB_VERSION_MAJOR[ \t]+([0-9]+)[^\\.]")
      set(${_pkg}_VERSION_MAJOR ${CMAKE_MATCH_1})
    endif()
    if("${_TBB_content}" MATCHES "#define[ \t]+TBB_VERSION_MINOR[ \t]+([0-9]+)[^\\.]")
      set(${_pkg}_VERSION_MINOR ${CMAKE_MATCH_1})
    endif()
    if("${_TBB_content}" MATCHES "#define[ \t]+TBB_VERSION_PATCH[ \t]+([0-9]+)[^\\.]")
      set(${_pkg}_VERSION_PATCH ${CMAKE_MATCH_1})
    endif()
  endif()

  if("${${_pkg}_VERSION_MAJOR}" STREQUAL "" AND "${${_pkg}_VERSION_MINOR}" STREQUAL "")
    message(WARNING "Unable to determine TBB's version since /oneapi/tbb/version.h file cannot be found!")
    set(${_pkg}_VERSION_MAJOR 99999)
    set(${_pkg}_VERSION_MINOR 9)
    set(${_pkg}_VERSION_PATCH 9)
  endif()
  set(${_pkg}_VERSION "${${_pkg}_VERSION_MAJOR}.${${_pkg}_VERSION_MINOR}.${${_pkg}_VERSION_PATCH}")

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

function(TBB_update_library_search_dirs_with_prebuilt_paths componentlibvar basedir)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_arch intel64)
  else()
    set(_arch ia32)
  endif()

  if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
    if(MSVC_TOOLSET_VERSION GREATER_EQUAL 140)
      # Shared libraries
      list(APPEND ${componentlibvar} ${basedir}/redist/${_arch}/vc14)
      list(APPEND ${componentlibvar} ${basedir}/redist/${_arch}/vc14_uwp)
      # Shared libraries (legacy)
      list(APPEND ${componentlibvar} ${basedir}/bin/${_arch}/vc14)
      list(APPEND ${componentlibvar} ${basedir}/bin/${_arch}/vc14_uwp)
    elseif(MSVC_TOOLSET_VERSION GREATER_EQUAL 120)
      # Shared libraries (legacy)
      list(APPEND ${componentlibvar} ${basedir}/bin/${_arch}/vc12)
      list(APPEND ${componentlibvar} ${basedir}/bin/${_arch}/vc12_ui)
    endif()
    list(APPEND ${componentlibvar} ${basedir}/redist/${_arch}/vc_mt)
    list(APPEND ${componentlibvar} ${basedir}/bin/${_arch}/vc_mt)
  elseif(UNIX)
    list(APPEND ${componentlibvar} ${basedir}/lib64)
    list(APPEND ${componentlibvar} ${basedir}/lib)
    if(NOT APPLE)
      foreach(v 4.8 4.7 4.4)
        list(APPEND ${componentlibvar} ${basedir}/lib/${_arch}/gcc${v})
      endforeach()
    endif()
  endif()

  set(${componentlibvar}
      ${${componentlibvar}}
      PARENT_SCOPE)
endfunction()

function(TBB_windows_set_import_library_path target basename)
  if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
    if(ARGC GREATER 2)
      set(property ${basename}_${ARGV2})
      set(implib_prop IMPORTED_IMPLIB_${ARGV2})
    else()
      set(property ${basename})
      set(implib_prop IMPORTED_IMPLIB)
    endif()
    get_target_property(_value ${target} ${property})
    if(_value)
      get_filename_component(_lib_name ${_value} NAME_WE)
      get_filename_component(_vc_dir ${_value} DIRECTORY)
      get_filename_component(_arch_dir ${_vc_dir} DIRECTORY)
      get_filename_component(_tbb_root ${_value}/../../../ ABSOLUTE)
      set_target_properties(${target} PROPERTIES ${implib_prop}
                                                 ${_tbb_root}/lib/${_arch_dir}/${_vc_dir}/${_lib_name}.lib)
    endif()
  endif()
endfunction()

include(_find_utils_end)

# ==============================================================================
