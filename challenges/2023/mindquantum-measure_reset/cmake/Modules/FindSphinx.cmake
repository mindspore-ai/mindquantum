#.rst:
# FindSphinx
# -------------
#
# Find the executable `sphinx-build`
#
# Use this module by invoking find_package with the form::
#
#   find_package(Sphinx
#     [REQUIRED]             # Fail with error if Sphinx is not found
#     [QUIET]                # Find library quietly
#     )
#
# This module defines::
#
#   Sphinx_FOUND            - True if headers and requested libraries were found
#   Sphinx_EXECUTABLE       - Full path to sphinx-build executable
#
# This module reads hints about search locations from variables
# (either CMake variables or environment variables)::
#
#   Sphinx_ROOT             - Preferred installation prefix for Sphinx
#   Sphinx_DIR              - Preferred installation prefix for Sphinx
#
# ==============================================================================
#
# Copyright 2020 <Huawei Technologies Co., Ltd>
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

# lint_cmake: -convention/filename,-package/stdargs

# cmake-lint: disable=C0103

set(Sphinx_SEARCH_PATHS
    ${Sphinx_ROOT}
    $ENV{Sphinx_ROOT}
    ${Sphinx_DIR}
    $ENV{Sphinx_DIR}
    ${CMAKE_PREFIX_PATH}
    $ENV{CMAKE_PREFIX_PATH}
    /usr
    /usr/local/
    /opt
    /opt/local)

# ==============================================================================

find_program(
  Sphinx_EXECUTABLE
  NAMES sphinx-build
  PATHS ${Sphinx_SEARCH_PATHS}
  PATH_SUFFIXES bin
  DOC "Path to sphinx-build executable")

# ==============================================================================

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Sphinx "Failed to find sphinx-build executable" Sphinx_EXECUTABLE)

# ==============================================================================
