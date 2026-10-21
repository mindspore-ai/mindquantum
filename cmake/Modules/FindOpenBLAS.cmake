# ==============================================================================
#
# Copyright 2025 <Huawei Technologies Co., Ltd>
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
# FindOpenBLAS
# ------------
#
# Locate an OpenBLAS build usable by the `mqmps` simulator backend, i.e. an LP64
# library exporting the Fortran BLAS/LAPACK ABI together with the OpenBLAS
# threading API (`openblas_get_num_threads`, `openblas_set_num_threads`).
#
# The API surface is verified at configure time: a generic BLAS (Accelerate,
# MKL, reference BLAS, an ILP64 OpenBLAS, ...) is rejected here with an
# actionable message instead of failing when `_mq_mps` is linked.
#
# Imported targets
# ^^^^^^^^^^^^^^^^
#
# ``OpenBLAS::OpenBLAS``
#   The library, plus the ``BLAS_SYMBOL_PREFIX`` compile definition when the
#   build uses prefixed symbol names.
#
# Cache variables
# ^^^^^^^^^^^^^^^
#
# ``OpenBLAS_LIBRARY``
#   Full path of the library to link against.
# ``OpenBLAS_SYMBOL_PREFIX``
#   Prefix carried by every exported symbol (``scipy_`` for scipy-openblas32;
#   empty for a stock OpenBLAS).
#
# Hints
# ^^^^^
#
# ``OpenBLAS_ROOT``
#   Installation prefix to search (standard CMake package root variable).
# ``MQ_OPENBLAS_LIB``, ``MQ_OPENBLAS_SYMBOL_PREFIX`` (environment)
#   Exported by the MindQuantum build scripts from the scipy-openblas32 wheel;
#   they seed the cache variables above.
#
# ==============================================================================

# lint_cmake: -convention/filename,-package/stdargs

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
include(FindPackageHandleStandardArgs)

# ------------------------------------------------------------------------------
# Library and symbol prefix

set(_OpenBLAS_prefix_doc "Prefix of the symbols exported by the OpenBLAS build (e.g. scipy_)")

# A library that was not found, or an explicitly empty value, leaves nothing to reuse: drop it so that the lookup below
# runs again instead of being short-circuited by the stale entry.
if(NOT OpenBLAS_LIBRARY)
  unset(OpenBLAS_LIBRARY CACHE)
endif()

# The build scripts hand over the scipy-openblas32 artifact through MQ_OPENBLAS_LIB / MQ_OPENBLAS_SYMBOL_PREFIX. The
# environment describes one artifact, so both values are taken together, and only while no library is set: an explicit
# OpenBLAS_LIBRARY is never overridden. An empty variable counts as unset.
if(NOT OpenBLAS_LIBRARY AND NOT "$ENV{MQ_OPENBLAS_LIB}" STREQUAL "")
  set(OpenBLAS_LIBRARY
      "$ENV{MQ_OPENBLAS_LIB}"
      CACHE FILEPATH "OpenBLAS library")
  set(OpenBLAS_SYMBOL_PREFIX
      "$ENV{MQ_OPENBLAS_SYMBOL_PREFIX}"
      CACHE STRING "${_OpenBLAS_prefix_doc}" FORCE)
endif()
set(OpenBLAS_SYMBOL_PREFIX
    ""
    CACHE STRING "${_OpenBLAS_prefix_doc}")

# Homebrew installs OpenBLAS keg-only, so it never appears on the default search path.
set(_OpenBLAS_hints)
if(APPLE)
  find_program(_OpenBLAS_brew brew)
  if(_OpenBLAS_brew)
    execute_process(
      COMMAND ${_OpenBLAS_brew} --prefix openblas
      RESULT_VARIABLE _OpenBLAS_brew_result
      OUTPUT_VARIABLE _OpenBLAS_brew_prefix
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_OpenBLAS_brew_result EQUAL 0)
      list(APPEND _OpenBLAS_hints "${_OpenBLAS_brew_prefix}")
    endif()
  endif()
endif()

find_library(
  OpenBLAS_LIBRARY
  NAMES openblas
  HINTS ${_OpenBLAS_hints}
  PATH_SUFFIXES lib lib64)
mark_as_advanced(OpenBLAS_LIBRARY OpenBLAS_SYMBOL_PREFIX)

# ------------------------------------------------------------------------------
# API verification
#
# Exactly the symbols referenced by simulator/mps/detail/linalg. Only the names matter for linking, so every symbol is
# declared with a dummy signature (the same technique as CMake's own CheckLibraryExists).

set(_OpenBLAS_required_symbols openblas_get_num_threads openblas_set_num_threads cgemm_ zgemm_ cgesvd_ zgesvd_)
set(_OpenBLAS_vars OpenBLAS_LIBRARY)
set(_OpenBLAS_reason "Set OpenBLAS_ROOT to an OpenBLAS installation prefix or OpenBLAS_LIBRARY to the library file, \
or configure with -DENABLE_MPS=OFF to skip the mqmps backend.")

if(OpenBLAS_LIBRARY AND NOT EXISTS "${OpenBLAS_LIBRARY}")
  # A path that no longer exists (typically a recreated virtualenv) or a typo: report that rather than a failed link
  # test.
  set(OpenBLAS_HAS_REQUIRED_API FALSE)
  list(APPEND _OpenBLAS_vars OpenBLAS_HAS_REQUIRED_API)
  set(_OpenBLAS_reason "${OpenBLAS_LIBRARY} does not exist. Point OpenBLAS_LIBRARY at the library file, or unset it \
to search again.")
elseif(OpenBLAS_LIBRARY)
  # Re-run the check whenever its inputs change instead of trusting a stale cached result.
  set(_OpenBLAS_check_inputs "${OpenBLAS_LIBRARY};${OpenBLAS_SYMBOL_PREFIX}")
  if(NOT "${OpenBLAS_API_CHECK_INPUTS}" STREQUAL "${_OpenBLAS_check_inputs}")
    unset(OpenBLAS_HAS_REQUIRED_API CACHE)
  endif()
  set(OpenBLAS_API_CHECK_INPUTS
      "${_OpenBLAS_check_inputs}"
      CACHE INTERNAL "Inputs of the cached OpenBLAS_HAS_REQUIRED_API result")

  set(_OpenBLAS_decls)
  set(_OpenBLAS_calls)
  foreach(_OpenBLAS_sym ${_OpenBLAS_required_symbols})
    string(APPEND _OpenBLAS_decls "void ${OpenBLAS_SYMBOL_PREFIX}${_OpenBLAS_sym}();\n")
    string(APPEND _OpenBLAS_calls "    ${OpenBLAS_SYMBOL_PREFIX}${_OpenBLAS_sym}();\n")
  endforeach()

  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LIBRARIES "${OpenBLAS_LIBRARY}")
  check_cxx_source_compiles("extern \"C\" {\n${_OpenBLAS_decls}}\nint main() {\n${_OpenBLAS_calls}    return 0;\n}\n"
                            OpenBLAS_HAS_REQUIRED_API)
  cmake_pop_check_state()

  list(APPEND _OpenBLAS_vars OpenBLAS_HAS_REQUIRED_API)
  if(NOT OpenBLAS_HAS_REQUIRED_API)
    string(REPLACE ";" ", " _OpenBLAS_symbol_list "${_OpenBLAS_required_symbols}")
    set(_OpenBLAS_reason
        "${OpenBLAS_LIBRARY} does not export the LP64 symbols required by mqmps (prefix '${OpenBLAS_SYMBOL_PREFIX}': \
${_OpenBLAS_symbol_list}). Point OpenBLAS_LIBRARY at an OpenBLAS build that includes LAPACK, set \
OpenBLAS_SYMBOL_PREFIX if its symbols are prefixed (scipy-openblas32 uses 'scipy_'), or configure with \
-DENABLE_MPS=OFF to skip the mqmps backend.")
  endif()
endif()

find_package_handle_standard_args(OpenBLAS REQUIRED_VARS ${_OpenBLAS_vars} REASON_FAILURE_MESSAGE "${_OpenBLAS_reason}")

# ------------------------------------------------------------------------------
# Imported target

if(OpenBLAS_FOUND AND NOT TARGET OpenBLAS::OpenBLAS)
  add_library(OpenBLAS::OpenBLAS UNKNOWN IMPORTED)
  set_target_properties(OpenBLAS::OpenBLAS PROPERTIES IMPORTED_LOCATION "${OpenBLAS_LIBRARY}")
  if(OpenBLAS_SYMBOL_PREFIX)
    set_property(TARGET OpenBLAS::OpenBLAS PROPERTY INTERFACE_COMPILE_DEFINITIONS
                                                    "BLAS_SYMBOL_PREFIX=${OpenBLAS_SYMBOL_PREFIX}")
  endif()
endif()

unset(_OpenBLAS_prefix_doc)
unset(_OpenBLAS_hints)
unset(_OpenBLAS_brew_result)
unset(_OpenBLAS_brew_prefix)
unset(_OpenBLAS_required_symbols)
unset(_OpenBLAS_vars)
unset(_OpenBLAS_reason)
unset(_OpenBLAS_check_inputs)
unset(_OpenBLAS_decls)
unset(_OpenBLAS_calls)
unset(_OpenBLAS_sym)
unset(_OpenBLAS_symbol_list)
