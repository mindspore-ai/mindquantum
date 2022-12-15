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

# lint_cmake: -whitespace/indent

include(CMakeParseArguments)

set(_allowed_build_types Release RelWithDebInfo MinSizeRel Debug)

if(ENABLE_SANITIZERS)
  if(SANITIZER_USE_O1)
    set(_sanitizer_compiler_flags "-O1")
  elseif(SANITIZER_USE_Og)
    set(_sanitizer_compiler_flags "-Og")
  else()
    set(_sanitizer_compiler_flags)
  endif()
endif()

# ==============================================================================

# ~~~
# Add a new build type
#
# add_new_build_type(<build_type_name>)
# ~~~
macro(add_new_build_type build_type_name)
  get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(_is_multi_config)
    if(NOT "${build_type_name}" IN_LIST CMAKE_CONFIGURATION_TYPES)
      list(APPEND CMAKE_CONFIGURATION_TYPES "${build_type_name}")
    endif()
  else()
    list(APPEND _allowed_build_types "${build_type_name}")
  endif()
  unset(_is_multi_config)
endmacro()

# ~~~
# Define new compiler/linker flags CMake variables for a new build type
#
# define_compiler_linker_flags(<build_type>
#                              <inherited_build_type>)
# ~~~
macro(define_compiler_linker_flags build_type inherited_build_type)
  cmake_parse_arguments(DCLF "" "COMPILER_FLAGS;LINKER_FLAGS" "" ${ARGN})

  string(TOUPPER "${build_type}" _build_type)
  string(TOUPPER "${inherited_build_type}" _inherited_build_type)

  set(CMAKE_C_FLAGS_${_build_type}
      "${CMAKE_C_FLAGS_${_inherited_build_type}} ${DCLF_COMPILER_FLAGS}"
      CACHE STRING "Flags used by the C compiler for Asan build type or configuration." FORCE)

  set(CMAKE_CXX_FLAGS_${_build_type}
      "${CMAKE_CXX_FLAGS_${_inherited_build_type}} ${DCLF_COMPILER_FLAGS}"
      CACHE STRING "Flags used by the C++ compiler for Asan build type or configuration." FORCE)

  set(CMAKE_EXE_LINKER_FLAGS_${_build_type}
      "${CMAKE_SHARED_LINKER_FLAGS_${_inherited_build_type}} ${DCLF_LINKER_FLAGS}"
      CACHE STRING "Linker flags to be used to create executables for Asan build type." FORCE)

  set(CMAKE_SHARED_LINKER_FLAGS_${_build_type}
      "${CMAKE_SHARED_LINKER_FLAGS_${_inherited_build_type}} ${DCLF_LINKER_FLAGS}"
      CACHE STRING "Linker lags to be used to create shared libraries for Asan build type." FORCE)

  set(CMAKE_MODULE_LINKER_FLAGS_${_build_type}
      "${CMAKE_SHARED_LINKER_FLAGS_${_inherited_build_type}} ${DCLF_LINKER_FLAGS}"
      CACHE STRING "Linker lags to be used to create modules for Asan build type." FORCE)

  mark_as_advanced(CMAKE_C_FLAGS_${_build_type} CMAKE_CXX_FLAGS_${_build_type} CMAKE_EXE_LINKER_FLAGS_${_build_type}
                   CMAKE_SHARED_LINKER_FLAGS_${_build_type} CMAKE_MODULE_LINKER_FLAGS_${_build_type})
endmacro()

# ==============================================================================
# Sanitizer build type

if(ENABLE_SANITIZERS)
  set(_build_type_name Sanitizer)
  string(TOUPPER "${_build_type_name}" _build_type_name)

  if(NOT DEFINED SANITIZER_INHERIT_TYPE)
    set(SANITIZER_INHERIT_TYPE
        RELWITHDEBINFO
        CACHE STRING "Build type that the Sanitizer build type inherit the flags from")
    mark_as_advanced(SANITIZER_INHERIT_TYPE)
  endif()

  add_new_build_type(Sanitizer)
  define_compiler_linker_flags(Sanitizer "${SANITIZER_INHERIT_TYPE}" COMPILER_FLAGS " ${_sanitizer_compiler_flags}")
endif()

# ==============================================================================

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
endif()
if(NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES
      "${_allowed_build_types}"
      CACHE INTERNAL "Allowed build types")
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${_allowed_build_types}")

# ==============================================================================

if("${CMAKE_BUILD_TYPE}" STREQUAL "Sanitizer")
  set(_cmake_interprocedural_optimization
      "${CMAKE_INTERPROCEDURAL_OPTIMIZATION}"
      CACHE INTERNAL "Original value of CMAKE_INTERPROCEDURAL_OPTIMIZATION")
  set(_modified_cmake_interprocedural_optimization
      TRUE
      CACHE INTERNAL "")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
elseif(_modified_cmake_interprocedural_optimization)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION "${_cmake_interprocedural_optimization}")
endif()

# ==============================================================================
