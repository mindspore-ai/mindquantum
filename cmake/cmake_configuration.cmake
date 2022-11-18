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

set(_allowed_build_types Debug Release RelWithDebInfo MinSizeRel)
get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

# ==============================================================================
# Asan build type

set(_build_type_name AsanAddr)
string(TOUPPER "${_build_type_name}" _build_type)

if(_is_multi_config)
  if(NOT "${_build_type_name}" IN_LIST CMAKE_CONFIGURATION_TYPES)
    list(APPEND CMAKE_CONFIGURATION_TYPES "${_build_type_name}")
  endif()
else()
  list(APPEND _allowed_build_types "${_build_type_name}")
endif()

set(_asan_common_flags "-fsanitize=address")
set(_asan_compiler_flags "${_asan_common_flags} -fno-omit-frame-pointer")
set(_asan_linker_flags "${_asan_common_flags}")

if(NOT DEFINED ASAN_INHERIT_TYPE)
  set(ASAN_INHERIT_TYPE
      RELWITHDEBINFO
      CACHE STRING "Build type that the Asan build type inherit the flags from")
  mark_as_advanced(ASAN_INHERIT_TYPE)
endif()
string(TOUPPER "${ASAN_INHERIT_TYPE}" ASAN_INHERIT_TYPE)

set(CMAKE_C_FLAGS_${_build_type}
    "${CMAKE_C_FLAGS_${ASAN_INHERIT_TYPE}} ${_asan_compiler_flags}"
    CACHE STRING "Flags used by the C compiler for Asan build type or configuration." FORCE)

set(CMAKE_CXX_FLAGS_${_build_type}
    "${CMAKE_CXX_FLAGS_${ASAN_INHERIT_TYPE}} ${_asan_compiler_flags}"
    CACHE STRING "Flags used by the C++ compiler for Asan build type or configuration." FORCE)

set(CMAKE_EXE_LINKER_FLAGS_${_build_type}
    "${CMAKE_SHARED_LINKER_FLAGS_${ASAN_INHERIT_TYPE}} ${_asan_linker_flags}"
    CACHE STRING "Linker flags to be used to create executables for Asan build type." FORCE)

set(CMAKE_SHARED_LINKER_FLAGS_${_build_type}
    "${CMAKE_SHARED_LINKER_FLAGS_${ASAN_INHERIT_TYPE}} ${_asan_linker_flags}"
    CACHE STRING "Linker lags to be used to create shared libraries for Asan build type." FORCE)

unset(_asan_common_flags)
unset(_asan_compiler_flags)
unset(_asan_linker_flags)

# ==============================================================================

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${_allowed_build_types}")

# ==============================================================================

if("${CMAKE_BUILD_TYPE}" STREQUAL "AsanAddr")
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

unset(_is_multi_config)
