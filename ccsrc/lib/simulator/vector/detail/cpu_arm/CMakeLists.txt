# ==============================================================================
#
# Copyright 2026 <Huawei Technologies Co., Ltd>
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

include(CheckCXXSourceCompiles)

target_sources(mqsim_vector_cpu PRIVATE ${CMAKE_CURRENT_LIST_DIR}/cpu_vector_core_dispatch.cpp)

set(_mq_build_arm_sve OFF)
set(_mq_arm_sve_compile_option "-march=armv8-a+sve")
if(MQ_VECTOR_ARM_SVE AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_mq_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_FLAGS "${_mq_saved_required_flags} ${_mq_arm_sve_compile_option}")
  file(READ "${PROJECT_SOURCE_DIR}/cmake/probes/arm_sve_fcma.cpp" _mq_arm_sve_probe)
  check_cxx_source_compiles("${_mq_arm_sve_probe}" MQ_VECTOR_COMPILER_HAS_ARM_SVE_FCMA)
  set(CMAKE_REQUIRED_FLAGS "${_mq_saved_required_flags}")
  unset(_mq_arm_sve_probe)
  unset(_mq_saved_required_flags)

  if(NOT MQ_VECTOR_COMPILER_HAS_ARM_SVE_FCMA)
    message(FATAL_ERROR "MQ_VECTOR_ARM_SVE=ON, but the C++ compiler cannot build the required SVE/FCMA intrinsics.")
  endif()
  set(_mq_build_arm_sve ON)
endif()

if(_mq_build_arm_sve)
  set(_mq_sve_sources ${CMAKE_CURRENT_LIST_DIR}/cpu_vector_core_circuit_tiling.cpp
                      ${CMAKE_CURRENT_LIST_DIR}/cpu_vector_core_matrix_gate.cpp)
  target_sources(mqsim_vector_cpu PRIVATE ${_mq_sve_sources})
  set_source_files_properties(${_mq_sve_sources} TARGET_DIRECTORY mqsim_vector_cpu
                              PROPERTIES COMPILE_OPTIONS "${_mq_arm_sve_compile_option}")
  target_compile_definitions(mqsim_vector_cpu PRIVATE MQ_VECTOR_HAS_ARM_SVE_BACKEND=1)
  message(
    STATUS "mqvector Arm float/double backends: baseline, SVE/FCMA (runtime dispatched; ${_mq_arm_sve_compile_option})")
else()
  target_compile_definitions(mqsim_vector_cpu PRIVATE MQ_VECTOR_HAS_ARM_SVE_BACKEND=0)
  message(STATUS "mqvector Arm float/double backends: baseline")
endif()
