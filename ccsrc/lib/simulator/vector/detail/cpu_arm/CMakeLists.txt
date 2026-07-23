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
  check_cxx_source_compiles(
    "#include <arm_sve.h>
     #if !defined(__ARM_FEATURE_SVE)
     #error SVE is unavailable
     #endif
     int main() {
       const auto pg32 = svptrue_b32();
       const auto value32 = svdup_n_f32(1.0F);
       const auto result32 = svcmla_f32_x(pg32, value32, value32, value32, 0);
       const auto pg64 = svptrue_b64();
       const auto value64 = svdup_n_f64(1.0);
       const auto result64 = svcmla_f64_x(pg64, value64, value64, value64, 0);
       return static_cast<int>(svlastb_f32(pg32, result32) + svlastb_f64(pg64, result64));
     }"
    MQ_VECTOR_COMPILER_HAS_ARM_SVE_FCMA)
  set(CMAKE_REQUIRED_FLAGS "${_mq_saved_required_flags}")
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
