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

# lint_cmake: -whitespace/indent

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/cpu_common)

# One backend directory per architecture. The matching MQ_VECTOR_ARCH_* definition is a usage requirement of the
# library, so every consumer resolves the policy types through simulator/vector/detail/cpu_vector_arch_policy.h.
if(X86_64)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/cpu_avx_double)
  target_compile_definitions(mqsim_vector_cpu PUBLIC MQ_VECTOR_ARCH_X86_64)
elseif(AARCH64)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/cpu_arm)
  target_compile_definitions(mqsim_vector_cpu PUBLIC MQ_VECTOR_ARCH_ARM64)
else()
  message(FATAL_ERROR "No CPU state-vector backend for target processor '${CMAKE_SYSTEM_PROCESSOR}'")
endif()

if(ENABLE_CUDA)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/gpu)
endif()
