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

target_sources(
  mqsim_vector_gpu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/gpu_vector_core_x_like.cu
          ${CMAKE_CURRENT_LIST_DIR}/gpu_vector_core_condition.cu
          ${CMAKE_CURRENT_LIST_DIR}/gpu_vector_core_dot_like.cu
          ${CMAKE_CURRENT_LIST_DIR}/gpu_vector_core_gate_expect.cu
          ${CMAKE_CURRENT_LIST_DIR}/gpu_vector_core_matrix_gate.cu
          ${CMAKE_CURRENT_LIST_DIR}/gpu_vector_core_other_gate.cu
          ${CMAKE_CURRENT_LIST_DIR}/gpu_vector_core_policy.cu
          ${CMAKE_CURRENT_LIST_DIR}/gpu_vector_core_rot_pauli.cu
          ${CMAKE_CURRENT_LIST_DIR}/gpu_vector_core_swap_like.cu
          ${CMAKE_CURRENT_LIST_DIR}/gpu_vector_core_z_like.cu)
