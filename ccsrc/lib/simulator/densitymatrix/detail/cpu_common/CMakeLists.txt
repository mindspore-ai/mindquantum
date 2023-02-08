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
  mqsim_densitymatrix_cpu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/cpu_densitymatrix_core_policy.cpp
          ${CMAKE_CURRENT_LIST_DIR}/cpu_densitymatrix_core_channel.cpp
          ${CMAKE_CURRENT_LIST_DIR}/cpu_densitymatrix_core_condition.cpp
          ${CMAKE_CURRENT_LIST_DIR}/cpu_densitymatrix_core_gate_expect.cpp
          ${CMAKE_CURRENT_LIST_DIR}/cpu_densitymatrix_core_matrix_gate.cpp
          ${CMAKE_CURRENT_LIST_DIR}/cpu_densitymatrix_core_other_gate.cpp
          ${CMAKE_CURRENT_LIST_DIR}/cpu_densitymatrix_core_rot_pauli.cpp
          ${CMAKE_CURRENT_LIST_DIR}/cpu_densitymatrix_core_swap_like.cpp
          ${CMAKE_CURRENT_LIST_DIR}/cpu_densitymatrix_core_utils.cpp
          ${CMAKE_CURRENT_LIST_DIR}/cpu_densitymatrix_core_x_like.cpp
          ${CMAKE_CURRENT_LIST_DIR}/cpu_densitymatrix_core_z_like.cpp)
