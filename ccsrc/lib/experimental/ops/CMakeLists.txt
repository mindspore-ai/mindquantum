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

target_sources(
  mq_cxx_nextgen
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/gates/details/std_complex_coeff_policy.cpp
          ${CMAKE_CURRENT_LIST_DIR}/gates/details/fermion_operator_term_policy.cpp
          ${CMAKE_CURRENT_LIST_DIR}/gates/details/qubit_operator_term_policy.cpp
          ${CMAKE_CURRENT_LIST_DIR}/gates/details/symengine_coeff_policy.cpp
          ${CMAKE_CURRENT_LIST_DIR}/gates/qubit_operator.cpp
          ${CMAKE_CURRENT_LIST_DIR}/gates/qubit_operator_parameter_resolver.cpp
          ${CMAKE_CURRENT_LIST_DIR}/parametric/register_gate_type.cpp
          ${CMAKE_CURRENT_LIST_DIR}/transform/jordan_wigner.cpp
          ${CMAKE_CURRENT_LIST_DIR}/transform/reverse_jordan_wigner.cpp
          ${CMAKE_CURRENT_LIST_DIR}/transform/parity.cpp
          ${CMAKE_CURRENT_LIST_DIR}/transform/fermion_number_operator.cpp
          ${CMAKE_CURRENT_LIST_DIR}/transform/transform_ladder_operator.cpp)

target_link_libraries(mq_cxx_nextgen PUBLIC mindquantum::symengine)
