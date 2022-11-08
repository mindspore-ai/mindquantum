# ==============================================================================
#
# Copyright 2021 <Huawei Technologies Co., Ltd>
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

if(ENABLE_CXX_EXPERIMENTAL)
  add_test_executable(test_parametric_gates LIBS mq_cxx_nextgen mindquantum::symengine)
  add_test_executable(test_angle_parametric_gates LIBS mq_cxx_nextgen mindquantum::symengine)
  add_test_executable(test_register_gate LIBS mq_cxx_nextgen mindquantum::symengine)
endif()

add_test_executable(test_terms_operators LIBS mq_base)
add_test_executable(test_qubit_operators LIBS mq_base DEFINES UNIT_TESTS)
add_test_executable(test_fermion_operators LIBS mq_base DEFINES UNIT_TESTS)
