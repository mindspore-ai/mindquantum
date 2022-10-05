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

add_test_executable(test_atom_storage LIBS mq_cxx_nextgen DEFINES UNIT_TESTS)
add_test_executable(test_apply_atoms LIBS mq_cxx_nextgen mindquantum::symengine DEFINES UNIT_TESTS)
add_test_executable(test_non_gate_decomposition_rules LIBS mq_cxx_nextgen DEFINES UNIT_TESTS)
add_test_executable(test_gate_decomposer LIBS mq_cxx_nextgen mindquantum::symengine DEFINES UNIT_TESTS)
add_test_executable(test_decomposition_rules LIBS mq_cxx_nextgen mindquantum::symengine DEFINES UNIT_TESTS)
