# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test Hamiltonian."""

from mindquantum.core.operators import Hamiltonian, QubitOperator


def test_hamiltonian():
    """
    Description: Test Hamiltonian
    Expectation:
    """
    ham = Hamiltonian(QubitOperator('Z0 Y1', 0.3))
    assert ham.ham_termlist == [(((0, 'Z'), (1, 'Y')), 0.3)]
