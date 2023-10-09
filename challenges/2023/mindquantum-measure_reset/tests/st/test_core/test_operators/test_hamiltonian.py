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
import numpy as np
import pytest

from mindquantum.core.operators import Hamiltonian, QubitOperator


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_hamiltonian():
    """
    Description: Test Hamiltonian
    Expectation:
    """
    ham = Hamiltonian(QubitOperator('Z0 Y1', 0.3))
    terms = ham.ham_termlist
    paulis = terms[0][0]
    coeff = terms[0][1]
    assert paulis == ((0, 'Z'), (1, 'Y'))
    assert np.allclose(coeff, 0.3, atol=1e-6)
