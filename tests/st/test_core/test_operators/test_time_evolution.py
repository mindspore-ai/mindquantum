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
"""Test TimeEvolution."""
import pytest

from mindquantum.core import gates as G
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import QubitOperator, TimeEvolution

import numpy as np
from scipy.linalg import expm


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_time_evolution():
    """
    Description: Test TimeEvolution
    Expectation: AssertionError
    """
    hamiltonian = QubitOperator('Z0 Z1', 'p') + QubitOperator('X0', 'q') + QubitOperator('', 'r')
    circ = TimeEvolution(hamiltonian, time=2).circuit
    circ_exp = Circuit(
        [G.X.on(1, 0), G.RZ({'p': 4}).on(1), G.X.on(1, 0), G.RX({'q': 4}).on(0), G.GlobalPhase({'r': 2}).on(0)]
    )
    assert repr(circ) == repr(circ_exp)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_time_evolution_matrix():
    """
    Description: Numeric validation: circuit matrix vs expm(-i H t) for mixed Pauli + constant.
    Expectation: AssertionError
    """
    ham = QubitOperator('X0', 1.23) + QubitOperator('', 0.5)
    t = 2.5
    circ = TimeEvolution(ham, time=t).circuit
    h_mat = ham.matrix().toarray()
    u_mat = expm(-1j * h_mat * t)
    circ_mat = circ.matrix()
    np.testing.assert_allclose(circ_mat, u_mat, atol=1e-7)
