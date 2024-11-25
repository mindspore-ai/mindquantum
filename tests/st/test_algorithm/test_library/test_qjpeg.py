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
"""test for qjpeg algorithm"""

import numpy as np

from mindquantum.simulator import Simulator
from mindquantum.utils import normalize
from mindquantum.algorithm.library import qjpeg


def test_qjpeg():
    """
    Description: Test the QJPEG algorithm.
    Expectation:
    """
    n_qubits = 4
    m_qubits = 2
    circ, remainder_qubits, discard_qubits = qjpeg(n_qubits, m_qubits)
    assert remainder_qubits == [0, 2]
    assert discard_qubits == [1, 3]

    data = np.array([[1, 0, 0, 0],
                     [1, 1, 0, 0],
                     [1, 1, 1, 0],
                     [1, 1, 1, 1]])
    state = normalize(data.reshape(-1))
    sim = Simulator('mqmatrix', n_qubits)
    sim.set_qs(state)
    sim.apply_circuit(circ)
    rho = sim.get_partial_trace(discard_qubits)
    sub_probs = rho.diagonal().real
    new_data = sub_probs.reshape((2 ** (m_qubits // 2), -1))
    assert np.allclose(new_data, np.array([[0.3, 0.0],
                                           [0.4, 0.3]]))
