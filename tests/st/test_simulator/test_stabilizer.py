# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Test stabilizer simulator."""
import numpy as np
from scipy.stats import entropy

from mindquantum import _mq_vector
from mindquantum.algorithm.error_mitigation import (
    query_double_qubits_clifford_elem,
    query_single_qubit_clifford_elem,
)
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import (
    Simulator,
    decompose_stabilizer,
    get_stabilizer_string,
    get_tableau_string,
)
from mindquantum.utils import random_clifford_circuit


def test_stabilizer():
    """
    Test stabilizer simulator.
    Description: Test stabilizer simulator.
    Expectation:
    """
    _mq_vector.stabilizer.verify()  # pylint: disable=no-member

    for i in range(11520):
        clifford = query_double_qubits_clifford_elem(i)
        circ = decompose_stabilizer(clifford)
        sim = Simulator('stabilizer', 2)
        sim.apply_circuit(circ)
        assert clifford.backend == sim.backend

    for i in range(24):
        clifford = query_single_qubit_clifford_elem(i)
        circ = decompose_stabilizer(clifford)
        sim = Simulator('stabilizer', 1)
        sim.apply_circuit(circ)
        assert clifford.backend == sim.backend


def test_stabilizer_tableau():
    """
    Description: Test tableau string of stabilizer.
    Expectation:
    """
    sim = Simulator('stabilizer', 2)
    sim.apply_circuit(Circuit().h(0).x(1, 0))
    tableau = get_tableau_string(sim)
    tableau_exp = '0 0 | 1 0 | 0\n0 1 | 0 0 | 0\n-------------\n1 1 | 0 0 | 0\n0 0 | 1 1 | 0\n'
    assert tableau == tableau_exp


def test_stabilizer_string():
    """
    Description: Test stabilizer string of stabilizer.
    Expectation:
    """
    sim = Simulator('stabilizer', 2)
    sim.apply_circuit(Circuit().h(0).x(1, 0))
    stabilizer = get_stabilizer_string(sim)
    stabilizer_exp = 'destabilizer:\n+IZ\n+XI\nstabilizer:\n+XX\n+ZZ'
    assert stabilizer == stabilizer_exp


def test_stabilizer_sampling():
    """
    Description: Test sampling of stabilizer simulator.
    Expectation:
    """
    for _ in range(50):
        n_qubits = np.random.randint(4, 6)
        length = np.random.randint(40, 60)
        clifford = random_clifford_circuit(n_qubits, length).measure_all()
        sim = Simulator('stabilizer', clifford.n_qubits)
        res_clifford = sim.sampling(clifford, shots=10000)
        res_state_vector = Simulator('mqvector', clifford.n_qubits).sampling(clifford, shots=10000)

        keys = set(res_clifford.data.keys()) | set(res_state_vector.data.keys())
        dis1 = np.array([res_clifford.data.get(key, 0) for key in keys]) / res_clifford.shots
        dis2 = np.array([res_state_vector.data.get(key, 0) for key in keys]) / res_state_vector.shots
        assert entropy(dis1, dis2) < 0.01
