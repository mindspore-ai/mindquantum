# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test for SABRE and MQSABRE mapping algorithms."""

import pytest

from mindquantum.algorithm.mapping import MQSABRE, SABRE
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, X
from mindquantum.device import GridQubits, LinearQubits, QubitNode, QubitsTopology


def test_sabre_basic():
    """
    Description: Test basic SABRE functionality.
    Expectation: Runs without error, returns correct types.
    """
    circ = Circuit([H(0), X(1, 0), X(2, 1)])
    topo = LinearQubits(3)
    solver = SABRE(circ, topo)
    new_circ, init_map, final_map = solver.solve()

    assert isinstance(new_circ, Circuit)
    assert isinstance(init_map, list)
    assert isinstance(final_map, list)
    assert len(init_map) == 3
    assert len(final_map) == 3
    assert len(new_circ) >= len(circ)


def test_sabre_non_contiguous_qubits():
    """
    Description: Test SABRE with non-contiguous qubit IDs. This should pass due to the fix.
    Expectation: Runs without crashing and maps to correct physical qubits.
    """
    q12 = QubitNode(12, poi_x=0, poi_y=0)
    q13 = QubitNode(13, poi_x=1, poi_y=0)
    q14 = QubitNode(14, poi_x=0, poi_y=1)
    q15 = QubitNode(15, poi_x=1, poi_y=1)

    topology = QubitsTopology([q12, q13, q14, q15])
    couplers = [[12, 13], [12, 14], [14, 15], [13, 15]]
    for pair in couplers:
        topology[pair[0]] >> topology[pair[1]]

    circ = Circuit().rx(1.23, 0).rx(2.13, 1).rx(3.12, 2).x(1, 0).x(2, 1).x(0, 2)
    solver = SABRE(circ, topology)
    new_circ, init_mapping, final_mapping = solver.solve(5, 0.5, 0.3, 0.2)

    assert isinstance(new_circ, Circuit)
    physical_qubits = {12, 13, 14, 15}
    for gate in new_circ:
        all_gate_qubits = set(gate.obj_qubits) | set(gate.ctrl_qubits)
        assert all_gate_qubits.issubset(physical_qubits)


def test_sabre_disconnected_topology():
    """
    Description: Test SABRE with a disconnected topology.
    Expectation: Raises ValueError.
    """
    q0 = QubitNode(0)
    q1 = QubitNode(1)
    q2 = QubitNode(2)
    q3 = QubitNode(3)
    q0 >> q1  # component 1
    q2 >> q3  # component 2
    topology = QubitsTopology([q0, q1, q2, q3])
    circ = Circuit().x(1, 0).x(3, 2)
    with pytest.raises(ValueError, match="SABRE only supports connected graphs"):
        SABRE(circ, topology)


def test_sabre_insufficient_physical_qubits():
    """
    Description: Test SABRE when logical qubits are more than physical qubits.
    Expectation: Raises RuntimeError because the C++ core will throw an exception.
    """
    circ = Circuit().x(1, 0).x(3, 2)  # Needs 4 logical qubits (0, 1, 2, 3)
    topology = LinearQubits(3)  # Only 3 physical qubits
    with pytest.raises(
        RuntimeError, match="The number of logical qubits .* cannot be greater than the number of physical qubits"
    ):
        SABRE(circ, topology)


def test_mqsabre_basic():
    """
    Description: Test basic MQSABRE functionality.
    Expectation: Runs without error and returns correct types.
    """
    circ = Circuit([H(0), X(1, 0), X(2, 1), X(3, 2)])
    topology = GridQubits(2, 2)
    cnot_data = [
        ((0, 1), [0.001, 250.0]),
        ((1, 0), [0.001, 250.0]),
        ((0, 2), [0.002, 300.0]),
        ((2, 0), [0.002, 300.0]),
        ((1, 3), [0.001, 250.0]),
        ((3, 1), [0.001, 250.0]),
        ((2, 3), [0.002, 300.0]),
        ((3, 2), [0.002, 300.0]),
    ]
    solver = MQSABRE(circ, topology, cnot_data)
    new_circ, init_map, final_map = solver.solve()

    assert isinstance(new_circ, Circuit)
    assert isinstance(init_map, list)
    assert isinstance(final_map, list)
    assert len(init_map) == 4
    assert len(final_map) >= 4  # Final map can be larger if idle qubits are mapped
    assert len(new_circ) >= len(circ)


def test_mqsabre_disconnected_topology():
    """
    Description: Test MQSABRE with a disconnected topology.
    Expectation: Raises ValueError.
    """
    q0 = QubitNode(0)
    q1 = QubitNode(1)
    q2 = QubitNode(2)
    q3 = QubitNode(3)
    q0 >> q1
    q2 >> q3
    topology = QubitsTopology([q0, q1, q2, q3])
    circ = Circuit().x(1, 0).x(3, 2)
    cnot_data = [((0, 1), [0.01, 100]), ((1, 0), [0.01, 100]), ((2, 3), [0.01, 100]), ((3, 2), [0.01, 100])]
    with pytest.raises(ValueError, match="MQSABRE only supports connected graphs"):
        MQSABRE(circ, topology, cnot_data)


def test_mqsabre_non_contiguous_qubits_fixed():
    """
    Description: Test MQSABRE with non-contiguous qubit IDs. This should pass after the fix.
    Expectation: Runs without crashing and maps to correct physical qubits.
    """
    q_map = {0: 10, 1: 12, 2: 14, 3: 16}

    nodes = [QubitNode(qid) for qid in q_map.values()]
    topology = QubitsTopology(nodes)
    topology[q_map[0]] >> topology[q_map[1]]
    topology[q_map[1]] >> topology[q_map[3]]
    topology[q_map[0]] >> topology[q_map[2]]
    topology[q_map[2]] >> topology[q_map[3]]

    circ = Circuit().x(1, 0).x(2, 1).x(3, 2)

    cnot_data = [
        ((q_map[0], q_map[1]), [0.001, 250.0]),
        ((q_map[1], q_map[0]), [0.001, 250.0]),
        ((q_map[1], q_map[3]), [0.002, 300.0]),
        ((q_map[3], q_map[1]), [0.002, 300.0]),
        ((q_map[0], q_map[2]), [0.003, 350.0]),
        ((q_map[2], q_map[0]), [0.003, 350.0]),
        ((q_map[2], q_map[3]), [0.004, 400.0]),
        ((q_map[3], q_map[2]), [0.004, 400.0]),
    ]

    # With the ID compression fix, this should now pass without raising an error.
    solver = MQSABRE(circ, topology, cnot_data)
    new_circ, init_map, final_map = solver.solve()

    assert isinstance(new_circ, Circuit)
    physical_qubits = set(q_map.values())
    for gate in new_circ:
        all_gate_qubits = set(gate.obj_qubits) | set(gate.ctrl_qubits)
        assert all_gate_qubits.issubset(physical_qubits)
