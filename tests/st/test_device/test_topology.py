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
"""Test topology of device."""
import pytest

from mindquantum.device import GridQubits, LinearQubits, QubitNode, QubitsTopology


def test_qubit_node_property():
    """
    Description: Test qubit node
    Expectation: success
    """
    qubit = QubitNode(1)
    qubit.set_poi(1.0, 2.0)
    qubit.set_color("#ababab")
    assert qubit.qubit_id == 1
    assert qubit.color == '#ababab'
    assert qubit.poi_x == 1.0
    assert qubit.poi_y == 2.0


def test_qubit_connection():
    """
    Description: Test qubit node connection
    Expectation: success
    """
    q0 = QubitNode(0)
    q1 = QubitNode(1)
    q2 = QubitNode(2)
    assert q0 == (q0 << q1)
    assert q1.qubit_id in q0.neighbor
    assert q0 == (q0 < q1)
    assert q1.qubit_id not in q0.neighbor
    assert q2 == (q1 >> q2)
    assert q2.qubit_id in q1.neighbor
    assert q2 == (q1 > q2)
    assert q2.qubit_id not in q1.neighbor


def test_topology():
    """
    Description: Test topology
    Expectation: success
    """
    topology = QubitsTopology([QubitNode(i, poi_x=i, poi_y=i) for i in range(4)])
    topology.add_qubit_node(QubitNode(4))
    assert topology.size() == 5
    assert topology.all_qubit_id() == set(range(5))
    _ = topology[0] >> topology[1] >> topology[2] >> topology[3] >> topology[4]
    topology.remove_qubit_node(4)
    assert topology.size() == 4
    assert not topology.has_qubit_node(4)
    assert topology.edges_with_id() == {(0, 1), (1, 2), (2, 3)}
    _ = topology[3] > topology[2]
    assert not topology.is_coupled_with(2, 3)
    topology.isolate_with_near(0)
    assert topology.n_edges() == 1
    topology.remove_isolate_node()
    assert topology.size() == 2
    topology.set_color(1, '#ababab')
    assert topology[1].color == '#ababab'
    topology.set_position(2, 3.0, 4.0)
    assert topology[2].poi_x + topology[2].poi_y == 7.0
    nodes = topology.choose([1, 2])
    assert nodes[1].qubit_id == 2
    assert topology.edges_with_poi() == {((1, 1), (3.0, 4.0))}
    topology_new = topology.select([1, 2])
    assert topology_new.edges_with_id() == {(1, 2)}
    topology = LinearQubits(4)
    topology.remove_qubit_node(0)
    topology, m = topology.compress()
    assert topology.all_qubit_id() == {0, 1, 2}
    assert m == {1: 0, 2: 1, 3: 2}


def test_linear_qubits():
    """
    Description: Test linear qubits topology
    Expectation: success
    """
    topology = LinearQubits(4)
    assert topology.size() == 4
    topology.isolate_with_near(2)
    topology.remove_isolate_node()
    assert topology.size() == 2


def test_grid_qubits():
    """
    Description: Test grid qubits topology
    Expectation: success
    """
    topology = GridQubits(4, 5)
    assert topology.size() == 20
    assert topology.n_row() == 4
    assert topology.n_col() == 5
    assert topology.n_edges() == 31
    topology.isolate_with_near(2)
    topology.remove_isolate_node()
    assert topology.size() == 19
    assert topology.n_edges() == 28


def test_set_edge_color():
    """
    Description: Test set color of topology edge.
    Expectation: success
    """
    topology = LinearQubits(4)
    _ = topology[1] > topology[2]
    with pytest.raises(ValueError, match="qubit 1 is not connected with qubit 2"):
        topology.set_edge_color(1, 2, '#ababab')
    topology.set_edge_color(0, 1, '#121212')
    assert topology.get_edge_color(1, 0) == '#121212'
