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
"""DAG Circuit."""
import typing
import numpy as np

from mindquantum.core import Circuit, gates
from mindquantum.utils.type_value_check import _check_input_type

# pylint: disable=invalid-name


class DAGNode:
    """Node object in Directed Acyclic Graph."""

    def __init__(self):
        """Initialize a DAGNode object."""
        self.child: typing.Dict[int, "DAGNode"] = {}  # key: local index, value: child DAGNode
        self.father: typing.Dict[int, "DAGNode"] = {}  # key: local index, value: father DAGNode
        self.local: typing.List[int] = []

    def insert_after(self, other_node: "DAGNode"):
        """Insert other node after this dag node."""
        _check_input_type("other_node", DAGNode, other_node)
        for local in self.local:
            if local in other_node.local:
                other_node.father[local] = self
                if local in self.child:
                    other_node.child[local] = self.child.get(local)
                    self.child.get(local).fathre[local] = other_node
                self.child[local] = other_node

    def insert_before(self, other_node: "DAGNode"):
        """Insert other node before this dag node."""
        _check_input_type("other_node", DAGNode, other_node)
        for local in self.local:
            if local in other_node.local:
                other_node.child[local] = self
                if local in self.father:
                    other_node.father[local] = self.father.get(local)
                    self.father.get(local).child[local] = other_node
                self.father[local] = other_node

    def clean(self):
        self.child = {}
        self.father = {}
        self.local = []


def connect_two_node(father_node: DAGNode, child_node: DAGNode, local_index: int):
    """Connect two dag node."""
    father_node.child[local_index] = child_node
    child_node.father[local_index] = father_node


class QubitNode(DAGNode):
    """DAG node that work as qubit."""

    def __init__(self, qubit: int):
        """Initialize a QubitNode object."""
        super().__init__()
        _check_input_type("qubit", int, qubit)
        self.qubit = qubit
        self.local = [qubit]

    def __str__(self):
        """Return a string representation of qubit node."""
        return f"q{self.qubit}"

    def __repr__(self):
        """Return a string representation of qubit node."""
        return self.__str__()


class GateNode(DAGNode):
    """DAG node that work as quantum gate."""

    def __init__(self, gate: gates.BasicGate):
        """Initialize a GateNode object."""
        super().__init__()
        _check_input_type("gate", gates.BasicGate, gate)
        self.gate = gate
        self.local = gate.obj_qubits + gate.ctrl_qubits

    def __str__(self):
        """Return a string representation of gate node."""
        return str(self.gate)

    def __repr__(self):
        """Return a string representation of gate node."""
        return self.__str__()


class BarrierNode(GateNode):
    """DAG node that work as barrier."""

    def __init__(self, gate, all_qubits):
        """Initialize a BarrierNode object."""
        super().__init__(gate)
        self.local = all_qubits


class DAGCircuit:
    """A Directed Acyclic Graph of a quantum circuit."""

    def __init__(self, circuit: Circuit):
        """Initialize a DAGCircuit object."""
        _check_input_type("circuit", Circuit, circuit)
        self.head_node = {i: QubitNode(i) for i in sorted(circuit.all_qubits.keys())}
        self.final_node = {i: QubitNode(i) for i in sorted(circuit.all_qubits.keys())}
        for i in self.head_node:
            self.head_node[i].insert_after(self.final_node[i])
        for gate in circuit:
            if isinstance(gate, gates.BarrierGate):
                self.append_node(BarrierNode(gate, sorted(circuit.all_qubits.keys())))
            else:
                self.append_node(GateNode(gate))

    @staticmethod
    def replace_node_with_dagcircuit(node: DAGNode, coming: "DAGCircuit"):
        """Replace a node with a DAGCircuit."""
        for local in node.local:
            connect_two_node(node.father[local], coming.head_node[local].child[local], local)
            connect_two_node(coming.final_node[local].father[local], node.child[local], local)

    def append_node(self, node: DAGNode):
        """Append a quantum gate node."""
        _check_input_type('node', DAGNode, node)
        for local in node.local:
            self.final_node[local].insert_before(node)

    def layerize(self):
        """Layerize the quantum circuit."""

        def _layerize(current_node: GateNode, depth_map):
            """Layerize the quantum circuit."""
            if current_node.father:
                prev_depth = []
                for father_node in current_node.father.values():
                    if father_node not in depth_map:
                        _layerize(father_node, depth_map)
                    prev_depth.append(depth_map[father_node])
                depth_map[current_node] = max(prev_depth) + 1
            for child in current_node.child.values():
                if not isinstance(child, QubitNode):
                    if child not in depth_map:
                        _layerize(child, depth_map)

        depth_map = {i: 0 for i in self.head_node.values()}
        for current_node in self.head_node.values():
            _layerize(current_node, depth_map)
        layer = [Circuit() for _ in range(len(set(depth_map.values())) - 1)]
        for k, v in depth_map.items():
            if v != 0:
                if not isinstance(k, BarrierNode):
                    layer[v - 1] += k.gate
        return [c for c in layer if len(c) != 0]

    def depth(self):
        """Return the depth of quantum circuit."""
        return len(self.layerize())

    def to_circuit(self):
        """Convert DAGCircuit to quantum circuit."""
        circuit = Circuit()
        consided = set(self.head_node.values())

        def adding_current_node(current_node, circuit, consided):
            if all(i in consided for i in current_node.father.values()) and not isinstance(current_node, QubitNode):
                circuit += current_node.gate
                consided.add(current_node)
            else:
                for node in current_node.father.values():
                    if node not in consided:
                        adding_current_node(node, circuit, consided)
                for node in current_node.child.values():
                    if node not in consided:
                        adding_current_node(node, circuit, consided)

        for current_node in self.final_node.values():
            adding_current_node(current_node, circuit, consided)
        return circuit

    def find_all_gate_node(self) -> typing.List[GateNode]:
        """Find all gate node in this DAG."""
        found = set(self.head_node.values())

        def _find(current_node: DAGNode, found):
            if current_node not in found:
                found.add(current_node)
                for node in current_node.father.values():
                    _find(node, found)
                for node in current_node.child.values():
                    _find(node, found)

        for head_node in self.head_node.values():
            for current_node in head_node.child.values():
                _find(current_node, found)
        return [i for i in found if not isinstance(i, QubitNode)]

    def to_tensor_network(self):
        """Convert DAG to hiq tensor style."""
        # pylint: disable=too-many-locals
        tensor = []
        open_lags = [i for i, _ in enumerate(self.head_node)]
        max_lags = len(open_lags)
        all_lags = set(open_lags)
        for layer in self.layerize():
            for gate in layer:
                if isinstance(gate, (gates.Measure, gates.PauliChannel)):
                    raise ValueError("Pauli Channel or Measure gate are not supported.")
                if gate.parameterized:
                    raise ValueError("Parameterized gate are not supported.")
                nc = len(gate.ctrl_qubits)
                no = len(gate.obj_qubits)
                n_total = nc + no
                m = gate.matrix()
                if nc:
                    c0 = np.zeros((2**nc, 2**nc))
                    c0[0, 0] = 1
                    c1 = np.zeros((2**nc, 2**nc))
                    c1[-1, -1] = 1
                    m = np.kron(np.identity(2**nc), c0) + np.kron(m, c1)
                m = np.reshape(m, [2 for _ in range(2 * n_total)])
                gate_lags = [None for _ in range(2 * n_total)]
                for idx, obj in enumerate(gate.obj_qubits + gate.ctrl_qubits):
                    gate_lags[idx] = open_lags[obj]
                    gate_lags[idx + n_total] = open_lags[obj] + n_total + max_lags
                    open_lags[obj] += n_total + max_lags
                    all_lags.add(open_lags[obj])
                    all_lags.add(open_lags[obj] + n_total + max_lags)
                max_lags = max(gate_lags)
                tensor.append([m, gate_lags])
        lag_map = {lag: idx for idx, lag in enumerate(list(all_lags))}
        tensor = [[np.array([1, 0]), [i]] for i, _ in enumerate(self.head_node)] + tensor
        tensor = [[i, [lag_map[k] for k in j]] for i, j in tensor]
        open_lags = [lag_map[i] for i in open_lags]
        return tensor, open_lags


def is_deletable(father_node: GateNode, child_node: GateNode) -> bool:
    if set(father_node.child.values()) != 1 or len(child_node.father.values()) != 1:
        return False
    for node in father_node.child.values():
        if node != child_node:
            return False
    if father_node.gate != child_node.gate.hermitian():
        return False
    return True


def try_delete_node(father_node: GateNode, child_node: GateNode) -> typing.Union[bool, typing.List[GateNode]]:
    if is_deletable(father_node, child_node):
        father_of_father = []
        for i in father_node.local:
            father_of_father.append(father_node.father[i])
            connect_two_node(father_node.father[i], child_node.child[i])
        father_node.clean()
        child_node.clean()
        return True, list(set(father_of_father))
    return False, [father_node]
