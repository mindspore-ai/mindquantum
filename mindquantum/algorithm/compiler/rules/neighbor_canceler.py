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
"""Neighbor cancler compiler rule."""

from mindquantum.algorithm.compiler.dag import DAGCircuit, connect_two_node
from mindquantum.algorithm.compiler.dag.dag import GateNode, QubitNode
from mindquantum.algorithm.compiler.rules.basic_rule import BasicCompilerRule
from mindquantum.utils.type_value_check import _check_input_type

# pylint: disable=invalid-name


def mergeable_params_gate(gn1, gn2):
    """Merge two parameterized gate."""
    g1, g2 = gn1.gate, gn2.gate
    if isinstance(g1, g2.__class__):
        if g1.parameterized and g2.parameterized:
            if g1.obj_qubits == g2.obj_qubits and set(g1.ctrl_qubits) == set(g2.ctrl_qubits):
                g = g1(g1.coeff + g2.coeff)
                return GateNode(g)
    return False


# pylint: disable=too-many-nested-blocks,too-many-branches,too-few-public-methods
class NeighborCancler(BasicCompilerRule):
    """Merge two nearby gate if possible."""

    def __init__(self, merge_parameterized_gate=True):
        """Initialize a neighbor cancler compiler rule."""
        super().__init__("NeighborCancler")
        self.merge_parameterized_gate = merge_parameterized_gate

    def do(self, dagcircuit: DAGCircuit):
        """Apply neighbor cancler compiler rule."""
        _check_input_type("dagcircuit", DAGCircuit, dagcircuit)

        def _cancler(current_node, fc_pair_consided):
            """Merge two gate."""
            for local in current_node.local:
                if current_node.child:
                    child_node = current_node.child[local]
                    fc_pair = (current_node, child_node)
                    if fc_pair not in fc_pair_consided:
                        fc_pair_consided.add(fc_pair)
                        if not isinstance(child_node, QubitNode):
                            if not isinstance(current_node, QubitNode):
                                if current_node.gate == child_node.gate.hermitian():
                                    if len(set(current_node.child.values())) == 1:
                                        for lo in current_node.local:
                                            connect_two_node(current_node.father[lo], child_node.child[lo], lo)
                                        _cancler(child_node.child[local], fc_pair_consided)
                                if self.merge_parameterized_gate:
                                    merged_node = mergeable_params_gate(current_node, child_node)
                                    if merged_node:
                                        for lo in current_node.local:
                                            connect_two_node(current_node.father[lo], merged_node, lo)
                                            connect_two_node(merged_node, child_node.child[lo], lo)
                                        _cancler(merged_node, fc_pair_consided)
                            _cancler(child_node, fc_pair_consided)
                if current_node.father:
                    father_node = current_node.father[local]
                    fc_pair = (father_node, current_node)
                    if fc_pair not in fc_pair_consided:
                        fc_pair_consided.add(fc_pair)
                        if not isinstance(father_node, QubitNode):
                            if not isinstance(current_node, QubitNode):
                                if current_node.gate == father_node.gate.hermitian():
                                    if len(set(current_node.father.values())) == 1:
                                        for lo in current_node.local:
                                            connect_two_node(father_node.father[lo], current_node.child[lo], lo)
                                        _cancler(father_node.father[local], fc_pair_consided)
                                if self.merge_parameterized_gate:
                                    merged_node = mergeable_params_gate(current_node, father_node)
                                    if merged_node:
                                        for lo in current_node.local:
                                            connect_two_node(father_node.father[lo], merged_node, lo)
                                            connect_two_node(merged_node, current_node.child[lo], lo)
                                        _cancler(merged_node, fc_pair_consided)
                            _cancler(father_node, fc_pair_consided)

        fc_pair_consided = set()
        for current_node in dagcircuit.head_node.values():
            _cancler(current_node, fc_pair_consided)
