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
import typing

from mindquantum.algorithm.compiler.dag import (
    DAGCircuit,
    connect_two_node,
    is_deletable,
    is_mergable,
    try_delete_node,
    try_merge_node,
)
from mindquantum.algorithm.compiler.dag.dag import DAGNode, GateNode, QubitNode
from mindquantum.algorithm.compiler.rules import KroneckerSeqCompiler
from mindquantum.algorithm.compiler.rules.basic_rule import BasicCompilerRule
from mindquantum.algorithm.compiler.rules.compiler_logger import CompileLog as CLog
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
class SimpleNeighborCancler(BasicCompilerRule):
    """Merge two nearby gate if possible."""

    def __init__(self, merge_parameterized_gate=True):
        """Initialize a neighbor cancler compiler rule."""
        super().__init__("SimpleNeighborCancler")
        self.merge_parameterized_gate = merge_parameterized_gate

    def do(self, dagcircuit: DAGCircuit):
        """Apply neighbor cancler compiler rule."""
        _check_input_type("dagcircuit", DAGCircuit, dagcircuit)
        global step
        step = 0

        def _cancler(current_node, fc_pair_consided):
            """Merge two gate."""
            compiled = False
            for local in current_node.local:
                if current_node.child:
                    child_node = current_node.child[local]
                    fc_pair = (current_node, child_node)
                    if fc_pair not in fc_pair_consided:
                        fc_pair_consided.add(fc_pair)
                        if not isinstance(child_node, QubitNode):
                            if not isinstance(current_node, QubitNode):
                                if is_deletable(current_node, child_node):
                                    CLog.IncreaceHeadBlock()
                                    CLog.log(
                                        f"{CLog.R1(self.rule_name)}: del {CLog.B(current_node.gate)} and {CLog.B(child_node.gate)}.",
                                        2,
                                        self.log_level,
                                    )
                                    CLog.DecreaseHeadBlock()
                                    compiled = True
                                    for lo in current_node.local:
                                        connect_two_node(current_node.father[lo], child_node.child[lo], lo)
                                        next_consider = child_node.child[lo]
                                        if lo in current_node.father:
                                            current_node.father.pop(lo)
                                        if lo in child_node.child:
                                            child_node.child.pop(lo)
                                        compiled = _cancler(next_consider, fc_pair_consided) or compiled
                                if self.merge_parameterized_gate:
                                    merged_node = mergeable_params_gate(current_node, child_node)
                                    if merged_node:
                                        compiled = True
                                        CLog.IncreaceHeadBlock()
                                        CLog.log(
                                            f"{CLog.R1(self.rule_name)}: merge {CLog.B(current_node.gate)} and {CLog.B(child_node.gate)}.",
                                            2,
                                            self.log_level,
                                        )
                                        CLog.DecreaseHeadBlock()
                                        for lo in current_node.local:
                                            connect_two_node(current_node.father[lo], merged_node, lo)
                                            connect_two_node(merged_node, child_node.child[lo], lo)
                                        compiled = _cancler(merged_node, fc_pair_consided) or compiled
                            compiled = _cancler(child_node, fc_pair_consided) or compiled
            return compiled

        fc_pair_consided = set()
        compiled = False
        CLog.log(f"Running {CLog.R1(self.rule_name)}.", 1, self.log_level)
        CLog.IncreaceHeadBlock()
        for current_node in dagcircuit.head_node.values():
            compiled = compiled or _cancler(current_node, fc_pair_consided)

        CLog.DecreaseHeadBlock()
        if compiled:
            CLog.log(f"{CLog.R1(self.rule_name)}: {CLog.P('successfule compiled')}.", 1, self.log_level)
        else:
            CLog.log(f"{CLog.R1(self.rule_name)}: nothing happened.", 1, self.log_level)
        return compiled


class FullyNeighborCancler(KroneckerSeqCompiler):
    def __init__(self):
        rule_set = [SimpleNeighborCancler()]
        super().__init__(rule_set)
        self.rule_name = "FullyNeighborCancler"
