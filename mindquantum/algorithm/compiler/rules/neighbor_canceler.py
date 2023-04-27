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
from mindquantum.algorithm.compiler.dag import DAGCircuit, connect_two_node
from mindquantum.algorithm.compiler.dag.dag import GateNode, QubitNode, DAGNode
from mindquantum.algorithm.compiler.rules.basic_rule import BasicCompilerRule
from mindquantum.utils.type_value_check import _check_input_type
from mindquantum.algorithm.compiler.rules.compiler_logger import CompileLog as CLog

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


def fu(node):
    out = ""
    for k, v in node.items():
        out += f"{k}, {v}, {id(v)}"
    return out


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

        def _cancler(current_node: DAGNode, fc_pair_consided: typing.List[typing.Tuple[GateNode, GateNode]]):
            compiled = False
            while True:
                if not current_node.local or not current_node.child:
                    break

        def _cancler1(current_node, fc_pair_consided):
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
                                if current_node.gate == child_node.gate.hermitian():
                                    CLog.IncreaceHeadBlock()
                                    CLog.log(
                                        f"{CLog.R1(self.rule_name)}: del {CLog.B(current_node.gate)} and {CLog.B(child_node.gate)}.",
                                        2, self.log_level)
                                    CLog.DecreaseHeadBlock()
                                    if len(set(current_node.child.values())) == 1:
                                        compiled = True
                                        for lo in current_node.local:
                                            connect_two_node(current_node.father[lo], child_node.child[lo], lo)
                                            next_consider = child_node.child[lo]
                                            if lo in current_node.father:
                                                current_node.father.pop(lo)
                                            if lo in child_node.child:
                                                child_node.child.pop(lo)
                                            # compiled = compiled or
                                            _cancler(next_consider, fc_pair_consided)
                                if self.merge_parameterized_gate:
                                    merged_node = mergeable_params_gate(current_node, child_node)
                                    if merged_node:
                                        compiled = True
                                        CLog.IncreaceHeadBlock()
                                        CLog.log(
                                            f"{CLog.R1(self.rule_name)}: merge {CLog.B(current_node.gate)} and {CLog.B(child_node.gate)}.",
                                            2, self.log_level)
                                        CLog.DecreaseHeadBlock()
                                        for lo in current_node.local:
                                            connect_two_node(current_node.father[lo], merged_node, lo)
                                            connect_two_node(merged_node, child_node.child[lo], lo)
                                        compiled = compiled or _cancler(merged_node, fc_pair_consided)
                            compiled = compiled or _cancler(child_node, fc_pair_consided)
                if current_node.father:
                    father_node = current_node.father[local]
                    fc_pair = (father_node, current_node)
                    if fc_pair not in fc_pair_consided:
                        fc_pair_consided.add(fc_pair)
                        if not isinstance(father_node, QubitNode):
                            if not isinstance(current_node, QubitNode):
                                if current_node.gate == father_node.gate.hermitian():
                                    CLog.IncreaceHeadBlock()
                                    CLog.log(
                                        f"{CLog.R1(self.rule_name)}: del {CLog.B(father_node.gate)} and {CLog.B(current_node.gate)}.",
                                        2, self.log_level)
                                    CLog.DecreaseHeadBlock()
                                    if len(set(current_node.father.values())) == 1:
                                        compiled = True
                                        for lo in current_node.local:
                                            connect_two_node(father_node.father[lo], current_node.child[lo], lo)
                                            next_consider = father_node.father[lo]
                                            if lo in father_node.father:
                                                father_node.father.pop(lo)
                                            if lo in current_node.child:
                                                current_node.child.pop(lo)
                                            compiled = compiled or _cancler(next_consider, fc_pair_consided)
                                if self.merge_parameterized_gate:
                                    merged_node = mergeable_params_gate(current_node, father_node)
                                    if merged_node:
                                        compiled = True
                                        CLog.IncreaceHeadBlock()
                                        CLog.log(
                                            f"{CLog.R1(self.rule_name)}: merge {CLog.B(current_node.gate)} and {CLog.B(child_node.gate)}.",
                                            2, self.log_level)
                                        CLog.DecreaseHeadBlock()
                                        for lo in current_node.local:
                                            connect_two_node(father_node.father[lo], merged_node, lo)
                                            connect_two_node(merged_node, current_node.child[lo], lo)
                                        compiled = compiled or _cancler(merged_node, fc_pair_consided)
                            compiled = compiled or _cancler(father_node, fc_pair_consided)
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
            CLog.log(f"{CLog.R1(self.rule_name)}: nothing happend.", 1, self.log_level)
        return compiled


if __name__ == "__main__":
    from mindquantum import *
    from mindquantum.algorithm.compiler import DAGCircuit
    circ = Circuit().h(0).x(1).x(1).h(0).h(0).h(0).x(0, 1).x(0).x(0).x(0)
    dag = DAGCircuit(circ)
    state = NeighborCancler().set_log_level(2).do(dag)
    new_circ = dag.to_circuit()

    import numpy as np
    from mindquantum import *
    from mindquantum.algorithm.compiler import *
    from mindquantum.algorithm.compiler.decompose.utils import *
    # circ = qft(range(3))
    np.random.seed(4133237)

    for i in range(3):
        circ = random_circuit(4, 40, ctrl_rate=0.5) + X.on(0) + X.on(0)

    dag_circ = DAGCircuit(circ + X.on(0) + X.on(0))
    print("origin depth: ", dag_circ.depth())
    c = SequentialCompiler([
        BasicDecompose(True),
        GateReplacer(SWAP.on([0, 1]),
                    Circuit().x(0, 1).x(1, 0).x(0, 1)),
        CXToCZ(),
        # KroneckerSeqCompiler([
        #     NeighborCancler(),
        # ]),
    ])
    c.do(dag_circ)
    circ = dag_circ.to_circuit()[-41:]
    circ = circ[:10] + circ[-10:]
    dag_circ = DAGCircuit(circ)
    # c = KroneckerSeqCompiler([NeighborCancler()]).set_all_log_level(2)
    c = NeighborCancler()
    c.do(dag_circ)
    # print("after compile depth: ", dag_circ.depth())

    new_circ = dag_circ.to_circuit()

    m1 = circ.matrix()
    m2 = new_circ.matrix()

    assert is_equiv_unitary(m1, m2)
    print("depth: ", dag_circ.depth())