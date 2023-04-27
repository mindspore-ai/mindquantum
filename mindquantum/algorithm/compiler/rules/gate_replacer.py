# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Convert cnot to cz."""
from mindquantum.algorithm.compiler.dag import DAGCircuit
from mindquantum.algorithm.compiler.rules import BasicCompilerRule, SequentialCompiler
from mindquantum.algorithm.compiler.rules.compiler_logger import CompileLog as CLog
from mindquantum.core import gates as G
from mindquantum.core.circuit import Circuit, apply


class GateReplacer(BasicCompilerRule):
    def __init__(
        self,
        ori_example_gate: G.BasicGate,
        wanted_example_circ: Circuit,
    ):
        if set(ori_example_gate.obj_qubits + ori_example_gate.ctrl_qubits) != set(
            wanted_example_circ.all_qubits.keys()
        ):
            raise ValueError("Qubit set not equal for given gate and circuit.")
        self.ori_example_gate = ori_example_gate
        self.wanted_example_circ = wanted_example_circ
        super().__init__("GateReplacer")

    def do(self, dag_circuit: DAGCircuit):
        compiled = False
        all_node = dag_circuit.find_all_gate_node()
        CLog.log(f"Running {CLog.R1(self.rule_name)}.", 1, self.log_level)
        CLog.IncreaceHeadBlock()
        for node in all_node:
            is_same = node.gate.__class__ == self.ori_example_gate.__class__
            is_same = is_same and (node.gate.name == self.ori_example_gate.name)
            is_same = is_same and (len(node.gate.obj_qubits) == len(self.ori_example_gate.obj_qubits))
            is_same = is_same and (len(node.gate.ctrl_qubits) == len(self.ori_example_gate.ctrl_qubits))
            if is_same:
                CLog.log(f"{CLog.R1(self.rule_name)}: gate {CLog.B(node.gate)} will be replaced.", 2, self.log_level)
                compiled = True
                new_circ = apply(self.wanted_example_circ, node.gate.obj_qubits + node.gate.ctrl_qubits)
                dag_circuit.replace_node_with_dagcircuit(node, DAGCircuit(new_circ))
        CLog.DecreaseHeadBlock()
        if compiled:
            CLog.log(f"{CLog.R1(self.rule_name)}: {CLog.P('successfule compiled')}.", 1, self.log_level)
        else:
            CLog.log(f"{CLog.R1(self.rule_name)}: nothing happened.", 1, self.log_level)
        return compiled

    def __repr__(self):
        """String expression of gate replacer."""
        strs = ['GateReplacer<']
        body = f"{Circuit(self.ori_example_gate)} -> \n{self.wanted_example_circ}"
        for s in body.split('\n'):
            if s:
                strs.append(f"  " + s)
        strs.append(">")
        return '\n'.join(strs)


class CXToCZ(SequentialCompiler):
    def __init__(self):
        rule_set = [
            GateReplacer(G.X.on(0, 1), Circuit().h(0).z(0, 1).h(0)),
            GateReplacer(G.CNOT(0, 1), Circuit().h(0).z(0, 1).h(0)),
        ]
        super().__init__(rule_set)
        self.rule_name = "CXToCZ"


class CZToCX(GateReplacer):
    def __init__(self):
        super().__init__(G.Z.on(0, 1), Circuit().h(0).x(0, 1).h(0))
        self.rule_name = "CZToCX"
