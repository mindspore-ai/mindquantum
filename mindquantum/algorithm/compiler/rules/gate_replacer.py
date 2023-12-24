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
"""Convert cnot to cz."""
from mindquantum.core.circuit import Circuit, apply
from mindquantum.core.gates import CNOT, BasicGate, X, Z

from ..dag import DAGCircuit
from .basic_rule import BasicCompilerRule, SequentialCompiler
from .compiler_logger import CompileLog as CLog
from .compiler_logger import LogIndentation


class GateReplacer(BasicCompilerRule):
    """
    Replace given gate with given circuit.

    Args:
        ori_example_gate (:class:`~.core.gates.BasicGate`): The gate you want to replace.
            Please note that every gate that belong to given gate together with same
            length of `obj_qubits` and `ctrl_qubits` will be matched.
        wanted_example_circ (:class:`~.core.circuit.Circuit`): The quantum circuit you want.

    Examples:
        >>> from mindquantum.algorithm.compiler import GateReplacer, compile_circuit
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import X
        >>> circ = Circuit().x(1, 0).h(1).x(1, 2)
        >>> circ
        q0: ────■─────────────────
                ┃
              ┏━┻━┓ ┏━━━┓ ┏━━━┓
        q1: ──┨╺╋╸┠─┨ H ┠─┨╺╋╸┠───
              ┗━━━┛ ┗━━━┛ ┗━┳━┛
                            ┃
        q2: ────────────────■─────
        >>> equivalent_cnot = Circuit().h(0).z(0, 1).h(0)
        >>> equivalent_cnot
              ┏━━━┓ ┏━━━┓ ┏━━━┓
        q0: ──┨ H ┠─┨ Z ┠─┨ H ┠───
              ┗━━━┛ ┗━┳━┛ ┗━━━┛
                      ┃
        q1: ──────────■───────────
        >>> compiler = GateReplacer(X.on(0, 1), equivalent_cnot)
        >>> compiler
        GateReplacer<
                ┏━━━┓
          q0: ──┨╺╋╸┠───
                ┗━┳━┛
                  ┃
          q1: ────■─────
           ->
                ┏━━━┓ ┏━━━┓ ┏━━━┓
          q0: ──┨ H ┠─┨ Z ┠─┨ H ┠───
                ┗━━━┛ ┗━┳━┛ ┗━━━┛
                        ┃
          q1: ──────────■───────────
        >
        >>> compile_circuit(compiler, circ)
        q0: ──────────■───────────────────────────────────
                      ┃
              ┏━━━┓ ┏━┻━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓
        q1: ──┨ H ┠─┨ Z ┠─┨ H ┠─┨ H ┠─┨ H ┠─┨ Z ┠─┨ H ┠───
              ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━┳━┛ ┗━━━┛
                                              ┃
        q2: ──────────────────────────────────■───────────
    """

    def __init__(
        self,
        ori_example_gate: BasicGate,
        wanted_example_circ: Circuit,
    ):
        """Initialize gate replacer compiler."""
        all_qubits = ori_example_gate.obj_qubits + ori_example_gate.ctrl_qubits
        if set(all_qubits) != set(wanted_example_circ.all_qubits.keys()):
            raise ValueError("Qubit set not equal for given gate and circuit.")
        self.ori_example_gate = ori_example_gate
        self.wanted_example_circ = wanted_example_circ
        super().__init__("GateReplacer")
        self.permute_map = dict(enumerate(all_qubits))

    def __repr__(self):
        """Get string expression of gate replacer."""
        strs = ['GateReplacer<']
        body = f"{Circuit(self.ori_example_gate)} ->\n{self.wanted_example_circ}"
        for string in body.split('\n'):
            if string:
                strs.append("  " + string)
        strs.append(">")
        return '\n'.join(strs)

    def do(self, dag_circuit: DAGCircuit) -> bool:
        """
        Do gate replacer rule.

        Args:
            dag_circuit (:class:`~.algorithm.compiler.DAGCircuit`): The DAG of quantum circuit you want to compile.
        """
        compiled = False
        all_node = dag_circuit.find_all_gate_node()
        CLog.log(f"Running {CLog.R1(self.rule_name)}.", 1, self.log_level)
        with LogIndentation() as _:
            for node in all_node:
                is_same = node.gate.__class__ == self.ori_example_gate.__class__
                is_same = is_same and (node.gate.name == self.ori_example_gate.name)
                is_same = is_same and (len(node.gate.obj_qubits) == len(self.ori_example_gate.obj_qubits))
                is_same = is_same and (len(node.gate.ctrl_qubits) == len(self.ori_example_gate.ctrl_qubits))
                if is_same:
                    CLog.log(
                        f"{CLog.R1(self.rule_name)}: gate {CLog.B(node.gate)} will be replaced.", 2, self.log_level
                    )
                    compiled = True
                    new_map = []
                    for idx, qid in enumerate(node.gate.obj_qubits + node.gate.ctrl_qubits):
                        new_map.append((self.permute_map[idx], qid))
                    new_map.sort()
                    new_circ = apply(self.wanted_example_circ, [i for _, i in new_map])
                    dag_circuit.replace_node_with_dag_circuit(node, DAGCircuit(new_circ))
        if compiled:
            CLog.log(f"{CLog.R1(self.rule_name)}: {CLog.P('successfully compiled')}.", 1, self.log_level)
        else:
            CLog.log(f"{CLog.R1(self.rule_name)}: nothing happened.", 1, self.log_level)
        return compiled


class CXToCZ(SequentialCompiler):
    """Convert cx to cz gate."""

    def __init__(self):
        """Initialize a CXToCZ compiler."""
        rule_set = [
            GateReplacer(X.on(0, 1), Circuit().h(0).z(0, 1).h(0)),
            GateReplacer(CNOT(0, 1), Circuit().h(0).z(0, 1).h(0)),
        ]
        super().__init__(rule_set)
        self.rule_name = "CXToCZ"


class CZToCX(GateReplacer):
    """Convert cz to cx gate."""

    def __init__(self):
        """Initialize a CZToCX."""
        super().__init__(Z.on(0, 1), Circuit().h(0).x(0, 1).h(0))
        self.rule_name = "CZToCX"
