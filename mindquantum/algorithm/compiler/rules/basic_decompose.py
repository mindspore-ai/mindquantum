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
"""Decompose gate with control qubits."""
# pylint: disable=unused-argument
from mindquantum.core.circuit import controlled
from mindquantum.core.gates import (
    RX,
    RY,
    RZ,
    BasicGate,
    CNOTGate,
    HGate,
    PhaseShift,
    Rxx,
    Ryy,
    Rzz,
    SGate,
    SWAPGate,
    TGate,
    UnivMathGate,
    XGate,
    YGate,
    ZGate,
)
from mindquantum.utils.type_value_check import _check_input_type

from ..dag import DAGCircuit
from ..decompose import (
    ccx_decompose,
    ch_decompose,
    crx_decompose,
    crxx_decompose,
    cry_decompose,
    cryy_decompose,
    crz_decompose,
    cs_decompose,
    cswap_decompose,
    ct_decompose,
    cu_decompose,
    cy_decompose,
    cz_decompose,
    rxx_decompose,
    ryy_decompose,
    rzz_decompose,
)
from .basic_rule import BasicCompilerRule
from .compiler_logger import CompileLog as CLog
from .compiler_logger import LogIndentation

# pylint: disable=invalid-name,too-many-branches,too-few-public-methods


def decom_swap(gate: SWAPGate, *args, **kwargs):
    """Decompose swap gate."""
    if len(gate.ctrl_qubits) == 1:
        return DAGCircuit(cswap_decompose(gate)[0])
    if len(gate.ctrl_qubits) > 1:
        ctrl_swap = cswap_decompose(gate.on(gate.obj_qubits, gate.ctrl_qubits[0]))[0]
        cctrl_swap = controlled(ctrl_swap)(gate.ctrl_qubits[1:])
        return DAGCircuit(cctrl_swap)
    return False


def decom_rz(gate: RZ, *args, prefer_u3=False, **kwargs):
    """Decompose rz gate."""
    if len(gate.ctrl_qubits) == 1:
        return DAGCircuit(crz_decompose(gate)[0])
    if len(gate.ctrl_qubits) > 1:
        if prefer_u3:
            return DAGCircuit(cu_decompose(gate))
        ctrl_rz = crz_decompose(gate.on(gate.obj_qubits, gate.ctrl_qubits[0]))[0]
        cctrl_rz = controlled(ctrl_rz)(gate.ctrl_qubits[1:])
        return DAGCircuit(cctrl_rz)
    return False


def decom_ry(gate: RY, *args, prefer_u3=False, **kwargs):
    """Decompose ry gate."""
    if len(gate.ctrl_qubits) == 1:
        return DAGCircuit(cry_decompose(gate)[0])
    if len(gate.ctrl_qubits) > 1:
        if prefer_u3:
            return DAGCircuit(cu_decompose(gate))
        ctrl_ry = cry_decompose(gate.on(gate.obj_qubits, gate.ctrl_qubits[0]))[0]
        cctrl_ry = controlled(ctrl_ry)(gate.ctrl_qubits[1:])
        return DAGCircuit(cctrl_ry)
    return False


def decom_rx(gate: RX, *args, prefer_u3=False, **kwargs):
    """Decompose rx gate."""
    if len(gate.ctrl_qubits) == 1:
        return DAGCircuit(crx_decompose(gate)[0])
    if len(gate.ctrl_qubits) > 1:
        if prefer_u3:
            return DAGCircuit(cu_decompose(gate))
        ctrl_rx = crx_decompose(gate.on(gate.obj_qubits, gate.ctrl_qubits[0]))[0]
        cctrl_rx = controlled(ctrl_rx)(gate.ctrl_qubits[1:])
        return DAGCircuit(cctrl_rx)
    return False


def decom_rzz(gate: Rzz, *args, prefer_u3=False, **kwargs):
    """Decompose zz gate."""
    if not gate.ctrl_qubits:
        return DAGCircuit(rzz_decompose(gate)[0])
    if gate.ctrl_qubits:
        if prefer_u3 and not gate.parameterized:
            return DAGCircuit(cu_decompose(gate))
        zz = rzz_decompose(gate.on(gate.obj_qubits))[0]
        ctrl_zz = controlled(zz)(gate.ctrl_qubits)
        return DAGCircuit(ctrl_zz)
    return False


def decom_ryy(gate: Ryy, *args, prefer_u3=False, **kwargs):
    """Decompose yy gate."""
    if not gate.ctrl_qubits:
        return DAGCircuit(ryy_decompose(gate)[0])
    if gate.ctrl_qubits:
        if prefer_u3 and not gate.parameterized:
            return DAGCircuit(cu_decompose(gate))
        return DAGCircuit(cryy_decompose(gate)[0])
    return False


def decom_rxx(gate: Rxx, *args, prefer_u3=False, **kwargs):
    """Decompose xx gate."""
    if not gate.ctrl_qubits:
        return DAGCircuit(rxx_decompose(gate)[0])
    if gate.ctrl_qubits:
        if prefer_u3 and not gate.parameterized:
            return DAGCircuit(cu_decompose(gate))
        return DAGCircuit(crxx_decompose(gate)[0])
    return False


def decom_h(gate: HGate, *args, prefer_u3=False, **kwargs):
    """Decompose h gate."""
    if len(gate.ctrl_qubits) == 1:
        return DAGCircuit(ch_decompose(gate)[0])
    if len(gate.ctrl_qubits) > 1:
        if prefer_u3 and not gate.parameterized:
            return DAGCircuit(cu_decompose(gate))
        ctrl_h = ch_decompose(gate.on(gate.obj_qubits, gate.ctrl_qubits[0]))[0]
        cctrl_h = controlled(ctrl_h)(gate.ctrl_qubits[1:])
        return DAGCircuit(cctrl_h)
    return False


def decom_x(gate: XGate, *args, prefer_u3=False, **kwargs):
    """Decompose x gate."""
    if len(gate.ctrl_qubits) == 2:
        return DAGCircuit(ccx_decompose(gate)[0])
    if len(gate.ctrl_qubits) > 2:
        if prefer_u3 and not gate.parameterized:
            return DAGCircuit(cu_decompose(gate))
        ctrl_x = ccx_decompose(gate.on(gate.obj_qubits, gate.ctrl_qubits[:2]))[0]
        cctrl_x = controlled(ctrl_x)(gate.ctrl_qubits[2:])
        return DAGCircuit(cctrl_x)
    return False


def decom_z(gate: ZGate, *args, prefer_u3=False, **kwargs):
    """Decompose z gate."""
    if len(gate.ctrl_qubits) == 2:
        return DAGCircuit(cz_decompose(gate)[0])
    if len(gate.ctrl_qubits) > 2:
        if prefer_u3 and not gate.parameterized:
            return DAGCircuit(cu_decompose(gate))
        ctrl_z = cz_decompose(gate.on(gate.obj_qubits, gate.ctrl_qubits[0]))[0]
        cctrl_z = controlled(ctrl_z)(gate.ctrl_qubits[1:])
        return DAGCircuit(cctrl_z)
    return False


def decom_y(gate: YGate, *args, prefer_u3=False, **kwargs):
    """Decompose y gate."""
    if len(gate.ctrl_qubits) == 1:
        return DAGCircuit(cy_decompose(gate)[0])
    if len(gate.ctrl_qubits) > 1:
        if prefer_u3 and not gate.parameterized:
            return DAGCircuit(cu_decompose(gate))
        ctrl_y = cy_decompose(gate.on(gate.obj_qubits, gate.ctrl_qubits[0]))[0]
        cctrl_y = controlled(ctrl_y)(gate.ctrl_qubits[1:])
        return DAGCircuit(cctrl_y)
    return False


def decom_s(gate: SGate, *args, prefer_u3=False, **kwargs):
    """Decompose s gate."""
    if len(gate.ctrl_qubits) == 1:
        if gate.hermitianed:
            return DAGCircuit(cs_decompose(gate.hermitian())[0].hermitian())
        return DAGCircuit(cs_decompose(gate)[0])
    if len(gate.ctrl_qubits) > 1:
        if prefer_u3 and not gate.parameterized:
            return DAGCircuit(cu_decompose(gate))
        if gate.hermitianed:
            ctrl_s = cs_decompose(gate.on(gate.obj_qubits, gate.ctrl_qubits[0]).hermitian())[0].hermitian()
        else:
            ctrl_s = cs_decompose(gate.on(gate.obj_qubits, gate.ctrl_qubits[0]))[0]
        cctrl_s = controlled(ctrl_s)(gate.ctrl_qubits[1:])
        return DAGCircuit(cctrl_s)
    return False


def decom_t(gate: SGate, *args, prefer_u3=False, **kwargs):
    """Decompose t gate."""
    if len(gate.ctrl_qubits) == 1:
        if gate.hermitianed:
            return DAGCircuit(ct_decompose(gate.hermitian())[0].hermitian())
        return DAGCircuit(ct_decompose(gate)[0])
    if len(gate.ctrl_qubits) > 1:
        if prefer_u3 and not gate.parameterized:
            return DAGCircuit(cu_decompose(gate))
        if gate.hermitianed:
            ctrl_t = ct_decompose(gate.on(gate.obj_qubits, gate.ctrl_qubits[0]).hermitian())[0].hermitian()
        else:
            ctrl_t = ct_decompose(gate.on(gate.obj_qubits, gate.ctrl_qubits[0]))[0]
        cctrl_t = controlled(ctrl_t)(gate.ctrl_qubits[1:])
        return DAGCircuit(cctrl_t)
    return False


def decom_ps(gate: PhaseShift, *args, **kwargs):
    """Decompose phase shift gate."""
    if gate.ctrl_qubits:
        return DAGCircuit(cu_decompose(gate))
    return False


def decom_univ_math_gate(gate: UnivMathGate, *args, **kwargs):
    """Decompose universal matrix gate."""
    return DAGCircuit(cu_decompose(gate))


def decom_cnot(gate: CNOTGate, *args, **kwargs):
    """Decompose cnot gate."""
    return decom_x(XGate().on(gate.obj_qubits[0], gate.obj_qubits[1:] + gate.ctrl_qubits), *args, **kwargs)


def decom_basic_gate(gate: BasicGate, prefer_u3=False):
    """Decompose all possible quantum gate."""
    res = False
    if isinstance(gate, XGate):
        res = decom_x(gate, prefer_u3=prefer_u3)
    elif isinstance(gate, YGate):
        res = decom_y(gate, prefer_u3=prefer_u3)
    elif isinstance(gate, ZGate):
        res = decom_z(gate, prefer_u3=prefer_u3)
    elif isinstance(gate, RX):
        res = decom_rx(gate, prefer_u3=prefer_u3)
    elif isinstance(gate, RY):
        res = decom_ry(gate, prefer_u3=prefer_u3)
    elif isinstance(gate, RZ):
        res = decom_rz(gate, prefer_u3=prefer_u3)
    elif isinstance(gate, Rxx):
        res = decom_rxx(gate, prefer_u3=prefer_u3)
    elif isinstance(gate, Ryy):
        res = decom_ryy(gate, prefer_u3=prefer_u3)
    elif isinstance(gate, Rzz):
        res = decom_rzz(gate, prefer_u3=prefer_u3)
    elif isinstance(gate, SGate):
        res = decom_s(gate, prefer_u3=prefer_u3)
    elif isinstance(gate, TGate):
        res = decom_t(gate, prefer_u3=prefer_u3)
    elif isinstance(gate, SWAPGate):
        res = decom_swap(gate, prefer_u3=prefer_u3)
    elif isinstance(gate, HGate):
        res = decom_h(gate, prefer_u3=prefer_u3)
    elif isinstance(gate, CNOTGate):
        res = decom_cnot(gate, prefer_u3=prefer_u3)
    elif isinstance(gate, UnivMathGate):
        res = decom_univ_math_gate(gate, prefer_u3=prefer_u3)
    elif isinstance(gate, PhaseShift):
        res = decom_ps(gate, prefer_u3=prefer_u3)
    else:
        if hasattr(gate, "__decompose__"):
            sub_circ = gate.__decompose__()
            if sub_circ:
                res = DAGCircuit(sub_circ)
    if res:
        return res
    return False


class BasicDecompose(BasicCompilerRule):
    """
    Decompose gate into a simple gate set.

    This decompose rule decompose multiple control gate, customized matrix gate
    and rotation gate with multi pauli word.

    Args:
        prefer_u3 (bool): whether prefer to use :class:`~.core.gates.U3` gate to decompose gate. Default: ``False``.

    Examples:
        >>> from mindquantum.algorithm.compiler import BasicDecompose, DAGCircuit
        >>> from mindquantum.core import gates as G
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit([G.Rxx('a').on([0, 1]), G.X(0, [1, 2])])
        >>> circ
              ┏━━━━━━━━┓ ┏━━━┓
        q0: ──┨        ┠─┨╺╋╸┠───
              ┃        ┃ ┗━┳━┛
              ┃ Rxx(a) ┃   ┃
        q1: ──┨        ┠───■─────
              ┗━━━━━━━━┛   ┃
                           ┃
        q2: ───────────────■─────
        >>> dag_circ = DAGCircuit(circ)
        >>> compiler = BasicDecompose()
        >>> compiler.do(dag_circ)
        >>> dag_circ.to_circuit()
              ┏━━━┓                       ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━━┓
        q0: ──┨ H ┠───■───────────────■───┨ H ┠─┨ H ┠─┨╺╋╸┠─┨ T† ┠─┨╺╋╸┠─┨ T ┠─┨╺╋╸┠─┨ T† ┠─↯─
              ┗━━━┛   ┃               ┃   ┗━━━┛ ┗━━━┛ ┗━┳━┛ ┗━━━━┛ ┗━┳━┛ ┗━━━┛ ┗━┳━┛ ┗━━━━┛
              ┏━━━┓ ┏━┻━┓ ┏━━━━━━━┓ ┏━┻━┓ ┏━━━┓         ┃            ┃           ┃
        q1: ──┨ H ┠─┨╺╋╸┠─┨ RZ(a) ┠─┨╺╋╸┠─┨ H ┠─────────■────────────╂───────────■──────────↯─
              ┗━━━┛ ┗━━━┛ ┗━━━━━━━┛ ┗━━━┛ ┗━━━┛                      ┃
                                                                     ┃
        q2: ─────────────────────────────────────────────────────────■──────────────────────↯─
              ┏━━━┓ ┏━━━┓ ┏━━━┓
        q0: ──┨╺╋╸┠─┨ T ┠─┨ H ┠────────────────
              ┗━┳━┛ ┗━━━┛ ┗━━━┛
                ┃   ┏━━━┓ ┏━━━┓ ┏━━━━┓ ┏━━━┓
        q1: ────╂───┨ T ┠─┨╺╋╸┠─┨ T† ┠─┨╺╋╸┠───
                ┃   ┗━━━┛ ┗━┳━┛ ┗━━━━┛ ┗━┳━┛
                ┃           ┃   ┏━━━┓    ┃
        q2: ────■───────────■───┨ T ┠────■─────
                                ┗━━━┛
    """

    def __init__(self, prefer_u3=False):
        """Initialize a control decompose rule."""
        super().__init__("BasicDecompose")
        self.prefer_u3 = prefer_u3

    def do(self, dag_circuit: DAGCircuit) -> bool:
        """
        Do control decompose rule and multi qubit gate decomposition.

        Args:
            dag_circuit (:class:`~.algorithm.compiler.DAGCircuit`): The DAG graph of quantum circuit.
        """
        _check_input_type("dag_circuit", DAGCircuit, dag_circuit)
        compiled = False
        all_node = dag_circuit.find_all_gate_node()
        CLog.log(f"Running {CLog.R1(self.rule_name)}.", 1, self.log_level)
        with LogIndentation() as _:
            for node in all_node:
                decompose_dag_circ = decom_basic_gate(node.gate, prefer_u3=self.prefer_u3)
                if decompose_dag_circ:
                    compiled = True
                    CLog.log(
                        f"{CLog.R1(self.rule_name)}: gate {CLog.B(node.gate)} will be compiled.", 2, self.log_level
                    )
                    with LogIndentation() as _:
                        self.do(decompose_dag_circ)
                    dag_circuit.replace_node_with_dag_circuit(node, decompose_dag_circ)

        if compiled:
            CLog.log(f"{CLog.R1(self.rule_name)}: {CLog.P('successfully compiled')}.", 1, self.log_level)
        else:
            CLog.log(f"{CLog.R1(self.rule_name)}: nothing happened.", 1, self.log_level)
        return compiled
