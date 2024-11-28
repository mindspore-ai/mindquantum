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
"""RZ gate related decompose rule."""

from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.utils.type_value_check import _check_control_num, _check_input_type


def crz_decompose(gate: gates.RZ):
    """
    Decompose controlled :class:`~.core.gates.RZ` gate.

    Args:
        gate (:class:`~.core.gates.RZ`): a :class:`~.core.gates.RZ` gate with one control qubits.

    Returns:
        List[:class:`~.core.circuit.Circuit`], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler import crz_decompose
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import X, RZ
        >>> crz = RZ(1).on(1, 0)
        >>> origin_circ = Circuit() + crz
        >>> decomposed_circ = crz_decompose(crz)[0]
        >>> origin_circ
        q0: ──────■───────
                  ┃
              ┏━━━┻━━━┓
        q1: ──┨ RZ(1) ┠───
              ┗━━━━━━━┛
        >>> decomposed_circ
        q0: ────────────────■──────────────────■─────
                            ┃                  ┃
              ┏━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━┓ ┏━┻━┓
        q1: ──┨ RZ(1/2) ┠─┨╺╋╸┠─┨ RZ(-1/2) ┠─┨╺╋╸┠───
              ┗━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━┛ ┗━━━┛
    """
    _check_input_type('gate', gates.RZ, gate)
    _check_control_num(gate.ctrl_qubits, 1)
    circuit = Circuit()
    control = gate.ctrl_qubits[0]
    target = gate.obj_qubits[0]
    circuit += gates.RZ(gate.coeff / 2).on(target)
    circuit += gates.X.on(target, control)
    circuit += gates.RZ(-gate.coeff / 2).on(target)
    circuit += gates.X.on(target, control)
    return [circuit]

def _cnrz_frame(ctrl_qubits, obj_qubit, angle):
    """The function used to construct iteratively the circuit of the cnrz_decompose compiler.

    Args:
        ctrl_qubits (List[int]): The remainder ctrl_qubits in current iteration.
        obj_qubit (int): The obj_qubit of the RZ gate to be decomposed.
        angle (:class:'~core.parameterresolver.ParameterResolver`): The rotation angle in current iteration.

    Returns:
        :class:`~.core.circuit.Circuit`, the decompose solution in current iteration.
    """
    circ = Circuit()
    if len(ctrl_qubits) == 1:
        circ.rz(angle/2, obj_qubit)
        circ.x(obj_qubit, ctrl_qubits[0])
        circ.rz(-angle/2, obj_qubit)
        circ.x(obj_qubit, ctrl_qubits[0])
    else:
        circ += _cnrz_frame(ctrl_qubits[1:], obj_qubit, angle/2)
        circ.x(obj_qubit, ctrl_qubits[0])
        circ += _cnrz_frame(ctrl_qubits[1:], obj_qubit, -angle/2)
        circ.x(obj_qubit, ctrl_qubits[0])
    return circ

def cnrz_decompose(gate: gates.RZ):
    """
    Decompose controlled :class:`~.core.gates.RZ` gate.

    Args:
        gate (:class:`~.core.gates.RZ`): A :class:`~.core.gates.RZ` gate with zero or more control qubits.

    Returns:
        List[:class:`~.core.circuit.Circuit`], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler import cnrz_decompose
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import RZ
        >>> crz = RZ(1).on(2, [0, 1])
        >>> origin_circ = Circuit() + crz
        >>> decomposed_circ = cnrz_decompose(crz)[0]
        >>> origin_circ
        q0: ──────■───────
                  ┃
                  ┃
        q1: ──────■───────
                  ┃
              ┏━━━┻━━━┓
        q2: ──┨ RZ(1) ┠───
              ┗━━━━━━━┛
        >>> decomposed_circ
        q0: ─────────────────────────────────────────■──────────────────────────────────────────■────
                                                     ┃                                          ┃
                                                     ┃                                          ┃
        q1: ────────────────■──────────────────■─────╂──────────────────■─────────────────■─────╂────
                            ┃                  ┃     ┃                  ┃                 ┃     ┃
              ┏━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━┓ ┏━┻━┓ ┏━┻━┓ ┏━━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━┓ ┏━┻━┓ ┏━┻━┓
        q2: ──┨ RZ(1/4) ┠─┨╺╋╸┠─┨ RZ(-1/4) ┠─┨╺╋╸┠─┨╺╋╸┠─┨ RZ(-1/4) ┠─┨╺╋╸┠─┨ RZ(1/4) ┠─┨╺╋╸┠─┨╺╋╸┠──
              ┗━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━┛ ┗━━━┛ ┗━━━┛
    """
    _check_input_type('gate', gates.RZ, gate)

    if not gate.ctrl_qubits:
        return [Circuit(gate)]

    circuit = _cnrz_frame(gate.ctrl_qubits, gate.obj_qubits[0], gate.coeff)
    return [circuit]


decompose_rules = ['crz_decompose', 'cnrz_decompose']

__all__ = decompose_rules
