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
"""Rxx gate related decompose rule."""

from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.utils.type_value_check import _check_control_num, _check_input_type


def rxx_decompose(gate: gates.Rxx):
    """
    Decompose :class:`~.core.gates.Rxx` gate.

    Args:
        gate (:class:`~.core.gates.Rxx`): a :class:`~.core.gates.Rxx` gate.

    Returns:
        List[:class:`~.core.circuit.Circuit`], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler import rxx_decompose
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import Rxx
        >>> rxx = Rxx(1).on([0, 1])
        >>> origin_circ = Circuit() + rxx
        >>> decomposed_circ = rxx_decompose(rxx)[0]
        >>> origin_circ
              ┏━━━━━━━━┓
        q0: ──┨        ┠───
              ┃        ┃
              ┃ Rxx(1) ┃
        q1: ──┨        ┠───
              ┗━━━━━━━━┛
        >>> decomposed_circ
              ┏━━━┓                       ┏━━━┓
        q0: ──┨ H ┠───■───────────────■───┨ H ┠───
              ┗━━━┛   ┃               ┃   ┗━━━┛
              ┏━━━┓ ┏━┻━┓ ┏━━━━━━━┓ ┏━┻━┓ ┏━━━┓
        q1: ──┨ H ┠─┨╺╋╸┠─┨ RZ(1) ┠─┨╺╋╸┠─┨ H ┠───
              ┗━━━┛ ┗━━━┛ ┗━━━━━━━┛ ┗━━━┛ ┗━━━┛
    """
    _check_input_type('gate', gates.Rxx, gate)
    _check_control_num(gate.ctrl_qubits, 0)
    return crxx_decompose(gate)


def crxx_decompose(gate: gates.Rxx):
    """
    Decompose :class:`~.core.gates.Rxx` gate with control qubits.

    Args:
        gate (:class:`~.core.gates.Rxx`): a :class:`~.core.gates.Rxx` gate.

    Returns:
        List[:class:`~.core.circuit.Circuit`], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler.decompose import crxx_decompose
        >>> from mindquantum.core import Circuit, Rxx
        >>> crxx = Rxx(2).on([0, 1], [2, 3])
        >>> origin_circ = Circuit() + crxx
        >>> decomposed_circ = crxx_decompose(crxx)[0]
        >>> origin_circ
              ┏━━━━━━━━┓
        q0: ──┨        ┠───
              ┃        ┃
              ┃ Rxx(2) ┃
        q1: ──┨        ┠───
              ┗━━━━┳━━━┛
                   ┃
        q2: ───────■───────
                   ┃
                   ┃
        q3: ───────■───────
        >>> decomposed_circ
              ┏━━━┓                                   ┏━━━┓
        q0: ──┨ H ┠─────────■───────────────■─────────┨ H ┠───
              ┗━┳━┛         ┃               ┃         ┗━┳━┛
                ┃   ┏━━━┓ ┏━┻━┓ ┏━━━━━━━┓ ┏━┻━┓ ┏━━━┓   ┃
        q1: ────╂───┨ H ┠─┨╺╋╸┠─┨ RZ(2) ┠─┨╺╋╸┠─┨ H ┠───╂─────
                ┃   ┗━┳━┛ ┗━┳━┛ ┗━━━┳━━━┛ ┗━┳━┛ ┗━┳━┛   ┃
                ┃     ┃     ┃       ┃       ┃     ┃     ┃
        q2: ────■─────■─────■───────■───────■─────■─────■─────
                ┃     ┃     ┃       ┃       ┃     ┃     ┃
                ┃     ┃     ┃       ┃       ┃     ┃     ┃
        q3: ────■─────■─────■───────■───────■─────■─────■─────
    """
    _check_input_type('gate', gates.Rxx, gate)
    q0 = gate.obj_qubits[0]
    q1 = gate.obj_qubits[1]

    circuit1 = Circuit()
    circuit1 += gates.H.on(q0, gate.ctrl_qubits)
    circuit1 += gates.H.on(q1, gate.ctrl_qubits)
    circuit1 += gates.X.on(q1, [q0] + gate.ctrl_qubits)
    circuit1 += gates.RZ(gate.coeff).on(q1, gate.ctrl_qubits)
    circuit1 += circuit1[:-1][::-1]

    circuit2 = Circuit()
    circuit2 += gates.H.on(q0, gate.ctrl_qubits)
    circuit2 += gates.H.on(q1, gate.ctrl_qubits)
    circuit2 += gates.X.on(q0, [q1] + gate.ctrl_qubits)
    circuit2 += gates.RZ(gate.coeff).on(q0, gate.ctrl_qubits)
    circuit2 += circuit2[:-1][::-1]

    return [circuit1, circuit2]


decompose_rules = ['rxx_decompose', 'crxx_decompose']
__all__ = decompose_rules
