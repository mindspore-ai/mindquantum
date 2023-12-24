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
"""Ryy gate related decompose rule."""

import numpy as np

from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.utils.type_value_check import _check_control_num, _check_input_type


def ryy_decompose(gate: gates.Ryy):
    """
    Decompose :class:`~.core.gates.Ryy` gate.

    Args:
        gate (:class:`~.core.gates.Ryy`): a :class:`~.core.gates.Ryy` gate.

    Returns:
        List[:class:`~.core.circuit.Circuit`], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler import ryy_decompose
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import Ryy
        >>> ryy = Ryy(1).on([0, 1])
        >>> origin_circ = Circuit() + ryy
        >>> decomposed_circ = ryy_decompose(ryy)[0]
        >>> origin_circ
              ┏━━━━━━━━┓
        q0: ──┨        ┠───
              ┃        ┃
              ┃ Ryy(1) ┃
        q1: ──┨        ┠───
              ┗━━━━━━━━┛
        >>> decomposed_circ
              ┏━━━━━━━━━┓                       ┏━━━━━━━━━━┓
        q0: ──┨ RX(π/2) ┠───■───────────────■───┨ RX(-π/2) ┠───
              ┗━━━━━━━━━┛   ┃               ┃   ┗━━━━━━━━━━┛
              ┏━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━┓
        q1: ──┨ RX(π/2) ┠─┨╺╋╸┠─┨ RZ(1) ┠─┨╺╋╸┠─┨ RX(-π/2) ┠───
              ┗━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━┛
    """
    _check_input_type('gate', gates.Ryy, gate)
    _check_control_num(gate.ctrl_qubits, 0)
    return cryy_decompose(gate)


def cryy_decompose(gate: gates.Ryy):
    """
    Decompose :class:`~.core.gates.Ryy` gate with control qubits.

    Args:
        gate (:class:`~.core.gates.Ryy`): a :class:`~.core.gates.Ryy` gate.

    Returns:
        List[:class:`~.core.circuit.Circuit`], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler import cryy_decompose
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import Ryy
        >>> cryy = Ryy(2).on([0, 1], [2, 3])
        >>> origin_circ = Circuit() + cryy
        >>> decomposed_circ = cryy_decompose(cryy)[0]
        >>> origin_circ
              ┏━━━━━━━━┓
        q0: ──┨        ┠───
              ┃        ┃
              ┃ Ryy(2) ┃
        q1: ──┨        ┠───
              ┗━━━━┳━━━┛
                   ┃
        q2: ───────■───────
                   ┃
                   ┃
        q3: ───────■───────
        >>> decomposed_circ
              ┏━━━━━━━━━┓                                                ┏━━━━━━━━━━┓
        q0: ──┨ RX(π/2) ┠───────────────■───────────────■────────────────┨ RX(-π/2) ┠───
              ┗━━━━┳━━━━┛               ┃               ┃                ┗━━━━━┳━━━━┛
                   ┃      ┏━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━┓       ┃
        q1: ───────╂──────┨ RX(π/2) ┠─┨╺╋╸┠─┨ RZ(2) ┠─┨╺╋╸┠─┨ RX(-π/2) ┠───────╂────────
                   ┃      ┗━━━━┳━━━━┛ ┗━┳━┛ ┗━━━┳━━━┛ ┗━┳━┛ ┗━━━━━┳━━━━┛       ┃
                   ┃           ┃        ┃       ┃       ┃         ┃            ┃
        q2: ───────■───────────■────────■───────■───────■─────────■────────────■────────
                   ┃           ┃        ┃       ┃       ┃         ┃            ┃
                   ┃           ┃        ┃       ┃       ┃         ┃            ┃
        q3: ───────■───────────■────────■───────■───────■─────────■────────────■────────
    """
    _check_input_type('gate', gates.Ryy, gate)
    q0 = gate.obj_qubits[0]
    q1 = gate.obj_qubits[1]
    controls = gate.ctrl_qubits

    circuit1 = Circuit()
    circuit1 += gates.RX(np.pi / 2).on(q0, controls)
    circuit1 += gates.RX(np.pi / 2).on(q1, controls)
    circuit1 += gates.X.on(q1, [q0] + controls)
    circuit1 += gates.RZ(gate.coeff).on(q1, controls)
    circuit1 += circuit1[-2]
    circuit1 += gates.RX(-np.pi / 2).on(q1, controls)
    circuit1 += gates.RX(-np.pi / 2).on(q0, controls)

    circuit2 = Circuit()
    circuit2 += gates.RX(np.pi / 2).on(q0, controls)
    circuit2 += gates.RX(np.pi / 2).on(q1, controls)
    circuit2 += gates.X.on(q0, [q1] + controls)
    circuit2 += gates.RZ(gate.coeff).on(q0, controls)
    circuit2 += circuit2[-2]
    circuit2 += gates.RX(-np.pi / 2).on(q1, controls)
    circuit2 += gates.RX(-np.pi / 2).on(q0, controls)

    return [circuit1, circuit2]


decompose_rules = ['ryy_decompose', 'cryy_decompose']
__all__ = decompose_rules
