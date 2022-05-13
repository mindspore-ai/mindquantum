# -*- coding: utf-8 -*-
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
"""
RY gate related decompose rule.
"""

from mindquantum.core import gates as G
from mindquantum.core import Circuit
from mindquantum.utils.type_value_check import _check_input_type, _check_control_num

def cry_decompose(gate: G.RYGate):
    """
    Decompose cry gate.

    Args:
        gate (RYGate): a RYGate with one control qubits.

    Returns:
        List[Circuit], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler.decompose import cry_decompose
        >>> from mindquantum.core import Circuit, RY
        >>> cry = RY.on(1, 0)
        >>> origin_circ = Circuit() + cry
        >>> decomposed_circ = cry_decompose(cry)[0]
        >>> origin_circ
        q0: ─────●─────
                 │
        q1: ───RY(1)───

        >>> decomposed_circ
        q0: ─────────────●────────────────●──
                         │                │
        q1: ──RY(1/2)────X────RY(-1/2)────X──
    """
    _check_input_type('gate', G.RYGate, gate)
    _check_control_num(gate.ctrl_qubits, 1)
    solutions = []
    c1 = Circuit()
    solutions.append(c1)
    q0 = gate.ctrl_qubits[0]
    q1 = gate.obj_qubits[0]
    c1 += G.RY(gate.coeff/2).on(q1)
    c1 += G.X.on(q1, q0)
    c1 += G.RY(-gate.coeff/2).on(q1)
    c1 += G.X.on(q1, q0)
    return solutions

decompose_rules = ['cry_decompose']
__all__ = decompose_rules
