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
H gate related decompose rule.
"""

from mindquantum.core import gates as G
from mindquantum.core import Circuit, T, X, H, S
from mindquantum.utils.type_value_check import _check_input_type, _check_control_num

def ch_decompose(gate: G.HGate):
    """
    Decompose ch gate.

    Args:
        gate (H): a H gate with one control qubits.

    Returns:
        List[Circuit], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler.decompose import ch_decompose
        >>> from mindquantum.core import Circuit, T, X, H, S
        >>> ch = H.on(1, 0)
        >>> origin_circ = Circuit() + ch
        >>> decomposed_circ = ch_decompose(ch)[0]
        >>> origin_circ
        q0: ──●──
              │
        q1: ──H──
        >>> decomposed_circ
        q0: ─────────────────●───────────────────
                             │
        q1: ──S────H────T────X────T†────H────S†──
    """
    _check_input_type('gate', G.HGate, gate)
    _check_control_num(gate.ctrl_qubits, 1)
    solutions = []
    c1 = Circuit()
    solutions.append(c1)
    q0 = gate.obj_qubits[0]
    q1 = gate.ctrl_qubits[0]
    c1 += S.on(q0)
    c1 += H.on(q0)
    c1 += T.on(q0)
    c1 += X.on(q0,q1)
    c1 += T.on(q0).hermitian()
    c1 += H.on(q0)
    c1 += S.on(q0).hermitian()
    return solutions

decompose_rules = ['ch_decompose']
__all__ = decompose_rules
