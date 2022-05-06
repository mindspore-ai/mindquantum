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
X gate related decompose rule.
"""

from mindquantum.core import gates as G
from mindquantum.core import Circuit
from mindquantum.core.gates.basicgate import XGate
from mindquantum.utils.type_value_check import _check_input_type, _check_control_num


def ccx_decompose(gate: G.XGate):
    """
    Decompose ccx gate.

    Args:
        gate (XGate): a XGate with two control qubits.

    Returns:
        List[Circuit], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler.decompose import ccx_decompose
        >>> from mindquantum.core import Circuit, X
        >>> ccx = X.on(2, [0,1])
        >>> origin_circ = Circuit() + ccx
        >>> decomposed_circ = ccx_decompose(ccx)[0]
        >>> origin_circ
        q0: ──●──
              │
        q1: ──●──
              │
        q2: ──X──
        >>> decomposed_circ
        q0: ───────●────────────────────●────T──────────X────T†────X──
                   │                    │               │          │
        q1: ───────┼──────────●─────────┼──────────●────●────T─────●──
                   │          │         │          │
        q2: ──H────X────T†────X────T────X────T†────X────T────H────────
    """
    _check_input_type('gate', XGate, gate)
    _check_control_num(gate.ctrl_qubits, 2)
    solutions = []
    c1 = Circuit()
    solutions.append(c1)
    q0 = gate.obj_qubits[0]
    q1 = gate.ctrl_qubits[0]
    q2 = gate.ctrl_qubits[1]
    c1 += G.H.on(q0)
    c1 += G.X.on(q0, q1)
    c1 += G.T.on(q0).hermitian()
    c1 += G.X.on(q0, q2)
    c1 += G.T.on(q0)
    c1 += G.X.on(q0, q1)
    c1 += G.T.on(q0).hermitian()
    c1 += G.X.on(q0, q2)
    c1 += G.T.on(q0)
    c1 += G.T.on(q1)
    c1 += G.X.on(q1, q2)
    c1 += G.H.on(q0)
    c1 += G.T.on(q2)
    c1 += G.T.on(q1).hermitian()
    c1 += G.X.on(q1, q2)
    return solutions


decompose_rules = ['ccx_decompose']
__all__ = decompose_rules
