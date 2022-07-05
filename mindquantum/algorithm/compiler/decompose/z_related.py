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
"""Z gate related decompose rule."""

from mindquantum.core import Circuit, gates
from mindquantum.utils.type_value_check import _check_control_num, _check_input_type


def cz_decompose(gate: gates.ZGate):
    """
    Decompose cz gate.

    Args:
        gate (ZGate): a ZGate with one control qubits.

    Returns:
        List[Circuit], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler.decompose import cz_decompose
        >>> from mindquantum.core import Circuit, Z
        >>> cz = Z.on(1, 0)
        >>> origin_circ = Circuit() + cz
        >>> decomposed_circ = cz_decompose(cz)[0]
        >>> origin_circ
        q0: ──●──
              │
        q1: ──Z──
        >>> decomposed_circ
        q0: ───────●───────
                   │
        q1: ──H────X────H──
    """
    _check_input_type('gate', gates.ZGate, gate)
    _check_control_num(gate.ctrl_qubits, 1)
    solutions = []
    c1 = Circuit()
    solutions.append(c1)
    q0 = gate.ctrl_qubits[0]
    q1 = gate.obj_qubits[0]
    c1 += gates.H.on(q1)
    c1 += gates.X.on(q1, q0)
    c1 += gates.H.on(q1)
    return solutions


decompose_rules = ['cz_decompose']
__all__ = decompose_rules
