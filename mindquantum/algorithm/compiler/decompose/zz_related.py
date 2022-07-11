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
"""ZZ gate related decompose rule."""

from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.utils.type_value_check import _check_control_num, _check_input_type


def zz_decompose(gate: gates.ZZ):
    """
    Decompose zz gate.

    Args:
        gate (ZZ): a ZZ gate with one control qubits.

    Returns:
        List[Circuit], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler.decompose import zz_decompose
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import ZZ
        >>> zz = ZZ(1).on([0, 1])
        >>> origin_circ = Circuit() + zz
        >>> decomposed_circ = zz_decompose(zz)[0]
        >>> origin_circ
        q0: ──ZZ(1)──
                │
        q1: ──ZZ(1)──
        >>> decomposed_circ
        q0: ──●─────────────●──
              │             │
        q1: ──X────RZ(1)────X──
    """
    _check_input_type('gate', gates.ZZ, gate)
    _check_control_num(gate.ctrl_qubits, 0)
    circuit = Circuit()
    q0 = gate.obj_qubits[0]
    q1 = gate.obj_qubits[1]
    circuit += gates.X.on(q1, q0)
    circuit += gates.RZ(2 * gate.coeff).on(q1)
    circuit += gates.X.on(q1, q0)
    return [circuit]


decompose_rules = ['zz_decompose']
__all__ = decompose_rules
