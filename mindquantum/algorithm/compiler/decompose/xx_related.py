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

"""XX gate related decompose rule."""

from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.utils.type_value_check import _check_input_type  # , _check_control_num


def _check_control_num(ctrl_qubits, require_n):
    if len(ctrl_qubits) != require_n:
        raise RuntimeError(f"requires {(require_n,'control qubit')}, but get {len(ctrl_qubits)}")


def xx_decompose(gate: gates.XX):
    """
    Decompose XX gate.

    Args:
        gate (XX): a XX gate.

    Returns:
        List[Circuit], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler.decompose import xx_decompose
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import XX
        >>> xx = XX(1).on([0, 1])
        >>> origin_circ = Circuit() + xx
        >>> decomposed_circ = xx_decompose(xx)[0]
        >>> origin_circ
        q0: ──XX(1)──
                │
        q1: ──XX(1)──
        >>> decomposed_circ
        q0: ──H────●─────────────●────H──
                   │             │
        q1: ──H────X────RZ(2)────X────H──
    """
    _check_input_type('gate', gates.XX, gate)
    _check_control_num(gate.ctrl_qubits, 0)
    return cxx_decompose(gate)


def cxx_decompose(gate: gates.XX):
    """
    Decompose xx gate with control qubits.

    Args:
        gate (XX): a XX gate.

    Returns:
        List[Circuit], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler.decompose import cxx_decompose
        >>> from mindquantum.core import Circuit, XX
        >>> cxx = XX(2).on([0, 1], [2, 3])
        >>> origin_circ = Circuit() + cxx
        >>> decomposed_circ = cxx_decompose(cxx)[0]
        >>> origin_circ
        q0: ──XX(2)──
                │
        q1: ──XX(2)──
                │
        q2: ────●────
                │
        q3: ────●────
        >>> decomposed_circ
        q0: ──H─────────●─────────────●─────────H──
              │         │             │         │
        q1: ──┼────H────X────RZ(4)────X────H────┼──
              │    │    │      │      │    │    │
        q2: ──●────●────●──────●──────●────●────●──
              │    │    │      │      │    │    │
        q3: ──●────●────●──────●──────●────●────●──
    """
    _check_input_type('gate', gates.XX, gate)
    q0 = gate.obj_qubits[0]
    q1 = gate.obj_qubits[1]

    circuit1 = Circuit()
    circuit1 += gates.H.on(q0, gate.ctrl_qubits)
    circuit1 += gates.H.on(q1, gate.ctrl_qubits)
    circuit1 += gates.X.on(q1, [q0] + gate.ctrl_qubits)
    circuit1 += gates.RZ(2 * gate.coeff).on(q1, gate.ctrl_qubits)
    circuit1 += circuit1[:-1][::-1]

    circuit2 = Circuit()
    circuit2 += gates.H.on(q0, gate.ctrl_qubits)
    circuit2 += gates.H.on(q1, gate.ctrl_qubits)
    circuit2 += gates.X.on(q0, [q1] + gate.ctrl_qubits)
    circuit2 += gates.RZ(2 * gate.coeff).on(q0, gate.ctrl_qubits)
    circuit2 += circuit2[:-1][::-1]

    return [circuit1, circuit2]


decompose_rules = ['xx_decompose', 'cxx_decompose']
__all__ = decompose_rules
