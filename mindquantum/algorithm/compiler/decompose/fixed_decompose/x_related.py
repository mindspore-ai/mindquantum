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
"""X gate related decompose rule."""

from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.utils.type_value_check import _check_control_num, _check_input_type


def ccx_decompose(gate: gates.XGate):
    """
    Decompose a `toffoli` gate.

    Args:
        gate (:class:`~.core.gates.XGate`): a :class:`~.core.gates.XGate` with two control qubits.

    Returns:
        List[:class:`~.core.circuit.Circuit`], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler.decompose import ccx_decompose
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import X
        >>> ccx = X.on(2, [0,1])
        >>> origin_circ = Circuit() + ccx
        >>> decomposed_circ = ccx_decompose(ccx)[0]
        >>> origin_circ
        q0: ────■─────
                ┃
                ┃
        q1: ────■─────
                ┃
              ┏━┻━┓
        q2: ──┨╺╋╸┠───
              ┗━━━┛
        >>> decomposed_circ
                                                   ┏━━━┓        ┏━━━┓ ┏━━━━┓ ┏━━━┓
        q0: ──────────■────────────────────────■───┨ T ┠────────┨╺╋╸┠─┨ T† ┠─┨╺╋╸┠───
                      ┃                        ┃   ┗━━━┛        ┗━┳━┛ ┗━━━━┛ ┗━┳━┛
                      ┃                        ┃                  ┃   ┏━━━┓    ┃
        q1: ──────────╂────────────■───────────╂────────────■─────■───┨ T ┠────■─────
                      ┃            ┃           ┃            ┃         ┗━━━┛
              ┏━━━┓ ┏━┻━┓ ┏━━━━┓ ┏━┻━┓ ┏━━━┓ ┏━┻━┓ ┏━━━━┓ ┏━┻━┓ ┏━━━┓ ┏━━━┓
        q2: ──┨ H ┠─┨╺╋╸┠─┨ T† ┠─┨╺╋╸┠─┨ T ┠─┨╺╋╸┠─┨ T† ┠─┨╺╋╸┠─┨ T ┠─┨ H ┠──────────
              ┗━━━┛ ┗━━━┛ ┗━━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛
    """
    _check_input_type('gate', gates.XGate, gate)
    _check_control_num(gate.ctrl_qubits, 2)
    circuit1 = Circuit()
    q0 = gate.obj_qubits[0]
    control1 = gate.ctrl_qubits[0]
    control2 = gate.ctrl_qubits[1]
    circuit1 += gates.H.on(q0)
    circuit1 += gates.X.on(q0, control1)
    circuit1 += gates.T.on(q0).hermitian()
    circuit1 += gates.X.on(q0, control2)
    circuit1 += gates.T.on(q0)
    circuit1 += gates.X.on(q0, control1)
    circuit1 += gates.T.on(q0).hermitian()
    circuit1 += gates.X.on(q0, control2)
    circuit1 += gates.T.on(q0)
    circuit1 += gates.T.on(control1)
    circuit1 += gates.X.on(control1, control2)
    circuit1 += gates.H.on(q0)
    circuit1 += gates.T.on(control2)
    circuit1 += gates.T.on(control1).hermitian()
    circuit1 += gates.X.on(control1, control2)

    circuit2 = Circuit()
    circuit2 += gates.H.on(q0)
    circuit2 += gates.T.on(control2)
    circuit2 += gates.T.on(q0)
    circuit2 += gates.X.on(q0, control2)
    circuit2 += gates.T.on(q0).hermitian()
    circuit2 += gates.X.on(q0, control2)
    circuit2 += gates.H.on(q0)
    circuit2 += gates.X.on(control2, control1)
    circuit2 += gates.Z.on(q0)
    circuit2 += gates.S.on(control2).hermitian()
    circuit2 += gates.H.on(q0)
    circuit2 += gates.T.on(control2)
    circuit2 += gates.T.on(q0)
    circuit2 += gates.X.on(q0, control2)
    circuit2 += gates.T.on(q0).hermitian()
    circuit2 += gates.X.on(q0, control2)
    circuit2 += gates.H.on(q0)
    circuit2 += gates.X.on(control2, control1)
    circuit2 += gates.Z.on(q0)
    circuit2 += gates.H.on(q0)
    circuit2 += gates.T.on(control1)
    circuit2 += gates.T.on(q0)
    circuit2 += gates.X.on(q0, control1)
    circuit2 += gates.T.on(q0).hermitian()
    circuit2 += gates.X.on(q0, control1)
    circuit2 += gates.H.on(q0)
    return [circuit1, circuit2]


decompose_rules = ['ccx_decompose']
__all__ = decompose_rules
