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
"""H gate related decompose rule."""

from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.utils.type_value_check import _check_control_num, _check_input_type


def ch_decompose(gate: gates.HGate):
    """
    Decompose controlled :class:`~.core.gates.HGate` gate.

    Args:
        gate (:class:`~.core.gates.HGate`): a :class:`~.core.gates.HGate` with one control qubits.

    Returns:
        List[:class:`~.core.circuit.Circuit`], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler import ch_decompose
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import T, X, H, S
        >>> ch = H.on(1, 0)
        >>> origin_circ = Circuit() + ch
        >>> decomposed_circ = ch_decompose(ch)[0]
        >>> origin_circ
        q0: ────■─────
                ┃
              ┏━┻━┓
        q1: ──┨ H ┠───
              ┗━━━┛
        >>> decomposed_circ
        q0: ──────────────────────■─────────────────────────
                                  ┃
              ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━┻━┓ ┏━━━━┓ ┏━━━┓ ┏━━━━┓
        q1: ──┨ S ┠─┨ H ┠─┨ T ┠─┨╺╋╸┠─┨ T† ┠─┨ H ┠─┨ S† ┠───
              ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━━┛ ┗━━━┛ ┗━━━━┛
    """
    _check_input_type('gate', gates.HGate, gate)
    _check_control_num(gate.ctrl_qubits, 1)
    circuit = Circuit()
    control = gate.ctrl_qubits[0]
    target = gate.obj_qubits[0]
    circuit += gates.S.on(target)
    circuit += gates.H.on(target)
    circuit += gates.T.on(target)
    circuit += gates.X.on(target, control)
    circuit += gates.T.on(target).hermitian()
    circuit += gates.H.on(target)
    circuit += gates.S.on(target).hermitian()
    return [circuit]


decompose_rules = ['ch_decompose']
__all__ = decompose_rules
