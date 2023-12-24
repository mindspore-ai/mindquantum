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
"""RX gate related decompose rule."""

from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.utils.type_value_check import _check_control_num, _check_input_type


def crx_decompose(gate: gates.RX):
    """
    Decompose controlled :class:`~.core.gates.RX` gate.

    Args:
        gate (:class:`~.core.gates.RX`): a :class:`~.core.gates.RX` gate with one control qubits.

    Returns:
        List[:class:`~.core.circuit.Circuit`], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler import crx_decompose
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import RX
        >>> crx = RX(1).on(1, 0)
        >>> origin_circ = Circuit() + crx
        >>> decomposed_circ = crx_decompose(crx)[0]
        >>> origin_circ
        q0: ──────■───────
                  ┃
              ┏━━━┻━━━┓
        q1: ──┨ RX(1) ┠───
              ┗━━━━━━━┛
        >>> decomposed_circ
        q0: ──────────■──────────────────■────────────────────────
                      ┃                  ┃
              ┏━━━┓ ┏━┻━┓ ┏━━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━┓ ┏━━━━┓
        q1: ──┨ S ┠─┨╺╋╸┠─┨ RY(-1/2) ┠─┨╺╋╸┠─┨ RY(1/2) ┠─┨ S† ┠───
              ┗━━━┛ ┗━━━┛ ┗━━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━┛ ┗━━━━┛
    """
    _check_input_type('gate', gates.RX, gate)
    _check_control_num(gate.ctrl_qubits, 1)
    circuit = Circuit()
    control = gate.ctrl_qubits[0]
    target = gate.obj_qubits[0]
    circuit += gates.S.on(target)
    circuit += gates.X.on(target, control)
    circuit += gates.RY(-gate.coeff / 2).on(target)
    circuit += gates.X.on(target, control)
    circuit += gates.RY(gate.coeff / 2).on(target)
    circuit += gates.S.on(target).hermitian()
    return [circuit]


decompose_rules = ['crx_decompose']
__all__ = decompose_rules
