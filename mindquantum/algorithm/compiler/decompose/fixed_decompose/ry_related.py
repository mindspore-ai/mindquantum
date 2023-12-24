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
"""RY gate related decompose rule."""

from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.utils.type_value_check import _check_control_num, _check_input_type


def cry_decompose(gate: gates.RY):
    """
    Decompose controlled :class:`~.core.gates.RY` gate.

    Args:
        gate (:class:`~.core.gates.RY`): A :class:`~.core.gates.RY` gate with one control qubits.

    Returns:
        List[:class:`~.core.circuit.Circuit`], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler import cry_decompose
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import RY
        >>> cry = RY(1).on(1, 0)
        >>> origin_circ = Circuit() + cry
        >>> decomposed_circ = cry_decompose(cry)[0]
        >>> origin_circ
        q0: ──────■───────
                  ┃
              ┏━━━┻━━━┓
        q1: ──┨ RY(1) ┠───
              ┗━━━━━━━┛
        >>> decomposed_circ
        q0: ────────────────■──────────────────■─────
                            ┃                  ┃
              ┏━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━┓ ┏━┻━┓
        q1: ──┨ RY(1/2) ┠─┨╺╋╸┠─┨ RY(-1/2) ┠─┨╺╋╸┠───
              ┗━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━┛ ┗━━━┛
    """
    _check_input_type('gate', gates.RY, gate)
    _check_control_num(gate.ctrl_qubits, 1)
    circuit = Circuit()
    control = gate.ctrl_qubits[0]
    target = gate.obj_qubits[0]
    circuit += gates.RY(gate.coeff / 2).on(target)
    circuit += gates.X.on(target, control)
    circuit += gates.RY(-gate.coeff / 2).on(target)
    circuit += gates.X.on(target, control)
    return [circuit]


decompose_rules = ['cry_decompose']
__all__ = decompose_rules
