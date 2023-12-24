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
"""CS gate related decompose rule."""

import numpy as np

from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.utils.type_value_check import _check_control_num, _check_input_type


def cs_decompose(gate: gates.SGate):
    """
    Decompose controlled :class:`~.core.gates.SGate` gate.

    Args:
        gate (:class:`~.core.gates.SGate`): a :class:`~.core.gates.SGate` with one control qubits.

    Returns:
        List[:class:`~.core.circuit.Circuit`], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler import cs_decompose
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import S
        >>> cs = S.on(1, 0)
        >>> origin_circ = Circuit() + cs
        >>> decomposed_circ = cs_decompose(cs)[0]
        >>> origin_circ
        q0: ────■─────
                ┃
              ┏━┻━┓
        q1: ──┨ S ┠───
              ┗━━━┛
        >>> decomposed_circ
              ┏━━━┓
        q0: ──┨ T ┠───■────────────■─────
              ┗━━━┛   ┃            ┃
              ┏━━━┓ ┏━┻━┓ ┏━━━━┓ ┏━┻━┓
        q1: ──┨ T ┠─┨╺╋╸┠─┨ T† ┠─┨╺╋╸┠───
              ┗━━━┛ ┗━━━┛ ┗━━━━┛ ┗━━━┛
    """
    _check_input_type('gate', gates.SGate, gate)
    _check_control_num(gate.ctrl_qubits, 1)
    circuit1 = Circuit()
    control = gate.ctrl_qubits[0]
    target = gate.obj_qubits[0]
    circuit1 += gates.T.on(control)
    circuit1 += gates.T.on(target)
    circuit1 += gates.X.on(target, control)
    circuit1 += gates.T.on(target).hermitian()
    circuit1 += gates.X.on(target, control)

    circuit2 = Circuit()
    circuit2 += gates.PhaseShift(np.pi / 4).on(control)
    circuit2 += gates.RZ(np.pi / 4).on(target)
    circuit2 += gates.X.on(target, control)
    circuit2 += gates.RZ(-np.pi / 4).on(target)
    circuit2 += gates.X.on(target, control)
    return [circuit1, circuit2]


decompose_rules = ['cs_decompose']
__all__ = decompose_rules
