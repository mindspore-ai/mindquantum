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
"""SWAP gate related decompose rule."""

from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.utils.type_value_check import _check_control_num, _check_input_type


def swap_decompose(gate: gates.SWAPGate):
    """
    Decompose :class:`~.core.gates.SWAPGate` gate.

    Args:
        gate (:class:`~.core.gates.SWAPGate`): A :class:`~.core.gates.SWAPGate` gate.

    Returns:
        List[:class:`~.core.circuit.Circuit`], All possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler.decompose import swap_decompose
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import SWAP
        >>> swap = SWAP.on([1, 0])
        >>> origin_circ = Circuit() + swap
        >>> decomposed_circ = swap_decompose(swap)[0]
        >>> origin_circ
        q0: ──╳───
              ┃
              ┃
        q1: ──╳───
        >>> decomposed_circ
              ┏━━━┓       ┏━━━┓
        q0: ──┨╺╋╸┠───■───┨╺╋╸┠───
              ┗━┳━┛   ┃   ┗━┳━┛
                ┃   ┏━┻━┓   ┃
        q1: ────■───┨╺╋╸┠───■─────
                    ┗━━━┛
    """
    _check_input_type('gate', gates.SWAPGate, gate)
    _check_control_num(gate.obj_qubits, 2)
    circuit = Circuit()
    q0 = gate.obj_qubits[0]
    q1 = gate.obj_qubits[1]
    circuit += gates.X.on(q1, q0)
    circuit += gates.X.on(q0, q1)
    circuit += gates.X.on(q1, q0)
    return [circuit]


def cswap_decompose(gate: gates.SWAPGate):
    """
    Decompose controlled :class:`~.core.gates.SWAPGate` gate.

    Args:
        gate (:class:`~.core.gates.SWAPGate`): a :class:`~.core.gates.SWAPGate` with
            one control qubit.

    Returns:
        List[:class:`~.core.circuit.Circuit`], all possible decompose solution.

    Examples:
        >>> from mindquantum.algorithm.compiler.decompose import cswap_decompose
        >>> from mindquantum.core import Circuit, SWAP
        >>> cswap = SWAP.on([1, 2], 0)
        >>> origin_circ = Circuit() + cswap
        >>> decomposed_circ = cswap_decompose(cswap)[0]
        >>> origin_circ
        q0: ──■───
              ┃
              ┃
        q1: ──╳───
              ┃
              ┃
        q2: ──╳───
        >>> decomposed_circ
        q0: ──────────■───────────
                      ┃
              ┏━━━┓   ┃   ┏━━━┓
        q1: ──┨╺╋╸┠───■───┨╺╋╸┠───
              ┗━┳━┛   ┃   ┗━┳━┛
                ┃   ┏━┻━┓   ┃
        q2: ────■───┨╺╋╸┠───■─────
                    ┗━━━┛
    """
    _check_input_type('gate', gates.SWAPGate, gate)
    _check_control_num(gate.ctrl_qubits, 1)
    circuit = Circuit()
    q0 = gate.ctrl_qubits[0]
    q1 = gate.obj_qubits[0]
    q2 = gate.obj_qubits[1]
    circuit += gates.X.on(q1, q2)
    circuit += gates.X.on(q2, [q0, q1])
    circuit += gates.X.on(q1, q2)
    return [circuit]


decompose_rules = ['swap_decompose', 'cswap_decompose']
__all__ = decompose_rules
