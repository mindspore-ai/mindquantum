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
"""Strongly entangling ansatz."""

from mindquantum.core.circuit import Circuit, add_prefix, add_suffix
from mindquantum.core.gates import BasicGate
from mindquantum.core.gates.basicgate import U3
from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_value_should_not_less,
)

from .._ansatz import Ansatz


class StronglyEntangling(Ansatz):  # pylint: disable=too-few-public-methods
    r"""
    Strongly entangling ansatz.

    Please refers `Circuit-centric quantum classifiers <https://arxiv.org/pdf/1804.00633.pdf>`_.

    Args:
        n_qubits (int): number of qubit that this ansatz act on.
        depth (int): the depth of ansatz.
        entangle_gate (BasicGate): a quantum gate to generate entanglement. If a single
            qubit gate is given, a control qubit will add, if a two qubits gate is given,
            the two qubits gate will act on different qubits.
        prefix (str): The prefix of parameters. Default: ``''``.
        suffix (str): The suffix of parameters. Default: ``''``.

    Examples:
        >>> from mindquantum.core.gates import X
        >>> from mindquantum.algorithm.nisq import StronglyEntangling
        >>> ansatz = StronglyEntangling(3, 2, X)
        >>> ansatz.circuit
              ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓             ┏━━━┓
        q0: ──┨ U3(θ=l0_a0, φ=l0_b0, λ=l0_c0) ┠───■─────────┨╺╋╸┠─↯─
              ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   ┃         ┗━┳━┛
              ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ ┏━┻━┓         ┃
        q1: ──┨ U3(θ=l0_a1, φ=l0_b1, λ=l0_c1) ┠─┨╺╋╸┠───■─────╂───↯─
              ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛ ┗━━━┛   ┃     ┃
              ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓       ┏━┻━┓   ┃
        q2: ──┨ U3(θ=l0_a2, φ=l0_b2, λ=l0_c2) ┠───────┨╺╋╸┠───■───↯─
              ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛       ┗━━━┛
              ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓       ┏━━━┓
        q0: ──┨ U3(θ=l1_a0, φ=l1_b0, λ=l1_c0) ┠───■───┨╺╋╸┠─────────
              ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   ┃   ┗━┳━┛
              ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓   ┃     ┃   ┏━━━┓
        q1: ──┨ U3(θ=l1_a1, φ=l1_b1, λ=l1_c1) ┠───╂─────■───┨╺╋╸┠───
              ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   ┃         ┗━┳━┛
              ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ ┏━┻━┓         ┃
        q2: ──┨ U3(θ=l1_a2, φ=l1_b2, λ=l1_c2) ┠─┨╺╋╸┠─────────■─────
              ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛ ┗━━━┛
    """

    # pylint: disable=too-many-arguments
    def __init__(self, n_qubits: int, depth: int, entangle_gate: BasicGate, prefix: str = '', suffix: str = ''):
        """Initialize a strongly entangling ansatz."""
        _check_int_type('n_qubits', n_qubits)
        _check_int_type('depth', depth)
        _check_value_should_not_less('n_qubits', 2, n_qubits)
        _check_value_should_not_less('depth', 1, depth)
        _check_input_type('prefix', str, prefix)
        _check_input_type('suffix', str, suffix)
        if not isinstance(entangle_gate, BasicGate) or entangle_gate.parameterized:
            raise ValueError(f"entangle gate requires a non parameterized gate, but get {entangle_gate}")
        m_dim = entangle_gate.matrix().shape[0]
        if m_dim == 2:
            self.gate_qubits = 1
        elif m_dim == 4:
            self.gate_qubits = 2
        else:
            raise ValueError(
                f"error gate, entangle_gate can only be one qubit or two qubits, but get dimension with {m_dim}"
            )
        self.prefix = prefix
        self.suffix = suffix
        super().__init__('Strongly Entangling', n_qubits, depth, entangle_gate)

    def _implement(self, depth, entangle_gate):  # pylint: disable=arguments-differ
        """Implement of strongly entangling ansatz."""
        rot_part_ansatz = Circuit([U3(f'a{i}', f'b{i}', f'c{i}').on(i) for i in range(self.n_qubits)])
        circ = Circuit()
        loop = 1
        for current_depth in range(depth):
            circ += add_prefix(rot_part_ansatz, f'l{current_depth}')
            if (current_depth + loop) % self.n_qubits == 0:
                loop += 1
            for idx in range(self.n_qubits):
                if self.gate_qubits == 1:
                    circ += entangle_gate.on((idx + current_depth + loop) % self.n_qubits, idx)
                else:
                    circ += entangle_gate.on([(idx + current_depth + loop) % self.n_qubits, idx])
        self._circuit = circ
        if self.prefix:
            self._circuit = add_prefix(self._circuit, self.prefix)
        if self.suffix:
            self._circuit = add_suffix(self._circuit, self.suffix)
