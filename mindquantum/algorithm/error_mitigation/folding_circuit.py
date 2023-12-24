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
"""Method to folding circuit."""
import numpy as np

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import NoiseGate, QuantumGate
from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_value_should_not_less,
)


def _fold_globally(circ: Circuit, factor: float) -> Circuit:
    """Folding circuit globally."""
    _check_value_should_not_less("Fold factor", 1, factor)
    if circ.has_measure_gate or circ.is_noise_circuit:
        raise ValueError("For globally folding, circuit cannot has measurement or noise channel.")
    n_pair = int((factor - 1) // 2)
    n_random_factor = (factor - 1) / 2 % 1
    folded_circ = Circuit()
    folded_circ += circ
    circ_herm_circ = circ + circ.hermitian()
    for _ in range(n_pair):
        folded_circ += circ_herm_circ
    quantum_gate_poi = []
    for idx, g in enumerate(folded_circ):
        if isinstance(g, QuantumGate) and not isinstance(g, NoiseGate):
            quantum_gate_poi.append(idx)
    np.random.shuffle(quantum_gate_poi)
    n_random = int(n_random_factor * len(quantum_gate_poi) // (2 * n_pair + 1))
    random_choice = quantum_gate_poi[:n_random]
    random_choice = set(random_choice)
    new_fold = Circuit()
    for idx, g in enumerate(folded_circ):
        new_fold += g
        if idx in random_choice:
            new_fold += Circuit([g, g.hermitian()])
    return new_fold


def _fold_locally(circ: Circuit, factor: float) -> Circuit:
    """Folding circuit locally."""
    _check_value_should_not_less("Fold factor", 1, factor)
    n_pair = int((factor - 1) // 2)
    n_random_factor = (factor - 1) / 2 % 1
    folded_circ = Circuit()

    quantum_gate_poi = []
    for idx, g in enumerate(circ):
        if isinstance(g, QuantumGate) and not isinstance(g, NoiseGate):
            quantum_gate_poi.append(idx)
    np.random.shuffle(quantum_gate_poi)
    n_random = int(n_random_factor * len(quantum_gate_poi))
    random_choice = quantum_gate_poi[:n_random]
    random_choice = set(random_choice)
    quantum_gate_poi = set(quantum_gate_poi)
    n_pairs = []
    for i in range(len(circ)):
        p = 0
        if i in quantum_gate_poi:
            p += n_pair
        if i in random_choice:
            p += 1
        n_pairs.append(p)
    for idx, g in enumerate(circ):
        folded_circ += g
        if n_pairs[idx] != 0:
            folded_circ += Circuit([g, g.hermitian()] * n_pairs[idx])
    return folded_circ


def fold_at_random(circ: Circuit, factor: float, method='locally') -> Circuit:
    r"""
    Folding circuit randomly.

    Folding a quantum circuit is going to increase the size of quantum circuit,
    while keeping the unitary matrix of circuit the same. We can implement it by
    inserting identity circuit after certain gate. For a very simple example, :math:`RX(1.2 \pi)`
    has the same unitary matrix as :math:`RX(1.2 \pi)RX(-1.2 \pi)RX(1.2 \pi)`, but the size of circuit
    increase to 3.

    Args:
        circ (:class:`~.core.circuit.Circuit`): The quantum circuit that will folding.
        factor (float): Folding factor, should greater than 1.
        method (str): The method for folding. ``method`` should be one of ``'globally'`` or ``'locally'``.
            ``'globally'`` means we extended circuit is append to the end of circuit while ``'locally'``
            means the identity part is randomly added after certain gate.

    Examples:
        >>> from mindquantum.algorithm.error_mitigation import fold_at_random
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit().h(0).x(1, 0)
        >>> circ
              ┏━━━┓
        q0: ──┨ H ┠───■─────
              ┗━━━┛   ┃
                    ┏━┻━┓
        q1: ────────┨╺╋╸┠───
                    ┗━━━┛
        >>> fold_at_random(circ, 3)
              ┏━━━┓ ┏━━━┓ ┏━━━┓
        q0: ──┨ H ┠─┨ H ┠─┨ H ┠───■─────■─────■─────
              ┗━━━┛ ┗━━━┛ ┗━━━┛   ┃     ┃     ┃
                                ┏━┻━┓ ┏━┻━┓ ┏━┻━┓
        q1: ────────────────────┨╺╋╸┠─┨╺╋╸┠─┨╺╋╸┠───
                                ┗━━━┛ ┗━━━┛ ┗━━━┛
        >>> fold_at_random(circ, 3, 'globally')
              ┏━━━┓       ┏━━━┓             ┏━━━┓
        q0: ──┨ H ┠───■───┨ H ┠───■─────■───┨ H ┠───
              ┗━━━┛   ┃   ┗━━━┛   ┃     ┃   ┗━━━┛
                    ┏━┻━┓       ┏━┻━┓ ┏━┻━┓
        q1: ────────┨╺╋╸┠───────┨╺╋╸┠─┨╺╋╸┠─────────
                    ┗━━━┛       ┗━━━┛ ┗━━━┛
    """
    _check_value_should_not_less("Fold factor", 1, factor)
    _check_input_type("method", str, method)
    supported_method = ['globally', 'locally']
    if method not in supported_method:
        raise ValueError(f"method should be one of {supported_method}, but get {method}")
    if method == 'globally':
        return _fold_globally(circ, factor)
    return _fold_locally(circ, factor)
