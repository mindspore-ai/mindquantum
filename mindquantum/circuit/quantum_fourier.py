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
"""Quantum fourier transform."""

import numpy as np
from mindquantum.gate import H, PhaseShift
from mindquantum.circuit import SwapParts, Circuit


def _rn(k):
    return PhaseShift(2 * np.pi / 2**k)


def _qft_unit(qubits):
    circ = Circuit(H.on(qubits[0]))
    for index, ctrl_qubit in enumerate(qubits[1:]):
        circ += _rn(index + 2).on(qubits[0], ctrl_qubit)
    return circ


def qft(qubits):
    """
    Quantum fourier transform.

    Note:
        Please refer Nielsen, M., & Chuang, I. (2010) for more information.

    Args:
        qubits (list[int]): Qubits you want to apply quantum fourier transform.

    Examples:
        >>> from mindquantum.circuit import qft
        >>> from mindquantum.circuit import StateEvolution
        >>> print(StateEvolution(qft([0, 1])).final_state(ket=True))
        0.5¦00⟩
        0.5¦01⟩
        0.5¦10⟩
        0.5¦11⟩
    """
    c = Circuit()
    n_qubits = len(qubits)
    for i in range(n_qubits):
        c += _qft_unit(qubits[i:])
    if n_qubits > 1:
        part1 = []
        part2 = []
        for j in range(n_qubits // 2):
            part1.append(qubits[j])
            part2.append(qubits[n_qubits - 1 - j])
        c += SwapParts(part1, part2)
    return c
