# -*- coding: utf-8 -*-
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
"""Bitphaseflip operator."""

from mindquantum.core.gates import Z
from mindquantum.core.circuit import Circuit


def bitphaseflip_operator(phase_inversion_qubit, n_qubits):
    """
    This operator could flip the phase of qubits that you set.

    Args:
        phase_inversion_qubit (list[int]): Qubits you want to flip phase.
        n_qubits (int): Total number of qubits.

    Examples:
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum import UN, H, Z
        >>> from mindquantum.algorithm.library import bitphaseflip_operator

        >>> circuit = Circuit()
        >>> circuit += UN(H, 3)
        >>> circuit += bitphaseflip_operator([2], 3)
        >>> circuit.get_qs()
        >>> print(circuit.get_qs(ket=True))
        √2/4¦000⟩
        √2/4¦001⟩
        -√2/4¦010⟩
        √2/4¦011⟩
        √2/4¦100⟩
        √2/4¦101⟩
        √2/4¦110⟩
        √2/4¦111⟩
    """
    s = [1 for i in range(1 << n_qubits)]
    for i in phase_inversion_qubit:
        s[i] = -1
    if s[0] == -1:
        for i in range(len(s)):
            s[i] = -1 * s[i]
    circuit = Circuit()
    length = len(s)
    cz = []
    for i in range(length):
        if s[i] == -1:
            cz.append([])
            current = i
            t = 0
            while current != 0:
                if (current & 1) == 1:
                    cz[-1].append(t)
                t += 1
                current = current >> 1
            for j in range(i + 1, length):
                if i & j == i:
                    s[j] = -1 * s[j]
    for i in cz:
        if i:
            if len(i) > 1:
                circuit += Z.on(i[-1], i[:-1])
            else:
                circuit += Z.on(i[0])
    return circuit
