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

# pylint: disable=duplicate-code

"""Bitphaseflip operator."""

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import Z
from mindquantum.utils.type_value_check import _check_input_type


def bitphaseflip_operator(phase_inversion_index, n_qubits):  # pylint: disable=too-many-branches
    """
    Generate a circuit that can flip the sign of any calculation bases.

    Args:
        phase_inversion_index (list[int]): Index of calculation bases want to flip phase.
        n_qubits (int): Total number of qubits.

    Examples:
        >>> from mindquantum.core.circuit import Circuit, UN
        >>> from mindquantum.core.gates import H, Z
        >>> from mindquantum.algorithm.library import bitphaseflip_operator
        >>> circuit = Circuit()
        >>> circuit += UN(H, 3)
        >>> circuit += bitphaseflip_operator([1, 3], 3)
        >>> print(circuit.get_qs(ket=True))
        √2/4¦000⟩
        -√2/4¦001⟩
        √2/4¦010⟩
        -√2/4¦011⟩
        √2/4¦100⟩
        √2/4¦101⟩
        √2/4¦110⟩
        √2/4¦111⟩

    Returns:
        Circuit, the bit phase flip circuit.
    """
    _check_input_type('n_qubits', int, n_qubits)
    _check_input_type('phase_inversion_index', (list, range), phase_inversion_index)
    state = [1 for i in range(1 << n_qubits)]
    for i in phase_inversion_index:
        state[i] = -1
    if state[0] == -1:
        for i, state_i in enumerate(state):
            state[i] = -1 * state_i
    circuit = Circuit()
    length = len(state)
    cz_list = []
    for i in range(length):
        if state[i] == -1:
            cz_list.append([])
            current = i
            qubit_id = 0
            while current != 0:
                if (current & 1) == 1:
                    cz_list[-1].append(qubit_id)
                qubit_id += 1
                current = current >> 1
            for j in range(i + 1, length):
                if i & j == i:
                    state[j] = -1 * state[j]
    for i in cz_list:
        if i:
            if len(i) > 1:
                circuit += Z.on(i[-1], i[:-1])
            else:
                circuit += Z.on(i[0])
    return circuit
