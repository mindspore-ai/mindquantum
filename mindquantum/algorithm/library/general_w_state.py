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
"""General W State."""

import numpy as np
from mindquantum.core.gates import X, RY
from mindquantum.core.circuit import Circuit
from mindquantum.utils.type_value_check import _check_input_type


def general_w_state(qubits):
    """
    General W State.

    Note:
        Please refer https://quantumcomputing.stackexchange.com/questions/4350/general-construction-of-w-n-state.

    Args:
        qubits (list[int]): Qubits you want to apply general W state.

    Examples:
        >>> from mindquantum.algorithm.library import general_w_state
        >>> print(general_w_state(range(3)).get_qs(ket=True))
        0.5773502691896257¦001⟩
        0.5773502691896258¦010⟩
        0.5773502691896257¦100⟩

    Returns:
        Circuit, circuit that can prepare w state.
    """
    _check_input_type('qubits', (list, range), qubits)
    circuit = Circuit()

    for i in range(len(qubits) - 1):
        angle_val = 2 * np.arccos(np.sqrt(1 / (len(qubits) - i)))

        if i == 0:
            circuit += RY(angle_val).on(qubits[i])
        else:
            circuit += RY(angle_val).on(qubits[i], qubits[i - 1])

    for j in reversed(range(len(qubits) - 1)):
        circuit += X.on(qubits[j + 1], qubits[j])

        if j == 0:
            circuit += X.on(qubits[j])

    return circuit
