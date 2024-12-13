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
"""Virtual Distillation Algorithm for error mitigation."""

from typing import Callable, Dict
from mindquantum.core.circuit import Circuit, apply
from mindquantum.core.gates import Givens, SWAP
import numpy as np


def virtual_distillation(
    circ: Circuit, executor: Callable[[Circuit], Dict[str, int]], little_endian: bool = True, **kwargs
) -> np.ndarray:
    """
    Error mitigation algorithm based on virtual distillation (arXiv:2011.07064).

    The algorithm calculates error-mitigated expectation values of Z_i Pauli operators
    for each qubit i. To measure expectation values of other Pauli operators (X_i or Y_i),
    appropriate basis rotation gates should be added at the end of the input circuit:

    - For X_i measurements: Add H gate on qubit i
    - For Y_i measurements: Add RX(π/2) gate on qubit i

    Args:
        circ (Circuit): The quantum circuit to be executed
        executor (Callable[[Circuit], dict[str, int]]): A callable object that executes quantum circuits
            and returns a dictionary mapping measurement result bitstrings to their counts.
            Note: executor must be able to handle twice the number of qubits as the input circuit
        little_endian (bool): Whether the bitstring returned by executor is little-endian.
            Default: True
        **kwargs: Additional arguments to be passed to the executor.

    Returns:
        np.ndarray, Error-mitigated expectation values <Z_i> for each qubit i. To obtain
        expectation values of other Pauli operators, add appropriate basis
        rotation gates to the input circuit before calling this function.

    Examples:
        >>> circ = Circuit([X.on(0), RY(1).on(1)])
        >>> sim = Simulator('mqvector', 4)  # Double number of qubits
        >>> def executor(circ):
        ...     res_dict = sim.sampling(circ, shots=10000).data
        ...     return res_dict
        >>> result = virtual_distillation(circ, executor)  # Returns <Z_i>
        >>> # To measure <X_1>:
        >>> circ_x = circ + H.on(1)
        >>> result_x = virtual_distillation(circ_x, executor)
    """
    circ_ = circ.remove_measure()
    num_qubits = circ_.n_qubits

    # Build extended circuit
    extended_circuit = Circuit()
    extended_circuit += apply(circ_, list(range(num_qubits)))
    extended_circuit += apply(circ_, [i + num_qubits for i in range(num_qubits)])

    # Add Givens rotations and SWAP gates
    for qubit_idx in range(num_qubits):
        extended_circuit += Givens(np.pi / 4).on([qubit_idx, qubit_idx + num_qubits])
        extended_circuit += SWAP([qubit_idx, qubit_idx + num_qubits])

    extended_circuit.measure_all()
    expectations = np.zeros(num_qubits)
    denominator = 0

    # Execute measurements and statistics
    counts_dict = executor(extended_circuit, **kwargs)

    # Parse measurement results
    for bitstring, count in counts_dict.items():
        if little_endian:
            z1 = np.array([1 if bit == '0' else -1 for bit in bitstring[num_qubits - 1 :: -1]])
            z2 = np.array([1 if bit == '0' else -1 for bit in bitstring[2 * num_qubits - 1 : num_qubits - 1 : -1]])
        else:
            z1 = np.array([1 if bit == '0' else -1 for bit in bitstring[:num_qubits]])
            z2 = np.array([1 if bit == '0' else -1 for bit in bitstring[num_qubits:]])

        # Calculate expectation values
        for i in range(num_qubits):
            parity = 1
            for j in range(num_qubits):
                if j != i:
                    parity *= (1 + z1[j] - z2[j] + z1[j] * z2[j]) / 2

            expectations[i] += (z1[i] + z2[i]) * parity * count / 2

        parity_i = (1 + z1[num_qubits - 1] - z2[num_qubits - 1] + z1[num_qubits - 1] * z2[num_qubits - 1]) / 2
        denominator += parity * parity_i * count

    return expectations / denominator
