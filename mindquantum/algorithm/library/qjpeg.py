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
"""QJPEG algorithm for quantum figure compression."""

from typing import Tuple, List
from mindquantum.algorithm.library.quantum_fourier import qft
from mindquantum.core.circuit import Circuit, dagger
from mindquantum.utils.type_value_check import _check_int_type, _check_value_should_not_less


def qjpeg(n_qubits: int, m_qubits: int) -> Tuple[Circuit, List[int], List[int]]:
    """
    Construct the circuit for compressing quantum figure with the QJPEG algorithm.

    Args:
        n_qubits (int): The number of qubits used to encode the quantum figure to be compressed.
        m_qubits (int): The number of qubits used to encode the compressed quantum figure.

    Note:
        The input arguments, n_qubits and m_qubits, should both be even, and the n_qubits must
        be not less than the m_qubits. Please refer to arXiv:2306.09323v2 for more information.

    Returns:
        - Circuit, The QJPEG circuit for quantum image compression
        - List[int], List of indices for remainder qubits that carry the compressed quantum image information
        - List[int], List of indices for discarded qubits

    Examples:
        >>> from mindquantum import Simulator, normalize
        >>> import numpy as np
        >>> n_qubits = 4
        >>> m_qubits = 2
        >>> circ, remainder_qubits, discard_qubits = qjpeg(n_qubits, m_qubits)
        >>> print(remainder_qubits, discard_qubits)
        [0, 2] [1, 3]
        >>> data = np.array([[1,0,0,0], [1,1,0,0], [1,1,1,0], [1,1,1,1]])
        >>> state = normalize(data.reshape(-1))
        >>> sim = Simulator('mqmatrix', n_qubits)
        >>> sim.set_qs(state)
        >>> sim.apply_circuit(circ)
        >>> rho = sim.get_partial_trace(discard_qubits)
        >>> sub_probs = rho.diagonal().real
        >>> new_data = sub_probs.reshape((2**(m_qubits//2), -1))
        >>> print(new_data)
        [[0.3, 0.],
         [0.4, 0.3]]
    """
    _check_int_type("n_qubits", n_qubits)
    _check_int_type("m_qubits", m_qubits)
    _check_value_should_not_less("n_qubits", 0, n_qubits)
    _check_value_should_not_less("m_qubits", 0, m_qubits)
    if n_qubits < m_qubits:
        raise ValueError("n_qubits should be not less than m_qubits.")
    if n_qubits % 2 != 0 or m_qubits % 2 != 0:
        raise ValueError("Both n_qubits and m_qubits should be even numbers.")

    half_diff = (n_qubits - m_qubits) // 2
    former_qubits = list(range(0, n_qubits // 2))
    latter_qubits = list(range(n_qubits // 2, n_qubits))
    mid_qubits = former_qubits[len(former_qubits) - half_diff :]
    last_qubits = latter_qubits[len(latter_qubits) - half_diff :]
    discard_qubits = mid_qubits + last_qubits
    remainder_qubits = list(set(range(n_qubits)).difference(discard_qubits))

    circ = Circuit()
    circ += qft(range(n_qubits))
    circ += dagger(qft(range(n_qubits - half_diff)))
    return circ, remainder_qubits, discard_qubits
