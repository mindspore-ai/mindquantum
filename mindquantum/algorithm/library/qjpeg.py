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
"""QJPEG algorithm for figure compression."""

import numpy as np
from mindquantum.utils import normalize
from mindquantum.algorithm import qft
from mindquantum.core.circuit import Circuit, dagger
from mindquantum.simulator import Simulator

def _scale_to_interval(data, new_min, new_max):
    """Scale the data into the specified interval."""
    old_min = np.min(data)
    old_max = np.max(data)
    return new_min + (new_max - new_min) * (data - old_min) / (old_max - old_min)

def qjpeg(data:np.ndarray, n_qubits:int, m_qubits:int) -> np.ndarray:
    """Compress the input square array with QJPEG algorithm.

    Note: Please refer to arXiv:2306.09323v2 for more information.

    Args:
        data (np.ndarray): An numpy array with square shape.
        n_qubits (int): The number of qubits used to encode the data to be compressed.
        m_qubits (int): The number of qubits used to encode the compressed data.

    Examples:
    ```python
        >>> data = np.array([[1,0,0,0], [1,1,0,0], [1,1,1,0], [1,1,1,1]])
        >>> n_qubits = 4
        >>> m_qubits = 2
        >>> qjpeg(data, n_qubits, m_qubits)
            array([[0.75, 0.  ],
                    [1.  , 0.75]])
    ```
    """
    origin_data = data.reshape(-1)
    if not isinstance(n_qubits, int) or not isinstance(m_qubits, int):
        raise ValueError("n_qubits and m_qubits should be positive int.")
    if n_qubits <= 0 or m_qubits <= 0:
        raise ValueError("n_qubits and m_qubits should be positive int.")
    if n_qubits <= m_qubits:
        raise ValueError("m_qubits should little than n_qubits.")
    if np.log2(origin_data.shape[0]) != n_qubits:
        raise ValueError("the pixel number of the input figure should equal to 2**n_qubits.")
    if (n_qubits - m_qubits) % 2 != 0:
        raise ValueError("the different between n_qubits and m_qubits should be even.")

    half_diff = (n_qubits - m_qubits) // 2 # Number of qubits will be discarded at a time
    former_qubits = list(range(0, n_qubits//2)) # Qubits in the first half
    latter_qubits = list(range(n_qubits//2, n_qubits)) # Qubits in the second half
    mid_qubits = former_qubits[len(former_qubits) - half_diff:] # Qubits to be discarded in the first half
    last_qubits = latter_qubits[len(latter_qubits) - half_diff:] # Qubits to be discarded in the second half

    state = normalize(origin_data)
    sim = Simulator('mqmatrix', n_qubits)
    sim.set_qs(state)

    circ = Circuit()
    circ += qft(range(n_qubits))
    circ += dagger(qft(range(n_qubits-half_diff)))
    sim.apply_circuit(circ)
    rho = sim.get_partial_trace(mid_qubits + last_qubits)
    sub_pros = rho.diagonal().real
    temp_data = sub_pros.reshape((2**(m_qubits//2), -1))
    new_data = _scale_to_interval(temp_data, np.min(origin_data), np.max(origin_data))
    return new_data
