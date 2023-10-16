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
"""Amplitude encoder for quantum machine learning."""

import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RY, RZ, X
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.utils.type_value_check import _check_input_type
from mindquantum.utils import normalize

def amp_circuit(n_qubits):
    '''Construct the quantum circuit of the amplitude_encoder.'''
    circ = Circuit()
    circ += RY(f'alpha_{n_qubits}0').on(n_qubits-1)
    circ += RZ(f'lambda_{n_qubits}0').on(n_qubits-1)
    for i in range(1, n_qubits):
        for j in range(int(2**i)):
            string = bin(j)[2:].zfill(i)
            for tem_qubit, bit in enumerate(string):
                qubit = int(n_qubits - tem_qubit)
                if bit == '0':
                    circ += X.on(qubit-1)

            circ += RY(f'alpha_{int(n_qubits-i)}{j}').on(int(n_qubits-1-i), list(range(n_qubits-i, n_qubits)))
            circ += RZ(f'lambda_{int(n_qubits-i)}{j}').on(int(n_qubits-1-i), list(range(n_qubits-i, n_qubits)))
            string = bin(j)[2:].zfill(i)

            for tem_qubit, bit in enumerate(string):
                qubit = int(n_qubits - tem_qubit)
                if bit == '0':
                    circ += X.on(qubit-1)
    return circ

# pylint: disable=too-many-locals
def amplitude_encoder(x, n_qubits):
    """
    Quantum circuit for amplitude encoding.

    Note:
        The length of classic data ought to be the power of 2, otherwise will be filled up with 0.
        The vector should be normalized.

    Args:
        x (list[float] or numpy.array(list[float])): the vector of data you want to encode, which should be normalized.
        n_qubits (int): the number of qubits of the encoder circuit.

    Returns:
        Circuit, the parameterized quantum circuit that do amplitude encoder.
        ParameterResolver, the parameter for parameterized quantum circuit to do
        amplitude encoder.

    Examples:
        >>> from mindquantum.algorithm.library import amplitude_encoder
        >>> from mindquantum.simulator import Simulator
        >>> sim = Simulator('mqvector', 2)
        >>> encoder, parameterResolver = amplitude_encoder([0.5, -0.5, -0.5j, -0.5j], 2)
        >>> sim.apply_circuit(encoder, parameterResolver)
        >>> print(sim.get_qs(True))
        1/2¦00⟩
        -1/2¦01⟩
        -1/2j¦10⟩
        -1/2j¦11⟩
        >>> sim.reset()
        >>> encoder, parameterResolver = amplitude_encoder([0, -0.5j, -0.5j, -0.5, 0.5], 3)
        >>> sim = Simulator('mqvector', 3)
        >>> sim.apply_circuit(encoder, parameterResolver)
        >>> print(sim.get_qs(True))
        -1/2j¦001⟩
        -1/2j¦010⟩
        -1/2¦011⟩
        1/2¦100⟩
    """
    _check_input_type('amplitude_encoder', (np.ndarray, list), x)
    _check_input_type('n_qubits', (int), n_qubits)
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if len(x) > 2**n_qubits:
        x = x[: (2**n_qubits)]
    while 2**n_qubits != len(x):
        x.append(0)

    vec = normalize(x)
    eta = [[]]
    for i in range(len(vec)):
        eta[0].append(abs(vec[i]))

    for v in range(1, n_qubits+1):
        eta.append([])
        for i in range(int(2**(n_qubits-v))):
            eta[v].append(np.sqrt(eta[v-1][2*i]**2 + eta[v-1][2*i+1]**2))

    omega_0 = []
    for i in range(len(vec)):
        omega_0.append(np.angle(vec[i]))

    omega = [[]]
    for i in range(len(vec)):
        omega[0].append(2*np.angle(vec[i]))

    for v in range(1, n_qubits+1):
        omega.append([])
        for i in range(2**(n_qubits-v)):
            omega[v].append(0.)
            for j in range(2**v):
                omega[v][i] += omega_0[i*2**v+j]/2**(v-1)

    alphas = {}
    for v in range(n_qubits, 0, -1):
        for i in range(2**(n_qubits-v)):
            if eta[v][i] < 1e-6:
                alphas[f'alpha_{v}{i}'] = 0
            else:
                alphas[f'alpha_{v}{i}'] = 2*np.arcsin(eta[v-1][2*i+1]/eta[v][i])

    lambs = {}
    for v in range(n_qubits, 0, -1):
        for i in range(2**(n_qubits-v)):
            lambs[f'lambda_{v}{i}'] = omega[v-1][2*i+1] - omega[v][i]

    return amp_circuit(n_qubits), ParameterResolver({**alphas, **lambs})
