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
'''Amplitude encoder for quantum machine learning'''

import math
import numpy as np
from mindquantum.utils.type_value_check import _check_input_type
from mindquantum.core import Circuit, ParameterResolver, X, RY

def controlled_gate(circuit, gate, tqubit, cqubits, zero_qubit):
    '''
    Extended quantum controlled gate
    '''
    tmp = []
    for i in range(len(cqubits)):
        tmp.append(cqubits[i])
        if cqubits[i] < 0 or (cqubits[i] == 0 and zero_qubit == 0):
            circuit += X.on(abs(cqubits[i]))
            tmp[i] = -tmp[i]

    circuit += gate.on(tqubit, tmp)

    for i in range(len(cqubits)):
        if cqubits[i] < 0 or (cqubits[i] == 0 and zero_qubit == 0):
            circuit += X.on(abs(cqubits[i]))

def amplitude_encoder(x, n_qubits):
    '''
    Quantum circuit for amplitude encoding

    Note:
        the length of classic data ought to be the power of 2, otherwise will be filled up with 0
        the vector should be normalized

    Args:
        x (list[float] or numpy.array(list[float]): the vector of data you want to encode, which should be normalized
        n_qubits (int): the number of qubits of the encoder circuit

    Examples:
        >>> from mindquantum.algorithm.library import amplitude_encoder
        >>> from mindquantum.simulator import Simulator
        >>> sim = Simulator('projectq', 8)
        >>> encoder, parameterResolver = amplitude_encoder([0.5, 0.5, 0.5, 0.5], 8)
        >>> sim.apply_circuit(encoder, parameterResolver)
        >>> print(sim.get_qs(True))
        1/2¦00000000⟩
        1/2¦01000000⟩
        1/2¦10000000⟩
        1/2¦11000000⟩
        >>> sim.reset()
        >>> encoder, parameterResolver = amplitude_encoder([0, 0, 0.5, 0.5, 0.5, 0.5], 8)
        >>> sim.apply_circuit(encoder, parameterResolver)
        >>> print(sim.get_qs(True))
        1/2¦00100000⟩
        1/2¦01000000⟩
        1/2¦10100000⟩
        1/2¦11000000⟩
    '''
    _check_input_type('amplitude_encoder', (np.ndarray, list), x)
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if len(x) > 2 ** n_qubits:
        x = x[ : (2 ** n_qubits)]
    while 2 ** n_qubits != len(x):
        x.append(0)

    c = Circuit()
    tree = []
    for i in range(len(x) - 1):
        tree.append(0)
    for i in range(len(x)):
        tree.append(x[i])
    for i in range(len(x) - 2, -1 ,-1):
        tree[i] += math.sqrt(tree[i * 2 + 1] * tree[i * 2 + 1] + tree[i * 2 + 2] * tree[i * 2 + 2])

    path = [[]]
    num = {}
    cnt = 0
    for i in range(1, 2 * len(x) - 1, 2):
        path.append(path[(i - 1) // 2] + [-1])
        path.append(path[(i - 1) // 2] + [1])

        tmp = path[(i - 1) // 2]
        controls = []
        for j in range(len(tmp)):
            controls.append(tmp[j] * j)
        theta = 0
        if tree[(i - 1) // 2] > 1e-10:
            amp_0 = tree[i] / tree[(i - 1) // 2]
            theta = 2 * math.acos(amp_0)
        num[f'alpha{cnt}'] = theta
        controlled_gate(c, RY(f'alpha{cnt}'), len(tmp), controls, (0 if tmp and tmp[0] == -1 else 1))
        cnt += 1

    return c, ParameterResolver(num)
