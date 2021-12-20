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

def amplitude_encoder(x):
    '''
    Quantum circuit for amplitude encoding

    Note:
        the length of classic data ought to be the power of 2, otherwise will be filled up with 0
        the vector should be normalized

    Args:
        x (list[double] or numpy.array(list[double]): the vector of data you want to encode, which should be normalized

    Examples:
        >>> from mindquantum.algorithm.library import amplitude_encoder
        >>> from mindquantum.simulator import Simulator
        >>> sim = Simulator('projectq', 8)
        >>> encoder, parameterResolver = amplitude_encoder([0.5, 0.5, 0.5, 0.5])
        >>> sim.apply_circuit(encoder, parameterResolver)
        >>> print(sim.get_qs(True))
        1/4¦00000000⟩
        1/4¦00000001⟩
        1/4¦00000010⟩
        1/4¦00000011⟩
    '''
    _check_input_type('amplitude_encoder', (np.ndarray, list), x)
    while 2 ** int(math.log2(len(x))) != len(x):
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
