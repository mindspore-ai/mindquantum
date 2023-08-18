# -*- coding: utf-8 -*-
"""
@NE
<Targeted>
"""
from mindquantum import *
import numpy as np
from utils import _pgate

def generate_ansatz():
    n_qubits = 3
    ansatz = Circuit()
    ansatz += X.on(0, 1)
    ansatz += X.on(1, 0)
    ansatz += X.on(0, 1)
    for i in range(n_qubits):
        ansatz += _pgate(RZ, f'a0_{i}', np.pi, i)
        ansatz += _pgate(RY, f'a1_{i}', np.pi, i)
        ansatz += _pgate(RZ, f'a2_{i}', np.pi, i)
    return ansatz, ansatz.params_name
