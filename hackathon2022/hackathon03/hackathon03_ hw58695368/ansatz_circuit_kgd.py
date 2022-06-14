# -*- coding: utf-8 -*-
"""
[1]Farrokh Vatan and Colin P. Williams. Realization of a general three-qubit quantum gate, 2004.

@NE
"""
from mindquantum import *
import numpy as np

def _g_a_zyz(prefix, n_qubits=3):
    circ = Circuit()
    for i in range(n_qubits):
        circ += RZ(f'{prefix}_{i*3}').on(i)
        circ += RY(f'{prefix}_{i*3+1}').on(i)
        circ += RZ(f'{prefix}_{i*3+2}').on(i)
    return circ
def _g_a_u(prefix):
    circ = Circuit()
    circ += RZ(- np.pi * .5).on(1)
    circ += H.on(2)
    circ += X.on(0, 1)
    circ += X.on(2, 1)
    circ += RY(f'{prefix}_0').on(1)
    circ += X.on(1, 0)
    circ += RY(f'{prefix}_1').on(1)
    circ += X.on(1, 0)
    circ += X.on(2, 1)
    circ += RZ(np.pi * .5).on(1)
    circ += H.on(2)
    circ += X.on(0, 1)
    circ += X.on(2, 0)
    circ += X.on(2, 1)
    circ += RZ(f'{prefix}_2').on(2)
    circ += X.on(2, 1)
    circ += X.on(2, 0)
    return circ
def _g_a_v(prefix):
    circ = Circuit()
    circ += X.on(0, 2)
    circ += X.on(1, 0)
    circ += X.on(1, 2)
    circ += RZ(- np.pi * .5).on(2)
    circ += RY(f'{prefix}_0').on(2)
    circ += X.on(2, 1)
    circ += RY(f'{prefix}_1').on(2)
    circ += X.on(2, 1)
    circ += RZ(np.pi * .5).on(2)
    circ += X.on(0, 2)
    circ += X.on(1, 0)
    circ += X.on(0, 1)
    circ += H.on(2)
    circ += X.on(2, 0)
    circ += RZ(f'{prefix}_2').on(2)
    circ += X.on(2, 0)
    circ += RZ(f'{prefix}_3').on(2)
    circ += H.on(2)
    circ += X.on(0, 1)
    return circ
def generate_ansatz():
    n_qubits = 3
    ansatz = Circuit()
    ansatz += _g_a_zyz('a0')
    ansatz += _g_a_u('a1')
    ansatz += _g_a_zyz('a2')
    ansatz += _g_a_v('a3')
    ansatz += _g_a_zyz('a4')
    ansatz += _g_a_u('a5')
    ansatz += _g_a_zyz('a6')
    #for i in range(n_qubits):
    #    ansatz += Circuit().phase_shift(f'a7', i)
    return ansatz, ansatz.params_name
