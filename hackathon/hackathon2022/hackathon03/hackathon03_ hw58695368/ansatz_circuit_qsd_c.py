# -*- coding: utf-8 -*-
"""
Quantum Shannon Decomposition
[1]Constructive Quantum Shannon Decomposition from Cartan Involutions B. Drury, P. Love, arXiv:0806.4015.

@NE
"""
from mindquantum import *
import numpy as np
from utils import _pgate, _phasegate

def _generate_ansatz_u4_part1(prefix):
    n_qubits = 2
    ansatz = Circuit()
    for i in range(n_qubits):
        ansatz += _pgate(RY, f'{prefix}_{i*3}', np.pi, i)
        ansatz += _pgate(RZ, f'{prefix}_{i*3+1}', np.pi, i)
        ansatz += _pgate(RY, f'{prefix}_{i*3+2}', np.pi, i)
    return ansatz
def _generate_ansatz_u4_part2(prefix):
    n_qubits = 2
    ansatz = Circuit()
    ansatz += X.on(0, 1)
    ansatz += _pgate(RZ, f'{prefix}_0', np.pi, 0)
    ansatz += _pgate(RY, f'{prefix}_1', np.pi, 1)
    ansatz += X.on(1, 0)
    ansatz += _pgate(RY, f'{prefix}_2', np.pi, 1)
    ansatz += X.on(0, 1)
    return ansatz
def _generate_ansatz_u4(prefix):
    n_qubits = 2
    ansatz = Circuit()
    ansatz += _generate_ansatz_u4_part1(f'{prefix}_0')
    ansatz += _generate_ansatz_u4_part2(f'{prefix}_1')
    ansatz += _generate_ansatz_u4_part1(f'{prefix}_2')
    return ansatz
def _g_a_z(prefix):
    ansatz = Circuit()
    ansatz += _pgate(RZ, f'{prefix}_0', np.pi, 2)
    ansatz += X.on(2, 1)
    ansatz += _pgate(RZ, f'{prefix}_1', np.pi, 2)
    ansatz += X.on(2, 0)
    ansatz += _pgate(RZ, f'{prefix}_2', np.pi, 2)
    ansatz += X.on(2, 1)
    ansatz += _pgate(RZ, f'{prefix}_3', np.pi, 2)
    return ansatz
def _g_a_x(prefix):
    ansatz = Circuit()
    ansatz += _pgate(RX, f'{prefix}_0', np.pi, 2)
    ansatz += Z.on(2, 1)
    ansatz += _pgate(RX, f'{prefix}_1', np.pi, 2)
    ansatz += Z.on(2, 0)
    ansatz += _pgate(RX, f'{prefix}_2', np.pi, 2)
    ansatz += Z.on(2, 1)
    ansatz += _pgate(RX, f'{prefix}_3', np.pi, 2)
    ansatz += Z.on(2, 0)
    return ansatz

def generate_ansatz():
    n_qubits = 3
    ansatz = Circuit()
    ansatz += _generate_ansatz_u4('a0')
    ansatz += _g_a_z('a1')
    ansatz += _generate_ansatz_u4('a2')
    ansatz += _g_a_x('a3')
    ansatz += _generate_ansatz_u4('a4')
    ansatz += _g_a_z('a5')
    ansatz += _generate_ansatz_u4('a6')
    for i in range(n_qubits):
        ansatz += _phasegate('a7', np.pi, i)
    return ansatz, ansatz.params_name
