# -*- coding: utf-8 -*-
"""
[1] https://doi.org/10.48550/arXiv.2106.13304

@NoEvaa
"""
import numpy as np
from mindquantum import Circuit, add_prefix
from mindquantum import RY, Z, UN
def _g_a_mpo(L, layer1, layer2, barrier):
    """Generate ansatz MPO."""
    ansatz = Circuit()
    for i in range(L // 2):
        ansatz += add_prefix(layer1, f'a_l{i}')
        if barrier:
            ansatz += Circuit().barrier()
        ansatz += layer2[i % 2]
        if barrier:
            ansatz += Circuit().barrier()
    if L % 2:
        ansatz += add_prefix(layer1, f'a_l{i+1}')
    return ansatz
def generate_ansatz_mpo(n, L, barrier=False):
    """
    Parameterized quantum circuit of MPO(matrix product operators) between N qubits.

    Args:
        N (int): Number of qubits.
        L (int): Depth of circuit.
        barrier (bool): Whether to add barrier.

    Returns:
        Circuit, a quantum circuit.
    """
    if n < 3:
        raise ValueError('The number of qubits must be greater than 2.')
    layer1 = Circuit()
    for i in range(n):
        layer1 += RY(f't{i}').on(i)
    if L == 1:
        return layer1
    layer2 = [UN(Z, list(range(1,n,2)), list(range(0,n,2))), 
              UN(Z, list(range(2,n,2)), list(range(1,n,2)))]
    return _g_a_mpo(L, layer1, layer2, barrier)
def generate_ansatz_mpo_v(n, L, barrier=False):
    """
    Parameterized quantum circuit of MPO(matrix product operators) between N qubits.
    <VARIANT>

    Args:
        N (int): Number of qubits.
        L (int): Depth of circuit.
        barrier (bool): Whether to add barrier.

    Returns:
        Circuit, a quantum circuit.
    """
    if n < 3:
        raise ValueError('The number of qubits must be greater than 2.')
    if n % 2:
        raise ValueError('The number of qubits must be even.')
    layer1 = Circuit()
    for i in range(n):
        layer1 += RY(f't{i}').on(i)
    if L == 1:
        return layer1
    layer2 = [UN(Z, list(range(1,n,2)), list(range(0,n,2))), 
              UN(Z, list(range(2,n,2))+[0], list(range(1,n,2))+[n-1])]
    return _g_a_mpo(L, layer1, layer2, barrier)

def test():
    """test"""
    n = 7
    L = 11
    circ = generate_ansatz_mpo(n, L, 1)
    print(circ)
if __name__ == '__main__':
    test()
