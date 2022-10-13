# -*- coding: utf-8 -*-
"""
[1] https://doi.org/10.48550/arXiv.2111.05076

@NoEvaa
"""
import numpy as np
from mindquantum import Circuit, ParameterResolver
from mindquantum import RX, RY, RZ, X
from mindquantum import XX, YY, ZZ
def _qga(qubits, prefix, p):
    """XYZ rotate gate between two qubits."""
    circ = Circuit()
    for i in range(2):
        circ += RX(prefix+f'_{i*3+p}').on(qubits[i])
        circ += RY(prefix+f'_{i*3+p+1}').on(qubits[i])
        circ += RZ(prefix+f'_{i*3+p+2}').on(qubits[i])
    return circ
def generate_ansata_conv(qubits, prefix):
    """Convolution between two qubits."""
    if len(qubits) != 2:
        raise
    prefix += '_c'
    circ = Circuit()
    circ += _qga(qubits, prefix, 0)
    circ += ZZ(prefix+f'_{6}').on(qubits)
    circ += YY(prefix+f'_{7}').on(qubits)
    circ += XX(prefix+f'_{8}').on(qubits)
    circ += _qga(qubits, prefix, 9)
    return circ
def generate_ansata_pool(qi, qio, prefix):
    """
    Pooling between two qubits.
    See Function `generate_qcnn_N2`.
    """
    if not isinstance(qi, int) or not isinstance(qio, int):
        raise
    if qi == qio:
        raise
    prefix += '_p'
    circ = Circuit()
    circ += _qga([qio, qi], prefix, 0)
    circ += X.on(qio, qi)
    circ += RZ({prefix+f'_{2}':-1}).on(qio)
    circ += RY({prefix+f'_{1}':-1}).on(qio)
    circ += RX({prefix+f'_{0}':-1}).on(qio)
    return circ
def generate_qcnn_N2(qi, qio, prefix):
    """
    QCNN between two qubits.

    Args:
        qi (list): Index of input qubit.
        qo (int): Index of input & output qubit.
        prefix (str): Layer prefix.

    Returns:
        Circuit, a quantum circuit.
    """
    if not isinstance(qi, int) or not isinstance(qio, int):
        raise
    if qi == qio:
        raise
    circ = Circuit()
    circ += generate_ansata_conv([qi, qio], prefix)
    circ += generate_ansata_pool(qi, qio, prefix)
    return circ
def _generate_qcnn_layer(circ, qlist, p):
    """
    Generate single layer of QCNN.

    ```
    qlist 0 1 2 3 4 5 6 7 8 9 10 11 12 13
    a	n   qo
    1	2   1
    2	3   1 2
    2	4   1 2
    3	5   1 2 4
    3	6   1 2 4
    4	7   1 2 5 6
    4	8   1 2 5 6
    5	9   1 2 5 6 8
    5	10  1 2 5 6 8
    6	11  1 2 5 6 9 10
    6	12  1 2 5 6 9 10
    7	13  1 2 5 6 9 10 12
    7	14  1 2 5 6 9 10 12
    ```
    """
    #print(qlist)
    n = len(qlist)
    #qlist = sorted(qlist)
    if n == 2:
        return circ + generate_qcnn_N2(qlist[0], qlist[1], f'l{p}_a0'), qlist[1]
    qo = []
    a = (n + 1) // 2
    for i in range(a - ((n % 2) | (a % 2))):
        qo.append(qlist[i*2+1-i%2])
        circ += generate_qcnn_N2(qlist[i*2+i%2], qo[-1], f'l{p}_a{i}')
    if n % 2:
        qo.append(qlist[-1])
    elif a % 2:
        qo.append(qlist[-2])
        circ += generate_qcnn_N2(qlist[-1], qo[-1], f'l{p}_a{i+1}')
    return _generate_qcnn_layer(circ, qo, p+1)
def generate_qcnn(N):
    """
    QCNN between N qubits.

    Args:
        N (int): Number of qubits.

    Returns:
        Circuit, a quantum circuit.
        Measurable qubit index.
    """
    if not isinstance(N, int) or N < 2:
        raise ValueError("Unreasonable parameter `N`.")
    circ = Circuit()
    return _generate_qcnn_layer(circ, list(range(N)), 0)

def _main():
    """test"""
    circ, qo = generate_qcnn(12)
    print(qo)
    print(circ)
if __name__ == '__main__':
    _main()
