# Importing the necessary libraries
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

#Importing mindquantum libraries
# pylint: disable=W0104
from mindquantum import *
from mindquantum.simulator import Simulator


def p_left(q, phi):  #right projector
    qc = Circuit()
    n = q
    ctrl_range = list(range(0, n - 1))

    for qubit in range(n - 1):  # Implement a simple multi 0-controlled
        qc += X.on(qubit)
    qc += X.on(
        n - 1, ctrl_range
    )  # 0-Controlled on all but the last qubits, acts on the last qubit
    for qubit in range(n - 1):
        qc += X.on(qubit)

    qc += RZ(phi).on(n - 1)  # RZ(phi) on the last qubit

    for qubit in range(n - 1):  # Reverse the effect of the first multi-control
        qc += X.on(qubit)
    qc += X.on(n - 1, ctrl_range)
    for qubit in range(n - 1):
        qc += X.on(qubit)

    # p_left_gate = UnivMathGate('P_left',qc.matrix())
    # return p_left_gate
    return qc + BarrierGate(True)


def p_right(
        phi):  # Left projector acts just on the signal and the ancillary qubit
    qc = Circuit()
    qc += X.on(1, 0)
    qc += RZ(phi).on(1)
    qc += X.on(1, 0)

    return qc + BarrierGate(True)


def U(q):
    qc = Circuit()
    n = q + 1

    for qubit in range(n - 2):
        qc += H.on(qubit)

    qc += X.on(n - 2, list(range(0, n - 2)))

    return qc + BarrierGate(True)


def qsvt_search(target):  # target = marked element, is a bit-string!

    systemqubits = len(target)
    nqubits = systemqubits + 2
    circuit = Circuit()

    d = (2 * systemqubits) - 1

    if systemqubits > 6 and systemqubits < 10:
        for i in range(1, systemqubits - 6 + 1):
            d += 2 * i

    u = U(nqubits - 1)
    u_dag = u.hermitian() + BarrierGate(True)

    p_right_range = [nqubits - 2, nqubits - 1]
    u_range = list(range(0, nqubits - 1))
    p_left_range = list(range(0, nqubits))

    circuit += p_left(nqubits, (1 - d) * pi)
    circuit += u

    for i in range((d - 1) // 2):
        circuit += apply(p_right(pi), p_right_range)
        circuit += u_dag
        circuit += p_left(nqubits, pi)
        circuit += u

    for i in range(len(
            target)):  # The operation for acquiring arbitrary marked element
        bts = target[::
                     -1]  # bitstring is reversed to be compatible with the reverse qubit order in Qiskit
        if bts[i] == '0':
            circuit += X.on(i)

    circuit += BarrierGate(True) + UN(Measure(), circuit.n_qubits)
    return circuit


circuit_test = qsvt_search('10')
circuit_test

# pylint: disable=W0104
sim_test = Simulator('mqvector',
                     circuit_test.n_qubits)  # 使用mqvector模拟器，命名为sim_test
#sim_test.apply_circuit(circuit_test)                      # 通过模拟器sim_test运行搭建好的量子线路circuit_test

sim_test.reset()  # 重置模拟器sim_test维护好的量子态，使得初始化的量子态为|00>

result = sim_test.sampling(
    circuit_test, shots=1000)  # 通过模拟器sim_test对量子线路circuit_test进行1000次的采样
result
