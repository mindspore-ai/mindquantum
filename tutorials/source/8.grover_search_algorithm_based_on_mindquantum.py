# -*- coding: utf-8 -*-


def bitphaseflip_operator(phase_inversion_qubit, n_qubits):
    s = [1 for i in range(1 << n_qubits)]
    for i in phase_inversion_qubit:
        s[i] = -1
    if s[0] == -1:
        for i in range(len(s)):
            s[i] = -1 * s[i]
    circuit = Circuit()
    length = len(s)
    cz = []
    for i in range(length):
        if s[i] == -1:
            cz.append([])
            current = i
            t = 0
            while current != 0:
                if (current & 1) == 1:
                    cz[-1].append(t)
                t += 1
                current = current >> 1
            for j in range(i + 1, length):
                if i & j == i:
                    s[j] = -1 * s[j]
    for i in cz:
        if i:
            if len(i) > 1:
                circuit += Z.on(i[-1], i[:-1])
            else:
                circuit += Z.on(i[0])

    return circuit


import mindquantum as mq
from mindquantum import Circuit, UN, H, Z
from mindquantum.simulator import Simulator

n_qubits = 3
sim = Simulator('projectq', n_qubits)

circuit = Circuit()
circuit += UN(H, n_qubits)

sim.apply_circuit(circuit)

circuit

print(sim.get_qs(True))

sim.reset()

phase_inversion_qubit = [4]
operator = bitphaseflip_operator(phase_inversion_qubit, n_qubits)

circuit += operator

sim.apply_circuit(circuit)

circuit

print(sim.get_qs(True))
print(int('100', 2))

n_qubits = 3
sim1 = Simulator('projectq', n_qubits)

operator1 = bitphaseflip_operator([i for i in range(1, pow(2, n_qubits))],
                                  n_qubits)

circuit1 = Circuit()
circuit1 += UN(H, n_qubits)
circuit1 += operator1

sim1.apply_circuit(circuit1)

circuit1

print(sim1.get_qs(True))


def G(phase_inversion_qubit, n_qubits):
    operator = bitphaseflip_operator(phase_inversion_qubit, n_qubits)
    operator += UN(H, n_qubits)
    operator += bitphaseflip_operator([i for i in range(1, pow(2, n_qubits))],
                                      n_qubits)
    operator += UN(H, n_qubits)
    return operator


import numpy as np
from numpy import pi, sqrt

n_qubits = 3
phase_inversion_qubit = [2]

N = 2**(n_qubits)
M = len(phase_inversion_qubit)

r = int(pi / 4 * sqrt(N / M))

sim2 = Simulator('projectq', n_qubits)

circuit2 = Circuit()
circuit2 += UN(H, n_qubits)

for i in range(r):
    circuit2 += G(phase_inversion_qubit, n_qubits)

sim2.apply_circuit(circuit2)

circuit2

print(sim2.get_qs(True))
from mindquantum import Measure

sim2.reset()

circuit2 += UN(Measure(), circuit2.n_qubits)

result = sim2.sampling(circuit2, shots=1000)
result

print(int('010', 2))

n_qubits = 5
phase_inversion_qubit = [5, 11]

N = 2**(n_qubits)
M = len(phase_inversion_qubit)

r = int(pi / 4 * sqrt(N / M))

sim3 = Simulator('projectq', n_qubits)

circuit3 = Circuit()
circuit3 += UN(H, n_qubits)

for i in range(r):
    circuit3 += G(phase_inversion_qubit, n_qubits)

sim3.apply_circuit(circuit3)

circuit3

print(sim3.get_qs(True))

sim3.reset()

circuit3 += UN(Measure(), circuit3.n_qubits)

result1 = sim3.sampling(circuit3, shots=1000)
print(result1)

print(int('00101', 2))
print(int('01011', 2))
