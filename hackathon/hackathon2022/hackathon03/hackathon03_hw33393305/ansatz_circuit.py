import numpy as np
from mindquantum import Circuit, RY, X, BarrierGate, PhaseShift


def build_ansatz_a():
    ansatz = Circuit()
    ansatz += RY('alpha0').on(2)
    ansatz += X.on(2)
    ansatz += RY('alpha1').on(1, 2)
    ansatz += X.on(2)
    ansatz += RY('alpha2').on(1, 2)
    ansatz += X.on(1)
    ansatz += X.on(2)
    ansatz += RY('alpha3').on(0, [1, 2])
    ansatz += X.on(1)
    ansatz += RY('alpha4').on(0, [1, 2])
    ansatz += X.on(1)
    ansatz += X.on(2)
    ansatz += RY('alpha5').on(0, [1, 2])
    ansatz += X.on(1)
    ansatz += RY('alpha6').on(0, [1, 2])
    for i in range(8):
        x = bin(i)[2:].zfill(3)
        ansatz += BarrierGate(True)
        for j in reversed(range(3)):
            if x[j] == '0':
                ansatz += X.on(j)
        ansatz += PhaseShift(f'beta{i}').on(2, [0, 1])
        for j in reversed(range(3)):
            if x[j] == '0':
                ansatz += X.on(j)

    return ansatz

def build_ansatz_b():
    ansatz_1 = Circuit()
    for i in range(8):
        x = bin(i)[2:].zfill(3)
        for j in reversed(range(3)):
            if x[j] == '0':
                ansatz_1 += X.on(j)
        ansatz_1 += PhaseShift(f'gama{i}').on(2, [0, 1])
        for j in reversed(range(3)):
            if x[j] == '0':
                ansatz_1 += X.on(j)
        ansatz_1 += BarrierGate(True)

    ansatz_1 += RY('zeta6').on(0, [1, 2])
    ansatz_1 += X.on(1)
    ansatz_1 += RY('zeta5').on(0, [1, 2])
    ansatz_1 += X.on(2)
    ansatz_1 += X.on(1)
    ansatz_1 += RY('zeta4').on(0, [1, 2])
    ansatz_1 += X.on(1)
    ansatz_1 += RY('zeta3').on(0, [1, 2])
    ansatz_1 += X.on(2)
    ansatz_1 += X.on(1)
    ansatz_1 += RY('zeta2').on(1, 2)
    ansatz_1 += X.on(2)
    ansatz_1 += RY('zeta1').on(1, 2)
    ansatz_1 += X.on(2)
    ansatz_1 += RY('zeta0').on(2)
# ansatz.svg()


def ansatz0(p):
    ansatz = Circuit()
    ansatz += RY(f'alpha0_{p}').on(2)
    ansatz += X.on(2)
    ansatz += RY(f'alpha1_{p}').on(1, 2)
    ansatz += X.on(2)
    ansatz += RY(f'alpha2_{p}').on(1, 2)
    ansatz += X.on(1)
    ansatz += X.on(2)
    ansatz += RY(f'alpha3_{p}').on(0, [1, 2])
    ansatz += X.on(1)
    ansatz += RY(f'alpha4_{p}').on(0, [1, 2])
    ansatz += X.on(1)
    ansatz += X.on(2)
    ansatz += RY(f'alpha5_{p}').on(0, [1, 2])
    ansatz += X.on(1)
    ansatz += RY(f'alpha6_{p}').on(0, [1, 2])
    for i in range(8):
        x = bin(i)[2:].zfill(3)
        ansatz.barrier()
        for j in reversed(range(3)):
            if x[j] == '0':
                ansatz += X.on(2 - j)
        ansatz += PhaseShift(f'beta{i}_{p}').on(2, [0, 1])
        for j in reversed(range(3)):
            if x[j] == '0':
                ansatz += X.on(2 - j)
    return ansatz


def ansatz1(p):
    ansatz_1 = Circuit()
    for i in range(8):
        x = bin(i)[2:].zfill(3)
        for j in reversed(range(3)):
            if x[j] == '0':
                ansatz_1 += X.on(2 - j)
        ansatz_1 += PhaseShift(f'gama{i}_{p}').on(2, [0, 1])
        for j in reversed(range(3)):
            if x[j] == '0':
                ansatz_1 += X.on(2 - j)
        ansatz_1.barrier()

    ansatz_1 += RY(f'zeta6_{p}').on(0, [1, 2])
    ansatz_1 += X.on(1)
    ansatz_1 += RY(f'zeta5_{p}').on(0, [1, 2])
    ansatz_1 += X.on(2)
    ansatz_1 += X.on(1)
    ansatz_1 += RY(f'zeta4_{p}').on(0, [1, 2])
    ansatz_1 += X.on(1)
    ansatz_1 += RY(f'zeta3_{p}').on(0, [1, 2])
    ansatz_1 += X.on(2)
    ansatz_1 += X.on(1)
    ansatz_1 += RY(f'zeta2_{p}').on(1, 2)
    ansatz_1 += X.on(2)
    ansatz_1 += RY(f'zeta1_{p}').on(1, 2)
    ansatz_1 += X.on(2)
    ansatz_1 += RY(f'zeta0_{p}').on(2)
    return ansatz_1


ansatz = Circuit()
for i in range(6):
    ansatz += ansatz1(i) + ansatz0(i)