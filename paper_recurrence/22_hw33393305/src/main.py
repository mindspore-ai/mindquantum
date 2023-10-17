import os
os.environ['OMP_NUM_THREADS'] = '2'

from mindquantum import *
import numpy as np
from numpy import random


def H_encoder(tar_qubit):
    # 初始化量子线路
    circuit = Circuit()

    # 随机生成两个01比特串作为密钥
    j, k = random.randint(0, 2, 2)

    # 根据随机生成的密钥，对量子态进行加密操作
    if j == 1:
        circuit += X.on(tar_qubit)
    if k == 1:
        circuit += Z.on(tar_qubit)

    # 执行期望的量子门
    circuit += BarrierGate(True)
    circuit += H.on(tar_qubit)
    circuit += BarrierGate(True)

    # 根据随机生成的密钥，对量子态进行相应的解密操作
    if k == 1:
        circuit += X.on(tar_qubit)
    if j == 1:
        circuit += Z.on(tar_qubit)
    return circuit


def Fredkin_encoder(tar_qubits, ctrl_qubit):
    # 初始化量子线路
    circuit = Circuit()

    # 随机生成六个01比特串作为密钥
    j, k, l, m, p, q = random.randint(0, 2, 6)

    # 根据随机生成的密钥，对量子态进行加密操作
    if j == 1:
        circuit += X.on(ctrl_qubit)
    if k == 1:
        circuit += Z.on(ctrl_qubit)

    circuit += BARRIER

    if l == 1:
        circuit += X.on(tar_qubits[0])
    if m == 1:
        circuit += Z.on(tar_qubits[0])

    circuit += BARRIER

    if p == 1:
        circuit += X.on(tar_qubits[1])
    if q == 1:
        circuit += Z.on(tar_qubits[1])

    circuit += BARRIER

    # 期望执行的量子门
    circuit += BarrierGate(True)
    circuit += SWAP.on(tar_qubits, ctrl_qubit)
    circuit += BarrierGate(True)

    # 根据随机生成的密钥，对量子态进行相应的解密操作
    if q == 1:
        circuit += Z.on(tar_qubits[1])
        circuit += BARRIER
        circuit += Z.on(tar_qubits[0], ctrl_qubit)
        circuit += BARRIER
        circuit += Z.on(tar_qubits[1], ctrl_qubit)
        circuit += BARRIER
    if p == 1:
        circuit += X.on(tar_qubits[1])
        circuit += BARRIER
        circuit += X.on(tar_qubits[0], ctrl_qubit)
        circuit += BARRIER
        circuit += X.on(tar_qubits[1], ctrl_qubit)
        circuit += BARRIER
    if m == 1:
        circuit += Z.on(tar_qubits[0])
        circuit += BARRIER
        circuit += Z.on(tar_qubits[0], ctrl_qubit)
        circuit += BARRIER
        circuit += Z.on(tar_qubits[1], ctrl_qubit)
        circuit += BARRIER
    if l == 1:
        circuit += X.on(tar_qubits[0])
        circuit += BARRIER
        circuit += X.on(tar_qubits[0], ctrl_qubit)
        circuit += BARRIER
        circuit += X.on(tar_qubits[1], ctrl_qubit)
        circuit += BARRIER
    if k == 1:
        circuit += Z.on(ctrl_qubit)
        circuit += BARRIER
    if j == 1:
        circuit += X.on(ctrl_qubit)
        circuit += BARRIER
        circuit += SWAP.on([tar_qubits[0], tar_qubits[1]])
        circuit += BARRIER
    return circuit


def CNOT_encoder(tar_qubit, ctrl_qubit):
    # 初始化量子线路
    circuit = Circuit()

    # 随机生成四个01比特串作为密钥
    j, k, l, m = random.randint(0, 2, 4)

    # 根据随机生成的密钥，对量子态进行加密操作
    if j == 1:
        circuit += X.on(ctrl_qubit)
    if k == 1:
        circuit += Z.on(ctrl_qubit)

    circuit += BARRIER

    if l == 1:
        circuit += X.on(tar_qubit)
    if m == 1:
        circuit += Z.on(tar_qubit)

    circuit += BARRIER

    # 期望执行的量子门
    circuit += BarrierGate(True)
    circuit += X.on(tar_qubit, ctrl_qubit)
    circuit += BarrierGate(True)

    # 根据随机生成的密钥，对量子态进行相应的解密操作
    if m == 1:
        circuit += Z.on(tar_qubit)
        circuit += BARRIER
        circuit += Z.on(ctrl_qubit)
        circuit += BARRIER
    if l == 1:
        circuit += X.on(tar_qubit)
        circuit += BARRIER
    if k == 1:
        circuit += Z.on(ctrl_qubit)
        circuit += BARRIER
    if j == 1:
        circuit += X.on(ctrl_qubit)
        circuit += BARRIER
        circuit += X.on(tar_qubit)
        circuit += BARRIER
    return circuit


def CZ_encoder(tar_qubit, ctrl_qubit):
    # 初始化量子线路
    circuit = Circuit()

    # 随机生成四个01比特串作为密钥
    j, k, l, m = random.randint(0, 2, 4)

    # 根据随机生成的密钥，对量子态进行加密操作
    if j == 1:
        circuit += X.on(ctrl_qubit)
    if k == 1:
        circuit += Z.on(ctrl_qubit)

    circuit += BARRIER

    if l == 1:
        circuit += X.on(tar_qubit)
    if m == 1:
        circuit += Z.on(tar_qubit)

    circuit += BARRIER

    # 期望执行的量子门
    circuit += BarrierGate(True)
    circuit += Z.on(tar_qubit, ctrl_qubit)
    circuit += BarrierGate(True)

    # 根据随机生成的密钥，对量子态进行相应的解密操作
    if m == 1:
        circuit += Z.on(tar_qubit)
        circuit += BARRIER
    if l == 1:
        circuit += X.on(tar_qubit)
        circuit += BARRIER
        circuit += Z.on(ctrl_qubit)
        circuit += BARRIER
    if k == 1:
        circuit += Z.on(ctrl_qubit)
        circuit += BARRIER
    if j == 1:
        circuit += X.on(ctrl_qubit)
        circuit += BARRIER
        circuit += Z.on(tar_qubit)
        circuit += BARRIER
    return circuit


def SWAP_encoder(tar_qubit, ctrl_qubit):
    # 初始化量子线路
    circuit = Circuit()

    # 随机生成四个01比特串作为密钥
    j, k, l, m = random.randint(0, 2, 4)

    # 根据随机生成的密钥，对量子态进行加密操作
    if j == 1:
        circuit += X.on(ctrl_qubit)
    if k == 1:
        circuit += Z.on(ctrl_qubit)

    circuit += BARRIER

    if l == 1:
        circuit += X.on(tar_qubit)
    if m == 1:
        circuit += Z.on(tar_qubit)

    circuit += BARRIER

    # 期望执行的量子门
    circuit += BarrierGate(True)
    circuit += SWAP.on([tar_qubit, ctrl_qubit])
    circuit += BarrierGate(True)

    # 根据随机生成的密钥，对量子态进行相应的解密操作
    if m == 1:
        circuit += Z.on(ctrl_qubit)
        circuit += BARRIER
    if l == 1:
        circuit += X.on(ctrl_qubit)
        circuit += BARRIER
    if k == 1:
        circuit += Z.on(tar_qubit)
        circuit += BARRIER
    if j == 1:
        circuit += X.on(tar_qubit)
        circuit += BARRIER
    return circuit


def Toffoli_encoder(tar_qubit, ctrl_qubits):
    # 初始化量子线路
    circuit = Circuit()

    # 随机生成六个01比特串作为密钥
    j, k, l, m, p, q = random.randint(0, 2, 6)

    # 根据随机生成的密钥，对量子态进行加密操作
    if j == 1:
        circuit += X.on(ctrl_qubits[0])
    if k == 1:
        circuit += Z.on(ctrl_qubits[0])

    circuit += BARRIER

    if l == 1:
        circuit += X.on(ctrl_qubits[1])
    if m == 1:
        circuit += Z.on(ctrl_qubits[1])

    circuit += BARRIER

    if p == 1:
        circuit += X.on(tar_qubit)
    if q == 1:
        circuit += Z.on(tar_qubit)

    circuit += BARRIER

    # 期望执行的量子门
    circuit += BarrierGate(True)
    circuit += X.on(tar_qubit, ctrl_qubits)
    circuit += BarrierGate(True)

    # 根据随机生成的密钥，对量子态进行相应的解密操作
    if q == 1:
        circuit += Z.on(tar_qubit)
        circuit += BARRIER
        circuit += Z.on(ctrl_qubits[1], ctrl_qubits[0])
        circuit += BARRIER
    if p == 1:
        circuit += X.on(tar_qubit)
        circuit += BARRIER
    if m == 1:
        circuit += Z.on(ctrl_qubits[1])
        circuit += BARRIER
    if l == 1:
        circuit += X.on(ctrl_qubits[1])
        circuit += BARRIER
        circuit += X.on(tar_qubit, ctrl_qubits[0])
        circuit += BARRIER
    if k == 1:
        circuit += Z.on(ctrl_qubits[0])
        circuit += BARRIER
    if j == 1:
        circuit += X.on(ctrl_qubits[0])
        circuit += BARRIER
        circuit += X.on(tar_qubit, ctrl_qubits[1])
        circuit += BARRIER
    return circuit


def Toffoli_encoder_fig11(tar_qubit, ctrl_qubits):
    # 初始化量子线路
    circuit = Circuit()

    # 随机生成六个01比特串作为密钥
    j, k, l, m, p, q = random.randint(0, 2, 6)

    # 根据随机生成的密钥，对量子态进行加密操作
    if j == 1:
        circuit += X.on(ctrl_qubits[0])
    if k == 1:
        circuit += Z.on(ctrl_qubits[0])

    circuit += BARRIER

    if l == 1:
        circuit += X.on(ctrl_qubits[1])
    if m == 1:
        circuit += Z.on(ctrl_qubits[1])

    circuit += BARRIER

    if p == 1:
        circuit += X.on(tar_qubit)
    if q == 1:
        circuit += Z.on(tar_qubit)

    circuit += BARRIER

    # 期望执行的量子门
    circuit += BarrierGate(True)
    circuit += X.on(tar_qubit, ctrl_qubits)
    circuit += BarrierGate(True)

    # 根据随机生成的密钥，对量子态进行相应的解密操作
    if q == 1:
        circuit += Z.on(tar_qubit)
        circuit += BARRIER
        circuit += H.on(ctrl_qubits[1])
        circuit += BARRIER
        circuit += X.on(ctrl_qubits[1], ctrl_qubits[0])
        circuit += BARRIER
        circuit += H.on(ctrl_qubits[1])
        circuit += BARRIER
    if p == 1:
        circuit += X.on(tar_qubit)
        circuit += BARRIER
    if m == 1:
        circuit += Z.on(ctrl_qubits[1])
        circuit += BARRIER
    if l == 1:
        circuit += X.on(ctrl_qubits[1])
        circuit += BARRIER
        circuit += X.on(tar_qubit, ctrl_qubits[0])
        circuit += BARRIER
    if k == 1:
        circuit += Z.on(ctrl_qubits[0])
        circuit += BARRIER
    if j == 1:
        circuit += X.on(ctrl_qubits[0])
        circuit += BARRIER
        circuit += X.on(tar_qubit, ctrl_qubits[1])
        circuit += BARRIER
    return circuit
