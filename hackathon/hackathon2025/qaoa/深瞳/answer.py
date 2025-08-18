import numpy as np
import random
import copy
import time
from multiprocessing import Pool, cpu_count

from samples.utils import build_ham_ising
from mindquantum.core.gates import I, X, Y, Z, S, H, CNOT, RX, Power
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.operators import Hamiltonian



# 函数输入输出不可修改
def solve(nqubit, Q_triu):
    """输入：问题比特数量nqubit，特定格式矩阵：Qtriu；
    输出：求解方法针对任意给定问题返回的求解线路。"""
    # 下面是可以修改的内容

    select_num = None
    repeat = 1
    if nqubit < 400:
        repeat = 1
    if 400 <= nqubit < 600:
        repeat = 2
    if 600 <= nqubit < 800:
        repeat = 2
    if 800 <= nqubit < 1000:
        repeat = 2
    if 1000 <= nqubit < 1100:
        repeat = 3
    best_circ = example4(nqubit, Q_triu, select_num=select_num, repeat=repeat)

    # 上面是可以修改的内容

    return best_circ



def apply_x_gates_and_get_lists(circuit, select_num, nqubit=None):
    bits = np.zeros(nqubit)

    for i in range(nqubit):
        if i in select_num:
            circuit += X.on(i)
            bits[i] = -1
        else:
            bits[i] = 1

    return circuit, bits


def example4(nqubit, Q_triu, select_num=None, repeat=1):
    qubits = nqubit + 1
    J = np.triu(Q_triu, k=1) + np.triu(Q_triu, k=1).T
    J = np.pad(J, ((1, 0), (1, 0)), mode='constant')
    h = np.diag(Q_triu)
    #  将一次项转化为二次项，扩展自旋0取值固定为1，注意到这破缺了模型的Z2对称性。
    for i in range(nqubit):
        J[0][i + 1] = h[i]
        J[i + 1][0] = h[i]

    best_circ = None
    best_energy = 100
    # best_depth = None
    sim = Simulator('stabilizer', nqubit)
    #  新线路应增加一个量子比特，但注意到run.py文件不可修改，故不扩展线路。
    #  事实上，应注意到原论文的方法本质上可以稍加修改而成为一个贪心算法，从而每次梯度计算后均定下了一个X门的位置，而无需使用原论文的双比特门。
    #  原论文抽象出的贪心算法的有效性证明尚未给出，但我们的实验表明它确实效果更好，如能研究清楚背后的理论机制，有可能可以向高阶伊辛模型推广。
    #  原论文的取法我们用注释标出，比如下面的两行。
    '''
    for i in range(qubits):
        circ += H.on(i)
    '''
    if select_num is None:
        num1 = repeat
        select_num = random.sample(range(qubits), num1)
    for index_k in select_num:
        circ = Circuit(UN(I, nqubit))
        #  若Z2对称性不被破缺，以下两个线路都是解，但事实上一次项导致只有一个可取。
        circk = Circuit(UN(I, nqubit))
        circj = Circuit(UN(I, nqubit))
        flag0 = 0    #  为0则选circk，为1则选circj
        if index_k == 0:
            flag0 = 1
        else:
            circk += X.on(index_k - 1)
        # circ += Z.on(index_k)
        # active.append(index_k)
        # active = []
        inactive = list(range(qubits))    #  还没有被激活的比特。
        inactive.remove(index_k)
        # depth = 1

        # stab = {}    #  用于存储稳定子的字典，由于最后不必构造出解，无需使用而注释掉。
        active_k = []    #  和 index_k 纠缠而激活的比特。
        active_j = []    #  和 index_j 纠缠而激活的比特。
        active_k.append(index_k)
        for r in range(1, qubits):
            gradients = {}    #  用于存储梯度的字典。
            if r == 1:
                for b in inactive:
                    gradients[b] = J[index_k][b]
                index_j = max(gradients, key=gradients.get)
                active_j.append(index_j)
                # active.append(index_j)
                inactive.remove(index_j)
                #  circ = circ + H.on(index_j) + Power(S, 3).on(index_k) + X.on(index_j, index_k) + RX(-np.pi/2).on(index_k) + X.on(index_j, index_k) + H.on(index_j) + S.on(index_k)
                # stab[index_k] = index_j
                if index_j != 0:
                    circj += X.on(index_j - 1)
            else:
                for b in inactive:
                    sum_weights_k = 0
                    for weights_index_k in active_k:
                        sum_weights_k += J[weights_index_k][b]
                    sum_weights_j = 0
                    for weights_index_j in active_j:
                        sum_weights_j += J[weights_index_j][b]
                    gradients[(index_k, b)] = sum_weights_j - sum_weights_k
                    gradients[(index_j, b)] = - gradients[(index_k, b)]
                (index_kj, index_max) = max(gradients, key=gradients.get)
                if index_kj == index_k:
                    active_k.append(index_max)
                    # stab[index_max] = index_k
                    if index_max != 0:
                        circk += X.on(index_max - 1)
                    else:
                        flag0 = 1
                else:
                    active_j.append(index_max)
                    # stab[index_max] = index_j
                    if index_max != 0:
                        circj += X.on(index_max - 1)
                inactive.remove(index_max)
                #  circ = circ + H.on(index_kj) + Power(S, 3).on(index_max) + X.on(index_kj, index_max) + RX(-np.pi / 2).on(index_max) + X.on(index_kj, index_max) + H.on(index_kj) + S.on(index_max)


        if flag0 == 0:
            circ = circk
        else:
            circ = circj

        ham = build_ham_ising(Q_triu)
        sim.reset()
        exp = sim.get_expectation(Hamiltonian(ham), circ)
        energy = exp.real

        if energy < best_energy:
            best_energy = energy
            best_circ = circ
            # best_depth = depth

    return best_circ


