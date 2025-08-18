import numpy as np
import random

from samples.utils import build_ham_ising
from mindquantum.core.gates import I, X, H, Z, S, CNOT
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

    best_circ = example2(nqubit, Q_triu, select_num=select_num, repeat=repeat)
    # 上面是可以修改的内容

    return best_circ


# 案例1：贪心算法


# baseline
# [272.0500472820889, 157.68115504314898, 1935.2187591994589, 239.864341524648, 293.970589078753, 763.3466463616536, 304.8340994518911, 6305.895720100031, 692.3105170455606, 902.5268239246739, 1446.9647103326306, 497.5170387350823, 11371.951256508195, 1316.7081453205585, 1706.9144726753561, 2272.013235589623, 637.1300147436655, 18022.404518741758, 2042.7468612592672, 2686.709718396147, 3282.074518309245, 787.4182952997319, 24734.13975772595, 2826.0067157865255, 3700.497537184302]
# use time： 132.4920153617859
# score: 89198.8955

def gradient(inaqubit, W, aqubits_k, aqubits_j):
    """
    Given an inactive qubit, this function computes the gradient of it with
    with respect to the two qubits defining the bipartition.

    inaqubit: the inactibe qubit under consideration
    W: a matrix of the edge weights
    aqubits_k: the active qubits which were entangled with qubit k
    aqubits_j: the active qubits which were entangled with qubitt j
    c: the current state encoded as a TableauSimulator
    """

    lindex_k = np.intersect1d(np.nonzero(W[:, inaqubit])[0], aqubits_k)
    lindex_j = np.intersect1d(np.nonzero(W[:, inaqubit])[0], aqubits_j)

    sum_weights_k = np.sum(W[ll, inaqubit] for ll in lindex_k)
    sum_weights_j = np.sum(W[ll, inaqubit] for ll in lindex_j)

    grad_k = -sum_weights_k + sum_weights_j
    return grad_k

def pos_max_grad(inaqubits, W, aqubits_k, aqubits_j):
    """
    This function finds the inactive qubit b with the largest gradient, and the
    corresponding initial qubit, k or j, with which this largest gradient occurs

    inaqubits: the vector of inactive qubits
    W: a matrix of the edge weights
    aqubits_k: the active qubits which were entangled with qubit k
    aqubits_j: the active qubits which were entangled with qubitt j
    c: the current state encoded as a TableauSimulator
    """

    all_grads_k = [gradient(inaqubit, W, aqubits_k, aqubits_j) for inaqubit in inaqubits]
    all_grads_j = -1.0 * np.array(all_grads_k)

    pos_max_k = np.argmax(all_grads_k)
    pos_max_j = np.argmax(all_grads_j)

    if all_grads_k[pos_max_k] > all_grads_j[pos_max_j]:
        return "k", inaqubits[pos_max_k], all_grads_k[pos_max_k]
    elif all_grads_k[pos_max_k] < all_grads_j[pos_max_j]:
        return "j", inaqubits[pos_max_j], all_grads_j[pos_max_j]
    else:
        char = np.random.choice([1, 2])
        if char == 1:
            return "k", inaqubits[pos_max_k], all_grads_k[pos_max_k]
        elif char == 2:
            return "j", inaqubits[pos_max_j], all_grads_j[pos_max_j]


def add_gate(q1, q2):
    """
    This function applies a e^(pi/4 YZ) gate to the state

    q1: the index of the first qubit
    q2: the index of the second qubit
    c: the current state in the form of a TableauSimulator
    """
    eg = Circuit()
    eg += S.on(int(q1))
    eg += S.on(int(q1))
    eg += S.on(int(q1))
    eg += H.on(int(q2))
    eg += CNOT.on(int(q2), int(q1))
    eg += Z.on(int(q1))

    eg += H.on(int(q1))
    eg += S.on(int(q1))
    eg += H.on(int(q1))
    eg += S.on(int(q1))
    eg += S.on(int(q1))

    eg += CNOT.on(int(q2), int(q1))
    eg += S.on(int(q1))
    eg += H.on(int(q2))

    #eg += X.on(int(q1))
    #eg += CNOT.on(int(q2), int(q1))

    # c.s_dag(q1)
    # c.h(q2)
    # c.cnot(q1, q2)
    # c.z(q1)
    # c.h_yz(q1)
    # c.cnot(q1, q2)
    # c.s(q1)
    # c.h(q2)

    return eg

def example2(nqubit, Q_triu, select_num=None, repeat=1):
    J = np.triu(Q_triu, k=1) + np.triu(Q_triu, k=1).T
    h = np.diag(Q_triu)
    W = Q_triu + Q_triu.T - np.diag(np.diag(Q_triu))
    #W = J
    bits = np.ones(nqubit)
    sim = Simulator('stabilizer', nqubit)

    circ = Circuit(UN(H, nqubit))

    #fqubit = 0
    #fqubit, qpair = np.unravel_index(np.argmax(W-np.diag(np.diag(Q_triu))), W.shape)
    #fqubit = 1
    # qpair = int(qpair)
    row_sums = np.sum(W, axis=1)
    fqubit = int(np.argsort(row_sums)[-1])
    fqubit = int(fqubit)
    #fqubit = int(np.argmax(np.sum(W, axis=1)))
    circ += Z.on(fqubit)
    bits[int(fqubit)] = -1

    active_qubits_k = []
    active_qubits_j = []
    inactive_qubits = list(range(nqubit))


    for nn in range(nqubit-1):
        if nn == 0:
            #qpair = np.argmax(W[:,fqubit])
            qpair = int(np.argsort(row_sums)[-2])
            #qpair = np.random.choice(W[:, fqubit].nonzero()[0])  #上三角得改
            while qpair == fqubit:
                qpair = np.random.choice(W[:, fqubit].nonzero()[0])
            gra = W[qpair, fqubit]
            qubits, grad = (fqubit, qpair), gra
            #qubits, grad = (qpair, fqubit), gra

            ##-- updating the records of active and inactive qubits
            active_qubits_j.append(qpair)
            active_qubits_k.append(fqubit)

            inactive_qubits.remove(qpair)
            inactive_qubits.remove(fqubit)
            bits[int(qpair)] = 1
            #eg = add_gate(qubits[0], qubits[1])
        else:
            aset, qpair, gra = pos_max_grad(inactive_qubits, W, active_qubits_k, active_qubits_j)

            if aset == "k":
                qubits = (qpair, fqubit)
                active_qubits_k.append(qpair)
                bits[int(qpair)] = -1
                # eg = add_gate(qubits[0], qubits[1])
                # circ = circ + eg
            elif aset == "j":
                qubits = (qpair, active_qubits_j[0])
                active_qubits_j.append(qpair)
                bits[int(qpair)] = 1
                #eg = add_gate(qubits[1], qubits[0])
            inactive_qubits.remove(qpair)

        eg = add_gate(qubits[0], qubits[1])
        circ = circ + eg
    while True:
        delta_E = (2 * J @ bits + h) * -2 * bits
        index = np.argmin(delta_E)
        bits[index] = -bits[index]
        if delta_E[index] < 0:
            circ += X.on(int(index))
        else:
            break

    return circ

