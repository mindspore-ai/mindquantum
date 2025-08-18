import numpy as np
import random

from samples.utils import build_ham_ising
from mindquantum.core.gates import I, X
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.operators import Hamiltonian

###
    ### pip install numba
###
from numba import njit

# 函数输入输出不可修改
def solve(nqubit, Q_triu):
    """输入：问题比特数量nqubit，特定格式矩阵：Qtriu；
    输出：求解方法针对任意给定问题返回的求解线路。"""
    # 下面是可以修改的内容

    if nqubit < 500:
        print(f"method: deterministic_customized, qubit: {nqubit}, sampling: {int(nqubit / 10)}")
        best_circ = deterministic_customized(nqubit, Q_triu, int(nqubit / 10))
    else:
        print(f"method: deterministic_customized, qubit: {nqubit}, sampling: {int(nqubit / 20)}")
        best_circ = deterministic_customized(nqubit, Q_triu, int(nqubit / 20))

    # 上面是可以修改的内容

    return best_circ

###
    # Adapt-Clifford 启发式的贪心算法
###
@njit
def gradient(nbit, W, active, inactive, qubits):
    ###
        ### speed up the gradient compute with numba
    ###
    grad = np.zeros(nbit)
    max_val = -1.0
    max_idx = -1
    for idx in range(len(inactive)):
        j = inactive[idx]
        s = W[j, j]
        for a in range(len(active)):
            i = active[a]
            w = 2 * W[min(i, j), max(i, j)]
            s += (1 if qubits[i] == 0 else -1) * w
        grad[j] = s
        abs_s = abs(s)
        if abs_s > max_val:
            max_val = abs_s
            max_idx = j
    return grad, max_idx

def greedy_adaptclifford_inspired(nbit, W, f):
    ###
        ### greedy algorithm inspired by adapt clifford 
    ###
    circ = Circuit().un(I, range(nbit))
    qubits = [0] * nbit

    if W[f,f] > 0:
        qubits[f] = 1
        circ += X.on(f) # update the circuit
    exp = -abs(W[f, f]) # get the expectation

    active = [f]
    inactive = [i for i in range(nbit) if i != f]
    while True:
        grad, max_key = gradient(nbit, W, active, inactive, qubits) # get the gradient for every inactive qubit
        exp -= abs(grad[max_key]) # get the expectation

        if grad[max_key] > 0:
            qubits[max_key] = 1
            circ += X.on(max_key) # update the circuit

        active.append(max_key)
        inactive.remove(max_key)
        if len(active) == nbit:
            return exp, circ

def randomized(nbit, W):
    f = random.choice(range(nbit))
    exp, circ = greedy_adaptclifford_inspired(nbit, W, f)
    return circ

def deterministic(nbit, W):
    exp_best = np.inf
    for f in range(nbit):
        exp, circ = greedy_adaptclifford_inspired(nbit, W, f)
        if exp<exp_best:
            exp_best = exp
            circ_best = circ
    return circ_best

def deterministic_customized(nbit, W, nums):
    exp_best = np.inf
    objs = random.sample(range(nbit), nums)
    for f in objs:
        exp, circ = greedy_adaptclifford_inspired(nbit, W, f)
        if exp<exp_best:
            exp_best = exp
            circ_best = circ
    return circ_best

