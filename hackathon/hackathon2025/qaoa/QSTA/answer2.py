import numpy as np
import random

from samples.utils import build_ham_ising
from mindquantum.core.gates import I, X
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.simulator import Simulator

###
    # pip install pymanopt
    # pip install autograd
###

import pymanopt
from pymanopt.manifolds import Product
from pymanopt.optimizers import TrustRegions
import autograd.numpy as anp

# 函数输入输出不可修改
def solve(nqubit, Q_triu):
    """输入：问题比特数量nqubit，特定格式矩阵：Qtriu；
    输出：求解方法针对任意给定问题返回的求解线路。"""
    # 下面是可以修改的内容
    O_opt = burer_monteiro_method(Q_triu, 3)
    num_samples = 10
    bits, _ = get_value(O_opt, Q_triu, samples=num_samples)
    select_num = np.where(bits == -1.0)[0]
    best_circ = example1(nqubit, Q_triu, select_num, repeat=1)

    # 上面是可以修改的内容

    return best_circ

###
    # 基于热启动的基线算法
###
def burer_monteiro_method(W, k):
    nbit = len(W)
    # Define Product manifolds
    Q = np.triu(W, k=1) + np.triu(W, k=1).T
    S = np.diag(W)

    Q_new = np.r_[np.c_[Q, S], [np.append(S, 1)]]

    manifold = Product([pymanopt.manifolds.Sphere(k) for _ in range(nbit+1)])

    @pymanopt.function.autograd(manifold)
    # Define the cost function
    def objective(*O):
        O = anp.stack(O, axis=0)  # matrix form
        return anp.trace(Q_new @ (O @ O.T))

    problem = pymanopt.Problem(manifold, objective)
    optimizer = TrustRegions(max_iterations=1000, min_gradient_norm=1e-6, verbosity=0)
    result = optimizer.run(problem)
    O_opt = anp.stack(result.point, axis=0)
    return O_opt[:-1, :]

def get_value(O_opt, W, samples):
    n, k = O_opt.shape
    best_bits = None
    best_value = np.inf
    for _ in range(samples):
        r = np.random.randn(k)
        r /= np.linalg.norm(r)
        x = np.sign(O_opt @ r)
        value = 0
        for i in range(n):
            for j in range(i+1, n):
                value += 2 * x[i] * x[j] * W[i, j]
        for i in range(n):
            value += x[i] * W[i, i]
        if value < best_value:
            best_value = value
            best_bits = x.copy()
    return best_bits, best_value

def apply_x_gates_and_get_lists(circuit, select_num, nqubit):
    bits = np.zeros(nqubit)

    for i in range(nqubit):
        if i in select_num:
            circuit += X.on(i)
            bits[i] = -1
        else:
            bits[i] = 1
    return circuit, bits

def example1(nqubit, Q_triu, select_num=None, repeat=1):
    J = np.triu(Q_triu, k=1) + np.triu(Q_triu, k=1).T
    h = np.diag(Q_triu)

    best_circ = None
    best_energy = 100
    best_depth = None
    sim = Simulator('stabilizer', nqubit)
    for ii in range(repeat):
        circ = Circuit().un(I, range(nqubit))
        if select_num is None:
            num1 = random.randint(0, nqubit)
            select_num = random.sample(range(nqubit), num1)

        circ, bits = apply_x_gates_and_get_lists(circ, select_num, nqubit)

        depth = 0
        while True:
            delta_E = (2*J @ bits + h) * -2 * bits #翻转比特对能量的变化
            index = np.argmin(delta_E)
            bits[index] = -bits[index]
            if delta_E[index] < 0:
                circ += X.on(int(index))
                depth += 1
            else:
                break

        ham = build_ham_ising(Q_triu)
        sim.reset()
        exp = sim.get_expectation(Hamiltonian(ham), circ)
        energy = exp.real

        if energy < best_energy:
            best_energy = energy
            best_circ = circ
            best_depth = depth
    return best_circ
