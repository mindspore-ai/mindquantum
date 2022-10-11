from mindquantum.core import Circuit, Hamiltonian, QubitOperator
from mindquantum.core import UN, H, X, RZ, RX
from mindquantum import ParameterResolver as PR
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator
import mindspore as ms
import mindspore.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

def random_clauses(n,m,k=3):
    s = []
    for i in range(m):
        l = random.sample(range(n), k)
        for j in range(k):
            t = random.randint(0, 1)
            l[j] += t * n
        s += [tuple(l)]
    return s

def calc_minV(n,s):
    ans = len(s)
    for cas in range(1<<n):
        penalty = 0
        for l in s:
            satsfied = False
            for x in l:
                if x < n and cas & (1<<x) != 0 :
                    satsfied = True
                if x >= n and cas & (1<<(x-n)) == 0 :
                    satsfied = True
            if not satsfied :
                penalty += 1
        ans = min(ans, penalty)
#     print('exact_ans=', ans)
    return ans

def build_Ug(n,s,d,k=3):
    """
    Args:
        n(int): variables
        s(list of tuple): clauses
        d(int): parameter notation
        k(int): k-sat
    """
    gd = f'g{d}'
    c = Circuit()
    for l in s:
        # 0 ~ n-1  means penalty on 0 (needed flipping)
        # n ~ 2n-1 means penalty on 1
        for x in l:
            if x < n:
                c += X.on(x)
        ll = [x%n for x in l]
        for j in range(k):
            c += RZ(PR({gd: 0.5**j})).on(ll[k-1-j], ll[:k-1-j])
        for x in l:
            if x < n:
                c += X.on(x)
    return c

def build_Ub(n,d):
    bd = f'b{d}'
    c = Circuit()
    for i in range(n):
        c += RX(bd).on(i)
    return c


def build_ansatz(n,s,p):
    """
    Args:
        n(int): variables
        s(list of tuple): clauses
        p(int): depth (2p parameters)
    """
    c = UN(H, n)
    for i in range(p):
        c += build_Ug(n, s, i)
        c += build_Ub(n, i)
    return c


def build_ham(n,s):
    ham = QubitOperator()
    for l in s:
        mono = QubitOperator('')
        for x in l:
            mono *= (QubitOperator('') + QubitOperator(f'Z{x%n}', (-1)**(x//n))) / 2
        ham += mono
    return ham

for p in range(15,40,10):
    random.seed(42)
    n = 6
    ave_list = []
    std_list = []
    var_list = []
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
    sim = Simulator('projectq', n)

    for m in range(1,60,8):
    #for m in range(1,60,2):
        f = []
        print(m, "clauses:")
        for _ in range(15):
            s = random_clauses(n, m)
            minV = calc_minV(n, s)
            ham = Hamiltonian(build_ham(n, s))
            circ = build_ansatz(n, s, p)
            grad_ops = sim.get_expectation_with_grad(ham, circ)
            net = MQAnsatzOnlyLayer(grad_ops)
            opti = nn.Adam(net.trainable_params(), learning_rate=0.05)
            train_net = nn.TrainOneStepCell(net, opti)
            lst = train_net().asnumpy()[0]
            while True:
                err = train_net().asnumpy()[0]
                print(err - minV)
                if lst > err and lst - err < 0.001 :
                    break
                lst = min(lst, err)
            err -= minV
            f.append(err)

        ave_list.append(np.mean(f))
        std_list.append(np.std(f))
        var_list.append(np.var(f))
        print(f)
        print("error =", np.mean(f), ", std =", np.std(f), ", var =", np.var(f))

        fp = open("statistic3_"+str(p)+".txt", "w")
        fp.write(str(ave_list))
        fp.write("\n")
        fp.write(str(std_list))
        fp.write("\n")
        fp.write(str(var_list))
        fp.close()
        