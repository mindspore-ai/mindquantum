from mindquantum import Circuit, RZ, RX, Z, BarrierGate, X, H, XX, YY, ZZ
from mindquantum import Hamiltonian, QubitOperator
from mindquantum import Simulator, add_prefix, apply
from scipy.optimize import minimize
from mindspore import Tensor, nn
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "4"

import mindspore as ms
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


def HeisenbergHam(J, d, num_qubits=4):
    """
    -J 是 总体 前面的系数
    -d 是 ZZ 前面的系数, XX, YY 前面的系数都是1
    num_qubits
    返回ham 和min eng
    """
    mat = np.zeros([2**num_qubits, 2**num_qubits])
    ham = QubitOperator()
    for i in range(num_qubits):
        if i+1 < num_qubits:
            ham += QubitOperator(f'X{i} X{i+1}', -J)
            ham += QubitOperator(f'Y{i} Y{i+1}', -J)
            ham += QubitOperator(f'Z{i} Z{i+1}', -J * d)
    mat = ham.matrix(num_qubits).todense()
    return Hamiltonian(ham), min(np.linalg.eigvals(mat))


def baseHVA(num_qubits=4, layers=4):
    ansatz = Circuit()
    for l in range(layers):
        for i in range(num_qubits-1):
            ansatz += XX(f"{l}x").on([i, i+1])
        for i in range(num_qubits-1):
            ansatz += YY(f"{l}y").on([i, i+1])
        for i in range(num_qubits-1):
            ansatz += ZZ(f"{l}z").on([i, i+1])
        ansatz += BarrierGate(False)
    return ansatz


def func(x, grad_ops, target, show_iter_val=False):
    f, g = grad_ops(x)
    if np.abs(f-target) < 1.6e-3:
        return np.real(np.squeeze(f)) , np.zeros_like(np.squeeze(g))
    if show_iter_val:
        print(np.squeeze(f).real)
    return np.real(np.squeeze(f)) , np.squeeze(g)


if __name__ == "__main__":
    num_qubits = 4
    base_layers = 4
    J = 1
    h = 2

    ansatz = Circuit()
    ansatz += baseHVA(num_qubits, base_layers)
    ansatz.as_ansatz()
    hams, target = HeisenbergHam(J, h, num_qubits)
    sim = Simulator('projectq', num_qubits)
    grad_ops = sim.get_expectation_with_grad(hams, ansatz)


    params_pool = []
    for k in range(100):
        # 不同的random方式
        x0 = np.random.random([len(ansatz.params_name)])*np.pi # [0, 1]
        # x0 = np.random.randn(len(ansatz.params_name))*np.pi  # standard normal  
        res = minimize(func, x0, args=(grad_ops, target, False), method='BFGS', jac=True, tol=1e-6)
        params_pool.append(res.x.real)
        print(f"{k+1} is finished", res.success, res.nit)
    np.save("pool3.npy", params_pool)
    params_pool = np.load('pool3.npy', allow_pickle=True)

    RR_var_list = []
    RT_var_list = []
    TR_var_list = []
    TT_var_list = []
    RR_avg_norm_list = []
    RT_avg_norm_list = []
    TR_avg_norm_list = []
    TT_avg_norm_list = []

    num_qubits_list = [4, 8, 12]
    for num_qubits in num_qubits_list:
        cir = baseHVA(num_qubits, 8)
        cir.as_ansatz()
        hams, _ = HeisenbergHam(J, h, num_qubits)
        sim = Simulator('projectq', num_qubits)
        grad_ops = sim.get_expectation_with_grad(hams, cir)

        # 随机初始化参数 采样500次
        partial_0_list = []
        normalized_norm_list = []
        for k in range(500):
            x = np.random.random(len(cir.params_name))
            f, g = grad_ops(x)
            g = g.squeeze() # 梯度
            # 把 partial 0 取出来
            partial_0_list.append(g[0])
            # 计算 normalized norm
            normalized_norm_list.append(np.linalg.norm(g) / len(g))
            if (k+1) % 50 == 0:
                print(f"{num_qubits} qubits system random sampling {k+1} is finished")
        
        RR_var_list.append(np.var(partial_0_list))
        RR_avg_norm_list.append(np.average(normalized_norm_list))

        partial_0_list = []
        normalized_norm_list = []
        for k in range(500):
            x1 = np.random.random([len(ansatz.params_name)])
            x2 = params_pool[np.random.randint(0, 100)]
            x = np.concatenate([x1, x2])
            f, g = grad_ops(x)
            g = g.squeeze() # 梯度
            # 把 partial 0 取出来
            partial_0_list.append(g[0])
            # 计算 normalized norm
            normalized_norm_list.append(np.linalg.norm(g) / len(g))
            if (k+1) % 50 == 0:
                print(f"{num_qubits} qubits system random + transfer sampling {k+1} is finished")
        
        RT_var_list.append(np.var(partial_0_list))
        RT_avg_norm_list.append(np.average(normalized_norm_list))

        partial_0_list = []
        normalized_norm_list = []
        for k in range(500):
            x1 = params_pool[np.random.randint(0, 100)]
            x2 = np.random.random(len(ansatz.params_name))
            x = np.concatenate([x1, x2])
            f, g = grad_ops(x)
            g = g.squeeze() # 梯度
            # 把 partial 0 取出来
            partial_0_list.append(g[0])
            # 计算 normalized norm
            normalized_norm_list.append(np.linalg.norm(g) / len(g))
            if (k+1) % 50 == 0:
                print(f"{num_qubits} qubits system transfer + random sampling {k+1} is finished")
        
        TR_var_list.append(np.var(partial_0_list))
        TR_avg_norm_list.append(np.average(normalized_norm_list))


        partial_0_list = []
        normalized_norm_list = []
        for k in range(500):
            x1 = params_pool[np.random.randint(0, 100)]
            x2 = params_pool[np.random.randint(0, 100)]
            x = np.concatenate([x1, x2])
            f, g = grad_ops(x)
            g = g.squeeze() # 梯度
            # 把 partial 0 取出来
            partial_0_list.append(g[0])
            # 计算 normalized norm
            normalized_norm_list.append(np.linalg.norm(g) / len(g))
            if (k+1) % 50 == 0:
                print(f"{num_qubits} qubits system transfer sampling {k+1} is finished")
        
        TT_var_list.append(np.var(partial_0_list))
        TT_avg_norm_list.append(np.average(normalized_norm_list))
    # 对于TT的base task 来说，不应该用优化后的参数
    TT_avg_norm_list[0] = RR_avg_norm_list[0]
    TR_avg_norm_list[0] = RR_avg_norm_list[0]
    RT_avg_norm_list[0] = RR_avg_norm_list[0]

    TR_var_list[0] = RR_var_list[0]
    RT_var_list[0] = RR_var_list[0]
    TT_var_list[0] = RR_var_list[0]

    plt.plot(num_qubits_list, np.log10(RR_var_list), 'o--', label="RR")
    plt.plot(num_qubits_list, np.log10(RT_var_list), 'o--', label="RT")
    plt.plot(num_qubits_list, np.log10(TR_var_list), 'o--', label="TR")
    plt.plot(num_qubits_list, np.log10(TT_var_list), 'o--', label="TT")
    plt.title("Heisenberg XXZ, HVA, structure transfer")
    plt.legend()
    plt.xlabel("number of qubits")
    plt.ylabel("log10 variance")
    plt.savefig('c.png')

    plt.clf()
    plt.plot(num_qubits_list, np.log10(RR_avg_norm_list), 'o--', label="RR")
    plt.plot(num_qubits_list, np.log10(RT_avg_norm_list), 'o--', label="RT")
    plt.plot(num_qubits_list, np.log10(TR_avg_norm_list), 'o--', label="TR")
    plt.plot(num_qubits_list, np.log10(TT_avg_norm_list), 'o--', label="TT")
    plt.title("Heisenberg XXZ, HVA, structure transfer")
    plt.legend()
    plt.xlabel("number of qubits")
    plt.ylabel("log10 average norm")
    plt.savefig('f.png')
