from mindquantum import Circuit, RZ, RX, Z, BarrierGate, X, H
from mindquantum import Hamiltonian, QubitOperator
from mindquantum import Simulator, add_prefix, apply
from scipy.optimize import minimize
from mindspore import Tensor, nn
import matplotlib.pyplot as plt
import numpy as np

import mindspore as ms

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


def IsingHam(J, h, num_qubits=4):
    """
    -J 是 ZZ 前面的系数
    -h 是 X  前面的系数
    num_qubits
    返回ham 和min eng
    """
    ham = QubitOperator()
    for i in range(num_qubits):
        ham += QubitOperator(f'X{i}', -h)
        if i + 1 < num_qubits:
            ham += QubitOperator(f'Z{i} Z{i+1}', -J)
    mat = ham.matrix(num_qubits).todense()

    return Hamiltonian(ham), min(np.linalg.eigvals(mat).real)


def baseHEA(num_qubits=4, layers=4):
    ansatz = Circuit()
    for l in range(layers):
        for i in range(num_qubits):
            ansatz += RZ(f'{l}{i}{0}').on(i)
            ansatz += RX(f'{l}{i}{1}').on(i)
            ansatz += RZ(f'{l}{i}{2}').on(i)
        for i in range(num_qubits - 1):
            ansatz += Z(i + 1, i)
        ansatz += BarrierGate(False)
    return ansatz


def targetHEA(num_qubits):
    """
    使用network transfer 生成多qubits线路
    """
    ansatz = Circuit()
    base = baseHEA(4, 4)
    for i in range(num_qubits):
        if i + 3 < num_qubits:
            ansatz += add_prefix(apply(base, [k for k in range(i, i + 4)]),
                                 f"{i}")
    return ansatz


def func(x, grad_ops, target, show_iter_val=False):
    f, g = grad_ops(x)
    if np.abs(f - target) < 1.6e-3:
        return np.real(np.squeeze(f)), np.zeros_like(np.squeeze(g))
    if show_iter_val:
        print(np.squeeze(f).real)
    return np.real(np.squeeze(f)), np.squeeze(g)


if __name__ == "__main__":

    # base task 运行100次，把最好的结果存储下来
    num_qubits = 4
    base_layers = 4
    J = 1
    h = 2
    ansatz = Circuit()
    ansatz += baseHEA(num_qubits, base_layers)
    ansatz.as_ansatz()
    hams, target = IsingHam(J, h, num_qubits)
    sim = Simulator('mqvector', num_qubits)
    grad_ops = sim.get_expectation_with_grad(hams, ansatz)

    params_pool = []
    for k in range(100):
        x0 = np.random.random([len(ansatz.params_name)]) * np.pi  # [0, 1] 改到pi

        res = minimize(func,
                       x0,
                       args=(grad_ops, target, False),
                       method='BFGS',
                       jac=True,
                       tol=1e-6)
        params_pool.append(res.x.real)
        print(f"base task {k+1} is finished", res.success, res.nit)

    np.save("pool1.npy", params_pool)

    params_pool = np.load("pool1.npy", allow_pickle=True)

    # 计算统计数据部分，计算var和 norm G
    RR_var_list = []
    RR_avg_norm_list = []
    TT_avg_norm_list = []
    num_qubits_list = [4, 6, 8, 10, 12]
    for num_qubits in num_qubits_list:
        partial_0_list = []
        normalized_norm_list = []
        cir = targetHEA(num_qubits)
        cir.as_ansatz()
        hams, _ = IsingHam(J, h, num_qubits)
        sim = Simulator('mqvector', num_qubits)
        grad_ops = sim.get_expectation_with_grad(hams, cir)

        # 随机初始化参数 采样500次
        for k in range(500):
            x = np.random.random(len(
                cir.params_name)) * np.pi  # 原来是随机初始化的[0, 1], 改到pi就可以了
            f, g = grad_ops(x)
            g = g.squeeze()  # 梯度
            # 把 partial 0 取出来
            partial_0_list.append(g[1])
            # 计算 normalized norm
            normalized_norm_list.append(np.linalg.norm(g) / len(g))
            if (k + 1) % 50 == 0:
                print(
                    f"{num_qubits} qubits system random sampling {k+1} is finished"
                )

        RR_var_list.append(np.var(partial_0_list))
        RR_avg_norm_list.append(np.average(normalized_norm_list))

        # TT structure transfer sampling
        partial_0_list = []
        normalized_norm_list = []
        for k in range(500):
            num_sub_cir = len(cir.params_name) // len(params_pool[0])
            idx = np.random.randint(0, 100, num_sub_cir)
            x = params_pool[idx].reshape(-1)

            f, g = grad_ops(x)
            g = g.squeeze()  # 梯度
            # 计算 normalized norm
            normalized_norm_list.append(np.linalg.norm(g) / len(g))
            if (k + 1) % 50 == 0:
                print(
                    f"{num_qubits} qubits system network transfer sampling {k+1} is finished"
                )
        TT_avg_norm_list.append(np.average(normalized_norm_list))
    # 对于TT的base task 来说，不应该用优化后的参数
    TT_avg_norm_list[0] = RR_avg_norm_list[0]

    plt.plot(num_qubits_list, RR_var_list, 'o--')
    plt.title("Ising model, HEA, network transfer")
    plt.legend(['random'])
    plt.xlabel("number of qubits")
    plt.ylabel("variance")
    plt.savefig("a.png")

    plt.clf()
    plt.plot(num_qubits_list, RR_avg_norm_list, 'o--', label="RR")
    plt.plot(num_qubits_list, TT_avg_norm_list, 'o--', label="TT")
    plt.title("Ising model, HEA, network transfer")
    plt.legend()
    plt.xlabel("number of qubits")
    plt.ylabel("average norm")
    plt.savefig("d.png")
