from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.algorithm.nisq import generate_uccsd
from mindquantum import Circuit, RZ, RX, Z, BarrierGate, X, H
from mindquantum import Hamiltonian
from mindquantum import Simulator, add_prefix, apply
from scipy.optimize import minimize
from mindspore import Tensor, nn
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = "4"

import mindspore as ms

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


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


def gene_molecule(num_H):
    DIST = 0.74
    geometry = []
    for k in range(num_H):
        geometry.append(["H", [0.0, 0.0, k * DIST]])
    basis = "sto3g"
    if num_H % 2 == 0:
        spin = 0
    else:
        spin = 0.5

    molecule_data = MolecularData(geometry,
                                  basis,
                                  multiplicity=int(2 * spin + 1))
    molecule_data = run_pyscf(molecule_data,
                              run_scf=1,
                              run_ccsd=1,
                              run_fci=1,
                              verbose=False)
    return molecule_data


def func(x, grad_ops, target, show_iter_val=False):
    f, g = grad_ops(x)
    if np.abs(f - target) < 1.6e-3:
        return np.real(np.squeeze(f)), np.zeros_like(np.squeeze(g))
    if show_iter_val:
        print(np.squeeze(f).real)
    return np.real(np.squeeze(f)), np.squeeze(g)


if __name__ == "__main__":

    # base task 运行100次，把最好的结果存储下来
    num_H = 2
    base_layers = 4

    molecule_data = gene_molecule(num_H)
    target = molecule_data.fci_energy
    _, _, _, ham_qubitOp, num_qubits, n_electrons = generate_uccsd(
        molecule_data)
    hams = Hamiltonian(ham_qubitOp)

    ansatz = Circuit()
    # 加了 h 门可以减少优化的步骤
    # ansatz.un(H, list(range(num_qubits)))
    ansatz += baseHEA(num_qubits, base_layers)
    ansatz.as_ansatz()

    sim = Simulator('mqvector', num_qubits)
    grad_ops = sim.get_expectation_with_grad(hams, ansatz)

    params_pool = []
    for k in range(100):
        x0 = np.random.random(len(ansatz.params_name)) * np.pi
        res = minimize(func,
                       x0,
                       args=(grad_ops, target, False),
                       method='BFGS',
                       jac=True,
                       tol=1e-6)
        params_pool.append(res.x.real)
        print(f"{k+1} is finished", res.success, res.nit)
    np.save("pool2.npy", params_pool)

    params_pool = np.load("pool2.npy", allow_pickle=True)

    # 计算统计数据部分，计算var和 norm G
    RR_var_list = []
    TT_var_list = []
    RT_var_list = []
    TR_var_list = []
    RR_avg_norm_list = []
    TT_avg_norm_list = []
    RT_avg_norm_list = []
    TR_avg_norm_list = []

    num_Hs = [2, 3, 4, 5]
    for num_H in num_Hs:
        partial_0_list = []
        normalized_norm_list = []

        molecule_data = gene_molecule(num_H)
        target = molecule_data.fci_energy
        _, _, _, ham_qubitOp, num_qubits, n_electrons = generate_uccsd(
            molecule_data)
        hams = Hamiltonian(ham_qubitOp)

        cir = baseHEA(num_qubits, 8)
        cir.as_ansatz()
        sim = Simulator('mqvector', num_qubits)
        grad_ops = sim.get_expectation_with_grad(hams, cir)

        # 随机初始化参数 采样500次
        for k in range(500):
            x = np.random.random(len(cir.params_name)) * np.pi
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

        # TT network transfer sampling
        partial_0_list = []
        normalized_norm_list = []
        cir_param_name = cir.params_name

        for k in range(500):
            x = np.random.random([len(cir_param_name)])
            pr = dict(zip(cir_param_name, x))
            idx = np.random.randint(0, 100, 1)
            params = params_pool[idx]
            for i, j in enumerate(ansatz.params_name):
                pr[j] = params[0, i]
                j = str(int(j) + 400)
                pr[j] = params[0, i]
            x = np.array(list(pr.values()))

            f, g = grad_ops(x)
            g = g.squeeze()  # 梯度
            # 把 partial 0 取出来
            partial_0_list.append(g[1])
            # 计算 normalized norm
            normalized_norm_list.append(np.linalg.norm(g) / len(g))
            if (k + 1) % 50 == 0:
                print(
                    f"{num_qubits} qubits system network transfer sampling {k+1} is finished"
                )
        TT_var_list.append(np.var(partial_0_list))
        TT_avg_norm_list.append(np.average(normalized_norm_list))
    # 对于TT的base task 来说，不应该用优化后的参数
    TT_avg_norm_list[0] = RR_avg_norm_list[0]
    TT_var_list[0] = RR_var_list[0]

    num_qubits_list = [2 * q for q in num_Hs]
    plt.plot(num_qubits_list, RR_var_list, 'o--', label="RR")
    plt.plot(num_qubits_list, TT_var_list, 'o--', label="TT")
    plt.title("hydrogen chain, HEA, structure transfer")
    plt.legend()
    plt.xlabel("number of qubits")
    plt.ylabel("variance")
    plt.savefig("b.png")

    plt.clf()
    plt.plot(num_qubits_list, RR_avg_norm_list, 'o--', label="RR")
    plt.plot(num_qubits_list, TT_avg_norm_list, 'o--', label="TT")
    plt.title("hydrogen chain, HEA, structure transfer")
    plt.legend()
    plt.xlabel("number of qubits")
    plt.ylabel("average norm")
    plt.savefig("e.png")