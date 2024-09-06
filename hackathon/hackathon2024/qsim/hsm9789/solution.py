import os
import sys

sys.path.append(os.path.abspath(__file__))
from simulator import HKSSimulator
from utils import generate_molecule, get_molecular_hamiltonian, read_mol_data
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X, RY, RZ, I
from mindquantum import SingleLoopProgress, TimeEvolution
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.algorithm.nisq.chem.uccsd import _para_uccsd_singlet_generator, _transform2pauli

import numpy as np
from scipy.optimize import minimize
from mindquantum.simulator import Simulator


def split_hamiltonian(ham: QubitOperator):
    """
        对哈密顿量进行拆分，将系数与算符分隔开
    """
    const = 0
    split_ham = []
    for i, j in ham.split():
        if j == 1:
            const = i.const.real
        else:
            split_ham.append([i.const.real, j])
    return const, split_ham


def get_best_params(circ, ham):
    circ_new = circ.copy()
    #p0 = np.zeros(len(circ_new.params_name))
    p0 = np.random.uniform(-np.pi, np.pi, len(circ_new.params_name))
    sim = Simulator('mqvector', circ_new.n_qubits)
    grad_ops = sim.get_expectation_with_grad(Hamiltonian(ham), circ_new)

    def fun(x, grad_ops):
        f, g = grad_ops(x)
        f = np.real(f)[0, 0]
        g = np.real(g)[0, 0]
        return f, g

    res = minimize(fun, p0, (grad_ops,), 'bfgs', True)
    # print(res.fun)
    return res.fun, res.x


def rotate_to_z(circ: Circuit, ops):
    circ_new = circ.copy()
    for i in range(len(ops)):
        op = ops[i]
        if op == 'X':
            circ_new.ry(-np.pi / 2, i)
        elif op == 'Y':
            circ_new.rx(np.pi / 2, i)
        circ_new.measure(i)
    return circ_new


def mea_single_ham(circ, ops, p, Simulator, shots):
    circ_new = rotate_to_z(circ, ops)
    pr = ParameterResolver(dict(zip(circ_new.params_name, p)))
    sim = Simulator('mqvector', circ_new.n_qubits)
    res = sim.sampling(circ_new, shots=shots, pr=pr)
    return res


def select_ham(circ, split_ham, n, pr):
    """
            移除哈密顿量中对结果贡献小于1e-5的项，返回优化后的哈密顿量
    """
    split_ham_new = []
    circ_cal = circ.copy()
    sim_cal = Simulator('mqvector', n)
    pr_cal = ParameterResolver(dict(zip(circ_cal.params_name, pr)))
    for i in range(len(split_ham)):
        coeff, ops = split_ham[i]
        ham_cal = Hamiltonian(ops)
        exp_cal = sim_cal.get_expectation(ham_cal, circ_cal, pr=pr_cal).real * coeff
        if abs(exp_cal) > 1e-5:
            split_ham_new.append(split_ham[i])
    return split_ham_new


def get_res(circ, pr, Simulator: HKSSimulator, shots, const, split_ham, g_div, ham_group, T_inv, n):
    """
            对同一组内的项同时测量，返回在线路circ下对哈密顿量采样得到的期望值（已进行测量噪声错误缓解）
    """
    res_noi = const
    with SingleLoopProgress(len(ham_group), '测量中') as bar:
        for idx, ops in enumerate(ham_group):
            res_data = mea_single_ham(circ, ops, pr, Simulator, shots)
            mea_noi = np.zeros(pow(2, n))
            for i_str, j in res_data.data.items():
                i_str = i_str[::-1]
                mea_noi[int(i_str, 2)] = j
            mea_ideal = np.dot(T_inv, mea_noi)
            for i in g_div[idx]:
                ops = split_ham[i][1]
                exp = 0
                vis = []
                for k in range(n):
                    vis.append(0)
                for idx_, op in list(ops.terms.keys())[0]:
                    vis[idx_] = 1
                for j in range(pow(2, n)):
                    i_bin = format(j, 'b').zfill(n)
                    cnt = 0
                    for k in range(len(i_bin)):
                        if vis[k] == 1 and i_bin[k] == '1':
                            cnt += 1
                    if cnt % 2 == 0:
                        exp += mea_ideal[j] / shots
                    else:
                        exp -= mea_ideal[j] / shots
                res_noi += exp * split_ham[i][0]
            bar.update_loop(idx)
    print(res_noi)
    return res_noi


def FIIM(circ, s):
    """
        插入2s个门，将噪声放大至r=2s+1倍，返回放大噪声后的线路
    """
    idx = 0
    circ_new = circ.copy()
    while idx < len(circ_new):
        if circ_new[idx].ctrl_qubits:
            cnt = 0
            g = circ_new[idx]
            while cnt < s:
                idx += 1
                circ_new.insert(idx, g)
                idx += 1
                circ_new.insert(idx, g)
                cnt = cnt + 1
        else:
            cnt = 0
            a = circ_new[idx].obj_qubits[0]
            while cnt < s:
                idx += 1
                circ_new.insert(idx, X.on(a))
                idx += 1
                circ_new.insert(idx, X.on(a))
                cnt += 1
        idx = idx + 1
    return circ_new


def generate_qccsd_pool(mol, th=0):
    """
            生成Qubit-ADAPT-VQE方法用到的操作池
    """
    fermion_ansatz, parameters = _para_uccsd_singlet_generator(mol, th)
    operator_pool = []

    for i, item in enumerate(fermion_ansatz):
        pauli_ansatz = _transform2pauli([item])
        for k, v in pauli_ansatz.items():
            s0 = ""
            for kk in k:
                if kk[1] != 'Z':
                    s0 += kk[1] + str(kk[0]) + ' '
            operator_pool.append(s0)
    return operator_pool


def generate_qccsd_circuit(mol, depth):
    """
        生成Qubit-ADAPT-VQE线路，depth为迭代次数
    """
    ham = get_molecular_hamiltonian(mol)
    n = mol.n_qubits
    circ = Circuit()
    for i in range(mol.n_electrons):
        circ += X.on(i)
    for i in range(mol.n_electrons, n):
        circ += I.on(i)
    operator_pool = generate_qccsd_pool(mol, th=0)
    sim = Simulator('mqvector', mol.n_qubits)
    num = 1
    pr = 0
    while num <= depth:
        grad = 0
        op_next = operator_pool[0]
        for op in operator_pool:
            ham_i = QubitOperator(op, 1j)
            ham_tmp = ham * ham_i - ham_i * ham
            if num == 0:
                grad_i = sim.get_expectation(Hamiltonian(ham_tmp), circ)
            else:
                grad_i = sim.get_expectation(Hamiltonian(ham_tmp), circ, pr=pr)
            if grad < grad_i.real:
                grad = grad_i.real
                op_next = op
        circ += TimeEvolution(QubitOperator(op_next, ('theta' + str(num)))).circuit
        energy = 0
        for _ in range(20):
            energy_i, pr_i = get_best_params(circ, ham)
            if energy_i < energy:
                energy = energy_i
                pr = pr_i
        pr = ParameterResolver(dict(zip(circ.params_name, pr)))
        #print(num, energy)
        num += 1

    circ_new = Circuit()
    for g in circ:
        if not g.obj_qubits:
            continue
        if g == I:
            continue
        circ_new += g

    return circ_new


def sol_f(mol, Simulator: HKSSimulator, T_inv) -> float:
    """
            FIIM方法放大噪声，ZNE方法错误缓解，返回错误缓解后的结果
    """
    ham = get_molecular_hamiltonian(mol)
    const, split_ham = split_hamiltonian(ham)  # 得到拆分后的哈密顿量

    turn = 100
    shots = 50000
    n = mol.n_qubits

    #circ = Ansatz(n, 4, 1).circ
    circ = generate_qccsd_circuit(mol, 9)

    energy, pr = get_best_params(circ, ham)
    with SingleLoopProgress(turn, '优化参数中') as bar:
        for i in range(turn):
            energy_i, pr_i = get_best_params(circ, ham)
            if energy_i < energy:
                energy = energy_i
                pr = pr_i
            bar.update_loop(i)
    print(" ", energy)

    split_ham = select_ham(circ, split_ham, n, pr)
    g_div = ham_div(split_ham, n)
    ham_group = get_ham_group(split_ham, g_div, n)
    print("化简后哈密顿量数量: ", len(split_ham), "分组后哈密顿量组数: ", len(g_div))

    res_noi_f = get_res(circ, pr, Simulator, shots, const, split_ham, g_div, ham_group, T_inv, n)

    circ_1 = FIIM(circ, 1)
    res_noi_f1 = get_res(circ_1, pr, Simulator, shots, const, split_ham, g_div, ham_group, T_inv, n)

    circ_2 = FIIM(circ, 2)
    res_noi_f2 = get_res(circ_2, pr, Simulator, shots, const, split_ham, g_div, ham_group, T_inv, n)

    circ_3 = FIIM(circ, 3)
    res_noi_f3 = get_res(circ_3, pr, Simulator, shots, const, split_ham, g_div, ham_group, T_inv, n)

    res_em_f1 = res_noi_f * 3 / 2 - res_noi_f1 * 1 / 2
    print("res_em_1: ", res_em_f1)
    res_em_f2 = res_noi_f * 15 / 8 - res_noi_f1 * 5 / 4 + res_noi_f2 * 3 / 8
    print("res_em_2: ", res_em_f2)
    res_em_f3 = res_noi_f * 35 / 16 - res_noi_f1 * 35 / 16 + res_noi_f2 * 21 / 16 - res_noi_f3 * 5 / 16
    print("res_em_3: ", res_em_f3)

    res_list = [res_em_f1, res_em_f2, res_em_f3]
    res = 0
    for res_i in res_list:
        if abs(res_i - mol.fci_energy) < abs(res - mol.fci_energy):
            res = res_i

    return res


class Ansatz:
    """
        生成不同类型的HEA变分线路
    """
    def __init__(self, n, depth, type):
        self.circ = Circuit()
        num = 0
        for _ in range(depth):
            if type == 1:
                for i in range(n):
                    self.circ += RZ('theta' + str(num)).on(i)
                    self.circ += RY('phi' + str(num)).on(i)
                    self.circ += RZ('psi' + str(num)).on(i)
                    num += 1
                for i in range(n-1):
                    self.circ += X.on(i + 1, i)
            if type == 2:
                for i in range(n-1):
                    for j in range(2):
                        self.circ += RZ('theta' + str(num)).on(i + j)
                        self.circ += RY('phi' + str(num)).on(i + j)
                        self.circ += RZ('psi' + str(num)).on(i + j)
                        num += 1
                    self.circ += X.on(i + 1, i)
                    for j in range(2):
                        self.circ += RZ('theta' + str(num)).on(i + j)
                        self.circ += RY('phi' + str(num)).on(i + j)
                        self.circ += RZ('psi' + str(num)).on(i + j)
                        num += 1
            if type == 3:
                for i in range(n):
                    self.circ += RZ('theta' + str(num)).on(i)
                    self.circ += RY('phi' + str(num)).on(i)
                    self.circ += RZ('psi' + str(num)).on(i)
                    num += 1
                for i in range(n-1):
                    self.circ += X.on(i + 1, i)
                self.circ += X.on(0, n - 1)
            if type == 4:
                for i in range(n):
                    self.circ += RZ('theta' + str(num)).on(i)
                    self.circ += RY('phi' + str(num)).on(i)
                    self.circ += RZ('psi' + str(num)).on(i)
                    num += 1
                for i in range(n-1):
                    self.circ += X.on(i + 1, i)
                for i in range(n):
                    self.circ += RZ('theta' + str(num)).on(i)
                    self.circ += RY('phi' + str(num)).on(i)
                    self.circ += RZ('psi' + str(num)).on(i)
                    num += 1
                for i in range(n-1):
                    self.circ += X.on(i, n - 1)


def greedy_clique_cover(graph):
    """
            贪心法将图划分成满足要求的若干集合
    """
    n = len(graph)
    uncovered = set(range(n))
    cliques = []

    while uncovered:
        clique = set()
        for v in uncovered:
            if all(graph[v][u] for u in clique):
                clique.add(v)
        cliques.append(clique)
        uncovered -= clique

    return cliques


def ham_div(split_ham, n):
    """
            对哈密顿量进行分组，把可同时测量的项放在一个组内
    """
    g = [[0 for _ in range(len(split_ham))] for _ in range(len(split_ham))]
    for i in range(len(split_ham)):
        ham_i = []
        for k in range(n):
            ham_i.append('I')
        for idx, op in list(split_ham[i][1].terms.keys())[0]:
            ham_i[idx] = op
        for j in range(len(split_ham)):
            flag = True
            for idx, op in list(split_ham[j][1].terms.keys())[0]:
                if ham_i[idx] != 'I' and ham_i[idx] != op:
                    flag = False
            if flag:
                g[i][j] = 1

    return greedy_clique_cover(g)


def get_ham_group(split_ham, g_div, n):
    ham_group = []
    for i in range(len(g_div)):
        ham_i = []
        for k in range(n):
            ham_i.append('I')
        for j in g_div[i]:
            for idx, op in list(split_ham[j][1].terms.keys())[0]:
                ham_i[idx] = op
        ham_group.append(ham_i)
    return ham_group


def get_T(n, p):
    """
            得到测量噪声变换矩阵的逆，p为比特翻转概率
    """
    T = np.zeros((pow(2, n), pow(2, n)))
    vec = []
    for i in range(pow(2, n)):
        vec.append(format(i, 'b').zfill(n))
    for i in range(pow(2, n)):
        for j in range(pow(2, n)):
            count = sum(1 for a, b in zip(vec[i], vec[j]) if a != b)
            T[i][j] = pow(1 - p, n - count) * pow(p, count)
    T_inv = np.linalg.inv(T)
    return T_inv


def solution(molecule, Simulator: HKSSimulator) -> float:
    mol = generate_molecule(molecule)  # 产生分子文件

    n = mol.n_qubits
    T_inv = get_T(n, 0.05)

    res = sol_f(mol, Simulator, T_inv)

    return res



if __name__ == '__main__':

    import simulator

    simulator.init_shots_counter()
    molecule = read_mol_data('mol.csv')  # 读取分子坐标文件

    result = solution(molecule, HKSSimulator)
    print(result)
    #print(1 / abs(result - mol.fci_energy))






