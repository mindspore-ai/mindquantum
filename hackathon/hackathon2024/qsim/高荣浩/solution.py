import os
import sys

import mindquantum.core.circuit

# sys.path.append(os.path.abspath(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from simulator import HKSSimulator
from mindquantum import *

from utils import generate_molecule, get_molecular_hamiltonian, read_mol_data
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit


def generate_hf_state(electrons, orbitals):
    r"""
    生成 HF 态
    """
    return [1] * electrons + [0] * (orbitals - electrons)

def excitations(electrons, orbitals, delta_sz=0):
    r"""
        根据 hf 参考态，生成所有的单双激励旋转门
        electrons: 电子数量
        orbitals: 自旋轨道数量，也是qubits 数量
        delta_sz: 电子总自旋，用于保持自旋守恒，可为{0, 1, -1, 2, -2}
    """
    # 根据 qubits 位置分配自旋量，偶数位向上自旋，奇数位向下自旋
    sz = np.array([0.5 if (i % 2 == 0) else -0.5 for i in range(orbitals)])

    # 所有 givens 单激发门要作用的 qubit 位置
    singles = [
        [r, p]
        for r in range(electrons)
        for p in range(electrons, orbitals)
        if sz[p] - sz[r] == delta_sz
    ]

    # 所有双激发门要做用的 qubit 位置
    doubles = [
        [s, r, q, p]
        for s in range(electrons - 1)
        for r in range(s + 1, electrons)
        for q in range(electrons, orbitals - 1)
        for p in range(q + 1, orbitals)
        if (sz[p] + sz[q] - sz[r] - sz[s]) == delta_sz
    ]

    return singles, doubles


def DoubleExcitation(wires, phi, circ):
    r"""
    双激发 givens 门
    wires: 作用的量子比特位
    """
    circ += CNOT.on(wires[3], wires[2])
    circ += CNOT.on(wires[2], wires[0])
    circ += H.on(wires[3])
    circ += H.on(wires[0])
    circ += CNOT.on(wires[3], wires[2])
    circ += CNOT.on(wires[1], wires[0])
    circ += RY(phi / 8).on(wires[1])
    circ += RY(-phi / 8).on(wires[0])
    circ += CNOT.on(wires[3], wires[0])
    circ += H.on(wires[3])
    circ += CNOT.on(wires[1], wires[3])
    circ += RY(phi / 8).on(wires[1])
    circ += RY(-phi / 8).on(wires[0])
    circ += CNOT.on(wires[1], wires[2])
    circ += CNOT.on(wires[0], wires[2])
    circ += RY(-phi / 8).on(wires[1])
    circ += RY(phi / 8).on(wires[0])
    circ += CNOT.on(wires[1], wires[3])
    circ += H.on(wires[3])
    circ += CNOT.on(wires[3], wires[0])
    circ += RY(-phi / 8).on(wires[1])
    circ += RY(phi / 8).on(wires[0])
    circ += CNOT.on(wires[1], wires[0])
    circ += CNOT.on(wires[0], wires[2])
    circ += H.on(wires[0])
    circ += H.on(wires[3])
    circ += CNOT.on(wires[2], wires[0])
    circ += CNOT.on(wires[3], wires[2])
    return circ


def SingleExcitation(wires, phi, circ):
    r"""
    单激发 Givens 门
    wires: 作用的量子比特位
    """
    circ += CNOT.on(wires[1], wires[0])
    circ += RY(phi / 2).on(wires[0])
    circ += CNOT.on(wires[0], wires[1])
    circ += RY(-phi / 2).on(wires[0])
    circ += CNOT.on(wires[0], wires[1])
    circ += CNOT.on(wires[1], wires[0])

    return circ


# 定义全局参数
electrons = 4  # H4 电子数量
orbitals = 8 # H4 自旋轨道数

# 生成 HF 参考态
hf_state = generate_hf_state(electrons, orbitals)

# 生成所有符合条件的单、双激发门
singles, doubles = excitations(electrons, orbitals, delta_sz=0)


def add_gate(wires, param, circ):
    r"""
    添加激发门
    wires: 作用的量子比特位
    """
    if len(wires) > 2:
        return DoubleExcitation(wires, param, circ)
    return SingleExcitation(wires, param, circ)


def get_givens_circuit(hf_state, doubles, singles):
    r"""
    构造完整的 givens ansatz
    先双后单，添加激发门
    """
    circ = Circuit()
    params = PRGenerator('theta')
    for i, x in enumerate(hf_state):
        if x == 1:
            circ += X.on(i)
    for i, wires in enumerate(doubles + singles):
        circ = add_gate(wires, params.new(), circ)
    return circ


def split_hamiltonian(ham: QubitOperator):
    const = 0
    split_ham = []
    for i, j in ham.split():
        if j == 1:
            const = i.const.real
        else:
            split_ham.append([i.const.real, j])
    return const, split_ham


def rotate_to_z_axis_and_add_measure(circ: Circuit, ops: QubitOperator):
    circ = circ.copy()
    assert ops.is_singlet
    for idx, o in list(ops.terms.keys())[0]:
        if o == 'X':
            circ.ry(-np.pi / 2, idx)
        elif o == 'Y':
            circ.rx(np.pi / 2, idx)
        circ.measure(idx)
    return circ


def get_best_params(ham):
    r"""
    在模拟器上学习初始 ansatz 最优的参数
    """
    circ = get_givens_circuit(hf_state, doubles, singles)
    p0 = np.zeros(len(circ.params_name))
    grad_ops = Simulator('mqvector', circ.n_qubits).get_expectation_with_grad(
        Hamiltonian(ham), circ)

    def fun(x, grad_ops):
        f, g = grad_ops(x)
        f = f.real[0, 0]
        g = g.real[0, 0]
        return f, g

    res = minimize(fun, p0, (grad_ops,), 'bfgs', True)
    # print('Optimization without noise done! Result :', res.fun)
    return res.x


def mea_single_ham(circ, ops, p, Simulator: HKSSimulator, shots=256):
    r"""
    在模拟器上进行单个 Pauli 项的测量，设定不同的测量次数 shots
    """
    circ = rotate_to_z_axis_and_add_measure(circ, ops)
    pr = ParameterResolver(dict(zip(circ.params_name, p)))
    sim = Simulator('mqvector', circ.n_qubits)
    result = sim.sampling(circ, shots=shots, pr=pr)
    expec = 0
    for i, j in result.data.items():
        expec += (-1) ** i.count('1') * j / shots
    return expec


def fit_function(x_points, y_points, method):
    r"""
    选择不同的拟合函数，应用在 ZNE 上，如多项式拟合，对数拟合
    x_points: 噪声水平
    y_points: 对应的测量值
    """
    if method == 'log':
        x = 0
        # 定义对数模型
        def log_model(x, a, b, c):
            return a * np.log(x + b) + c
    
        # 进行对数拟合
        params, _ = curve_fit(log_model, x_points, y_points, maxfev=2000)
        a, b, c = params
        return log_model(x, a, b, c)

    elif method == 'log_2':
        x = 0
        # 定义对数模型
        def log_model(x, a, b, c):
            return a * (np.log1p(x)**2) + b * np.log1p(x) + c
    
        # 进行对数拟合
        params, _ = curve_fit(log_model, x_points, y_points)
        a, b, c = params
        return log_model(x, a, b, c)

    elif method == 'poly':
        # 使用numpy的polyfit进行多项式拟合
        # degree = 2 二次曲线拟合
        
        x = 0
        degree = 2
        coeffs = np.polyfit(x_points, y_points, degree)
        # 使用 polyval 来计算多项式在x处的值
        return np.polyval(coeffs, x)


def get_zne_circuit(p, scalars, delta = 0.1):
    r"""
    生成零噪声外推 (ZNE) 量子电路。

    参数:
    p: list
        参数列表，用于生成 Givens 旋转电路。
    scalars: list
        整数标量列表，每个标量对应一个 ZNE 电路。
    delta: float, 可选
        阈值，用于过滤掉绝对值小于该阈值的参数。默认为 0.1。

    返回: list
        返回一个元组列表，每个元组包含一个 ZNE 电路和相应的简化参数列表。
    """
    simplify_doubles = []
    simplify_singles = []
    simplify_p = []
    for i, param in enumerate(p):
        # 给定阈值，删减参数较小的 Givens 门
        if abs(param) < delta:
            continue
        if i >= len(doubles):
            simplify_singles.append(singles[i - len(doubles)])
        else:
            simplify_doubles.append(doubles[i])
        simplify_p.append(param)
        
    zne_circuit_list = []
    circ = get_givens_circuit(hf_state, simplify_doubles, simplify_singles)
    
    # 电路的共轭转置
    dagger_circ = dagger(circ)
    
    # 根据给定的不同的噪声水平，进行电路折叠
    for s in scalars:
        zne_circuit = Circuit()
        for i in range(0, s - 1, 2):
            zne_circuit += circ.__deepcopy__(None)
            zne_circuit += dagger_circ.__deepcopy__(None)
        zne_circuit += circ.__deepcopy__(None)
        zne_circuit_list.append((zne_circuit, simplify_p))
    # print('ZNE circuit preparation done')
    
    # 返回折叠后的 list
    return zne_circuit_list


def solution(molecule, Simulator: HKSSimulator) -> float:
    mol = generate_molecule(molecule)
    ham = get_molecular_hamiltonian(mol)
    
    # 获得初始 Givens ansatz 的最优参数
    p = get_best_params(ham)
    # print('Best params :', p)
    const, split_ham = split_hamiltonian(ham)
    
    # 给定噪声拟合水平
    zne_scalars = [1, 3, 5, 7, 9]
    
    # 获得根据噪声水平折叠后的电路
    zne_circuit_list = get_zne_circuit(p, zne_scalars)
    
    # 用于存储不同折叠电路的测量值
    result = []

    # print("Start noisy measurement")
    # 判断模拟器
    if Simulator is HKSSimulator:
    
        for zne_circuit, zne_param in zne_circuit_list:
            tmp = const
            with SingleLoopProgress(len(split_ham), '哈密顿量测量中') as bar:
                for idx, (coeff, ops) in enumerate(split_ham):
                    tmp += mea_single_ham(zne_circuit, ops, zne_param, Simulator) * coeff
                    bar.update_loop(idx)
                result.append(tmp)
        print("result = ", result)

        # 返回 ZNE 外推噪声为 0 时的值
        return fit_function(zne_scalars, result, method='log_2')
    
    else:
        result = const
        with SingleLoopProgress(len(split_ham), '哈密顿量测量中') as bar:
            for idx, (coeff, ops) in enumerate(split_ham):
                result += mea_single_ham(zne_circuit_list[0][0], ops, zne_circuit_list[0][1], Simulator) * coeff
                bar.update_loop(idx)
        
        print("result = ", result)
        return result   
        


if __name__ == '__main__':
    import simulator

    simulator.init_shots_counter()
    molecule = read_mol_data('mol.csv')
    for sim in [HKSSimulator, Simulator]:
        result = solution(molecule, sim)
        print(sim, result)

        # 计算分数
        score = 1/(np.abs(result+ 2.1663874498494664))
        print('score:',score )
