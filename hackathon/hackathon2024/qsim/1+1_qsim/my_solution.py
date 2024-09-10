import os
import sys

sys.path.append(os.path.abspath(__file__))
from simulator import HKSSimulator
from utils import generate_molecule, get_molecular_hamiltonian, read_mol_data
from mindquantum.core.operators import QubitOperator, Hamiltonian, TimeEvolution
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import X
from mindquantum import uccsd_singlet_generator, SingleLoopProgress
from mindquantum.core.parameterresolver import ParameterResolver
import typing
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from mindquantum.simulator import Simulator
from mindquantum.algorithm.nisq import Transform
from mindquantum.core.operators import FermionOperator
from mindquantum.core.gates import NoiseGate, QuantumGate
from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_value_should_not_less,
)

import warnings
warnings.filterwarnings('ignore', message='install "ipywidgets" for Jupyter support')

#其他辅助函数
#################################

def extract_bracketed_parts(s):
    start = 0  # 初始化开始索引
    result = []  # 初始化结果列表
    while start < len(s):
        # 查找下一个方括号开放的位置
        start = s.find('[', start)
        if start == -1:  # 如果没有找到，结束循环
            break
        # 找到对应的方括号闭合位置
        end = s.find(']', start + 1)
        if end == -1:  # 如果没有找到闭合的方括号，结束循环
            break
        # 提取方括号中的部分，并添加到结果列表
        result.append(s[start+1:end])
        # 移动开始索引到闭合方括号之后
        start = end + 1

    return result

def extract_before_bracketed_parts(s):
    start = 0  # 初始化开始索引
    result = []  # 初始化结果列表
    while start < len(s):
        # 查找方括号开放的位置
        end = s.find('[', start + 1)
        if end == -1:  # 如果没有找到闭合的方括号，结束循环
            break
        # 提取方括号中的部分，并添加到结果列表
        result.append(s[start:end-1])
        # 移动开始索引到闭合方括号之后
        start = end + 1

    return result

# 精简H门和RX门
def compress_H_AND_RX(circuit):
    # 精简连续的H门
    # 初始化一个空字典，用于跟踪每个量子位上最后一个哈达玛门
    last_hadamard = {}
    gates_to_remove = []

    # 遍历电路以识别同一量子位上的连续哈达玛门
    for index, gate in enumerate(circuit):
        
        if gate.name == "BARRIER":
            continue
        if gate.name == "H":
            qubit = gate.obj_qubits[0]
            if qubit in last_hadamard:
                # 标记当前和上一个哈达玛门以供移除
                gates_to_remove.append(last_hadamard[qubit])
                gates_to_remove.append(index)
                 # 删除上一个哈达玛门的记录，因为两个连续的H门会相互抵消
                del last_hadamard[qubit]
            else:
                # 记录该量子位的最后一个哈达玛门的位置
                last_hadamard[qubit] = index
        else:
            # 任何其他类型的门都应清除其量子位的记录
            if (gate.obj_qubits[0] in last_hadamard) :
                del last_hadamard[gate.obj_qubits[0]]
                
            if len(gate.ctrl_qubits)>0:
                if (gate.ctrl_qubits[0] in last_hadamard):
                    del last_hadamard[gate.ctrl_qubits[0]]
    
    last_RX = {}
    # 精简RX门，合并相邻的Rx门
    for index, gate in enumerate(circuit):
        if gate.name == "BARRIER":
            continue
        if gate.name == "RX":
            qubit = gate.obj_qubits[0]
            
            if (qubit in last_RX):
                if (((circuit[last_RX[qubit]].coeff.const+circuit[index].coeff.const)-4*np.pi)<0.1):
                    gates_to_remove.append(last_RX[qubit])
                    gates_to_remove.append(index)    
                    del last_RX[qubit]
            else:
                # 记录该量子位的最后一个哈达玛门的位置
                last_RX[qubit] = index

        else:
            # 任何其他类型的门都应清除其量子位的记录
            if (gate.obj_qubits[0] in last_RX) :
                del last_RX[gate.obj_qubits[0]]
                
            if len(gate.ctrl_qubits)>0:
                if (gate.ctrl_qubits[0] in last_RX):
                    del last_RX[gate.ctrl_qubits[0]]
    
    gates_to_remove.sort(reverse=True)
    for  id in gates_to_remove:
        del circuit[id]
    return circuit
#################################

# 精简合并测量次数
#################################
def new_split_ham(sorted_split_ham):

    # 假设sorted_split_ham已经是一个排序后的列表，并且每个元素是一个(coeff, ops)元组
    # 初始化子集列表
    subset_keys = []
    # 将sorted_split_ham[0]的terms.keys的第一个元素添加到subset_keys
    if sorted_split_ham and sorted_split_ham[0][1].terms:  # 检查sorted_split_ham是否非空且第一个元素有terms
        subset_keys.append(sorted_split_ham[0])

    sorted_split_ham_new=sorted_split_ham.copy()
    # 遍历sorted_split_ham，从索引1开始
    for j in range(len(sorted_split_ham)-1,0,-1):
        # 检查sorted_split_ham[j][1].terms.keys()是否非空
        if sorted_split_ham[j][1].terms:
            current_keys = list(sorted_split_ham[j][1].terms.keys())[0]
            # 检查current_keys是否是subset_keys最后一个元素的子集
            if set(current_keys).issubset(set(list(sorted_split_ham[0][1].terms.keys())[0])):
                subset_keys.append(sorted_split_ham[j])
                # 删除sorted_split_ham中的当前元素
                del sorted_split_ham_new[j]
                # 由于列表已修改，这里不需要调整索引j

    # 遍历完成后，删除sorted_split_ham中的第一个元素
    if sorted_split_ham_new:  # 检查sorted_split_ham是否非空
        del sorted_split_ham_new[0]


    return subset_keys,sorted_split_ham_new
    
def solution_one(circ,molecule, Simulator: HKSSimulator) -> float:
    mol = generate_molecule(molecule)
    ham = get_molecular_hamiltonian(mol)
    const, split_ham = split_hamiltonian(ham)
    ucc = circ
    
    if circ.params_name==[]:
        p=[]
    else:
        p = get_best_params(circ, ham)

    print(p)
    result = const
    
    # 按照len(list(ops.terms.keys())[0])的大小，将split_ham重新从大到小排序
    sorted_split_ham = sorted(split_ham, key=lambda item: len(list(item[1].terms.keys())[0]), reverse=True)

    id=0
    with SingleLoopProgress(len(split_ham), '哈密顿量测量中') as bar:  
        while sorted_split_ham!=[]:
            subset_keys,sorted_split_ham=new_split_ham(sorted_split_ham)
            result+= new_mea_single_ham(ucc, subset_keys, p, Simulator) 

            for m in range(len(subset_keys)):
                bar.update_loop(id)
                id+=1
    return result        
#################################

# 读出错误缓解 （REM）
#################################
def kron_A_k_inv(k, e):
    # 定义矩阵 A
    A = np.array([[1-e, e],
                  [e, 1-e]])

    # 初始化为 A
    kron_A_k = A.copy()
    
    # 计算 A 的 k 次直积
    for _ in range(k - 1):
        kron_A_k = np.kron(kron_A_k, A)
    
    # 计算逆矩阵
    kron_A_k_inv = np.linalg.inv(kron_A_k)
    
    return kron_A_k_inv

def bitstrings_to_probability_vector(bitstrings):

    pv = np.zeros(2 ** len(list(bitstrings.keys())[0]))
    for bs in bitstrings:
        index = int("".join(map(str, bs)), base=2)
        pv[index] += bitstrings[bs]
    pv /= sum(pv)

    return pv

def closest_positive_distribution(quasi_probabilities):

    quasi_probabilities = np.maximum(quasi_probabilities, 0)
    quasi_probabilities /= np.sum(quasi_probabilities)

    def distance(probabilities):
        return np.linalg.norm(probabilities - quasi_probabilities)

    num_vars = len(quasi_probabilities)
    bounds = Bounds(np.zeros(num_vars), np.ones(num_vars))
    normalization = LinearConstraint(np.ones(num_vars), 1, 1)

    result = minimize(
        distance,
        quasi_probabilities,
        method='SLSQP',
        bounds=bounds,
        constraints=normalization
    )

    return result.x

def sample_probability_vector(probability_vector, samples):

    # 根据概率分布进行抽样
    num_values = len(probability_vector)
    choices = np.random.choice(num_values, size=samples, p=probability_vector)

    # 计算比特宽度并转换为二进制字符串
    bit_width = int(np.log2(num_values))
    binary_strings = [np.binary_repr(choice, width=bit_width) for choice in choices]

    # 手动统计每个比特字符串的出现次数
    count_dict = {}
    for string in binary_strings:
        if string in count_dict:
            count_dict[string] += 1
        else:
            count_dict[string] = 1

    return count_dict
#################################

#  零噪声外推（ZNE）
#################################
def _fold_globally(circ: Circuit, factor: float) -> Circuit:
    """Folding circuit globally."""
    _check_value_should_not_less("Fold factor", 1, factor)
    if circ.has_measure_gate or circ.is_noise_circuit:
        raise ValueError("For globally folding, circuit cannot has measurement or noise channel.")
    n_pair = int((factor - 1) // 2)
    n_random_factor = (factor - 1) / 2 % 1
    folded_circ = Circuit()
    folded_circ += circ
    circ_herm_circ = circ + circ.hermitian()
    for _ in range(n_pair):
        folded_circ += circ_herm_circ
    quantum_gate_poi = []
    for idx, g in enumerate(folded_circ):
        if isinstance(g, QuantumGate) and not isinstance(g, NoiseGate):
            quantum_gate_poi.append(idx)
    np.random.shuffle(quantum_gate_poi)
    n_random = int(n_random_factor * len(quantum_gate_poi) // (2 * n_pair + 1))
    random_choice = quantum_gate_poi[:n_random]
    random_choice = set(random_choice)
    new_fold = Circuit()
    for idx, g in enumerate(folded_circ):
        new_fold += g
        if idx in random_choice:
            new_fold += Circuit([g, g.hermitian()])
    return new_fold

def _fold_locally(circ: Circuit, factor: float) -> Circuit:
    """Folding circuit locally."""
    _check_value_should_not_less("Fold factor", 1, factor)
    n_pair = int((factor - 1) // 2)
    n_random_factor = (factor - 1) / 2 % 1
    folded_circ = Circuit()

    quantum_gate_poi = []
    for idx, g in enumerate(circ):
        if isinstance(g, QuantumGate) and not isinstance(g, NoiseGate):
            quantum_gate_poi.append(idx)
    np.random.shuffle(quantum_gate_poi)
    n_random = int(n_random_factor * len(quantum_gate_poi))
    random_choice = quantum_gate_poi[:n_random]
    random_choice = set(random_choice)
    quantum_gate_poi = set(quantum_gate_poi)
    n_pairs = []
    for i in range(len(circ)):
        p = 0
        if i in quantum_gate_poi:
            p += n_pair
        if i in random_choice:
            p += 1
        n_pairs.append(p)
    for idx, g in enumerate(circ):
        folded_circ += g
        if n_pairs[idx] != 0:
            folded_circ += Circuit([g, g.hermitian()] * n_pairs[idx])
    return folded_circ

def fold_at_random(circ: Circuit, factor: float, method='locally') -> Circuit:

    _check_value_should_not_less("Fold factor", 1, factor)
    _check_input_type("method", str, method)
    supported_method = ['globally', 'locally']
    if method not in supported_method:
        raise ValueError(f"method should be one of {supported_method}, but get {method}")
    if method == 'globally':
        return _fold_globally(circ, factor)
    return _fold_locally(circ, factor)

def zne(
    circuit: Circuit,
    scaling: typing.List[float] = None,
    order=None,
    method="R",
    a=0,
    molecule=None, 
    Simulator=HKSSimulator,
) -> float:

    y = []
    mitigated = 0
    if scaling is None:
        scaling = [1.0, 2.0, 3.0]
    for factor in scaling:
        expectation = solution_one(fold_at_random(circuit, factor,method='locally'), molecule, Simulator)
#         expectation = solution_one(fold_at_random(circuit, factor,method='globally'), molecule, Simulator)
        print("scaling:",factor,"expectation:",expectation)
        y.append(expectation)
    if method == "R":
        for k, y_k in enumerate(y):
            product = 1
            for i in range(0, len(y)):
                if k != i:
                    try:
                        product = product * (scaling[i] / (scaling[i] - scaling[k]))
                    except ZeroDivisionError as exc:
                        raise ZeroDivisionError(f"Error scaling: {scaling}") from exc
            mitigated = mitigated + y_k * product
        return mitigated
    if order is None:
        raise ValueError("For polynomial and poly exponential, order cannot be None.")
    if method == "P":
        z = np.polyfit(scaling, y, (order - 1))
        f = np.poly1d(z)
        mitigated = f(0)
        mitigated = a + mitigated

        return mitigated
    if method == "PE":
        y = np.exp(y)
        z = np.polyfit(scaling, y, (order - 1))
        f = np.poly1d(z)
        mitigated = f(0)  
        mitigated=np.log(mitigated)
        mitigated = a + mitigated
    else:
        print("Provide a valid extrapolation scheme R, PE, P")

    return mitigated
#################################

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

def get_ucc_circ(mol):
    ucc = Transform(uccsd_singlet_generator(mol.n_qubits, mol.n_electrons)).jordan_wigner().imag

    ucc = TimeEvolution(ucc).circuit

    return UN(X, mol.n_electrons) + ucc

def get_best_params(circ, ham):

    p0 = np.zeros(len(circ.params_name))

    grad_ops = Simulator('mqvector', NUM_QUBITS,seed=2024).get_expectation_with_grad(Hamiltonian(ham), circ)

    def fun(x, grad_ops):
        f, g = grad_ops(x)
        f = f.real[0, 0]
        g = g.real[0, 0]
        print(f"tenergy: {f}")
        return f, g

    res = minimize(fun, p0, (grad_ops, ), 'bfgs', True)
    return res.x

# def mea_single_ham(circ, ops, p, Simulator: HKSSimulator, shots=1000):
#     circ = rotate_to_z_axis_and_add_measure(circ, ops)
#     pr = ParameterResolver(dict(zip(circ.params_name, p)))
#     sim = Simulator('mqvector', circ.n_qubits,seed=2024)
#     result = sim.sampling(circ, shots=shots, pr=pr,seed=2024)
#     expec = 0

#     for i, j in result.data.items():

#         expec += (-1)**i.count('1') * j / shots
#         print(i,j,i.count('1'),(-1)**i.count('1') * j,(-1)**i.count('1') * j / shots)
#     return expec

def new_mea_single_ham(circ, subset_keys, p, Simulator: HKSSimulator, shots=10000):
    
    # 使用列表推导式提取数字并创建数组
    array = [item[0] for item in list(subset_keys[0][1].terms.keys())[0]]

    # 创建一个字典，将array中的元素映射到range(len(array), 0, -1)
    mapping_dict = mapping_dict = {item: len(array)-idx-1 for idx, item in enumerate(array)}

    circ = rotate_to_z_axis_and_add_measure(circ, list(subset_keys[0][1])[0])
    pr = ParameterResolver(dict(zip(circ.params_name, p)))
    sim = Simulator('mqvector', circ.n_qubits,seed=2024)
    result = sim.sampling(circ, shots=shots, pr=pr,seed=2024)
    expec = 0
    
    # 读出错误缓解 （REM）
    A_order=len(mapping_dict)
    e=0.05 #测量来源误差
    res_str=result.data
    kron_A=kron_A_k_inv(A_order, e)
    pro_res=bitstrings_to_probability_vector(res_str)
    
    adjusted_quasi_dist = (kron_A @ pro_res.T).T
    adjusted_prob_dist = closest_positive_distribution(adjusted_quasi_dist)
    REM_result = sample_probability_vector(adjusted_prob_dist, shots)
    

    for k in range(len(subset_keys)):
        k_array = [item[0] for item in list(subset_keys[k][1].terms.keys())[0]]
        
        # 读出错误缓解 （REM）
        for i, j in REM_result.items():
            # 初始化计数器  
            count_one=0
            for l in k_array:
                # 检查第1位（索引为0，因为索引从0开始）
                if i[mapping_dict[l]] == '1':
                    count_one += 1

            expec += (-1)**count_one * j / shots  *subset_keys[k][0]

    return expec

def solution(molecule, Simulator: HKSSimulator) -> float:
    
    np.random.seed(2024)  # 你可以选择任何整数值作为种子

    mol = generate_molecule(molecule)
    ham = get_molecular_hamiltonian(mol)
    const, split_ham = split_hamiltonian(ham)
    
    ucc = get_ucc_circ(mol)

    global NUM_QUBITS
    NUM_QUBITS=ucc.n_qubits

    p = get_best_params(ucc, ham)
    
    params_dict = dict(zip(ucc.params_name, p))
    # print(params_dict)
   
    # 阈值舍弃
    Fermion0=uccsd_singlet_generator(mol.n_qubits, mol.n_electrons)
    Fermion1=None
    for i in range(len(list(Fermion0))):
        for j in range(len(list(Fermion0)[i].params_name)):
            if abs(params_dict[list(Fermion0)[i].params_name[j]])>0.9*1e-1:
                Fermion1+=FermionOperator((extract_bracketed_parts(str(list(Fermion0)[i]))[0]),extract_before_bracketed_parts(str(list(Fermion0)[i]))[0])
            
    if  Fermion1==None:
        ucc_new = UN(X, mol.n_electrons)
    else:
        ucc_new = Transform(Fermion1).jordan_wigner().imag
        ucc_new = TimeEvolution(ucc_new).circuit
        ucc_new = UN(X, mol.n_electrons) + ucc_new

    # 精简H门和RX门
    ucc_new = compress_H_AND_RX(ucc_new)
    
    #  零噪声外推（ZNE）
    result=zne(ucc_new,scaling=[1,2,3],order=3,method="PE",a=-0.05, molecule=molecule, Simulator=HKSSimulator)

    return result
