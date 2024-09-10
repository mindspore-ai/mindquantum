import os
import sys
import copy

import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告

from openfermion.chem import MolecularData
from mindquantum import InteractionOperator, FermionOperator, Transform
sys.path.append(os.path.abspath(__file__))
from simulator import HKSSimulator
from utils import generate_molecule, read_mol_data
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import X, BarrierGate, NoiseGate
from mindquantum.core.circuit.circuit import CollectionMap
from typing import Optional, Union
from mindquantum import *
import random
from mindquantum.algorithm.compiler import *

import numpy as np
from scipy.optimize import minimize


# 配置随机数种子
seed = 1
np.random.seed(seed)
random.seed(seed)

def circ_depth(circuit:Circuit, 
               with_single:bool=False, 
               with_barrier:bool=False):
    """统计线路深度。
    
    Argments:
    circ(Circuit): 待统计的线路；
    with_single(bool): 是否统计单量子比特门。默认值 False。
    with_barrier(bool): 是否将栅栏门视为分隔。
    
    Examples:
    >>> circ = Circuit().x(0).x(0,1).x(1,2).x(2,3)
    >>> circ_depth(circ, single=True)
    4
    >>> circ_depth(circ, single=False)
    3
    """
    depth_stack = {i: 0 for i in range(circuit.n_qubits)}
    if not depth_stack:
        return 0
    
    for gate in circuit: # type: ignore
        if isinstance(gate, NoiseGate):
            continue
        
        qubits = gate.obj_qubits + gate.ctrl_qubits
        
        # for single-qubit gate
        if len(qubits) == 1:
            if isinstance(gate, BarrierGate):
                continue
            if with_single:
                depth_stack[qubits[0]] += 1
            continue
        
        # for multi-qubit gate
        tmp = 0
        for i in qubits:
            tmp = max(tmp, depth_stack[i])
            
        if isinstance(gate, BarrierGate) and with_barrier:
            for i in qubits:
                depth_stack[i] = tmp
        else:
            for i in qubits:
                depth_stack[i] = tmp + 1
         
    return max(depth_stack.values())


def n_gates(circuit:Circuit):
    """统计线路中单比特门和多比特门的数量。"""
    single, multiple = 0, 0
    for gate in circuit: # type: ignore
        if isinstance(gate, BarrierGate):
            continue
        n = len(gate.ctrl_qubits + gate.obj_qubits)
        if n == 1:
            single += 1
        else:
            multiple += 1
    return single, multiple


def get_hamiltonian(mol: MolecularData, # 分子文件
                    occupied_indices:Optional[list[int]]=None, # 占据的空间轨道指标
                    active_indices:Optional[list[int]]=None, # 活跃的空间轨道指标
                    abs_tol:Optional[float] = None, # 系数小于阈值的项将被忽略
                    ) -> tuple[QubitOperator, int]:
    """根据分子文件生成哈密顿量。这里面加入了活性空间的参数。"""
    ham_of = mol.get_molecular_hamiltonian(occupied_indices, active_indices)
    inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
    ham_hiq = FermionOperator(inter_ops) # type: ignore
    qubit_ham = Transform(ham_hiq).jordan_wigner()
    
    # 忽略哈密顿量中非常小的部分，降低采样成本
    if abs_tol is not None:
        qubit_ham = qubit_ham.compress(abs_tol)
        
    # 计算哈密顿量所用的量子比特数，后续作为创建线路和模拟器的全局参数
    n_qubits = 0
    for term in qubit_ham.terms.keys():
        for op in term:
            n_qubits = max(n_qubits, op[0]) # type: ignore
            
    return qubit_ham, n_qubits + 1 # type: ignore


def split_hamiltonian(ham: QubitOperator) -> tuple[float, list]:
    """将算符的系数跟算符本身分开。
    
    Example:
    >>> ham = 2 + QubitOperator('X0 Z1', 1.2) + QubitOperator('Y0 X1', 0.3)
    >>> a, b = split_hamiltonian(ham)
    >>> print(a)
    2.0
    >>> print(b)
    [[1.2, [(0, 'X'), (1, 'Z')]], [0.3, [(0, 'Y'), (1, 'X')]]]
    """
    const = 0 # ham 的常数项
    terms = []
    for i, j in ham.split(): # i: PR(1.2), j: QubitOperator('X0 Y1', 1)
        if j == 1:
            const = i.const.real # type: ignore
        else:
            terms.append([i.const.real, list(*j.terms.keys())])  # type: ignore
    return float(const), terms


def grouping_ops(terms:list):
    """将测量操作分组，从而避免重复采样。
    
    Example:
    >>> terms = [[1.2, [(0, 'X'), (1, 'Z')]], [0.3, [(0, 'Y'), (1, 'X')]]]
    >>> print(grouping_ops(ham))
    [[(0, 'X'), (1, 'X')], [(0, 'Y'), (1, 'Z')]]
    """
    ops_set = set() # 用集合去除掉重复元素
    for term in terms:
        ops_set |= set(term[1])
        
    # 根据比特大小进行排序
    sored_ops = sorted(list(ops_set)) # [(0, 'X'), (0, 'Y'), (1, 'X'), (1, 'Z')]
    ops_dict = CollectionMap() # 底层是字典，可以保持元素顺序
    ops_dict.collect(sored_ops) # {(0, 'X'): 1, (0, 'Y'): 1, (1, 'X'): 1, (1, 'Z'): 1}
    
    grouped_ops = [] # [[(0, 'Y'), (1, 'Z')], [(0, 'X'), (1, 'X')]]
    # 分组，将可以放在同一次测量中的操作放在一个组里
    while len(ops_dict.map) > 0:
        ops = []
        min_qubit = -1
        tmp = list(ops_dict.map.keys())
        for op in tmp:
            if op[0] > min_qubit:
                ops.append(op)
                min_qubit = op[0]
                ops_dict.delete(op)         
        grouped_ops.append(ops)
    return grouped_ops


def map_outcome(ops:list[tuple[int, str]], 
                string:str):
    """根据测量结果获取测量对应的数值。
    若测量结果为 0, 则数值为 1, 若测量结果为 1 则数值为 -1。
    
    Example:
    >>> ops = [(0,'X'), (1, 'Y')]
    >>> string = '01'
    >>> map_outcome(ops, string)
    {(0, 'X'): -1, (1, 'Y'): 1}
    """
    string = string[::-1] # mq 是 little-endian 表示法，所以要逆序
    map = {}
    for i, op in enumerate(ops):
        map[op] = 1 if string[i] == '0' else -1
    return map


def ham_value(const:float, # 哈密顿量中的常数项
              terms:list, # 哈密顿量中的其它项
              map:dict) -> float:
    """根据统计好的每个测量的结果, 计算总哈密顿量的值。
    
    Example
    >>> const = 1.2
    >>> terms =  [[0.8, [(0, 'X'), (1, 'Z')]], [0.3, [(0, 'Y'), (1, 'X')]]]
    >>> map = {(0, 'X'):-1, (1, 'Z'): 1, (0, 'Y'): -1, (1, 'X'): -1}
    >>> ham_value(const, terms, map)
    0.7 # = 1.2 + 0.8 * (-1) * 1 + 0.3 * (-1) * (-1)
    """    
    for weight, term in terms:
        sign = 1 # 最终要么为 +1 要么为 -1
        for op in term:
            sign *= map[op]
        const += sign * weight # 系数 * sign
    return float(const)


def rotate_to_z_axis_and_add_measure(circuit: Circuit, 
                                     ops:list[tuple[int, str]],
                                     zne_n:int=0):
    """根据需要，在线路末尾根据哈密顿量添加旋转门，从而实现在 Z 基底下的测量。
    
    Example:
    >>> circ = Circuit()
    >>> a = [(0,'X'), (1, 'Y')]
    >>> circ = rotate_to_z_axis_and_add_measure(circ, a)
    >>> print(circ)
          ┏━━━━━━━━━━┓ ┍━━━━━━┑   
    q0: ──┨ RY(-π/2) ┠─┤ M q0 ├───
          ┗━━━━━━━━━━┛ ┕━━━━━━┙   
          ┏━━━━━━━━━┓ ┍━━━━━━┑    
    q1: ──┨ RX(π/2) ┠─┤ M q1 ├────
          ┗━━━━━━━━━┛ ┕━━━━━━┙    
    """
    circ = circuit.copy() # 后面使用 .ry 会直接修改线路。为复用，需先 copy
    for op in ops:
        if op[1] == 'X':
            circ.ry(-np.pi / 2, op[0])
        elif op[1] == 'Y':
            circ.rx(np.pi / 2, op[0])
            
    circ = zne_circ(circ, zne_n)
    for op in ops:
        circ.measure(op[0])
    return circ

def rxy_circ(p:Union[float,str], q0:int, q1:int):
    """将 Rxy 门分解为 RX RZ CNOT 门。"""
    circ = Circuit()
    circ += H.on(q0)
    circ += RX(np.pi/2).on(q1)
    circ += X.on(q1, q0)
    circ += RZ(p).on(q1)
    circ += X.on(q1, q0)
    circ += H.on(q0)
    circ += RX(-np.pi/2).on(q1)
    return circ

def get_ansatz_circ(n_hf:int, ham) -> Circuit:
    """根据分子结构，获取量子线路。
    只考虑了空间轨道 1 和 2 之间跃迁。
    """
    ansatz = Circuit()
    ansatz += rxy_circ("a", 0, 1)
    ansatz += X.on(2,0)
    ansatz += X.on(3,1)
    ansatz += X.on(0)
    ansatz += X.on(1)
    circ = UN(X, n_hf) + ansatz
    
    # 保证线路和哈密顿量的 n_qubits 一致
    global n_qubits
    if n_qubits is not None:
        circ.all_qubits.collect(n_qubits-1) 
    # print("unsimplified ansatz depth:",  circ_depth(circ), circ_depth(circ, with_single=True))
    # single, multiple = n_gates(circ)
    # print("unsimplified ansatz gates: 1-qubit ", single, " 2-qubit ", multiple)
    
    circ = compile_circuit(FullyNeighborCanceler(), circ) # 尽量融合相邻的量子门
    # print("simplified ansatz depth:",  circ_depth(circ), circ_depth(circ, with_single=True))
    # single, multiple = n_gates(circ)
    # print("simplified ansatz gates: 1-qubit ", single, " 2-qubit ", multiple)
    return circ

def get_hf_circ(n_hf:int, # hf 线路所应该施加的 X 门数量 
                    ham) -> Circuit:
    """获取 HF 线路。"""
    circ = UN(X, n_hf)
    return circ
 

def zne_circ(circuit:Circuit, 
             zne_n:int=0) -> Circuit:
    """将 circuit 修改为对应的 ZNE 线路。
    zne_n: 对合次数。默认值: 0, 不施加 ZNE。
    
    Example:
    >>> circ = Circuit().rx(1, 0)
    >>> circ = zne_circ(circ, zne_n=2)
    >>> print(circ)
          ┏━━━━━━━┓   ┏━━━━━━━┓ ┏━━━━━━━━┓   ┏━━━━━━━┓ ┏━━━━━━━━┓     
    q0: ──┨ RX(1) ┠─▓─┨ RX(1) ┠─┨ RX(-1) ┠─▓─┨ RX(1) ┠─┨ RX(-1) ┠─▓───
          ┗━━━━━━━┛   ┗━━━━━━━┛ ┗━━━━━━━━┛   ┗━━━━━━━┛ ┗━━━━━━━━┛  
    """
    circ = Circuit()
    circ += circuit
    for _ in range(zne_n):
        circ += BarrierGate()
        circ += circuit + circuit.hermitian()
    circ += BarrierGate()
    return circ


def get_expect(circuit:Circuit, 
               pr:Union[dict, np.ndarray, None]=None,
               sim=None,
               shots:int=100,
               zne_n:int=0):
    """获取哈密顿量的期望值。zne_n:使用 ZNE 法时的对合数, 若为 0, 则不使用 ZNE。
    
    Example:
    >>> ham = QubitOperator('Z0 Z1', 1.2)
    >>> circ = Circuit().h(0).h(1)
    >>> print(get_expect(circ))
    -0.019
    """            
    circuit = circuit.apply_value(pr=pr) # 根据 pr 将含参线路转换为固定线路
    global n_qubits
    if n_qubits is not None: # 修改 circ 的 n_qubits
        circuit.all_qubits.collect(n_qubits-1)
        
    if sim is None:
        sim = Simulator('mqvector', circuit.n_qubits)
    else:
        sim = sim('mqvector', circuit.n_qubits)
        
    global const, terms, grouped_ops
    
    zne_axis = [] # ZNE 拟合的横轴
    zne_values = [] # ZNE 拟合的数据
    for n in range(zne_n + 1):
        expect = 0
        zne_axis.append(2*n + 1)
        
        ## 先分组采样
        maps = {} 
        # maps = {((0, 'X'), (1, 'Y')): ['01', '10', '01',], ....}
        # 意味着对 'X0 Y1' 采样三次，其中结果 '01' 出现了两次， '10' 出现了一次
        for ops in grouped_ops:
            circ = rotate_to_z_axis_and_add_measure(circuit, ops, zne_n=n)
            res = sim.sampling(circ, shots=shots)
            recover = [] # recover = ['01', '01', '10']
            for k, v in res.data.items(): # res.data = {'01':2, '10':1}
                recover += [k] * v
            random.shuffle(recover) # 要打乱，因为采样时结果是随机的，而 res.data 是有序的
            maps[tuple(ops)] = recover
            
        ## 再综合分析
        for i in range(shots):
            map = {}
            for k, v in maps.items():
                map |= map_outcome(k, v[i])
            expect += ham_value(const, terms, map)
        expect /= shots
        zne_values.append(float(expect))
    
    if zne_n == 0: # 不使用 ZNE
        return zne_values[0]
    else: # 使用 ZNE
        z = np.polyfit(zne_axis, zne_values, 3) # 多项式拟合
        p = np.poly1d(z)
        return p(0) # 返回 0 点的拟合值
    
    
def get_expect_with_grad(circ:Circuit,
                         pr:Union[dict, np.ndarray],
                         sim=None, 
                         shots:int=100,
                         zne_n:int=0):
    """获取哈密顿量的期望值及相对参数的梯度。
    
    Example:
    >>> ham = QubitOperator('Z0', 1.2)
    >>> circ = Circuit().ry('theta', 0)
    >>> pr = np.array([np.pi/2])
    >>> value, grads = get_expect_with_grad(ham, circ, pr)
    >>> print(value, grads)
    -0.019, [-1.2]
    """
    if isinstance(pr, np.ndarray):
        pr = {i:k for i, k in zip(circ.params_name, pr)}
    
    expect = get_expect(circ, pr, sim, shots=shots, zne_n=zne_n)
    grads = []
    h = np.pi / 2
    for key, value in pr.items():
        pr_p = copy.deepcopy(pr)
        pr_p[key] = value + h
        expect_p = get_expect(circ, pr_p, sim, shots=shots, zne_n=zne_n)
        
        pr_n = copy.deepcopy(pr)
        pr_n[key] = value - h
        expect_n = get_expect(circ, pr_n, sim, shots=shots, zne_n=zne_n)
        
        grad = (expect_p - expect_n) / 2
        grads.append(grad)
    return expect, np.array(grads)


def get_best_params(circuit:Circuit, 
                    ham: QubitOperator,
                    ) -> tuple:
    """在无噪声和随机采样的情况下获取最佳结果。"""
    
    
    p0 = np.random.uniform(-np.pi, np.pi, len(circuit.params_name))
    grad_ops = Simulator('mqvector', circuit.n_qubits).get_expectation_with_grad(Hamiltonian(ham), circuit)

    def fun(x, grad_ops):
        f, g = grad_ops(x)
        f = f.real[0, 0]
        g = g.real[0, 0]
        return f, g

    res = minimize(fun, p0, (grad_ops, ), 'bfgs', True)
    return res.fun, res.x


def opti_under_noise(circ:Circuit, # 待优化的参数化线路
                     sim,
                     init_params:Optional[np.ndarray]=None, # 参数初猜值
                     tol:Optional[float]=None, # 退出时的更新阈值
                     max_step:Optional[int]=None, # 最大优化次数
                     shots:int=100, # 期望值评估是所用 shots
                     zne_n:int=0): # ZNE 法的对合数
    
    """使用 scipy.minimize 优化参数。"""
    
    if init_params is None:
        init_params = np.zeros(len(circ.params_name),)
        
    def fun(x):
        f, g = get_expect_with_grad(circ, x, sim, shots, zne_n)
        return f, g
    
    # print('optimizing ...') # 显示一次就是求一次梯度
    res = minimize(fun, 
                   init_params, 
                   method='bfgs',
                   jac=True, 
                   tol=tol, 
                   options={'maxiter':max_step,'disp':False})
    return res.fun, res.x
 

def solution(molecule, Simulator: HKSSimulator) -> float:
    """含参拟设, 在噪声环境下进行训练, 并得到结果。"""
    sim = Simulator
    mol = generate_molecule(molecule)
    
    ## 配置活性空间
    occupied_indices = [0]
    active_indices = [1, 2]
    abs_tol = 1e-2
    
    global n_qubits # 确定后续线路和模拟器所用的比特数
    ham, n_qubits = get_hamiltonian(mol, 
                                    occupied_indices, 
                                    active_indices, 
                                    abs_tol)
    
    global const, terms, grouped_ops # 设为全局变量，避免重复执行
    const, terms = split_hamiltonian(ham)
    grouped_ops = grouping_ops(terms)
    
    # 采用活性空间之后，hf 线路需发生改变
    n_hf = mol.n_electrons - 2 * len(occupied_indices) 
    shots = 100
    zne_n = 4

    ## HF 线路
    # circ_hf = get_hf_circ(n_hf=n_hf, ham=ham)
    # expect = Simulator('mqvector', n_qubits).get_expectation(Hamiltonian(ham), circ_hf).real
    # print("HF theory expect:", expect)
    
    # expect = get_expect(circ_hf, sim=sim, shots=shots, zne_n=zne_n)
    # print("HF noise expect:", expect)
    
    ## 含参 线路
    circ = get_ansatz_circ(n_hf=n_hf, ham=ham)
    
    ## 使用 mqvector 模拟器来计算 ansatz 理论上的最优参数和最低能量
    expect, params = get_best_params(circ, ham=ham) # 获取理论上的最佳值
    # print("ansatz theory expect:", expect)
    # print("ansatz best params:", params)
    
    
    ## 使用以上最有参数在噪声环境下获取期望
    # pr = dict(zip(circ.params_name, params))
    # expect = get_expect(circ, pr, sim=sim, shots=shots, zne_n=zne_n)
    # print("ansatz noise expect:", expect)
    
    ## 直接在噪声环境训练，获取期望
    expect, params = opti_under_noise(circ, sim=sim,
                                      init_params=params,
                                      shots=shots, 
                                      zne_n=zne_n, max_step=100)
    # print("ansatz noise training expect:", expect, "\nnoise params:", params)
    return expect


if __name__ == '__main__':
    import simulator
    
    global n_qubits
    n_qubits = None
    shots_counter = simulator.init_shots_counter()
    molecule = read_mol_data('mol.csv')
    res = solution(molecule, HKSSimulator)
    print(res)
    