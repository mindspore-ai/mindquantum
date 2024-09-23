import numpy as np                           
from mindquantum.core.gates import X, Y, Z, I, H, Rzz, RZ, RY, RX, Rxx, Ryy   
from mindquantum.simulator import Simulator  
from mindquantum.core.circuit import Circuit 
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.core.operators import QubitOperator, Hamiltonian, commutator
import networkx as nx
import re
import copy
import matplotlib.pyplot as plt
from openqaoa.utilities import ground_state_hamiltonian
from openqaoa.problems import MaximumCut
from openqaoa.utilities import plot_graph
from mindquantum.core.gates import gene_univ_parameterized_gate
import numba
from scipy.optimize import minimize


'''
   定义 Rzy 门，Rzy(θ) = exp(-i θ/2 (Y⊗Z) )
   .on([a,b])
   当 a>b, 作用 exp(-i θ/2 (Y⊗Z) )
   当 a<b, 作用 exp(-i θ/2 (Z⊗Y) )
   因为mindquantum 0.9.0 中的双比特旋转门无法实现 非对称。只能通过 该接口构建
'''
def matrix(alpha):
    ep = np.cos(alpha/2)
    em = np.sin(alpha/2)
    return np.array([
        [ep + 0.0j, 0.0j, -em + 0.0j, 0.0j],
        [0.0j, ep + 0.0j, 0.0j, em + 0.0j],
        [em + 0.0j, 0.0j, ep + 0.0j, 0.0j],
        [0.0j, -em + 0.0j, 0.0j, ep + 0.0j]
    ])
def diff_matrix(alpha):
    ep = -0.5 * np.sin(alpha/2)
    em = 0.5 * np.cos(alpha/2)
    return np.array([
        [ep + 0.0j, 0.0j, -em + 0.0j, 0.0j],
        [0.0j, ep + 0.0j, 0.0j, em + 0.0j],
        [em + 0.0j, 0.0j, ep + 0.0j, 0.0j],
        [0.0j, -em + 0.0j, 0.0j, ep + 0.0j]
    ])
Rzy = gene_univ_parameterized_gate('Rzy', matrix, diff_matrix)

'''
   对当前线路 circ 加上一层 Rzz 门
   Rzz门的参数由两种可能，str 或者 float/int
'''
def qaoa_hamil(circ, qubo, gamma):
    ising = qubo.terms
    weight = qubo.weights
    if isinstance(gamma, str):
        for index, term in enumerate(ising):
            pr = ParameterResolver({gamma: weight[index]})
            circ += Rzz(pr).on([term[0], term[1]])

    if isinstance(gamma, (int, float)):
        for index, term in enumerate(ising):
            circ += Rzz(gamma * weight[index]).on([term[0], term[1]])
    circ.barrier()

'''
   根据 mixer，对当前线路 circ 加上一层对应的 U
   文献中的 Pop 池子中的所有可能
'''
def qaoa_mixer(circ, mixer_str, beta, qubits):
    letters = re.findall(r'[A-Z]', mixer_str)
    numbers = re.findall(r'\d+', mixer_str)
    letters_str = ''.join(letters)
    nums = [int(num) for num in numbers]

    pr = ParameterResolver({beta: 2})

    if letters_str == 'XX':
        circ += Rxx(pr).on(nums)
    if letters_str == 'YY':
        circ += Ryy(pr).on(nums)
    if letters_str == 'YZ':
        circ += Rzy(pr).on([max(nums), min(nums)])
    if letters_str == 'ZY':
        circ += Rzy(pr).on([min(nums), max(nums)])
    if letters_str == 'X' * qubits:
        for i in range(qubits):
            circ += RX(pr).on(i)
    if letters_str == 'Y' * qubits:
        for i in range(qubits):
            circ += RY(pr).on(i)
    if letters_str == 'X':
        circ += RX(pr).on(nums)
    if letters_str == 'Y':
        circ += RY(pr).on(nums)
    circ.barrier()

'''
   单比特的 mixer：1.{Xj, Yj}(j=1,...,N)
   全比特的 mixer：2.{∑Xi, ∑Yi}
'''
def mixer_pool_single(qubits):
    pool = []

    single_X = [f'X{i}' for i in range(qubits)]
    pool.extend(single_X)

    all_X = ' '.join([f'X{i}' for i in range(qubits)])
    pool.extend([all_X])

    single_Y = [f'Y{i}' for i in range(qubits)]
    pool.extend(single_Y)

    all_Y = ' '.join([f'Y{i}' for i in range(qubits)])
    pool.extend([all_Y])

    return pool
'''
   双比特的 mixer：{XX, YY, YZ, ZY}
'''
def mixer_pool_multi(qubits):
    number_pairs = []
    for i in range(qubits):
        for j in range(i + 1, qubits):
            number_pairs.append(f'{i}{j}')

    letter_pairs = ['Y Z','Z Y','X X','Y Y']

    pool = []
    for num in number_pairs:
        for let in letter_pairs:
            pool.append(f'{let[0]}{num[0]} {let[2]}{num[1]}')
    return pool

'''
   得到问题的哈密顿量，Hc = 1/2 ∑(wZZ)
'''
def hamilC(qubo):
    def hamil_str(pos):
        result = f'Z{pos[0]} Z{pos[1]}'
        return result
    hamil_op = QubitOperator()
    for index, term in enumerate(qubo.terms):
        hamil_op += QubitOperator(hamil_str(term), qubo.weights[index] / 2)
    return hamil_op

'''
   构建问题的哈密顿量，Hc = 1/2 ∑(wZZ)
   qubo 是通过 openqaoa.problems 的 MaximumCut 生成的 maxcut graph
'''
def derivative(qubo, qubits, mixer, circ):
    commu = (-1j) * commutator(hamilC(qubo), QubitOperator(mixer))
    ham = Hamiltonian(commu)

    sim = Simulator("mqvector", qubits)
    expectation = sim.get_expectation(ham, circ)
    return expectation.real

'''
   传统 minimize COBYLA 和 bfgs 方法优化搜索，返回收敛值和参数值
   每层优化前的初始值 为 上一层的优化参数 和 这一层的[0.01,0]
'''
def opt(qubo, circ, qubits, theta, method):
    ham = Hamiltonian(hamilC(qubo))
    sim = Simulator('mqvector', qubits)
    grad_ops = sim.get_expectation_with_grad(ham, circ)

    if len(theta) < 2:
        p0 = np.array([0.01, 0.0])
    else:
        p = np.array(list(theta.values()))
        p0 = np.append(p, [0.01, 0.0])

    def fun(p, grad_ops):
        f, g = grad_ops(p)
        f = np.real(f)[0, 0]
        g = np.real(g)[0, 0]
        return f
    res = minimize(fun, p0, args=(grad_ops,), method=method)
    return dict(zip(circ.params_name, res.x)), res.fun

'''
    通过 derivative 计算寻找下一层的 mixer 之前，需要将线路 deepcopy
    theta: 当前所有参数（dict）
    mixers_used: 所有使用过的 mixer
    allthemin: 每层的优化结果，也就是每层的期望
'''
def ADAPT_QAOA(nodes, qubo, pool, layers, method):
    circ = Circuit()
    for qubit in range(nodes):
        circ += H.on(qubit)
    circ.barrier()

    k = 0
    theta = {}
    mixers_used = []
    allthemin = []

    while True:
        gradients = []

        circ_grad = copy.deepcopy(circ.apply_value(theta))

        qaoa_hamil(circ_grad, qubo, 0.01)

        for mixer in pool:
            gradients.append(derivative(qubo, nodes, mixer, circ_grad))
        mixers_used.append(pool[np.argmax(gradients)])

        if k == layers:
            return theta, allthemin, circ
        k += 1

        qaoa_hamil(circ, qubo, f'g{k}')
        qaoa_mixer(circ, mixers_used[-1], f'b{k}', nodes)

        result, qubo_min = opt(qubo, circ, nodes, theta, method)
        theta = result
        allthemin.append(qubo_min)
        
'''
    按要求，为了实现标准 QAOA 与 ADAPT 的可比性，也需要用到相同的 迭代策略
'''
def QAOA(qubo, qubits, layers, method):
    weight = qubo.weights
    circ = Circuit()
    for qubit in range(qubits):
        circ += H.on(qubit)
    circ.barrier()

    def build_hc(circ, g):
        for index, value in enumerate(qubo.terms):
            pr = ParameterResolver({g: weight[index]})
            circ += Rzz(pr).on(value)
        circ.barrier()

    def build_hb(circ, b):
        for i in range(qubits):
            pr = ParameterResolver({b: 2})
            circ += RX(pr).on(i)
        circ.barrier()

    allthemin = []
    k = 0
    theta = {}
    while True:
        if k == layers:
            return theta, allthemin, circ
        k += 1

        build_hc(circ, f'g{k}')
        build_hb(circ, f'b{k}')
        result, qubo_min = opt(qubo, circ, qubits, theta, method)
        theta = result
        allthemin.append(qubo_min)
'''
   用 MaximumCut 接口生成自定义的 maxcut graph
   qubo.weights 和 qubo.terms 可以获取该 graph 的权重列表和连线列表
   四种权值分布：uniform、exponential、normal、uniform2
   draw_graph: 图的绘制
'''
def graph_complete(nodes, distribution):
    G = nx.complete_graph(nodes)

    if distribution == 'uniform':
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = np.random.uniform(0, 1)
    if distribution == 'exponential':
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = np.random.exponential(1)
    if distribution == 'normal':
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = np.random.normal(0, 1)
    if distribution == 'uniform2':
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = np.random.uniform(-1, 1)
    return MaximumCut(G).qubo

import csv
def save_lists_to_csv(file_path, data_lists):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for data_list in data_lists:
            writer.writerow(data_list)




