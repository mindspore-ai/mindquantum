import numpy as np
import random

from samples.utils import build_ham_ising
from mindquantum.core.gates import I, X,H
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.operators import Hamiltonian




# 函数输入输出不可修改
def solve(nqubit, Q_triu):
    """输入：问题比特数量nqubit，特定格式矩阵：Qtriu；
    输出：求解方法针对任意给定问题返回的求解线路。"""
    # 下面是可以修改的内容

    #可以进一步调节参数值
    best_circ = YJC002(nqubit, Q_triu, T_max=5,T_min=0.01, sweeps=20000*1, sweeps_per_beta=1, seed=0,pre_depth=int(len(Q_triu)/10))

    # 上面是可以修改的内容
    return best_circ

# 案例1：贪心算法
def apply_x_gates_and_get_lists(circuit, select_num, nqubit=None):
    bits = np.zeros(nqubit)

    for i in range(nqubit):
        if i in select_num:
            circuit += X.on(i)
            bits[i] = -1
        else:
            bits[i] = 1

    return circuit, bits

# 案例1：贪心算法
def apply_x_gates_and_get_lists(circuit, select_num, nqubit=None):
    bits = np.zeros(nqubit)

    for i in range(nqubit):
        if i in select_num:
            circuit += X.on(i)
            bits[i] = -1
        else:
            bits[i] = 1

    return circuit, bits

def example1_1(nqubit, Q_triu, select_num=None, repeat=1):
    J = np.triu(Q_triu, k=1) + np.triu(Q_triu, k=1).T
    h = np.diag(Q_triu)

    best_circ = None
    best_energy = 100
    best_depth = None
    sim = Simulator('stabilizer', nqubit)
    for ii in range(repeat):
        circ = Circuit(UN(I, nqubit))
        if select_num is None:
            num1 = random.randint(0, nqubit)
            select_num = random.sample(range(nqubit), num1)

        circ, bits = apply_x_gates_and_get_lists(circ, select_num, nqubit)

        depth = 0
        flag = 0
        while True:
            delta_E = (2*J @ bits + h) * -2 * bits
            index = np.argmin(delta_E)
            bits[index] = -bits[index]
            if delta_E[index] < 0:
                circ += X.on(int(index))
                depth += 1
            elif flag<10:
                circ += X.on(int(index))
                depth += 1
                flag+=1
            else:
                break

        ham = build_ham_ising(Q_triu)
        sim.reset()
        exp = sim.get_expectation(Hamiltonian(ham), circ)
        energy = exp.real

        if energy < best_energy:
            best_energy = energy
            best_circ = circ.copy()
            best_depth = depth

        print('depth:',best_depth)
    return best_circ

def convert_to_sparse(J, h):
    """
    将矩阵J和向量h转换为稀疏表示形式。
    
    参数:
        J (numpy.ndarray): 对称矩阵，表示变量之间的耦合关系。
        h (numpy.ndarray): 一维数组，表示变量的局部场。
    
    返回:
        ldata (list): h的稀疏表示。
        irow (list): 非零元素的行索引。
        icol (list): 非零元素的列索引。
        qdata (list): 非零元素的值。
    """
    J = np.array(J)
    num_vars = len(h)
    ldata = list(h) # h的稀疏表示

    irow, icol, qdata = [], [], []
    for i in range(num_vars):
        for j in range(i, num_vars):  # 只遍历上三角部分，因为J是对称的
            if abs(J[i, j]) > 1e-10:  # 检查是否为非零元素
                irow.append(i)
                icol.append(j)
                qdata.append(J[i, j])

    return ldata, irow, icol, qdata

def get_flip_energy(var, state, h, degrees, neighbors, neighbour_couplings):
    """
    计算翻转变量var时的能量变化。
    
    参数:
        var (int): 要翻转的变量索引。
        state (list): 当前变量的状态。
        h (list): 局部场的稀疏表示。
        degrees (list): 每个变量的度数。
        neighbors (list): 每个变量的邻居列表。
        neighbour_couplings (list): 每个变量与其邻居的耦合权重。
    
    返回:
        energy (float): 翻转变量var时的能量变化。
    """

    energy = h[var]
    for n_i in range(degrees[var]):
        energy += state[neighbors[var][n_i]] * neighbour_couplings[var][n_i]
    return -2 * state[var] * energy

def YJC002(nqubit, Q_triu, T_max=5,T_min=1, sweeps=5000, sweeps_per_beta=100, seed=0,pre_depth=0):
    """
    实现基于量子退火的优化算法。
    
    参数:
        nqubit (int): 变量的数量。
        Q_triu (numpy.ndarray): 上三角矩阵，表示问题的哈密顿量。
        T_max (float): 初始温度。
        T_min (float): 最小温度。
        sweeps (int): 总迭代次数。
        sweeps_per_beta (int): 每个温度下的迭代次数。
        seed (int): 随机种子。
        pre_depth (int): 预处理阶段的最大深度。
    
    返回:
        best_circ (Circuit): 最优解对应的量子电路。
    """
    J = np.triu(Q_triu, k=1) + np.triu(Q_triu, k=1).T
    h = np.diag(Q_triu)
    # 归一化J和h
    J_max=np.max(np.abs(J))
    J0=J/J_max
    h0=h/J_max

    hdata, coupler_starts, coupler_ends, coupler_weights = convert_to_sparse(2*J0, h0)
    # 构造温度调度
    start = T_max
    end = T_min
    num = int(sweeps/sweeps_per_beta)

    # 计算几何平均分布的值
    ratio = (end / start) ** (1 / (num - 1))
    beta_schedule = [start * (ratio ** i) for i in range(num)]

    num_vars = nqubit
    # 初始化变量的度数、邻居列表和耦合权重
    degrees = [0] * num_vars
    neighbors = [[] for _ in range(num_vars)]
    neighbour_couplings = [[] for _ in range(num_vars)]
    
    for cplr in range(len(coupler_starts)):
        u = coupler_starts[cplr]
        v = coupler_ends[cplr]
        neighbors[u].append(v)
        neighbors[v].append(u)
        neighbour_couplings[u].append(coupler_weights[cplr])
        neighbour_couplings[v].append(coupler_weights[cplr])
        degrees[u] += 1
        degrees[v] += 1
    
    # 初始化量子模拟器和随机状态
    sim = Simulator('stabilizer', nqubit)

    random.seed(seed)
    state_rondom = [random.choice([-1, 1]) for _ in range(num_vars)]
    state=state_rondom.copy()
    circ = Circuit(UN(I, nqubit))
    for i,value in enumerate(state):
        if value == -1:
            circ += X.on(i)
    # 预处理阶段：通过局部更新降低能量
    depth=0
    while depth<pre_depth:
        delta_E = (2*J @ state + h) * -2 * state # 计算每个变量的能量变化
        index = np.argmin(delta_E) # 找到能量变化最小的变量
        state[index] = -state[index] # 翻转该变量
        if delta_E[index] < 0: # 如果能量降低，则更新电路
            circ += X.on(int(index))
            depth += 1
        else:
            break

    best_circ = circ.copy() # 记录当前最优电路
    best_x = state.copy()  # 记录当前最优状态
    current_energy = state_rondom@J0@state_rondom+h0@state_rondom # 计算当前能量
    best_energy = current_energy # 初始化最优能量
    # 计算每个变量翻转后的能量变化
    delta_energy = [0.0] * num_vars
    for var in range(num_vars):
        delta_energy[var] = get_flip_energy(var, state, hdata, degrees, neighbors, neighbour_couplings)
    # 模拟退火过程
    for beta_idx in range(len(beta_schedule)):
        beta = beta_schedule[beta_idx]  # 当前温度的倒数
        for _ in range(sweeps_per_beta):

            threshold = 5 * beta # 设置能量变化的阈值

            for var in range(num_vars):
                if delta_energy[var] >= threshold: # 如果能量变化大于阈值，跳过
                    continue
                 # 判断是否接受翻转
                if delta_energy[var] <= 0.0 or np.exp(-delta_energy[var] / beta) > random.random():

                    current_energy += delta_energy[var]
                    # 更新邻居的能量变化
                    multiplier = 4 * state[var]
                    for n_i in range(degrees[var]):
                        neighbor = neighbors[var][n_i]
                        delta_energy[neighbor] += multiplier * neighbour_couplings[var][n_i] * state[neighbor]
                    state[var] *= -1 # 翻转变量
                    delta_energy[var] *= -1 # 更新该变量的能量变化
                # 如果找到更好的解，更新最优解
                if current_energy < best_energy:
                    best_energy = current_energy

                    differing_indices = [i for i in range(len(state)) if state[i] != best_x[i]]
                    for ii in differing_indices:
                        circ += X.on(int(ii))
                        depth += 1

                    best_x = state.copy()
                    best_circ = circ.copy()

    print('depth:',depth) # 输出最终的电路深度

    return best_circ # 返回最优解对应的量子电路
