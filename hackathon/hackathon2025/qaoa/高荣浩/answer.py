import numpy as np
import random

from samples.utils import build_ham_ising
from mindquantum.core.gates import I, X, CNOT, Z, H
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.operators import Hamiltonian


def solve(nqubit, Q_triu):
    """
    输入：问题比特数量 nqubit，特定格式矩阵 Q_triu；
    输出：返回用于求解任意给定 Ising 问题的 Clifford 电路。
    
    本方案使用候选算子池策略：在每次迭代中同时考察单比特（利用 X 门）和双比特更新，
    结合模拟退火准则逐步改善比特状态，每一步都在电路中添加对应的算符更新，
    最终返回的线路必为 Clifford 线路。
    """
    init_strategies = [
        ('no_flips', lambda: get_no_flips_init(nqubit)),
        ('all_flips', lambda: get_all_flips_init(nqubit)),        
        ('random', lambda: get_random_init(nqubit)),
        ('h', lambda: get_h_based_init(nqubit, np.diag(Q_triu)))
    ]

    best_circ = None
    best_energy = float('inf')

    for name, init_func in init_strategies:
        circ = example_new(nqubit, Q_triu, select_num=init_func())
        sim = Simulator('stabilizer', nqubit)
        ham = build_ham_ising(Q_triu)
        energy = sim.get_expectation(Hamiltonian(ham), circ).real
        
        if energy < best_energy:
            best_energy = energy
            best_circ = circ

    return best_circ


def get_random_init(nqubit):
    # 随机选择若干比特作为初始翻转（对应于应用 X 门后的比特），随机数量介于0与 nqubit 之间
    num = random.randint(0, nqubit)
    return random.sample(range(nqubit), num)


def get_no_flips_init(nqubit: int):
    """返回空列表，代表全 |0> 态"""
    return []


def get_all_flips_init(nqubit: int):
    """返回所有比特索引，代表全 |1> 态"""
    return list(range(nqubit))

def get_h_based_init(nqubit, h):
    # 根据 h 的大小选择初始态
    return [i for i in range(nqubit) if h[i] > 0]


def apply_initial_x_gates(circuit, select_num, nqubit):
    """
    根据输入的 select_num 对初始态（全 |0>，对应经典状态 +1）执行 X 门操作，
    并构造经典比特状态 bits，其中 |0> 对应 +1，|1> 对应 -1。
    """
    bits = np.ones(nqubit, dtype=int)
    for i in range(nqubit):
        if i in select_num:
            circuit += X.on(i)
            bits[i] = -1
    return circuit, bits


def example_new(nqubit, Q_triu, select_num):
    """
    新的优化方案：
    
    1. 初始化量子线路和经典比特状态（通过 X 门确定初始割分），构造耦合矩阵 J（对称）
       以及局部磁场 h（取自 Q_triu 的对角线）。
    2. 对于每个比特 i，计算单比特翻转的能量变化：
          delta[i] = -2 * bits[i] * (h[i] + dot(J[i, :], bits))
    3. 在循环迭代过程中，引入候选算子池：
         - 单比特候选：选择使 delta 最小（即能量变化最有利）的比特。
         - 每隔一定迭代次数（pair_interval），利用向量化方式计算所有比特对的能量变化，
           双比特候选的能量变化为
             delta_pair = delta[i] + delta[j] + 4 * J[i, j] * bits[i] * bits[j]
           该项考虑了比特之间的耦合效应。
    4. 根据候选操作的能量变化，如果能降低能量 (delta < 0)，则必然接受该操作；
       否则以模拟退火准则（概率 exp(-delta/temperature)）接受能量提升更新，
       从而增加跳出局部最优的可能性。
    5. 每一次更新后，都在电路中添加相应的算符更新（单比特使用 X 门，
       双比特更新时对每个比特均添加 X 门），同时更新经典状态 bits。注意：
       单比特更新时利用局部公式对 delta 向量进行修正，而双比特更新后重新计算整个 delta 向量。
    6. 随着迭代和温度下降，算法从探索转向稳定，直到满足提前退出条件或达到最大迭代次数。
    """
    # 构造耦合矩阵 J 和局部磁场 h
    J = np.triu(Q_triu, k=1) + np.triu(Q_triu, k=1).T
    h = np.diag(Q_triu)
    
    # 初始化量子线路和经典比特状态
    circ = Circuit(UN(I, nqubit))
    circ, bits = apply_initial_x_gates(circ, select_num, nqubit)
    
    # 计算初始的单比特能量变化 delta
    delta = np.empty(nqubit)
    for i in range(nqubit):
        delta[i] = -2 * bits[i] * (h[i] + np.dot(J[i, :], bits))
    
    # 模拟退火与迭代参数
    temperature = 1.0
    cooling_rate = 0.993
    max_iter = 5000
    pair_interval = 3 # 每 pair_interval 次迭代计算一次双比特候选

    for iteration in range(max_iter):
        # 单比特候选：选出使 delta 最小的比特
        best_single_index = int(np.argmin(delta))
        best_single_delta = delta[best_single_index]

        # 双比特候选：每 pair_interval 次计算一次候选
        best_pair_delta = np.inf
        best_pair = None
        if (iteration % pair_interval) == 0:
            indices_i, indices_j = np.triu_indices(nqubit, k=1)
            # 计算所有比特对翻转后的能量变化
            pair_delta_candidates = (delta[indices_i] + delta[indices_j]
                                     + 4 * J[indices_i, indices_j] * bits[indices_i] * bits[indices_j])
            min_pair_idx = np.argmin(pair_delta_candidates)
            best_pair_delta = pair_delta_candidates[min_pair_idx]
            best_pair = (int(indices_i[min_pair_idx]), int(indices_j[min_pair_idx]))
        
        # 从单比特和双比特候选中选择更优者
        if best_single_delta <= best_pair_delta:
            candidate_type = "single"
            candidate_delta = best_single_delta
            candidate = best_single_index
        else:
            candidate_type = "double"
            candidate_delta = best_pair_delta
            candidate = best_pair
        
        # 模拟退火准则：若候选能量变化小于0（能量降低）则必选，
        # 否则以 exp(-delta/temperature) 概率接受
        if candidate_delta < 0 or random.random() < np.exp(-candidate_delta / temperature):
            if candidate_type == "single":
                i = candidate
                # 在量子线路中添加单比特更新：X 门
                circ += X.on(i)
                old_val = bits[i]
                bits[i] = -bits[i]
                # 局部更新：对于所有 j (j ≠ i)，delta[j] 增加 4 * J[j, i] * bits[j] * old_val
                delta = delta + 4 * (J[:, i] * bits * old_val)
                # 重新计算被翻转比特 i 的 delta
                delta[i] = -2 * bits[i] * (h[i] + np.dot(J[i, :], bits))
            else:  # 双比特候选更新
                i, j = candidate
                # 在量子线路中分别为 i 和 j 添加 X 门，满足“每次经典更新后添加算符更新”
                circ += H.on(i)
                circ += H.on(j)
                circ += CNOT(j,i)
                circ += Z.on(j)
                circ += CNOT(j,i)
                circ += H.on(i)
                circ += H.on(j)
                bits[i] = -bits[i]
                bits[j] = -bits[j]
                # 双比特更新后，为保证 delta 精度，重新计算所有 delta
                for k in range(nqubit):
                    delta[k] = -2 * bits[k] * (h[k] + np.dot(J[k, :], bits))
        # 降温
        temperature *= cooling_rate

        # 如果所有候选均不产生改善，且温度极低，则提前退出
        if np.all(delta >= 0) and temperature < 1e-3:
            break

    return circ