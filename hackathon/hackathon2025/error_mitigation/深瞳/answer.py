import numpy as np
import h5py
import networkx as nx
import random
import json
import itertools
from scipy.optimize import minimize

TRAIN_SAMPLE_NUM = 0
BITSTRINGS_BASE = np.load('./samples/data/bitstrings_base.npz')['arr_0']


def get_data(state, qubits_number_list, random_seed=2025):
    """
    获取基本线路的“态制备-测量”实验结果；

    Args:
        state: 表示制备的量子态，为0-511整数，分别对应将比特制备在'000000000'-'000000001'-...-'111111111'态，最右边为q0状态，
        qubits_number_list: 形如[[比特组合, 测量结果数量], ...]的列表，
        比特组合形如：[0, 4, 6]，测返回测量结果'|q6,q4,q0>'；
        返回的测量结果为随机抽取的，可以设定随机数种子；对同一个态，不同比特组合的测量结果总数最大为50000；

    Returns:
        list: 比特串列表，

    Example:
        state = 5
        random_seed = 2025
        qubits_number_list = [
            [[0, 4, 6], 1000],
            [[1, 2, 7], 1000],
        ]
    """
    global TRAIN_SAMPLE_NUM

    _count = 0
    for _qubits, _number in qubits_number_list:
        if np.max(_qubits) >= 9:
            raise ValueError('The max index of qubit is 8.')
        _count += _number
    if _count > 50000:
        raise ValueError('The total number of samples obtained for each state should be no more than 50,000.')

    # 获取数据量计数
    for _qubits, _number in qubits_number_list:
        TRAIN_SAMPLE_NUM += len(_qubits) * _number

    # 随机采样顺序
    select_order = np.arange(50000)
    np.random.seed(random_seed)
    np.random.shuffle(select_order)

    acquired_data = [None, ] * len(qubits_number_list)

    for _idx, (_qubits, _number) in enumerate(qubits_number_list):
        bitstring_arr = BITSTRINGS_BASE[state]
        # 获取前_number个测量中_qubits比特的结果
        acquired_data[_idx] = bitstring_arr[select_order[:_number]][:, _qubits]
        # 去掉已获取的数据
        bitstring_arr = bitstring_arr[_number:]

    return acquired_data


# ************************************************************************** 请于以下区域内作答 **************************************************************************

random_seed = 2025
q_num = 9
max_sample = 2 ** 12

def num2bit_string(nqubits, number, reverse=False) -> str:
    # MSB is on the right
    """convert iteragal number to bit string

    Args:
        nqubits (_type_): system qubit numbers
        number (_type_): iteragal number
        reverse (bool, optional): MSB on the right side or left side. Defaults to False.

    Returns:
        str: bit string
    """
    if reverse:
        return bin(number)[2:].zfill(nqubits)[::-1]
    return bin(number)[2:].zfill(nqubits)


def dict2vec(nqubits, res_dict, reverse=False) -> np.ndarray:
    """convert a dict to a numpy vector

    Args:
        nqubits (int): an integer, the number of qubits
        res_dict (dict): a dict, the result of a quantum circuit
        reverse (bool, optional): MSB on the right side or left side. Defaults to False.

    Returns:
        np.ndarray: a numpy vector
    """
    N = 2 ** nqubits
    vec = np.zeros(N)
    for i in range(N):
        if num2bit_string(nqubits, i) in res_dict:
            vec[i] = res_dict[num2bit_string(nqubits, i, reverse=reverse)]
    return vec


def bitstring_arr2vec(bitstring_arr, q_num=9):
    """
    将一组比特串转换成概率，最右边为第一个比特；
    即q0=1其他比特为0时 = [0, 0, 0, 0, 0, 0, 0, 0, 1]；

    Example:
        bitstring_arr = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0],
            ...
        ]
        q_num = 9 # 比特数量
    """
    # 用base_count_dict储存所有的比特串
    # 所有测量态的计数初始化为0
    base_count_dict = {}
    for _idx in range(2 ** q_num):
        base_count_dict[bin(_idx)[2:].zfill(q_num)] = 0
    # 遍历并对所有测量结果计数
    for _arr in bitstring_arr:
        bitstring = ''.join(_arr.astype(str))
        base_count_dict[bitstring] += 1
    # 总数
    _count = 0
    for _key, _value in base_count_dict.items():
        _count += base_count_dict[_key]
    # 除以总数得到概率
    base_prob_dict = {}
    for _key, _value in base_count_dict.items():
        base_prob_dict[_key] = base_count_dict[_key] / _count
    return base_prob_dict


def normalize(v):
    return v / np.sum(v)


# QuFEM权重构造
def get_max_weight_edge(G):
    max_weight = -float('inf')
    max_edge = None
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        if weight > max_weight:
            max_weight = weight
            max_edge = (u, v)
    return max_edge, max_weight


def get_error_single(x, i, circuits, sample_num):
    error = 0
    k = 0  # 用来计数，看有贡献的线路有多少
    P = []
    for circuit in circuits:
        circuit_x = int(num2bit_string(q_num, circuit)[x])
        if circuit_x == i:
            bitstring_arr = get_data(state=circuit, qubits_number_list=[[[x], sample_num]],
                                     random_seed=random_seed)
            base_prob_dict = bitstring_arr2vec(bitstring_arr[0], q_num=1)
            P = dict2vec(nqubits=1, res_dict=base_prob_dict)
            error += P[1 - circuit_x]
            k += 1
    error /= k
    return error


def get_error_double(x, i, y, j, circuits, sample_num):
    error = 0
    k = 0
    P = []
    for circuit in circuits:
        circuit_x = int(num2bit_string(q_num, circuit)[x])
        circuit_y = int(num2bit_string(q_num, circuit)[y])
        if circuit_x == i and circuit_y == j:
            bitstring_arr = get_data(state=circuit, qubits_number_list=[[[x], sample_num]],
                                     random_seed=random_seed)
            base_prob_dict = bitstring_arr2vec(bitstring_arr[0], q_num=1)
            P = dict2vec(nqubits=1, res_dict=base_prob_dict)
            error += P[1 - circuit_x]
            k += 1
    if k != 0:
        error /= k
    return error


def get_weight(x, y, circuits, sample_num):
    # 交互项一共有8项
    weight = 0
    for i in range(2):
        error_single_1 = get_error_single(x, i, circuits, sample_num)  # 表示x比特理想为i时的误差
        for j in range(2):
            error_single_2 = get_error_single(y, j, circuits, sample_num)
            error_double_1_2 = get_error_double(x, i, y, j, circuits, sample_num)  # y 对 x 的影响
            error_double_2_1 = get_error_double(y, j, x, i, circuits, sample_num)  # x 对 y 的影响
            weight += np.abs(error_double_1_2 - error_single_1)
            weight += np.abs(error_double_2_1 - error_single_2)
    return weight

# 计算组间割权重
def calculate_inter_group_cut(G, groups):
    # 将分组列表转换为字典：{节点: 组ID}
    group_dict = {}
    for group_id, node_list in enumerate(groups):
        for node in node_list:
            group_dict[node] = group_id

    total_weight = 0
    for u, v in G.edges():
        if group_dict.get(u) != group_dict.get(v):  # 检查节点是否在不同组
            # 获取边权重（默认1）
            total_weight += G.edges[u, v].get('weight', 1)
    return total_weight


def generate_partitions(remaining_nodes, groups_size):
    if not groups_size:
        return [[]]
    first_size = groups_size[0]
    if len(remaining_nodes) < first_size:
        return []
    first_node = remaining_nodes[0]
    other_nodes = remaining_nodes[1:]
    if len(other_nodes) < first_size - 1:
        return []
    partitions = []
    for subset in itertools.combinations(other_nodes, first_size - 1):
        current_group = [first_node] + list(subset)
        new_remaining = [n for n in remaining_nodes if n not in current_group]
        for sub_partition in generate_partitions(new_remaining, groups_size[1:]):
            partitions.append([current_group] + sub_partition)
    return partitions

def calculate_cut_weight(partition, edges):
    node_to_group = {}
    for group_idx, group in enumerate(partition):
        for node in group:
            node_to_group[node] = group_idx
    cut_weight = 0
    for u, v, *data in edges:
        weight = 1
        if data:
            if isinstance(data[0], dict):
                weight = data[0].get('weight', 1)
            else:
                weight = data[0]
        if node_to_group[u] != node_to_group[v]:
            cut_weight += weight
    return cut_weight

def find_optimal_partition(G, groups_size):
    nodes = sorted(G.nodes)
    all_partitions = generate_partitions(nodes, groups_size)
    if not all_partitions:
        raise ValueError("No valid partition exists for the given group sizes.")
    min_cut = float('inf')
    best_partition = None
    for partition in all_partitions:
        current_cut = calculate_cut_weight(partition, G.edges(data=True))
        if current_cut < min_cut:
            min_cut = current_cut
            best_partition = partition
    return best_partition


def generate_valid_reshuffled_groupings(original_groups):
    """
    生成所有符合条件的新分组：每个新子集包含原每个子集的一个元素，且结构相同
    """
    # 将原子组转换为列表的列表以便处理
    original = [list(group) for group in original_groups]

    # 生成每个原子组元素的所有排列
    perms = [list(itertools.permutations(group)) for group in original]

    # 所有排列的笛卡尔积组合
    all_perm_combos = itertools.product(*perms)

    valid_groupings = []

    for combo in all_perm_combos:
        # 构造新分组：每个新子集由各原子组的对应位置元素组成
        new_groups = [
            sorted([combo[i][j] for i in range(len(original))])
            for j in range(len(original[0]))
        ]
        # 规范化为排序后的分组
        sorted_groups = sorted([sorted(group) for group in new_groups], key=lambda x: x[0])
        valid_groupings.append(sorted_groups)

    # 去重
    unique_groupings = []
    seen = set()
    for group in valid_groupings:
        t = tuple(tuple(sub) for sub in group)
        if t not in seen:
            seen.add(t)
            unique_groupings.append(group)

    return unique_groupings


def find_min_cut_reshuffled_group(G, original_groups):
    """
    找到割权重最小的新分组（排除原分组）
    """
    # 生成所有有效的新分组
    candidate_groups = generate_valid_reshuffled_groupings(original_groups)

    # 将原分组规范化为可比较的格式
    original_sorted = sorted([sorted(group) for group in original_groups], key=lambda x: x[0])

    min_cut = float('inf')
    best_group = None

    for group in candidate_groups:
        # 排除原分组
        if group == original_sorted:
            continue

        # 转换为节点到组ID的映射


        # 计算割权重
        cut = calculate_inter_group_cut(G, group)

        if cut < min_cut:
            min_cut = cut
            best_group = group

    return best_group, min_cut

def get_sub_matrix(sub_group, sample):
    # L = []
    L = np.zeros((2 ** len(sub_group), 2 ** len(sub_group)))
    for true_state in range(2 ** q_num):
        true_string = num2bit_string(q_num, true_state)
        # 获得该态在子矩阵上的下标
        index = 0
        k = len(sub_group)
        for i in sub_group:
            index += int(true_string[i]) * (2 ** (k - 1))
            k -= 1
        bitstring_arr = get_data(state=true_state, qubits_number_list=[[sub_group, sample]],
                                 random_seed=random_seed)
        base_prob_dict = bitstring_arr2vec(bitstring_arr[0], q_num=len(sub_group))
        '''
        if true_state == 0:
            L[index] = dict2vec(nqubits=len(sub_group), res_dict=base_prob_dict)

        # 转换为arr
        else:
            L[index] += dict2vec(nqubits=len(sub_group), res_dict=base_prob_dict)
        '''
        L[index] += dict2vec(nqubits=len(sub_group), res_dict=base_prob_dict)
    # 矩阵列归一化

    L /= L.sum(axis=1, keepdims=True)
    return L

# ************************************************************************** 请于以上区域内作答 **************************************************************************


def correct(measure_prob_list):
    """
        将六个量子线路的测量结果measure_prob_list分别修正并返回，
        请参赛选手保持本函数的输入measure_prob_list和输出corrected_prob_list格式不变；

        输入值measure_prob_list为包含6个量子线路“原始”测量概率的列表，
        返回值corrected_prob_list为包含6个量子线路“修正后”测量概率的列表，
    """

    global TRAIN_SAMPLE_NUM
    # ************************************************************************** 请于以下区域内作答 **************************************************************************

    # 获取“态制备-测量”结果，并建立修正矩阵
    sample_num_1 = 2 ** 7
    power = 6
    N_sum = q_num * power
    sum = range(2 ** q_num)
    selected = random.sample(sum, N_sum)

    # 构造加权图
    G = nx.Graph()
    for i in range(q_num):
        for j in range(i+1, q_num):
            weight = get_weight(i, j, selected, sample_num_1)
            G.add_edge(i, j, weight=weight)

    # 对图进行切割分组
    group_size = [3, 3, 3]
    groups = []
    group1 = find_optimal_partition(G, group_size)
    groups.append(group1)
    group2, min_cut2 = find_min_cut_reshuffled_group(G, group1)
    groups.append(group2)
    sample_num = []
    for i in range(len(groups)):
        sample_num_2 = np.zeros(len(groups[i]))
        for j in range(len(groups[i])):
            sample_num_2[j] = int(max_sample * (9 ** 2) / (len(groups[i][j]) ** 2) * (2 ** (len(groups[i][j]) - 9)))
        sample_num.append(sample_num_2)
    # 对每个组建立子噪声矩阵
    W = []
    for _id, group in enumerate(groups):
        d = 1 / (2 ** _id)
        M = []
        for id, grou in enumerate(group):
            J = get_sub_matrix(grou, int(sample_num[_id][id]))
            for i in range(len(J)):
                for j in range(len(J[i])):
                    if i != j:
                        J[i][j] *= d
                        J[i][i] -= J[i][j] * (d - 1)
            M.append(J)
        W.append(M)
    X = []
    B = []
    V = []
    for group in groups:
        R = []
        L = []
        Q = []
        for j in range(len(group)):
            for k in range(len(group[j])):
                R.append(group[j][k])
                Q.append(group[j][k])
            L.append(2 ** len(group[j]))
        for k in range(q_num):
            Q[R[k]] = k
        X.append(R)
        B.append(L)
        V.append(Q)

    # 将六个量子线路的测量结果measure_prob_list分别修正并返回

    corrected_prob_list = []
    beta = 2 / 50000  # 用于剪枝

    for measure_prob in measure_prob_list:
        # 将向量写成张量形式，并排序方便运算
        sub_measure_prob = measure_prob
        for t in range(len(groups)):
            sub_measure_prob = sub_measure_prob.reshape([2] * q_num)
            sub_measure_prob = sub_measure_prob.transpose(X[t])
        # sub_measure_prob = sub_measure_prob.flatten()
            sub_measure_prob = sub_measure_prob.reshape(B[t])
            for i, (group, matrix) in enumerate(zip(groups[t], W[t])):
                inv_matrix = np.linalg.inv(matrix.T)
            # 模拟张量积校准
                sub_measure_prob = np.tensordot(inv_matrix, sub_measure_prob, axes=(1, i))
            # 稀疏剪枝 (有用)
                sub_measure_prob[sub_measure_prob < beta] = 0  # 将负概率抹去
                sub_measure_prob = np.moveaxis(sub_measure_prob, 0, [i])
            sub_measure_prob = sub_measure_prob.flatten()
            sub_measure_prob = sub_measure_prob.reshape([2] * q_num)
            sub_measure_prob = sub_measure_prob.transpose(V[t])
            sub_measure_prob = sub_measure_prob.flatten()
        corrected_prob_list.append(sub_measure_prob)

    # ************************************************************************** 请于以上区域内作答 **************************************************************************

    return corrected_prob_list, TRAIN_SAMPLE_NUM