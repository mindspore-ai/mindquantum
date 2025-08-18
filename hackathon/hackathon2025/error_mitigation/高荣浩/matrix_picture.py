import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter, context
from mindspore.common.initializer import initializer, XavierUniform

TRAIN_SAMPLE_NUM = 0
# BITSTRINGS_BASE = np.load('./samples/data/bitstrings_base.npz')['arr_0']
BITSTRINGS_BASE = np.load('/Users/fangaoming/Desktop/比赛/HuaWei_2025/hackathon-readout/samples/data/bitstrings_base.npz')['arr_0']

import matplotlib.pyplot as plt
# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统常用中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

def single_circuit_score(P, Q):
    # 归一化函数
    def normalize(v):
        return v / np.sum(v)

    # 检查并归一化 P
    if not np.isclose(np.sum(P), 1.0):  # 检查是否归一化
        print(f"P is not normalized. sum(P) = {np.sum(P)}; Normalizing P...")
        P = normalize(P)

    # 检查并归一化 Q
    if not np.isclose(np.sum(Q), 1.0):  # 检查是否归一化
        print(f"Q is not normalized. sum(Q) = {np.sum(Q)}; Normalizing Q...")
        Q = normalize(Q)

    # 计算总变差距离 (TVD)
    TVD = np.sum(np.abs(P - Q)) / 2

    # 计算 score
    return 1 - TVD


# 读取 .npy 矩阵文件
matrix_1 = np.load("/Users/fangaoming/Desktop/比赛/HuaWei_2025/hackathon-readout/随机矩阵法结果/rabdom_matrix_ground_state.npy")
matrix_2 = np.load("/Users/fangaoming/Desktop/比赛/HuaWei_2025/hackathon-readout/随机矩阵法结果/rabdom_matrix_mixed.npy")
matrix_3 = np.load("/Users/fangaoming/Desktop/比赛/HuaWei_2025/hackathon-readout/随机矩阵法结果/matrix_inversion_complex_state.npy")

matrix_1_score_list = []
matrix_2_score_list = []
matrix_3_score_list = []

q_num = 9
dim = 2 ** q_num
samples_per_state = 50000
for true_state in range(2 ** q_num):
    bitstring_arr = get_data(state=true_state, qubits_number_list=[[np.arange(q_num), samples_per_state]], random_seed=2025)
    base_prob_dict = bitstring_arr2vec(bitstring_arr[0], q_num=q_num)
    noise_state = dict2vec(nqubits=q_num, res_dict=base_prob_dict)

    # 理想概率分布(one-hot向量)
    ideal_prob = np.zeros(dim)
    ideal_prob[true_state] = 1.0

    # 矩阵校准 1 
    matrix_1_prob = matrix_1 @ noise_state
    matrix_1_prob = np.maximum(matrix_1_prob, 0)  # 确保概率非负
    matrix_1_prob /= np.sum(matrix_1_prob)  # 归一化
    matrix_1_score = single_circuit_score(P=ideal_prob, Q=matrix_1_prob)
    matrix_1_score_list.append(matrix_1_score)

    # 矩阵校准 2
    matrix_2_prob = matrix_2 @ noise_state
    matrix_2_prob = np.maximum(matrix_2_prob, 0)
    matrix_2_prob /= np.sum(matrix_2_prob)
    matrix_2_score = single_circuit_score(P=ideal_prob, Q=matrix_2_prob)
    matrix_2_score_list.append(matrix_2_score)

    # 矩阵校准 3
    matrix_3_prob = matrix_3 @ noise_state
    matrix_3_prob = np.maximum(matrix_3_prob, 0)
    matrix_3_prob /= np.sum(matrix_3_prob)
    matrix_3_score = single_circuit_score(P=ideal_prob, Q=matrix_3_prob)
    matrix_3_score_list.append(matrix_3_score)


# 计算均值
value_1 = np.mean(np.array(matrix_1_score_list))
value_2 = np.mean(np.array(matrix_2_score_list))
value_3 = np.mean(np.array(matrix_3_score_list))

print(f'matrix_1_score_list = {value_1}')
print(f'matrix_2_score_list = {value_2}')
print(f'matrix_3_score_list = {value_3}')


# 在print语句后添加绘图代码
plt.figure(figsize=(10, 6))
# 将原来的plt.plot改为plt.scatter
# 修改散点图标记样式
plt.scatter(range(512), matrix_1_score_list, label='随机矩阵法(1)', marker='o', s=10)  # 原点
plt.scatter(range(512), matrix_2_score_list, label='随机矩阵法(2)', marker='*', s=10)  # 星点
plt.scatter(range(512), matrix_3_score_list, label='随机矩阵法(3)', marker='^', s=10)  # 三角形
plt.xlabel('Index (0-511)')
plt.ylabel('1-TVD')
# plt.title('各基态在随机矩阵法下的 1-TVD 值')
# 修改图例位置到左上角
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


