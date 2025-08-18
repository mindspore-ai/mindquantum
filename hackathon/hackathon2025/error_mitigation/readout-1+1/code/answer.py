import numpy as np
import h5py

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
from scipy.optimize import minimize
from scipy.sparse.linalg import gmres, bicgstab
from scipy.sparse import diags
def normalize(v):
    return v / np.sum(v)

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


def MatrixInversion(ymes, Rin):
    # Rin is a matrix where the first coordinate is the measured value and the second coordinate is the true value.
    # Rin[m][t],measured value, truth value, ymes means m, m=Rt
    return np.ravel(np.matmul(np.linalg.inv(np.matrix(Rin)), ymes))

def calculate_moments(arr):

    # 一阶矩：均值
    mean = np.mean(arr)
    
    # 二阶矩：方差（总体方差，除以 N）
    variance = np.var(arr)
    
    # 三阶矩：偏度（手动计算）
    # 计算三阶中心矩
    centered = arr - mean
    third_moment = np.mean(centered ** 3)
    
    # 计算标准差（用于归一化偏度）
    std = np.std(arr)
    skewness = third_moment / (std ** 3) if std != 0 else 0
    
    return mean, variance, skewness,std


def get_a(skewness):
    #可以修正高阶a函数
    a = 0.40

    return a

def IBU_2(ymes: np.ndarray, t0: np.ndarray, Rin: np.ndarray, n: int) -> np.ndarray:
    """
    使用 NumPy 向量化优化的迭代贝叶斯展开（Iterative Bayesian Unfolding, IBU）方法。

    参数：
    `ymes`：测量的概率分布（形状：`(m,)`）
    `t0`：对真实分布的初始猜测（形状：`(k,)`）
    `Rin`：响应矩阵（测量值与真实值之间的关系，形状：`(m, k)`）
    `n`：迭代次数

    返回值：
    - 修正后的概率分布（形状：`(k,)`）
    """
    tn = t0.copy()
    for _ in range(n):
        print(f'IBU iteration = {_}')
        # 计算每个测量结果的预测概率
        myden = Rin.dot(tn)  # 形状: (m,)
        # 仅处理分母大于零的情况
        valid = myden > 0
        # 向量化计算每个项的贡献
        term = (Rin[valid] * tn) * ymes[valid][:, np.newaxis] / myden[valid][:, np.newaxis]
        # 对有效项求和得到新的概率分布
        tn = term.sum(axis=0)
    return tn

def ignis_method(R, m):
    """
    使用优化方法求解线性方程 R @ t = m 的解 t。

    参数:
        R (numpy.ndarray): 矩阵 R，形状为 (n_bins, n_bins)，表示系统的响应矩阵。
        m (numpy.ndarray): 向量 m，形状为 (n_bins,)，表示测量结果。

    返回:
        numpy.ndarray: 优化得到的解 t，形状为 (n_bins,)。
    """
    n_bins = R.shape[0]
    
    # 定义优化目标函数
    def objective(t):
        """
        计算目标函数值，即 R @ t 与测量结果 m 的平方误差之和。

        参数:
            t (numpy.ndarray): 当前的解向量，形状为 (n_bins,)。

        返回:
            float: 目标函数值。
        """
        # 确保t的形状为 (n_bins,)
        t = t.reshape(-1)
        # 计算R @ t与测量结果m的差异
        return np.sum((R @ t - m) ** 2)

    # 初始猜测为均匀分布
    t0 = np.ones(n_bins) / n_bins
    # 约束条件：t的L1范数等于m的L1范数
    constraints = [
        {'type': 'eq', 'fun': lambda t: np.sum(t) - np.sum(m)}
    ]
    
    # 边界条件：t的每个元素必须非负
    bounds = [(0.0, None) for _ in range(n_bins)]
    
    # 使用SLSQP优化器求解
    result = minimize(
        objective, 
        t0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    # 返回优化结果
    return result.x

def vec2dict(vec, reverse=False) -> dict:
    """Convert a numpy vector to a dictionary where keys are binary strings and values are vector elements.

    Args:
        vec (np.ndarray): The numpy vector to convert.
        reverse (bool, optional): If True, the binary strings are reversed. Defaults to False.

    Returns:
        dict: A dictionary mapping binary strings to vector values.
    """
    if not isinstance(vec, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    nqubits = int(np.log2(len(vec)))
    res_dict = {}
    
    for i in range(len(vec)):
        # if vec[i] != 0:
            bit_string = num2bit_string(nqubits, i, reverse=reverse)
            res_dict[bit_string] = vec[i]
    
    return res_dict


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

    # # 获取“态制备-测量”结果，并建立修正矩阵
    # q_num = 9
    # sample_num = 1000
    # random_seed = 2025
    # R = np.zeros((2 ** q_num, 2 ** q_num))
    # for true_state in range(2 ** q_num):
    #     bitstring_arr = get_data(state=true_state, qubits_number_list=[[np.arange(q_num), sample_num]], random_seed=random_seed)
    #     # print(bitstring_arr)
    #     base_prob_dict = bitstring_arr2vec(bitstring_arr[0], q_num=q_num)
    #     # print(base_prob_dict)
    #     # 转换为arr
    #     R[true_state] = dict2vec(nqubits=q_num, res_dict=base_prob_dict)

########################
    sample_num = 5000
    random_seed = 114

    all_num=9 # 总的量子比特数量
    add_num=0  # 用于记录已经处理过的量子比特数量
    R_all=[] # 用于存储每个子矩阵R0
    q_nums=[4,5] # 每次处理的量子比特数量列表
    for q_num in q_nums:
        R0 = np.zeros((2 ** q_num, 2 ** q_num))     # 初始化一个2^q_num x 2^q_num的零矩阵R0，用于存储概率矩阵
        for true_state in range(2 ** q_num):
            bitstring_arr = get_data(state=true_state*(2**add_num), qubits_number_list=[[list(range(all_num-q_num,all_num)), sample_num]], random_seed=random_seed)
            base_prob_dict = bitstring_arr2vec(bitstring_arr[0], q_num=q_num)        # 将bitstring数组转换为概率向量
            R0[true_state] = dict2vec(nqubits=q_num, res_dict=base_prob_dict)        # 将概率字典转换为向量，并存储到矩阵R0的对应行
        add_num+=q_num     # 更新add_num和all_num，为下一次循环做准备
        all_num-=q_num
        R_all.append(R0)     # 将当前的R0矩阵添加到R_all列表中

    R = np.ones((1, 1)) # 初始化一个1x1的单位矩阵R
    for matrix in R_all: # 遍历R_all中的所有子矩阵，通过Kronecker积将它们组合成最终的概率矩阵R
        R = np.kron(R, matrix)

    R[R<0.00001]=0 # 将矩阵R中小于0.00001的值设为0，以去除极小值的影响

    corrected_prob_list = []

    for measure_prob in measure_prob_list: # 遍历测量概率列表measure_prob_list
        _prob = ignis_method(R, measure_prob)
        _prob=normalize(_prob)

        mean, variance, skewness,std = calculate_moments(_prob)     # 计算修正后概率分布的统计特征（均值、方差、偏度、标准差）
        squares = [x ** 2 for x in _prob]
        mean_of_squares = sum(squares) / len(_prob)
        a=get_a(skewness)     # 根据偏度计算参数a
        L=1+a*(mean_of_squares-1/(len(_prob)**2))/(1/(len(_prob))-mean_of_squares)     # 计算L值，用于调整概率分布
        _prob=_prob**L/sum(_prob**L)    # 根据L值对概率分布进行调整
        _prob=normalize(_prob)    # 再次对概率分布进行归一化

        corrected_prob_list.append(_prob)    # 将修正后的概率分布添加到列表中
    
########################

    # ************************************************************************** 请于以上区域内作答 **************************************************************************

    return corrected_prob_list, TRAIN_SAMPLE_NUM