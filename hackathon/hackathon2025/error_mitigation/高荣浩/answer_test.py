r"""

仅仅测量全 0 和全 1 态，构造单个比特的校准矩阵，再进行IBU，获得最终的校准
测量结果应该包含了部分串扰影响

重新采样 10000 次，然后获得分数，并进行绘图对比

"""

import time
import numpy as np
import matplotlib.pyplot as plt
# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统常用中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

TRAIN_SAMPLE_NUM = 0
BITSTRINGS_BASE = np.load('./samples/data/bitstrings_base.npz')['arr_0']

def get_data(state, qubits_number_list, random_seed=2025):
    """
    获取基本线路的“态制备-测量”实验结果；

    Args:
        state: 表示制备的量子态，为0-511整数，分别对应将比特制备在'000000000'-'000000001'-...-'111111111'态，最右边为q0状态，
        qubits_number_list: 形如[[比特组合, 测量结果数量], ...]的列表，
        比特组合形如：[0, 4, 6]，则返回 8 - [0, 4, 6] = [8, 4, 2] -> '|q8,q4,q2>'的测量结果；
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

# 定义 score 函数，包含归一化检查
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
    
def normalize(v):
    return v / np.sum(v)
    
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

# ************************************************************************** 请于以上区域内作答 **************************************************************************

# 对 ibu 函数进行优化
# 原始 IBU
# IBU Algorithm for Error Mitigation
def IBU(ymes: np.ndarray, t0: np.ndarray, Rin: np.ndarray, n: int) -> np.ndarray:
    """
    Iterative Bayesian Unfolding method for error mitigation.

    NACHMAN B, URBANEK M, DE JONG W A, 等. Unfolding quantum computer readout noise[J/OL]. 
    npj Quantum Information, 2020, 6(1): 1-7. DOI:[10.1038/s41534-020-00309-7](https://doi.org/10.1038/s41534-020-00309-7).
    
    Args:
        ymes: Measured probability distribution
        t0: Initial guess for true distribution
        Rin: Response matrix (measured vs true)
        n: Number of iterations
    
    Returns:
        Mitigated probability distribution
    """
    tn = t0
    for _i in range(n):
        print(f'IBU iteration = {_i}')
        out = np.zeros(t0.shape)
        for j in range(len(t0)):
            mynum = 0.
            for i in range(len(ymes)):
                myden = sum(Rin[i][k] * tn[k] for k in range(len(t0)))
                if myden > 0:
                    mynum += Rin[i][j] * tn[j] * ymes[i] / myden
            out[j] = mynum
        tn = out
    return tn


# 向量化 IBU
def IBU_vectorized(ymes: np.ndarray, t0: np.ndarray, Rin: np.ndarray, n: int) -> np.ndarray:
        """ 向量化实现的 IBU 算法 """
        tn = t0.copy()
        for i in range(n):
            # print(f'IBU iteration = {i}')
            # 计算分母 (对每个 i 计算 sum(Rin[i][k] * tn[k]))
            den = Rin @ tn  # 矩阵乘法计算所有分母
            
            # 避免除零
            den = np.where(den > 0, den, 1e-10)
            
            # 计算 ymes[i] / den[i] 的比率，测量噪声 ymes
            ratio = ymes / den
            
            # 计算更新后的 tn
            # tn = np.sum(Rin.T * ratio, axis=1) * tn
            tn = (Rin.T @ ratio) * tn            
        return tn    


# 单比特校准矩阵使用的 IBU 函数
def apply_response_matrix(vector, single_qubit_mats):
    """
    快速应用张量积结构的校准矩阵
    
    参数:
        vector: 输入向量，形状为(2^9,)
        single_qubit_mats: 9个单量子比特校准矩阵的列表
        
    返回:
        应用校准矩阵后的向量，形状与输入相同
    """
    # 将输入向量重塑为9维张量，每个维度大小为2
    # print(vector.shape)
    # print(vector[0], vector[1])
    tensor = vector.reshape([2]*9)
    # print(tensor[0][0][0][0][0][0][0][0])
    # print(tensor[0][0][0][0][0][0][0][0].shape)
    # 应用每个量子比特的校准矩阵（按构造时的顺序）
    for k in range(9):
        mat = single_qubit_mats[k]
        # 将当前量子比特对应的轴移到最后一个位置
        tensor = np.moveaxis(tensor, k, -1)
        # 重塑为 (..., 2) 并进行矩阵乘法
        original_shape = tensor.shape
        tensor = tensor.reshape(-1, 2) @ mat.T
        # 恢复形状并移回轴位置
        tensor = tensor.reshape(original_shape)
        tensor = np.moveaxis(tensor, -1, k)
    return tensor.reshape(-1)

def IBU_optimized(ymes, t0, single_qubit_mats, n_iter=5):
    """
    优化后的IBU(迭代贝叶斯展开)算法
    
    参数:
        ymes: 测量结果向量
        t0: 初始概率分布
        single_qubit_mats: 9个单量子比特校准矩阵的列表
        n_iter: 迭代次数，默认为5
        
    返回:
        校准后的概率分布
    """
    # 预处理转置矩阵（用于后续计算）
    mats_T = [mat.T for mat in single_qubit_mats]
    
    tn = t0.copy()
    for _ in range(n_iter):
        # 计算分母：R @ tn
        den = apply_response_matrix(tn, single_qubit_mats)
        den = np.where(den > 0, den, 1e-10)
        
        # 计算比率：ymes / den
        ratio = ymes / den
        
        # 计算分子：R.T @ ratio
        rt_ratio = apply_response_matrix(ratio, mats_T)
        
        # 更新估计值
        tn = tn * rt_ratio
        tn /= tn.sum()  # 保持归一化
    
    return tn    
    
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
    q_num = 9
    dim = 2 ** q_num
    
    start = time.time()
    # 单比特 ibu 法
    samples_per_state = 10000  # 每个态的最大测量次数
    measured_data = {}
    # 测量全0态
    bitstrings0 = get_data(state=0,
                         qubits_number_list=[[list(range(q_num)), samples_per_state]],
                         random_seed=1024)[0]
    measured_data[0] = bitstrings0
    # 测量全1态 (state=511 = 2^9-1)
    bitstrings1 = get_data(state=511,
                         qubits_number_list=[[list(range(q_num)), samples_per_state]],
                         random_seed=2048)[0]
    measured_data[511] = bitstrings1

    single_qubit_mats = []
    for idx in range(q_num):
        # idx = 0 时，得到的实际是 q8 的测量结果

        # 使用贝叶斯估计改进参数计算
        alpha, beta = 25, 25  # 先验参数

        # # 符合测量结果的顺序，q8,...,q0 的校准矩阵依次存储
        p0_given0 = (np.sum(bitstrings0[:, idx] == 0) + alpha) / (samples_per_state + alpha + beta)
        p1_given1 = (np.sum(bitstrings1[:, idx] == 1) + alpha) / (samples_per_state + alpha + beta)

        # # 但实际测试，将张量积顺序更换，分数会高一点点，大约13分左右
        # physical_qubit = 8 - idx  # 物理比特索引
        # p0_given0 = (np.sum(bitstrings0[:, physical_qubit] == 0) + alpha) / (samples_per_state + alpha + beta)
        # p1_given1 = (np.sum(bitstrings1[:, physical_qubit] == 1) + alpha) / (samples_per_state + alpha + beta)
        
        p1_given0 = 1 - p0_given0
        p0_given1 = 1 - p1_given1
        
        R_qubit = np.array([
            [p0_given0, p0_given1],
            [p1_given0, p1_given1]
        ])
        
        # 依次存储得到的是 q8,q7,...,q0 的校准矩阵
        single_qubit_mats.append(R_qubit)
    end = time.time()
    print("time:", end - start)


    start = time.time()
    # 矩阵取逆法
    R = np.zeros((2 ** q_num, 2 ** q_num))
    for true_state in range(2 ** q_num):
        bitstring_arr = get_data(state=true_state, qubits_number_list=[[np.arange(q_num), samples_per_state]], random_seed=2025)
        base_prob_dict = bitstring_arr2vec(bitstring_arr[0], q_num=q_num)
        # 转换为arr
        R[:,true_state] = dict2vec(nqubits=q_num, res_dict=base_prob_dict)
    # 取逆矩阵
    R_inv = np.linalg.inv(np.matrix(R))
    end = time.time()
    print("time:", end - start)

    vec_size = 2 ** q_num
    IBU_iterations = 5    
    random_prob = np.ones(vec_size) / vec_size

    ibu_score_list = [] # 局部 ibu score
    inv_score_list = [] # 矩阵取逆 score
    # 重新进行采样，查看各方案校准效果
    for true_state in range(2 ** q_num):
        bitstring_arr = get_data(state=true_state, qubits_number_list=[[np.arange(q_num), samples_per_state]], random_seed=2048+true_state)
        base_prob_dict = bitstring_arr2vec(bitstring_arr[0], q_num=q_num)
        noise_state = dict2vec(nqubits=q_num, res_dict=base_prob_dict)

        # 理想概率分布(one-hot向量)
        ideal_prob = np.zeros(dim)
        ideal_prob[true_state] = 1.0

        # 矩阵校准
        inv_prob = np.ravel(R_inv @ np.array(noise_state).reshape(-1,1))
        inv_prob = np.maximum(inv_prob, 0)  # 确保概率非负
        inv_prob /= np.sum(inv_prob)  # 归一化
        inv_score = single_circuit_score(P=ideal_prob, Q=inv_prob)
        inv_score_list.append(inv_score)

        # 单比特 ibu 校准
        ibu_prob = IBU_optimized(noise_state, random_prob, single_qubit_mats, IBU_iterations)        
        ibu_score = single_circuit_score(P=ideal_prob, Q=ibu_prob)
        ibu_score_list.append(ibu_score)

    print(f'inv_score_list = {np.mean(np.array(inv_score_list))}')
    print(f'ibu_score_list = {np.mean(np.array(ibu_score_list))}')

    # 将上面两个列表绘制成图

    # 在print语句后添加绘图代码
    plt.figure(figsize=(10, 6))
    # 将原来的plt.plot改为plt.scatter
    # 修改散点图标记样式
    plt.scatter(range(512), inv_score_list, label='矩阵求逆法', marker='o', s=10)  # 原点
    plt.scatter(range(512), ibu_score_list, label='局部 IBU', marker='*', s=10)  # 星点
    plt.xlabel('Index (0-511)')
    plt.ylabel('1-TVD')
    plt.title('各基态在矩阵求逆法和局部 IBU 下的 1-TVD 值')
    # 修改图例位置到左上角
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


    # 以下内容忽略
    # 将六个量子线路的测量结果measure_prob_list分别修正并返回
    corrected_prob_list = []
    for i in range(len(measure_prob_list)):
        measure_prob = measure_prob_list[i]
        # print(measure_prob.shape)
        # measure_prob = measure_prob.reshape(-1,1)
        _prob = MatrixInversion(measure_prob, R)
            
        corrected_prob_list.append(normalize(_prob))    
    # ************************************************************************** 请于以上区域内作答 **************************************************************************
   
    return corrected_prob_list, TRAIN_SAMPLE_NUM







