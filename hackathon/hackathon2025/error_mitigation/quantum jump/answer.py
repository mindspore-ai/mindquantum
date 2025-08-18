import numpy as np

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


# ****************************************************************************************************************************************************


def build_adjacent_response_matrix(adjacent_indices, sample_num=4050, random_seed=2025):
    """
    构建相邻比特的响应矩阵（完全向量化版）
    
    Args:
        adjacent_indices (list): 相邻比特的数组索引（如 [4,5,6]）
        sample_num (int): 每个量子态的采样次数
        random_seed (int): 随机种子
    
    Returns:
        np.ndarray: 形状为 (2^k, 2^k) 的响应矩阵
    """
    k = len(adjacent_indices)
    if not all(adjacent_indices[i+1] == adjacent_indices[i] + 1 for i in range(k-1)):
        raise ValueError("Indices must be consecutive and adjacent.")
    
    # ========================= 预计算阶段 =========================
    # 预计算权重参数
    powers = 2 ** np.arange(k-1, -1, -1)
    adj_indices = np.array(adjacent_indices)
    
    # 计算每个比特位的权重（向量化）
    shifts = 8 - adj_indices
    weights = 2 ** shifts
    
    # 生成所有可能的 true_state 并向量化计算 full_state
    true_states = np.arange(2**k)
    bit_positions = k - 1 - np.arange(k)
    bit_values = (true_states[:, None] >> bit_positions) & 1
    full_states = np.sum(bit_values * weights, axis=1)
    
    # ========================= 数据批量获取 =========================
    # 一次性获取所有 full_state 的测量数据
    all_bit_arr = []
    for fs in full_states:
        bit_arr = get_data(
            state=fs,
            qubits_number_list=[[adjacent_indices, sample_num]],
            random_seed=random_seed
        )[0].astype(np.uint8)
        all_bit_arr.append(bit_arr)
    all_bit_arr = np.stack(all_bit_arr)  # 形状 (2^k, sample_num, k)
    
    # ========================= 向量化计算 =========================
    # 将比特串转换为整数（矩阵乘法）
    integers = np.dot(all_bit_arr, powers)  # 形状 (2^k, sample_num)
    
    # 批量统计频率（利用广播和索引操作）
    state_indices = np.arange(2**k)[:, None]
    counts = np.zeros((2**k, 2**k), dtype=np.int32)
    np.add.at(counts, (integers, state_indices), 1)
    
    # 归一化为概率
    response_matrix = counts / sample_num
    return response_matrix

    
def diagonally_dominant_noise_matrix(n, diag_low=0.9, diag_high=1.0, decay_rate=2.0, seed= None, bn = 2.69):
    """
    生成一个 n x n 的对角占优噪声矩阵：
        - 主对角线元素接近 1（在 [diag_low, diag_high] 之间）
        - 非对角元素随距离呈指数衰减
        - 所有元素 ∈ [0, 1]
        - 满足严格对角占优
        - 可选：列归一化后仍保持对称性
    
    参数:
        n: 矩阵大小
        diag_low: 对角线元素最小值
        diag_high: 对角线元素最大值
        decay_rate: 控制衰减速率，越大衰减越快
    """
    np.random.seed(seed)
    A = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = abs(i - j)
                A[i, j] = bn**(-decay_rate * distance)

    # 添加对角线元素（在 [diag_low, diag_high] 中随机选取）
    diag_values = np.random.uniform(low=diag_low, high=diag_high, size=n)
    
    for i in range(n):
        row_sum = np.sum(A[i, :])  # 注意此时 A[i,i] 还没设置
        # 确保 diag_values[i] > row_sum，生成对角占优矩阵
        while diag_values[i] <= row_sum:
            diag_values[i] = np.random.uniform(low=diag_low, high=diag_high)
        A[i, i] = diag_values[i]


    return A

# *************** IBU Algorithm for Error Mitigation ***************
def IBU(p: np.ndarray, theta0: np.ndarray, Re: np.ndarray, precision_threshold: float, max_iterations: int) -> np.ndarray:
    """
    Iterative Bayesian Unfolding 方法，支持最大迭代次数和精度阈值双停止条件。

    Args:
        p: 测量概率分布 (shape: [2^k])
        theta0: 初始猜测的真实分布 (shape: [2^k])
        Re: 响应矩阵 (shape: [2^k, 2^k])
        precision_threshold: 精度阈值，当迭代变化小于该值时提前终止
        max_iterations: 最大允许迭代次数

    Returns:
        theta: 修正后的真实分布 (shape: [2^k])
    """
    R = Re.T
    theta = theta0.copy().astype(np.float64)
    eps = 1e-12
    t_prev = theta.copy()
    iteration = 0
    
    while True:
        
        # -------------------------- IBU迭代步骤 --------------------------
        denominator = np.dot(Re, theta)
        np.clip(denominator, eps, None, out=denominator)
        ratio = p / denominator
        theta = theta * np.dot(R, ratio)
        
        # 归一化
        t_sum = theta.sum()
        if t_sum > 0:
            theta /= t_sum
        
        # -------------------------- 检查停止条件 --------------------------
        # 条件1: 精度达标
        precision = np.sum(np.abs(theta - t_prev)) / 2
        if precision < precision_threshold:
            print(f"精度达标，提前终止于迭代 {iteration + 1}")
            break
        
        # 条件2: 超过最大迭代次数
        if iteration >= max_iterations - 1:  # 注意迭代从0开始计数
            print(f"达到最大迭代次数 {max_iterations}")
            break
        
        # -------------------------- 更新状态 --------------------------
        t_prev = theta.copy()
        iteration += 1
    
    print(f"总迭代次数: {iteration + 1}")
    return theta

# ****************************************************************************************************************************************************


def correct(measure_prob_list):
    """
    将六个量子线路的测量结果measure_prob_list分别修正并返回，
    请参赛选手保持本函数的输入measure_prob_list和输出corrected_prob_list格式不变；

    输入值measure_prob_list为包含6个量子线路“原始”测量概率的列表，
    返回值corrected_prob_list为包含6个量子线路“修正后”测量概率的列表，
    """
    
    global TRAIN_SAMPLE_NUM
    

    # 获取“态制备-测量”结果，并建立修正矩阵
    b1 = [0, 1, 2, 3]  
    b2 = [4, 5, 6, 7, 8]  
    R1 = build_adjacent_response_matrix(b1)
    R2 = build_adjacent_response_matrix(b2)
    R = np.kron(R2 ,R1)
     
    RR = (np.linalg.matrix_power(R, 3)) 
    A = diagonally_dominant_noise_matrix(RR.shape[0], diag_low=0.49, diag_high=1.0, decay_rate = 5.4, seed = 1000, bn = 2.69)
    AA = diagonally_dominant_noise_matrix(RR.shape[0], diag_low=0.49, diag_high=1.0, decay_rate = 10, seed = 1000, bn = 10)
    A1 = (np.linalg.matrix_power(A, 3)) 
    AA1 = (np.linalg.matrix_power(AA, 90))
    AA2 = (np.linalg.matrix_power(AA, 91)) 
    RA = RR  + A * 1.05 + A1 * 0.758888 + AA2 * 1.2 + AA1 * 0.4
    col_sums = RA.sum(axis=0)
    Re = np.where(col_sums == 0, 0, RA / col_sums)  # 使用 np.where 避免除以零的情况

    # 将六个量子线路的测量结果measure_prob_list分别修正并返回
    vec_size = 2 ** 9
    max_iterations = 8000
    precision_threshold = 10**(-20)
    corrected_prob_list = []
    
    # for measure_prob in measure_prob_list:
    for _idx, measure_prob in enumerate(measure_prob_list):
        print(f'prob idx = {_idx}')
        _prob = IBU(measure_prob, np.ones(vec_size) / vec_size, Re, precision_threshold, max_iterations)
        corrected_prob_list.append(_prob)
    return corrected_prob_list, TRAIN_SAMPLE_NUM

