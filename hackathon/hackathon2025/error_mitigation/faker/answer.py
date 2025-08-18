import numpy as np


# 运行说明:将测试线路的数据按numpy数组格式存储在./samples/data/bitstrings_base.npz中，运行时调用correct函数,将待修正的测量结果以列表形式传入。

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


def correct(measure_prob_list):
    """
    将六个量子线路的测量结果measure_prob_list分别修正并返回，
    请参赛选手保持本函数的输入measure_prob_list和输出corrected_prob_list格式不变；

    输入值measure_prob_list为包含6个量子线路“原始”测量概率的列表，
    返回值corrected_prob_list为包含6个量子线路“修正后”测量概率的列表，
    """
    
    global TRAIN_SAMPLE_NUM

    # 获取“态制备-测量”结果，并建立修正矩阵
    q_num = 9
    random_seed = 2025
    sample_num = 2000  # 每个量子线路的测量结果数量

    def compare(n, m, q_num = 9):         #计算n,m态之间的史明距离
        n_b = num2bit_string(q_num, n)
        m_b = num2bit_string(q_num, m)
        i = 0
        for j in range(q_num):
            if n_b[j] != m_b[j]:
                i += 1
        return i
    
    # 按高斯函数形式a*exp(-b*x^2) 初始化R矩阵, 每行用2000组数据
    R = np.zeros((2 ** q_num, 2 ** q_num))
    for true_state in range(2 ** q_num):
        # 用get_data函数获取测量结果，并用bitstring_arr2vec函数将其转换成概率
        bitstring_arr = get_data(state=true_state, qubits_number_list=[[np.arange(q_num), sample_num]], random_seed=random_seed)
        random_seed += 10
        base_prob_dict = bitstring_arr2vec(bitstring_arr[0], q_num=q_num)
        base_prob = dict2vec(nqubits=q_num, res_dict=base_prob_dict)

        differ = 0 # 记录与真实态汉明距离为1的概率之和
        for i in range(2 ** q_num):
            if compare(true_state, i) == 1:
                differ += base_prob[i]

        a =  base_prob[true_state] # 根据理想态的概率设置c
        b =  -np.log(differ/q_num/a) # 根据相差一个比特的概率设置b，共有九个相差一个比特的态
        
        for i in range(2 ** q_num):
            R[true_state][i] = a * np.exp(-b * compare(true_state, i) ** 2) # 根据高斯函数设置R矩阵
        R[true_state] = R[true_state] / np.sum(R[true_state]) #将R各行向量归一化
    
    R = np.transpose(R)

    # 将六个量子线路的测量结果measure_prob_list分别修正并返回
    corrected_prob_list = []
    for measure_prob in measure_prob_list:
        _prob = MatrixInversion(measure_prob, R)
        corrected_prob_list.append(_prob)
    

    return corrected_prob_list, TRAIN_SAMPLE_NUM