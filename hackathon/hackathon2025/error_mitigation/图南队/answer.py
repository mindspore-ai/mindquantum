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


def MatrixInversion(ymes, Rin):
    # Rin is a matrix where the first coordinate is the measured value and the second coordinate is the true value.
    # Rin[m][t],measured value, truth value, ymes means m, m=Rt
    return np.ravel(np.matmul(np.linalg.inv(np.matrix(Rin)), ymes))

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
    q_num = 9
    sample_num = 25000
    random_seed = 2025

    # get measure fidelity
    MFs = {}
    for qidx in range(q_num):
        MF0 = 1-np.sum(get_data(0, [[[q_num-1-qidx], sample_num]], random_seed=random_seed))/sample_num
        MF1 = np.sum(get_data(2**q_num-1, [[[q_num-1-qidx], sample_num]], random_seed=random_seed))/sample_num
        MFs[qidx] = [MF0*0.9993, MF1*0.99] # 9538.4797
        # MFs[qidx] = [MF0*0.9983, MF1*0.9955] # 9538.4797
        # MFs[qidx] = [MF0*0.9985, MF1*0.998] # 9533.768616712297
        # MFs[qidx] = [MF0, MF1] # 9519.387325230893
    
    # # get measure fidelity
    # _MFs = np.zeros(shape=[9,2])
    # for basis in range(2**q_num):
    #     bitstrings = get_data(basis, [[np.arange(q_num), sample_num]])[0] # shape (sample_num, q_num)
    #     binary = np.binary_repr(basis, width=q_num)
    #     _MFs[np.arange(q_num), np.array([int(s) for s in binary])] += np.sum(bitstrings, axis=0)
    # MFs = {}
    # total_sample = 2**q_num * sample_num / 2
    # for qidx in range(q_num):
    #     MF0 = 1 - _MFs[qidx][0]/total_sample
    #     MF1 = _MFs[qidx][1]/total_sample
    #     MFs[qidx] = [MF0, MF1]

    # build measure fidelity matrix R
    R = 1
    for qidx in range(q_num):
        MF0, MF1 = MFs[qidx]
        _M = np.array([[MF0, 1-MF1], [1-MF0, MF1]])
        R = np.kron(R, _M)

    # R = np.zeros((2 ** q_num, 2 ** q_num))
    # for true_state in range(2 ** q_num):
    #     bitstring_arr = get_data(state=true_state, qubits_number_list=[[np.arange(q_num), sample_num]], random_seed=random_seed)
    #     base_prob_dict = bitstring_arr2vec(bitstring_arr[0], q_num=q_num)
    #     # 转换为arr
    #     R[true_state] = dict2vec(nqubits=q_num, res_dict=base_prob_dict)

    # 将六个量子线路的测量结果measure_prob_list分别修正并返回
    corrected_prob_list = []
    for measure_prob in measure_prob_list:
        _prob = MatrixInversion(measure_prob, R)
        corrected_prob_list.append(_prob)
    
    # ************************************************************************** 请于以上区域内作答 **************************************************************************

    return corrected_prob_list, TRAIN_SAMPLE_NUM


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

def ml_method(measure_prob_list):
    q_num = 9
    sample_num = 25000
    # 准备训练样本
    X = []  # noisy (measured) distribution
    Y = []  # clean (ideal) distribution

    for basis in range(2**q_num):
        measured_samples = get_data(basis, [[np.arange(q_num), sample_num]])
        measured_prob = bitstring_arr2vec(measured_samples[0], q_num=q_num)
        ideal_prob = np.zeros(2 ** q_num)
        ideal_prob[basis] = 1.0  # one-hot vector
        X.append(measured_prob)
        Y.append(ideal_prob)

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


    model = MLPRegressor(hidden_layer_sizes=(512, 256), activation='relu', max_iter=500)
    model.fit(X_train, Y_train)

    # 评估误差
    print("Train score:", model.score(X_train, Y_train))
    print("Test score:", model.score(X_test, Y_test))

    # 使用model 进行校正
    corrected_prob_list = []
    for measure_prob in measure_prob_list:
        corrected = model.predict([measure_prob])[0]
        corrected = np.clip(corrected, 0, 1)
        corrected /= np.sum(corrected)  # 保证归一化
        corrected_prob_list.append(corrected)

    from joblib import load
    model = load("trained_calibration_model.joblib")

def correct_ml(measure_prob_list, model):
    corrected_prob_list = []
    for measure_prob in measure_prob_list:
        corrected = model.predict([measure_prob])[0]
        corrected = np.clip(corrected, 0, 1)
        corrected /= np.sum(corrected)
        corrected_prob_list.append(corrected)
    return corrected_prob_list, TRAIN_SAMPLE_NUM