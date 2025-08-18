import time
import h5py
import numpy as np
from mindquantum import Circuit, Simulator

from answer import correct


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


IDEAL_PROB_LIST = []
MEASURE_PROB_LIST = []

for circuit_idx in [1, 2, 5, 6, 8, 'ghz']:
    # 读取线路数据
    # 打开文件并读取内容
    with open(f'./samples/circuit/circuit_{circuit_idx}.txt', 'r') as file:
        qasm_content = file.read()

    # 利用mindquantum计算理想结果
    circuit = Circuit().from_openqasm(qasm_content)
    sim = Simulator('mqvector', 9)    #声明一个9比特的mqvector模拟器
    sim.apply_circuit(circuit)
    ideal_prob = np.abs(sim.get_qs()) ** 2

    IDEAL_PROB_LIST.append(ideal_prob)
    
    # 读取测量结果
    bitstring_arr = np.load(f'./samples/data/bitstrings_circuit_{circuit_idx}.npz')['arr_0']
    prob_dict = bitstring_arr2vec(bitstring_arr, q_num=9)
    # 转换为arr
    measure_prob = dict2vec(nqubits=9, res_dict=prob_dict, reverse=False)

    MEASURE_PROB_LIST.append(measure_prob)


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

def get_score():
    corrected_prob_list, train_sample_num = correct(MEASURE_PROB_LIST)
    score = 0
    for circuit_idx in range(len(corrected_prob_list)):
	    score += single_circuit_score(P=corrected_prob_list[circuit_idx], Q=IDEAL_PROB_LIST[circuit_idx])
    print(f'raw score = {score / len(corrected_prob_list)}, train_sample_num = {train_sample_num}')
    score = score / len(corrected_prob_list) + 0.005 * (50000 * 512 * 9 - train_sample_num) / 50000. / 512. / 9.
    score = score * 10000
    return score

def judgment():
    score = get_score()
    print(score)
    return score


if __name__ == '__main__':
    start = time.time()
    score = judgment()
    end = time.time()
    print("use time：", str(end - start))
    print("score:", "%.4f" % score)