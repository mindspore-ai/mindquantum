from mindquantum import *
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import trange


#定义随机Pauli旋转门
def RP(rotation_angle):
    a = np.random.randint(0, 3)
    if a == 0:
        return RX(rotation_angle)
    elif a == 1:
        return RY(rotation_angle)
    elif a == 2:
        return RZ(rotation_angle)
    else:
        print("error in random Pauli gates")


#定义barren plateau的ansatz，线路图参考readme或者main.ipynb文件
def bpansatz(n, p):
    par = np.array(range(n * p)).astype(np.str)
    init_state_circ = UN(RY(np.pi / 4), n)
    qc = Circuit() + init_state_circ
    for j in range(p):
        for i in range(n):
            theta = par[i * p + j]
            qc += RP(theta).on(i)
        for ii in range(n - 1):
            qc += Z(ii, ii + 1)
    return qc


#定义偏导数方差的计算函数，使用采样的方法，程序自动判断收敛性，可设置误差要求
def get_var_partial_exp(
    circuit,
    hamiltonian,
    number_of_attempts=10,  #初始采样次数，不是实际的采样次数，程序会根据error_reques自动调整（小概率会影响精度，如果发现跑出来与预期不符，可以适当调大此项）
    error_request=0.2):  #要求最低的采样误差，这个在很大程度上影响代码运行时间，默认0.2会比较快，而且精度也够
    simulator = Simulator('mqvector', circuit.n_qubits)
    grad_ops = simulator.get_expectation_with_grad(hamiltonian, circuit)
    avg_variance = 0
    error = 1  #当前误差，初始值设置为最大值1
    iteration = 0  #标记迭代次数

    #接下里是初次采样
    for kk in range(number_of_attempts):
        rdm = np.random.rand(len(circuit.params_name)) * 2 * np.pi
        avg_variance += grad_ops(rdm)[1][0, 0,
                                         0].real**2 / (number_of_attempts)

    #这里是做审敛的条件循环，若不满足误差要求，会自动增加样本容量，不断采样计算，直到结果满足要求。
    while error > error_request:
        iteration += 1
        variance = 0
        for jj in trange(number_of_attempts * (2**(iteration - 1))):
            rdm = np.random.rand(len(circuit.params_name)) * 2 * np.pi
            variance += grad_ops(rdm)[1][0, 0,
                                         0].real**2 / (number_of_attempts *
                                                       (2**(iteration - 1)))
        avg_variance_i = avg_variance
        avg_variance = (variance + avg_variance) / 2
        error = abs(avg_variance - avg_variance_i) / (avg_variance)
        print("已迭代次数：", iteration, "\t"
              "当前采样误差", error)
    return avg_variance


#测试代码
if __name__ == '__main__':
    #Part1 计算梯度方差与qubit数量的关系
    n_max = 16  # maximal number of  qubits
    p = 200  # number of layers
    xxx = np.arange(4, n_max + 1)  #用于绘图时候的横坐标
    yyy = np.zeros(n_max - 3)  #储存梯度方差的向量
    for n in range(4, n_max + 1):
        circuit = bpansatz(n, p)
        hamiltonian = Hamiltonian(QubitOperator('Z0 Z1'))
        print("n = ", n)
        yyy[n - 4] = get_var_partial_exp(circuit, hamiltonian)

    #在半指数图上绘制图像
    plt.semilogy(xxx, yyy)
    plt.title('Var(∂E) vs number of qubits')
    plt.xlabel('number of qubits')
    plt.ylabel('Var(∂E)')
    plt.savefig('result1_var_vs_qubis.svg')
    plt.clf()

    #Part2 计算梯度方差和层数的关系
    for n in range(2, 16, 2):
        p_max = 400  # maximal number of layers
        xxx = np.arange(20, p_max, 20)  #用于绘图时候的横坐标
        yyy = np.zeros(int((p_max - 20) / 20))  #储存梯度方差的向量
        for p in range(20, p_max, 20):
            circuit = bpansatz(n, p)
            hamiltonian = Hamiltonian(QubitOperator('Z0 Z1'))
            print("p = ", p)
            yyy[int((p - 20) / 20)] = get_var_partial_exp(circuit, hamiltonian)

    #在半指数图上绘制图像
        plt.semilogy(xxx, yyy)
    plt.title('Var(∂E) vs # of layers')
    plt.xlabel('number of layers')
    plt.ylabel('Var(∂E)')
    plt.savefig('result2_var_vs_layers.svg')
