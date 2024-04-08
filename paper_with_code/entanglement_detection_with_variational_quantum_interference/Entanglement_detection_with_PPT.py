import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
import mindquantum as mq
from mindquantum.core.gates import *
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator
from mindquantum.core.operators import Hamiltonian
from mindquantum.core.parameterresolver import ParameterResolver
import qutip
from qutip import Qobj
import matplotlib.pyplot as plt
from math import cos,sin

import warnings
warnings.filterwarnings("ignore")

'''根据k随机Sample一个量子态'''
def rho_sampling(bit_1,bit_2,k):
    dimension_1 = 2**bit_1
    dimension_2 = 2**bit_2
    #在haar测度下sample量子态
    state = qutip.rand_ket_haar(N=dimension_1*dimension_2*k, dims=[[dimension_1,dimension_2,k],[1,1,1]])
    #对量子态取偏迹得到sample的密度矩阵
    rho = state.ptrace(sel = [0,1])
    rho_PPT = qutip.partial_transpose(rho,[1,0])
    rho_ptrace = rho.ptrace(sel = [0])
    rho_list = [np.array(rho),np.array(rho_PPT),np.array(rho_ptrace)]
    return rho_list

def circuit_build(bit, depth):
    # 构造一个变分量子线路
    circuit = Circuit()
    for i in range(depth):
        cnot_index = 1 - i % 2
        for j in range(2*bit):
            circuit += RX(f'd{i}_x{j}').on(j)
            circuit += RY(f'd{i}_y{j}').on(j)
        for j in range(bit):
            if (cnot_index == 1 and j == bit - 1):
                circuit += X.on(0, 2 * j + cnot_index)
            else:
                circuit += X.on(2 * j + cnot_index + 1, 2 * j + cnot_index)
    return circuit

'''搜索算法需要优化的函数'''
def op_function(bit, depth, rho):
    circuit = circuit_build(bit, depth)
    ham = Hamiltonian(csr_matrix(rho))
    sim = Simulator('mqvector', circuit.n_qubits)
    # 生成求期望值和梯度的算子
    grad_ops = sim.get_expectation_with_grad(ham, circuit)
    def fun(para):
        f, g = grad_ops(para)
        f = f.real[0, 0]
        g = g.real[0, 0]
        return f, g
    return fun, len(circuit.params_name)

'''纠缠判据的优化函数'''
def witness_function(bit, depth, rho):
    def fun(para):
        circuit = circuit_build(bit, depth)
        sim = Simulator('mqvector',2*bit)
        #设置量子线路参数
        pr = ParameterResolver()
        for i in range(depth):
            for j in range(2*bit):
                index = 2*((2*bit)*i+j)
                pr[f'd{i}_x{j}'] = para[index]
                pr[f'd{i}_y{j}'] = para[index+1]
        #模拟量子线路并获得量子态
        sim.apply_circuit(circuit,pr)
        state = sim.get_qs()
        #计算witness
        svd = np.linalg.svd(state.reshape(2**bit,2**bit))
        alpha = max(svd[1])**2
        witness = alpha - np.matmul(state.conjugate(),rho@state)
        return witness.real
    return fun, 4*bit*depth

'''对于确定的k,进行sample,repeat代表sample的次数'''
def PPT_sample_result(bit,k,repeat,Judges):
    counter = 0
    PPT_counter = 0
    sample_counter1 = 0
    sample_counter2 = 0
    sample_counter3 = 0
    purity_counter = 0
    witness_counter1 = 0
    witness_counter2 = 0
    witness_counter3 = 0
    while(counter < repeat):
        print(counter                                            ,end='\r')
        rho_list = rho_sampling(bit,bit,k)
        rho = rho_list[0]
        rho_PPT = rho_list[1]
        rho_ptrace = rho_list[2]
        counter += 1

        #Judges[0]=True则统计PPT理论结果
        if(Judges[0]):
            eigen_min = np.min(np.linalg.eig(rho_PPT)[0].real)
            if(eigen_min < 0):
                PPT_counter += 1
        
        #Judges[1]=True则统计Depth=1的线路
        if(Judges[1]):
            fun, n_params = op_function(bit, 1, rho_PPT)
            para = np.ones(n_params)
            result = minimize(fun, para, method='bfgs', jac=True)
            if(result.fun < 0):
                sample_counter1 += 1
        
        #Judges[2]=True则统计Depth=2的线路
        if(Judges[2]):
            fun, n_params = op_function(bit, 2, rho_PPT)
            para = np.ones(n_params)
            result = minimize(fun, para, method='bfgs', jac=True)
            if(result.fun < 0):
                sample_counter2 += 1
        
        #Judges[3]=True则统计Depth=3的线路
        if(Judges[3]):
            fun, n_params = op_function(bit, 3, rho_PPT)
            para = np.ones(n_params)
            result = minimize(fun, para, method='bfgs', jac=True)
            if(result.fun < 0):
                sample_counter3 += 1
        
        #Judges[4]=True则统计Purity
        if(Judges[4]):
            purity1 = (rho@rho).trace().real
            purity2 = (rho_ptrace@rho_ptrace).trace().real
            if(purity1 > purity2):
                purity_counter += 1
        
        #Judges[5]=True则统计depth=1的Witness
        if(Judges[5]):
            fun, n_params = witness_function(bit, 1, rho)
            para = np.ones(n_params)
            result = minimize(fun, para, method='bfgs')
            if(result.fun.real < 0):
                witness_counter1 += 1
        
        #Judges[6]=True则统计depth=2的Witness
        if(Judges[6]):
            fun, n_params = witness_function(bit, 2, rho)
            para = np.ones(n_params)
            result = minimize(fun, para, method='bfgs')
            if(result.fun.real < 0):
                witness_counter2 += 1
        
        #Judges[7]=True则统计depth=3的Witness
        if(Judges[7]):
            fun, n_params = witness_function(bit, 3, rho)
            para = np.ones(n_params)
            result = minimize(fun, para, method='bfgs')
            if(result.fun.real < 0):
                witness_counter3 += 1

    return [counter,PPT_counter,sample_counter1,sample_counter2,sample_counter3,
            purity_counter,witness_counter1,witness_counter2,witness_counter3]

#实验的bit数,如bit=3则我们是在6bit系统中探究两个3bit子系统之间的纠缠
bit = 3

#repeat代表sample的次数
repeat = 1000

#Judges用于决定PPT_sample_result函数做探测方法的仿真
Judges = [1,1,1,1,1,1,1,1]

#K是sample量子态的参数组成的列表
K = [4,8,12,16,20,24,28]

#以下列表分别储存各个k下实验1000次测出纠缠的成功次数
PPT = []
D1 = []
D2 = []
D3 = []
Purity = []
W1 = []
W2 = []
W3 = []


for k in K:
    begin = time.time()
    counter,PPT_counter,sample_counter1,sample_counter2,sample_counter3,purity_counter,witness_counter1,witness_counter2,witness_counter3 = PPT_sample_result(bit,k,repeat,Judges)
    end = time.time()
    print(f'k={k},用时{(end-begin)/60}分,  {PPT_counter,sample_counter1,sample_counter2,sample_counter3,purity_counter,witness_counter1,witness_counter2,witness_counter3}')
    PPT.append(PPT_counter)
    D1.append(sample_counter1)
    D2.append(sample_counter2)
    D3.append(sample_counter3)
    Purity.append(purity_counter)
    W1.append(witness_counter1)
    W2.append(witness_counter2)
    W3.append(witness_counter3)

with open(f'data.txt', 'w') as f:
    for i in range(len(K)):
        f.write(f'{K[i]} {PPT[i]} {D1[i]} {D2[i]} {D3[i]} {Purity[i]}\n')