from itertools import product
from mindquantum.core.circuit import Circuit
from mindquantum.core.circuit import UN
from mindquantum.simulator import Simulator
import random 
import numpy as np   
import math                          # 导入numpy库并简写为np
from mindquantum.core.gates import X,Y,Z, H, RY,Measure,I
from mindquantum import *
from mindquantum.core.operators import Hamiltonian    # 引入哈密顿量定义模块
from mindquantum.core.operators import QubitOperator 
from mindquantum import Circuit, X
from mindquantum.core.gates import DepolarizingChannel, X
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import U3
# 其他的函数定义和代码继续


n_qubits=3 #比特数

UGates_num = 10 # 随机门个数


#定义G门[-1,0],[0,1]
G_mat=np.array([[-1,0],[0,1]])
G_Gate=UnivMathGate('G',G_mat)
G=G_Gate



#G(θ)门的参数θ随机取值
kk=2#G^kk --Nnum个基矢→Nnum个G(θ)
Nnum1 = kk**2
Nnum2 = (kk+1)**2  # 想要生成的随机数的数量=基态ρ的数量，如2G4ρ，那么Nnum=4
minvalue = 0  # 随机数范围的下限
maxvalue = 8  # 随机数范围的上限
theta_values1 = minvalue + (maxvalue - minvalue) * np.random.rand(Nnum1)  # 生成N个[a, b)区间内的随机数作为旋转角度
theta_values2 = minvalue + (maxvalue - minvalue) * np.random.rand(Nnum2)  # 生成N个[a, b)区间内的随机数作为旋转角度

#定义G(θ)门[exp(iθΠ),0],[0,1]

def matrix(theta):
    ep = np.exp(1j * np.pi * theta)
    return np.array([[ep, 0.0j], [0.0j, 1.0 + 0.0j]])

def diff_matrix(theta):
    ep = 1j * np.pi * np.exp(1j * np.pi * theta)
    return np.array([[ep, 0.0j], [0.0j, 0.0j]])
GG = gene_univ_parameterized_gate('GG', matrix, diff_matrix)


# 创建参数化门和它们的dagger版本
# 注意：这里我们直接使用theta值，而不是字符串
G_theta1 = [GG(theta) for theta in theta_values1]
G_theta_dagger1 = [GG(-theta) for theta in theta_values1]
# 给定的参数集

G_theta2 = [GG(theta) for theta in theta_values2]
G_theta_dagger2 = [GG(-theta) for theta in theta_values2]

G_theta=[G_theta1,G_theta2]
G_theta_dagger=[G_theta_dagger1,G_theta_dagger2]

parameters = []

for _ in range(UGates_num):
    alpha_coeff = np.random.rand()
    phi_coeff = np.random.rand()
    lam_coeff = np.random.rand()
    alpha = alpha_coeff * math.pi
    phi = phi_coeff * math.pi
    lam = lam_coeff * math.pi
    parameters.append((alpha, phi, lam))



# 创建一个空列表来存储U3门
random_U = []
random_U_d=[]

# 使用循环来添加U3门到列表中
for alpha, phi, lam in parameters:
    u3_gate = U3(alpha, phi, lam)
    u3_gate_d=u3_gate.hermitian()
    random_U.append(u3_gate)
    random_U_d.append(u3_gate_d)



def rho(i,j,T,L):
    
    circuit = Circuit()
    circuit += UN(random_U[j],n_qubits)
    circuit += UN(random_U_d[i],n_qubits)
    circuit += G_theta_dagger[T][L].on(2)
    circuit += G_theta[T][L].on(2,0)
    circuit += G_theta[T][L].on(2,1)
    circuit += G_theta_dagger[T][L].on(2,[1,0]) 
    circuit += UN(random_U[i],n_qubits)
    return circuit

def rho_d(i,j,T,L):
    
    circuit = Circuit()
    circuit += UN(random_U_d[j],n_qubits)
    circuit += UN(random_U[i],n_qubits)
    circuit += G_theta_dagger[T][L].on(2,[1,0])
    circuit += G_theta[T][L].on(2,1)
    circuit += G_theta[T][L].on(2,0)
    circuit += G_theta_dagger[T][L].on(2)  
    circuit += UN(random_U_d[i],n_qubits)
    return circuit

def g_ig_j(*indices):
    circuit = Circuit()
    n_qubits = 3  # 假设量子比特数是已知的

    # 循环遍历每个索引，这里的 indices 将自动接收所有传入的参数
    for index in indices:
        # 应用 U_d 门
        circuit += UN(random_U_d[index],n_qubits)
        circuit += G.on(2)
        circuit += G.on(2,0)
        circuit += G.on(2,1)
        circuit += G.on(2,[1,0])
        circuit += UN(random_U[index],n_qubits)#Gj

    return circuit

def calculate_matrices_and_trace(UGates_num, kk):
    
    A = np.zeros((UGates_num,) * kk)
    B = np.zeros((UGates_num,) * kk)
    
    def generate_rho_combinations(T, kk, UGates_num):

        if T == 0:  # 第一部分计算：基于 kk^2 索引
            indices = [range(UGates_num) for _ in range(kk)]
            for combination in product(*indices):
                if len(set(combination)) == kk:  # 确保所有索引都是唯一的
                    index_pairs = list(product(combination, repeat=2))
                    rho_all = [rho(i, j, T, l) for l, (i, j) in enumerate(index_pairs)]
                    rho_all_d = [rho(i, j, T, l) for l, (i, j) in enumerate(index_pairs)]
                    yield rho_all, rho_all_d, combination
        else:  # 第二部分计算：基于 (kk+1)^2 索引
            extended_indices = [range(UGates_num) for _ in range(kk+1)]
            for extended_combination in product(*extended_indices):
                if len(set(extended_combination[:-1])) == len(extended_combination[:-1]) and all(extended_combination[-1] != item for item in extended_combination[:-1]):
                    extended_index_pairs = list(product(extended_combination, repeat=2))
                    extended_rho_all = [rho(i, j, T, l) for l, (i, j) in enumerate(extended_index_pairs)]
                    extended_rho_all_d = [rho(i, j, T, l) for l, (i, j) in enumerate(extended_index_pairs)]
                    yield extended_rho_all, extended_rho_all_d, extended_combination

    for T in [0, 1]:  # Process both matrices A and B
        matrix = A if T == 0 else B
        Nnum = kk**2 if T == 0 else (kk+1)**2
        for rho_all, rho_all_d, combination in generate_rho_combinations(T, kk, UGates_num):
            #i, j ,k= combination[:kk]  # Safer access
            variables = combination[:kk] # Safer access
            ptm_array = np.zeros((Nnum, Nnum))
            g_array = np.zeros((Nnum, Nnum))
            for m in range(Nnum):
                for n in range(Nnum):
                    circuit = Circuit()
                    circuit += rho_all[n]
                    circuit += g_ig_j(*variables)
                    circuit += rho_all_d[m]
                    sim = Simulator('mqvector',n_qubits)   #Declare a 3-qubit mqvector simulator named 'sim'.
                    sim           
                    sim.apply_circuit(circuit)

                    encoder = Circuit()
                    encoder += UN(I,n_qubits)
                    encoder += rho_all[n]
                    encoder += rho_all_d[m]

                    sim1 = Simulator('mqvector', n_qubits)  #Declare a 3-qubit mqvector simulator named 'sim1'.
                    sim1
                    sim1.apply_circuit(encoder)

                    ptm_array[m,n]=(sim.get_qs()[0])*(sim.get_qs()[0]).conjugate()
                
                    g_array[m,n]=(sim1.get_qs()[0])*(sim1.get_qs()[0]).conjugate()  
                    
                  

            U_svd, s_svd, V_svd = np.linalg.svd(g_array)
    
            if np.isclose(min(s_svd), 0, atol=1e-25):
                matrix[tuple(variables)] = 0
            else:
                g_array_inv = np.linalg.inv(g_array)
                product1 = np.dot(g_array_inv, ptm_array)
                matrix[tuple(variables)] = np.trace(product1)
                #print(f"Processing element of matrix {'A' if T == 0 else 'B'}")

    print("Matrix A and B calculations completed.")
    return A, B



    # 计算 G_q



    # 计算 G_q

# 定义参数
UGates_num = 10  # 单位门的数量
kk=2

p = np.random.rand(UGates_num - 1)  # 生成UGates_num-1个随机数
p = p / np.sum(p)  # 归一化使得总和为1
p = np.append(p, 1 - np.sum(p))  # 添加剩余的概率值使得总和为1

n = 3                      # 计算G_q矩阵的指数因子

# 计算矩阵A和B
A, B = calculate_matrices_and_trace(UGates_num, kk)
print("矩阵 A 和 B 的计算已经完成，开始计算迹。")
# 输出计算结果
print("矩阵 A:")
print(A)
sum_of_elements1 = np.sum(A)
print("The sum of all elements in matrix A is:", sum_of_elements1)     
print("矩阵 B:")
print(B)
sum_of_elements = np.sum(B)
print("The sum of all elements in matrix B is:", sum_of_elements)

# Read files and store numbers as 2D lists A and B



# Calculate G_q
G_q = [[0 for _ in range(UGates_num)] for _ in range(UGates_num)]
for i in range(UGates_num):
    for j in range(UGates_num):
        if i != j:
            T = 0.5 * (B[i][j] - A[i][j] - 1)
            G_q[i][j] = p[i] * p[j] * (T + 2**n - 2)
        else:
            G_q[i][j] = p[i] * p[j] * 2**n
# Calculate the total G_q
total_G_q = sum(sum(G_q[i]) for i in range(UGates_num))

tr2 = (total_G_q - 4) / 4
print(total_G_q)
print(tr2) #Tr(\rho^2)
            



