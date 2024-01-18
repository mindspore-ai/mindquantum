from mindquantum import *
from typing_extensions import Union
import numpy as np

def init_table(n_qubits: int) -> np.ndarray:
    '''生成初始的表格。'''
    table = np.identity(2*n_qubits, dtype=np.int32) # X_ij = delta_ij, Z_ij = delta_(i-n)j
    table = np.concatenate((table, np.zeros((2*n_qubits, 1))), axis=1) # 最右边加上全 0 的 r_i 列
    table = np.concatenate((table, np.zeros((1, 2*n_qubits+1))), axis=0) # 最下边加上全 0 的 scratch space
    return table

def g(x1: int, z1: int, x2: int, z2: int) -> int:
    '''rowsum 操作中的 g 函数。'''
    if x1 == 0 and z1 == 0:
        return 0
    elif x1 == 1 and z1 == 1:
        return z2 - x2
    elif x1 == 1 and z1 == 0:
        return z2 * (2 * x2 - 1)
    elif x1 == 0 and z1 == 1:
        return x2 * (1 - 2*z2)
    else:
        raise ValueError("The input is illegal.")
    
def mod2(*args):
    '''返回 mod 2 的结果。
    args: 整数, 列表, 一维数组。for examples:
    mod2(2, 3)                               -> np.int(1)
    mod2(np.array([0, 1]), np.array([2, 4])) -> np.array([0, 1])
    mod2([0, 1], [2, 4])                     -> np.array([0, 1])
    mod2(np.array([0, 1]), 1))               -> np.array([1, 0])
    '''
    return np.sum(args, axis=0)%2

def apply_gate(gate) -> None:
    if isinstance(gate, HGate) and gate.ctrl_qubits == []: # H 门
        a = gate.obj_qubits[0]
        
        # r_i = r_i mod2 (X_ia * Z_ia), for i in {1, ..., 2n}
        table[:-1, -1] = mod2(table[:-1, -1], table[:-1, a] * table[:-1, n_qubits + a])
        
        # X_ia <--> Z_ia, for i in {1, ..., 2n}
        table[:-1, [a, n_qubits + a]] = table[:-1, [n_qubits + a, a]]
        
        
    elif isinstance(gate, SGate) and gate.ctrl_qubits == []: # S 门
        a = gate.obj_qubits[0]
        
        # r_i = r_i mod2 (X_ia * Z_ia), for i in {1, ..., 2n}
        table[:-1, -1] = mod2(table[:-1, -1], table[:-1, a] * table[:-1, n_qubits + a])
        
        # Z_ia = Z_ia mod2 X_ia, for i in {1, ..., 2n}
        table[:-1, n_qubits + a] = mod2(table[:-1, n_qubits + a], table[:-1, a])
        
    # 高度怀疑 bug 由此 CNOT 引起
    elif isinstance(gate, XGate) and len(gate.ctrl_qubits) == 1: # CNOT 门, a 控 b
        a = gate.ctrl_qubits[0]
        b = gate.obj_qubits[0]
        
        # r_i = r_i mod2 [(X_ia * Z_ib) * (X_ib mod2 Z_ia mod2 1)], for i in {1, ..., 2n}.
        table[:-1, -1] = mod2(table[:-1, -1], 
                              (table[:-1, a] * table[:-1, n_qubits + b]) 
                              * mod2(table[:-1, b], table[:-1, n_qubits + a], 1)) # 报警是由这条命令导致的
        
        # X_ib = X_ib mod2 X_ia, for i in {1, ..., 2n}
        table[:-1, b] = mod2(table[:-1, b], table[:-1, a])
        
        # Z_ia = Z_ia mod2 Z_ib, for i in {1, ..., 2n}
        table[:-1, n_qubits + a] = mod2(table[:-1, n_qubits + a], table[:-1, n_qubits + b])
        
    elif isinstance(gate, Measure):
        a = gate.obj_qubits[0]
        if np.sum(table[n_qubits:-1, a]) == 0: # 确定性结果 (无 p 存在)
            table[-1] = 0 # 最后一行的值都设为 0
            for i in range(n_qubits): # 施加 rowsum(2n+1, i+n) 操作，for i in {1, ..., n} 且 X_ia == 1
                if table[i, a] == 1: # X_ia == 1
                    # rowsum(2n+1, i+n) 操作
                    sum_g = np.sum([g(table[n_qubits + i, j],            # X_ij
                                      table[n_qubits + i, n_qubits + j], # Z_ij
                                      table[-1, j],                      # X_hj
                                      table[-1, n_qubits + j])           # Z_hj
                                      for j in range(n_qubits)])
                    label = 2 * table[-1, -1] + 2 * table[i + n_qubits, -1] + sum_g
                    if label % 4 == 0:
                        table[-1, -1] = 0 
                    elif label % 4 == 2:
                        table[-1, -1] = 1
                    else:
                        raise ValueError("The lable is wrong.")
                    table[-1, :n_qubits] = mod2(table[i, :n_qubits], table[-1, :n_qubits]) # X_hj = X_ij mod2 X_hj
                    table[-1, n_qubits:-1] = mod2(table[i, n_qubits:-1], table[-1, n_qubits:-1]) # Z_hj = Z_ij mod2 Z_hj
            measure_result = int(table[-1, -1]) # 测量结果为 r_2n+1
            
        else: # 随机结果 (有 p 存在)
            p = int(np.argmax(table[n_qubits:-1, a])) + n_qubits # p 取最小值
            for i in range(2 * n_qubits):
                if i == p:
                    continue
                if table[i, a] == 1:
                    # rowsum(i, p) 操作
                    sum_g = np.sum([g(table[p, j],            # X_ij
                                      table[p, n_qubits + j], # Z_ij
                                      table[i, j],            # X_hj
                                      table[i, n_qubits + j]) # Z_hj
                                      for j in range(n_qubits)])
                    label = 2 * table[i, -1] + 2 * table[p, -1] + sum_g
                    if label % 4 == 0:
                        table[-1, -1] = 0 
                    elif label % 4 == 2:
                        table[-1, -1] = 1
                    else:
                        raise ValueError("The lable is wrong.")
                    table[i, :n_qubits] = mod2(table[p, :n_qubits], table[i, :n_qubits]) # X_hj = X_ij mod2 X_hj
                    table[i, n_qubits:-1] = mod2(table[p, n_qubits:-1], table[i, n_qubits:-1]) # Z_hj = Z_ij mod2 Z_hj
                    
            table[p - n_qubits] = table[p] # 第 p-n 行 = 第 p 行
            table[p, :-1] = 0 # 除 r_p 外，第 p 行的值全设置为 0
            table[p, -1] = np.random.randint(0, 2) # r_p 以等概率随机赋值 0 或者 1
            table[p, n_qubits + a] = 1
            measure_result = int(table[p, -1]) # 测量结果为 r_2n+1
                    
        print('测量结果为：', measure_result)
    
    else:
        raise ValueError(f'The gate {gate} is not support.')


n_qubits = 2
table = init_table(n_qubits)
# apply_gate(S.on(1))
apply_gate(H.on(0))
# apply_gate(H.on(1))
apply_gate(X.on(1,0))
apply_gate(Measure().on(0))
apply_gate(Measure().on(1)) 
