# 引入相关库
from mindquantum.core.circuit import Circuit
import numpy as np
from mindquantum.core.gates import H,CNOT,RX,RZ
from mindquantum.core.circuit import dagger
from graph import draw_graph
from rule import simplify
from instantiate import verify_by_para



# two local circular
def build_ansatz(n_qubits,depth):
    circ = Circuit()
    for i in range(depth):
        for j in range(n_qubits):
            circ += RX(f'theta{i*n_qubits+j}').on(j)
        circ += CNOT.on(0,n_qubits-1)
        for j in range(n_qubits-1):
            circ += CNOT.on(j+1,j)
    for j in range(n_qubits):
        circ += RX(f'theta{depth*n_qubits+j}').on(j)
    return circ



# 编译后的电路只含有H，CNOT，RZ门
def compile_circuit(circ):
    circ_compiled = Circuit()
    for gate in circ:
        if gate.name == 'H' or gate.name == 'CNOT' or gate.name == 'RZ':
            circ_compiled += gate
        elif gate.name == 'RX':
            circ_compiled += H.on(gate.obj_qubits)
            circ_compiled += RZ(gate.coeff).on(gate.obj_qubits)
            circ_compiled += H.on(gate.obj_qubits)

    return circ_compiled



def main():
    print('=============TwoLocal-Circular=============')
    # 完整线路共3层127个量子比特
    n_qubits = 127
    depth = 3
    circ1 = build_ansatz(n_qubits,depth)
    circ1_inv = dagger(circ1)    # dagger将量子线路左右逆转
    circ2 = compile_circuit(circ1)
    circ_all = circ1_inv + circ2
    
    g = draw_graph(circ_all)
    
    # 化简TwoLocal-Circular
    print("化简之前：")
    g.equiv()
    simplify(g)
    print("化简之后：")
    g.equiv()
    
    print('=============Not Equivalent Example=============')
    neq_circ1 = Circuit()
    neq_circ1 += H.on(1)
    neq_circ1 += RX(f'theta{0}').on(2)
    neq_circ1 += CNOT.on(0,1)
    neq_circ1 += RZ(f'theta{1}').on(0)
    neq_circ1 += CNOT.on(2,1)
    neq_circ1 += CNOT.on(0,1)
    neq_circ1 += RX(f'theta{2}').on(2)
    
    neq_circ2 = Circuit()
    neq_circ2 += H.on(1)
    neq_circ2 += RX(f'theta{0}').on(2)
    neq_circ2 += CNOT.on(0,1)
    neq_circ2 += RZ(f'theta{1}').on(0)
    neq_circ2 += CNOT.on(2,1)
    neq_circ2 += CNOT.on(0,1)
    # 不同之处
    neq_circ2 += RX({f'theta{0}': 1, f'theta{1}': 1, f'theta{2}': 1}).on(2)
    
    neq_circ1_inv = dagger(neq_circ1)
    neq_circ_all = neq_circ1_inv + neq_circ2
    
    neq_g = draw_graph(neq_circ_all)
    
    print("化简之前：")
    neq_g.equiv()
    simplify(neq_g)
    print("化简之后：")
    neq_g.equiv()
    
    # 实例化参数之后，验证发现线路不等价
    print("实例化参数验证：")
    verify_by_para(neq_circ1,neq_circ2,5)



# 合并成完整的ZX-calculus函数
def ZXcalculus (circ1, circ2):
    circ1_inv = dagger(circ1)
    circ = circ1_inv + circ2
    g = draw_graph(circ)
    print("化简之前：")
    g.equiv()
    simplify(g)
    print("化简之后：")
    if len(g.vertices) == 0:
        g.equiv()
    else:
        g.equiv()
        print("实例化参数验证：")
        verify_by_para(circ1,circ2,5)



if __name__ == "__main__":
    main()












