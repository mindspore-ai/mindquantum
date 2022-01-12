from mindquantum import Circuit, UN, H, Z, X, Measure
from mindquantum.simulator import Simulator


def circuits():

    n_qubits = 5                               # 设定量子比特数为3
    sim = Simulator('projectq', n_qubits)        # 使用projectq模拟器，命名为sim
    circuit = Circuit()                          # 初始化量子线路，命名为circuit

    circuit += UN(H, n_qubits-1) 
    circuit += UN(Z, n_qubits-2) 

    circuit += Z.on([2],[1])
    circuit += Z.on([2],[0])
    circuit += Z.on([1],[0])
    circuit += Z.on([2],[0,1])
    circuit += Z.on(1)
    circuit += Z.on(2)
    circuit += Z.on([3],[1])
    circuit += Z.on([2],[0])
    circuit += Z.on([1],[0])
    circuit += Z.on([3],[1,2])
    circuit += Z.on([3],[0,1])
    circuit += Z.on([3],[0,1,2])

    circuit += UN(H, n_qubits-1) 
    circuit += UN(X, n_qubits-1) 
    circuit += X.on([4],[0,1,2,3])

    circuit += Measure().on(4)

    return circuit
