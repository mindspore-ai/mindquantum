from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RY,X
from mindquantum.framework import MQAnsatzOnlyLayer
from mindspore import Tensor
import numpy as np

def build_ham(num_qubits,h):
    ham = QubitOperator()

    temp_range = range(num_qubits)
    for i in range(num_qubits-1):
        ham = ham+QubitOperator(f'Z{temp_range[i]} Z{temp_range[i+1]}',-1.0)

    ham = ham+QubitOperator(f'Z{temp_range[0]} Z{temp_range[num_qubits-1]}',-1.0)

    for i in range(num_qubits):
        ham = ham+QubitOperator(f'X{temp_range[i]}',-h)

    return Hamiltonian(ham)

def build_ansatz(num_qubits,para):
    circ = Circuit()

    for i in range(num_qubits):
        circ = circ+RY(para+f'{i}').on(i)
    circ.barrier()
    
    circ = circ+X.on(0,num_qubits-1)
    for i in range(1,num_qubits):
        circ= circ+X.on(i,i-1)
    circ.barrier()  

    for i in range(num_qubits):
        circ = circ+RY(para+f'{i+12}').on(i)

    return circ

def ising_value_fn(x_init,ham,simulator,ansatz):
  x_init = x_init.astype(np.float32)
  grad_ops = simulator.get_expectation_with_grad(ham, ansatz)
  net = MQAnsatzOnlyLayer(grad_ops,Tensor(x_init))
  return net().asnumpy()


