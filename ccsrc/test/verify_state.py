from cmath import pi
from qutip.qip.circuit import QubitCircuit, Gate, snot
from qutip import *
import numpy as np
from qutip.qip.operations import rx

def user_gate1(arg_value):
     # controlled rotation X
    mat = np.zeros((4, 4), dtype=np.complex)
    mat[0, 0] = mat[1, 1] = 1.
    mat[2:4, 2:4] = rx(arg_value)
    return Qobj(mat, dims=[[2, 2], [2, 2]])


qc = QubitCircuit(N=2)

qc.user_gates = {"CTRLRX": user_gate1}

qc.add_gate("SNOT", 1)
# qc.add_gate("RX", 1, arg_value=pi)
# qc.add_gate("CTRLRX", targets=[1,2], arg_value=np.pi)
qc.add_gate("CY", 0, 1)
zero_state = tensor(basis(2, 0), basis(2, 0))
result = qc.run(state=zero_state)
print(ket2dm(result))
