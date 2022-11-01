from qutip.qip.circuit import QubitCircuit, Gate, snot
from qutip import *
qc = QubitCircuit(N=2)

qc.add_gate("SNOT", 0)
# qc.add_gate("X", 0)
zero_state = tensor(basis(2, 0), basis(2, 0))
result = qc.run(state=zero_state)
print(ket2dm(result))
