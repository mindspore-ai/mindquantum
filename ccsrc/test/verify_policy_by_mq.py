from mindquantum import *
import numpy as np
circ = Circuit()

pr = ParameterResolver({'a':3})
mat = RX('a').diff_matrix(pr)
print(mat)
diff = UnivMathGate('diff', mat)
circ += diff.on(0)
sim = Simulator('projectq', 3)
ham = Hamiltonian(QubitOperator("X0", 1))
sim.apply_circuit(circ)
a = sim.get_expectation(ham)

print(a)