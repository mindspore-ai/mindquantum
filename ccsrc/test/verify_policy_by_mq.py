from mindquantum import *
import numpy as np
circ = Circuit()

pr = ParameterResolver({'a':3})
# mat = RX('a').diff_matrix(pr)
# print(mat)
# diff = UnivMathGate('diff', mat)
# circ += diff.on(0)
circ += RX('a').on(0)
sim = Simulator('projectq', 3)
ham = Hamiltonian(QubitOperator("X0", 1))
# sim.apply_circuit(circ)
grad_ops = sim.get_expectation_with_grad(ham, circ)
a = grad_ops(np.array([3]))

print(__version__)