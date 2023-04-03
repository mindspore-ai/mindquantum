import _math
import numpy as np
import openfermion as of

pauli = ['X', 'Y', 'Z']
n_terms = 2000

n_qubits = 30

paulis = []
np.random.seed(42)
for i in range(n_terms):
    tmp = ""
    for j in range(np.random.randint(n_qubits)):
        tmp += pauli[np.random.randint(3)]
        tmp += str(np.random.randint(n_qubits))
        tmp += " "
    paulis.append(tmp)


def generate_ops(paulis, fun):
    out = fun(paulis[0])
    for i in paulis[1:]:
        out += fun(i)
    return out


op1 = generate_ops(paulis[: n_terms // 2], _math.ops.QubitOperator)
op2 = generate_ops(paulis[n_terms // 2 :], _math.ops.QubitOperator)

of_op1 = generate_ops(paulis[: n_terms // 2], of.QubitOperator)
of_op2 = generate_ops(paulis[n_terms // 2 :], of.QubitOperator)
