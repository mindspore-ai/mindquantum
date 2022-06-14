import numpy as np
from mindquantum.core.operators import QubitOperator
from mindquantum import Hamiltonian

def pr2array(pr):
    parameters = []
    k_list = []
    for k in pr.keys():
        k_list.append(k)
        parameters.append(pr[k])

    parameters = np.array(parameters)
    return parameters, k_list

def array2pr(parameters, k_list):
    _pr = {}
    for k, p in zip(k_list, parameters.tolist()):
        _pr[k] = p
    pr = PR(_pr)
    return pr


class Ising_like_ham:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def vac_Hamiltonian(self):
        ham = QubitOperator('')
        ham = Hamiltonian(ham)
        return ham

    def local_Hamiltonian(self):
        ham = None
        g, h = 0.5, 0.32
        for i in range(self.n_qubits):
            if ham is None:
                ham = QubitOperator('X{}'.format(i), g)
            else:
                ham += QubitOperator('X{}'.format(i), g)
                
            ham += QubitOperator('Z{}'.format(i), h)
            if i<self.n_qubits-1:
                ham += QubitOperator('Z{} Z{}'.format(i, i+1), 1)

        ham = Hamiltonian(ham)
        return ham