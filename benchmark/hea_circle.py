import os

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import time

from mindquantum.algorithm.nisq.chem import HardwareEfficientAnsatz
from mindquantum.core import Hamiltonian, RX, RY, RZ, X
from mindquantum import Simulator
from scipy.optimize import minimize
from q_ham import q_ham_producer
from functools import partial
from mole import get_H2_geo, get_LiH_geo


def energy_obj(ansatz_circuit, sparsed_q_ham, n_paras):
    mol_pqc = Simulator('projectq',
                        ansatz_circuit.n_qubits).get_expectation_with_grad(
                            sparsed_q_ham, ansatz_circuit)

    ansatz_data = np.array(n_paras)
    e, grad = mol_pqc(ansatz_data)
    return np.real(e[0, 0]), np.real(grad[0, 0])


# step 1: load qubit hamiltonian form openfermion or hiqfermion
# molecule info
geometry = get_LiH_geo(1.4)
basis = 'sto3g'
charge = 0
multiplicity = 1
transform = 'bravyi_kitaev'
# end of molecule info

n_qubits,n_electrons, \
hf_energy, ccsd_energy, \
fci_energy, q_ham = q_ham_producer(geometry, basis, charge, multiplicity, transform)
sparsed_q_ham = Hamiltonian(q_ham)
sparsed_q_ham.sparse(n_qubits)

start = time.process_time()
# step 2: constructe hardware efficient ansatz circuit
# Setting for hardware efficient ansatz

n_paras = []
n_p = 0
for i in range(6):
    single_rot_gate_seq, entangle_gate, \
    entangle_mapping, depth = [RY, RZ], X, 'linear', i+1
    # Create such a hardware efficient circuit
    hea = HardwareEfficientAnsatz(n_qubits, single_rot_gate_seq, entangle_gate,
                                  entangle_mapping, depth)
    hea_circuit = hea.circuit

    # output details of the circuit
    # circuit shown with code
    #print(hea_circuit)
    # summary for the circuit
    #print(hea_circuit.summary())
    # a list of string for all parameters
    #print(hea_circuit.params_name)

    # step 3: objective function
    para_only_energy_obj = partial(energy_obj, hea_circuit, sparsed_q_ham)

    # step 4: optimization step.
    # Randomly initialized parameters

    n_paras.extend([
        np.random.uniform(-np.pi, np.pi)
        for j in range(len(hea_circuit.params_name) - n_p)
    ])
    res = minimize(para_only_energy_obj, n_paras, jac=True, tol=1e-6)
    print("VQE energy with HEA ansatz:{}".format(float(res.fun)))
    print("Corresponding parameters:{}".format(res.x.tolist()))
    n_paras = res.x.tolist()
    n_p = len(hea_circuit.params_name)
    print(n_p)

t_cost = time.process_time() - start

print("HF energy:{}".format(hf_energy))
print("CCSD energy:{}".format(ccsd_energy))
print("FCI energy:{}".format(fci_energy))
print("Time consumed:{}".format(t_cost))
