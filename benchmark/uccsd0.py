import os

os.environ['OMP_NUM_THREADS'] = '4'
import time
import numpy as np
from scipy.optimize import minimize
from openfermion.utils import count_qubits
from mindquantum.simulator.simulator import Simulator
from mindquantum import Circuit
from mindquantum.core import Hamiltonian, RX, RY, RZ, X
from mindquantum.algorithm.nisq.chem import UCCAnsatz
from q_ham import q_ham_producer


def energy_obj(n_paras, mol_pqc):
    ansatz_data = np.array(n_paras)
    e, grad = mol_pqc(ansatz_data)
    return np.real(e[0, 0]), np.real(grad[0, 0])


# step 1: load qubit hamiltonian form openfermion or hiqfermion
# molecule info
atom_1 = 'H'
atom_2 = 'Li'
coordinate_1 = (0.0, 0.0, 0.0)
coordinate_2 = (1.4, 0.0, 0.0)
geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2)]
basis = 'sto3g'
charge = 0
multiplicity = 1
transform = 'jordan_wigner'
# end of molecule info

n_qubits,n_electrons, \
hf_energy, ccsd_energy, \
fci_energy, q_ham = q_ham_producer(geometry, basis, charge, multiplicity, transform)
sparsed_q_ham = Hamiltonian(q_ham)
sparsed_q_ham.sparse(n_qubits)

start = time.time()
# step 2: constructe UCC ansatz circuit

# Create such a UCCSD0 circuit

initial_circuit = Circuit([X.on(i) for i in range(n_electrons)])

uccsd0 = UCCAnsatz(n_qubits, n_electrons)

uccsd0_circuit = uccsd0.circuit

total_circuit = initial_circuit + uccsd0_circuit
# output details of the circuit
# circuit shown with code
#print(total_circuit)
# summary for the circuit
print(total_circuit.summary())
# a list of string for all parameters
print(total_circuit.params_name)

# step 3: objective function
# generate a circuit that have right number of qubits
total_pqc = Simulator('projectq', n_qubits).get_expectation_with_grad(
    sparsed_q_ham, total_circuit)

# step 4: optimization step.
n_paras = [0.0 for j in range(len(total_circuit.params_name))]
res = minimize(energy_obj,
               n_paras,
               args=(total_pqc, ),
               method='bfgs',
               jac=True,
               tol=1e-6)
print("VQE energy with UCCSD0 ansatz:{}".format(float(res.fun)))
print("Corresponding parameters:{}".format(res.x.tolist()))
n_paras = res.x.tolist()
n_p = len(total_circuit.params_name)
print(n_p)

t_cost = time.time() - start

print("HF energy:{}".format(hf_energy))
print("CCSD energy:{}".format(ccsd_energy))
print("FCI energy:{}".format(fci_energy))
print("Time consumed:{}".format(t_cost))
