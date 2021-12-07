import os
os.environ['OMP_NUM_THREADS'] = '4'
import mindspore as ms
from mindspore import Tensor
from openfermion.utils import count_qubits
from mindquantum.circuit import Circuit
from mindquantum.gate import Hamiltonian, RX, RY, RZ, X
from mindquantum.nn import generate_pqc_operator
from mindquantum.ansatz import UCCAnsatz
from q_ham import q_ham_producer
import numpy as np
from scipy.optimize import minimize
import time

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="CPU")


def energy_obj(n_paras, mol_pqc):
    encoder_data = Tensor(np.array([[0]]).astype(np.float32))
    ansatz_data = Tensor(np.array(n_paras).astype(np.float32))
    e, _, grad = mol_pqc(encoder_data, ansatz_data)
    return e.asnumpy()[0, 0], grad.asnumpy()[0, 0]


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
print(total_circuit.para_name)

# step 3: objective function
# generate a circuit that have right number of qubits
total_pqc = generate_pqc_operator(["null"], total_circuit.para_name, \
                                RX("null").on(count_qubits(q_ham)-1) + total_circuit, \
                                Hamiltonian(q_ham))

# step 4: optimization step.
n_paras = [0.0 for j in range(len(total_circuit.para_name))]
res = minimize(energy_obj,
               n_paras,
               args=(total_pqc, ),
               method='bfgs',
               jac=True,
               tol=1e-6)
print("VQE energy with UCCSD0 ansatz:{}".format(float(res.fun)))
print("Corresponding parameters:{}".format(res.x.tolist()))
n_paras = res.x.tolist()
n_p = len(total_circuit.para_name)
print(n_p)

t_cost = time.time() - start

print("HF energy:{}".format(hf_energy))
print("CCSD energy:{}".format(ccsd_energy))
print("FCI energy:{}".format(fci_energy))
print("Time consumed:{}".format(t_cost))

