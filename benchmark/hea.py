import os
os.environ['OMP_NUM_THREADS'] = '4'
import time

import numpy as np
from scipy.optimize import minimize
from mindquantum.simulator.simulator import Simulator
from mindquantum import HardwareEfficientAnsatz
from mindquantum.core import Hamiltonian, RY, RZ, X
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms.opconversions import (jordan_wigner, bravyi_kitaev,
                                                  get_fermion_operator,
                                                  normal_ordered)

def q_ham_producer(geometry, basis, charge, multiplicity, fermion_transform):
    mol = MolecularData(geometry=geometry,
                        basis=basis,
                        charge=charge,
                        multiplicity=multiplicity)
    py_mol = run_pyscf(mol, run_scf=1, run_ccsd=1, run_fci=1)
    # print("Hartree-Fock energy: %20.16f Ha" % (py_mol.hf_energy))
    # print("CCSD energy: %20.16f Ha" % (py_mol.ccsd_energy))
    # print("FCI energy: %20.16f Ha" % (py_mol.fci_energy))

    # Get fermion hamiltonian
    molecular_hamiltonian = py_mol.get_molecular_hamiltonian()
    fermion_hamiltonian = normal_ordered(
        get_fermion_operator(molecular_hamiltonian))

    # Get qubit hamiltonian for a given mapping
    if fermion_transform == 'jordan_wigner':
        q_ham = jordan_wigner(fermion_hamiltonian)
        q_ham.compress()
        # print(q_ham)
    elif fermion_transform == 'bravyi_kitaev':
        q_ham = bravyi_kitaev(fermion_hamiltonian)
        q_ham.compress()
        # print(q_ham)

    return (py_mol.n_qubits, py_mol.n_electrons, py_mol.hf_energy,
            py_mol.ccsd_energy, py_mol.fci_energy, q_ham)


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
transform = 'bravyi_kitaev'
# end of molecule info

n_qubits,n_electrons, \
hf_energy, ccsd_energy, \
fci_energy, q_ham = q_ham_producer(geometry, basis, charge, multiplicity, transform)
sparsed_q_ham = Hamiltonian(q_ham)
sparsed_q_ham.sparse(n_qubits)

start = time.time()
# step 2: constructe hardware efficient ansatz circuit
# Setting for hardware efficient ansatz

n_paras = []
n_p = 0
single_rot_gate_seq, entangle_gate, \
entangle_mapping, depth = [RY, RZ], X, 'linear', 4
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
mol_pqc = Simulator('projectq', n_qubits).get_expectation_with_grad(
    sparsed_q_ham, hea_circuit)
# para_only_energy_obj = partial(energy_obj, hea_circuit, q_ham)

# step 4: optimization step.
# Randomly initialized parameters
n_paras.extend([
    np.random.uniform(-np.pi, np.pi)
    for j in range(len(hea_circuit.params_name) - n_p)
])
res = minimize(energy_obj,
               n_paras,
               args=(mol_pqc, ),
               method='bfgs',
               jac=True,
               tol=1e-6)
print("VQE energy with HEA ansatz:{}".format(float(res.fun)))
# print("Corresponding parameters:{}".format(res.x.tolist()))
n_paras = res.x.tolist()
n_p = len(hea_circuit.params_name)
print(n_p)

t_cost = time.time() - start

print("HF energy:{}".format(hf_energy))
print("CCSD energy:{}".format(ccsd_energy))
print("FCI energy:{}".format(fci_energy))
print("Time consumed:{}".format(t_cost))
