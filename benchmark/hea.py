import os
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import time
from mindspore import Tensor
import mindspore as ms
from mindquantum.ansatz import HardwareEfficientAnsatz, QubitUCCAnsatz
from mindquantum.gate import Hamiltonian, RX, RY, RZ, X
from mindquantum.nn import generate_pqc_operator
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms.opconversions import (jordan_wigner, bravyi_kitaev,
                                                  get_fermion_operator,
                                                  normal_ordered)
from scipy.optimize import minimize
from functools import partial


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
transform = 'bravyi_kitaev'
# end of molecule info

n_qubits,n_electrons, \
hf_energy, ccsd_energy, \
fci_energy, q_ham = q_ham_producer(geometry, basis, charge, multiplicity, transform)

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
#print(hea_circuit.para_name)

# step 3: objective function
mol_pqc = generate_pqc_operator(["null"], hea_circuit.para_name, \
                                RX("null").on(0) + hea_circuit, \
                                Hamiltonian(q_ham))
# para_only_energy_obj = partial(energy_obj, hea_circuit, q_ham)

# step 4: optimization step.
# Randomly initialized parameters
n_paras.extend([
    np.random.uniform(-np.pi, np.pi)
    for j in range(len(hea_circuit.para_name) - n_p)
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
n_p = len(hea_circuit.para_name)
print(n_p)

t_cost = time.time() - start

print("HF energy:{}".format(hf_energy))
print("CCSD energy:{}".format(ccsd_energy))
print("FCI energy:{}".format(fci_energy))
print("Time consumed:{}".format(t_cost))
