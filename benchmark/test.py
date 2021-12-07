import os

os.environ['OMP_NUM_THREADS'] = '4'
import time
import numpy as np

from mindquantum import Circuit, X, RX, Hamiltonian
from mindquantum import generate_uccsd
from openfermionpyscf import run_pyscf
from openfermion.chem import MolecularData


def energy_obj(n_paras, mol_pqc):

    ansatz_data = np.array(n_paras)
    e, grad = mol_pqc(ansatz_data)
    return np.real(e[0, 0]), np.real(grad[0, 0])


mole_name = 'N2'
basis = 'sto3g'
charge = 0
multiplicity = 1
transform = 'jordan_wigner'
method = "1-UpCCGSD"

bond_lengths = [0.1 * i + 0.6 for i in range(15)]
bbond_lengths = [0.2 * i + 2.2 for i in range(5)]
bond_lengths.extend(bbond_lengths)
for bond_len in bond_lengths:
    print(bond_len)
    atom_1 = 'N'
    atom_2 = 'N'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (bond_len, 0.0, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2)]

    molecule_of = MolecularData(geometry, basis, multiplicity)
    molecule_of = run_pyscf(molecule_of, run_scf=1, run_ccsd=1, run_fci=0)

    start = time.time()
    hartreefock_wfn_circuit = Circuit(
        [X.on(i) for i in range(molecule_of.n_electrons)])

    ansatz_circuit, \
    init_amplitudes, \
    ansatz_parameter_names, \
    hamiltonian_QubitOp, \
    n_qubits, n_electrons = generate_uccsd(molecule_of, th=-1)

    ham_termlist = [(i, j) for i, j in hamiltonian_QubitOp.terms.items()]
    for term, coeff in ham_termlist:
        if coeff.imag != 0:
            print(type(coeff.real), term)
