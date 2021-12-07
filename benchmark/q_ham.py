from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.algorithm.nisq.chem import Transform
from mindquantum.core.operators.hamiltonian import Hamiltonian
from mindquantum.third_party.interaction_operator import InteractionOperator
from mindquantum.core.operators import get_fermion_operator


class MolInfoProduce(object):
    def __init__(self,
                 geometry,
                 basis,
                 charge,
                 multiplicity,
                 fermion_transform,
                 sparse=True):
        self.geometry = geometry
        self.basis = basis
        self.multiplicity = multiplicity
        self.charge = charge
        self.fermion_transform = fermion_transform
        self.sparse = sparse
        self.n_qubits = None
        self.n_electrons = None
        self.hf_energy = None
        self.ccsd_energy = None
        self.fci_energy = None
        self.qubit_hamiltonian = None
        self.sparsed_qubit_hamiltonian = None
        self.mol_info_producer()

    def mol_info_producer(self):
        mol = MolecularData(geometry=self.geometry,
                            basis=self.basis,
                            charge=self.charge,
                            multiplicity=self.multiplicity)

        py_mol = run_pyscf(mol, run_scf=1, run_ccsd=1, run_fci=1)

        self.hf_energy = py_mol.hf_energy
        self.ccsd_energy = py_mol.ccsd_energy
        self.fci_energy = py_mol.fci_energy
        self.n_qubits = py_mol.n_qubits
        self.n_electrons = py_mol.n_electrons

        # Get fermion hamiltonian
        molecular_hamiltonian = py_mol.get_molecular_hamiltonian()
        inter_ops = InteractionOperator(
            *molecular_hamiltonian.n_body_tensors.values())
        ham_hiq = get_fermion_operator(inter_ops)

        # Get qubit hamiltonian for a given mapping
        if self.fermion_transform == 'jordan_wigner':
            q_ham = Transform(ham_hiq).jordan_wigner()
            q_ham.compress()
            self.qubit_hamiltonian = q_ham.real
            # print(q_ham)
        elif self.fermion_transform == 'bravyi_kitaev':
            q_ham = Transform(ham_hiq).bravyi_kitaev()
            q_ham.compress()
            self.qubit_hamiltonian = q_ham.real
        self.sparsed_qubit_hamiltonian = Hamiltonian(self.qubit_hamiltonian)
        if self.sparse:
            self.sparsed_qubit_hamiltonian.sparse(self.n_qubits)


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
    inter_ops = InteractionOperator(
        *molecular_hamiltonian.n_body_tensors.values())
    ham_hiq = get_fermion_operator(inter_ops)

    # Get qubit hamiltonian for a given mapping
    if fermion_transform == 'jordan_wigner':
        q_ham = Transform(ham_hiq).jordan_wigner()
        q_ham.compress()
        # print(q_ham)
    elif fermion_transform == 'bravyi_kitaev':
        q_ham = Transform(ham_hiq).bravyi_kitaev()
        q_ham.compress()
        # print(q_ham)

    return (py_mol.n_qubits, py_mol.n_electrons, py_mol.hf_energy,
            py_mol.ccsd_energy, py_mol.fci_energy, q_ham.real)


if __name__ == "__main__":
    geometry = [['H', (0, 0, 0)], ['H', (0, 0, 0.74)]]
    n_qubits, n_electrons, \
    hf_energy, ccsd_energy, \
    fci_energy, q_ham = q_ham_producer(geometry, 'sto3g', 0, 1, 'bravyi_kitaev')
    print(n_qubits)
