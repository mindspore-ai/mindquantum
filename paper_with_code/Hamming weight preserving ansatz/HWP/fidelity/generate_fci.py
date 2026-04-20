import openfermion
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.utils import save_operator
import pickle
import os

# 定义H2分子的几何结构
def generate_H2_geometry(dist):
    geometry = [('H', (0, 0, 0)), ('H', (0, 0, dist))]
    return geometry

# 加载哈密顿量并获取FCI能量和量子态
def load_hamiltonian_and_fci_state(geometry, name, save_dir):
    basis = 'sto-3g'
    charge = 0
    multiplicity = 1

    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule, run_scf=1, run_mp2=0, run_cisd=0, run_ccsd=0, run_fci=1, verbose=0)
    print(f"molecule.fci_energy: {molecule.fci_energy}")
    # 获取并保存哈密顿量
    molecule_hamiltonian = molecule.get_molecular_hamiltonian()
    fermion_hamiltonian = get_fermion_operator(molecule_hamiltonian)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    hamiltonian_path = os.path.join(save_dir, f'{name}_hamiltonian.pkl')
    with open(hamiltonian_path, 'wb') as f:
        pickle.dump(qubit_hamiltonian, f)
    print(f"Hamiltonian saved to {hamiltonian_path}")


    sparse_hamiltonian = openfermion.get_sparse_operator(molecule_hamiltonian)
    gs_energy, gs_eigvec = openfermion.get_ground_state(sparse_hamiltonian)
    print(f"gs_energy: {gs_energy}")
    print(f"gs_eigvec: {gs_eigvec}")
    energy_path = os.path.join(save_dir, f'{name}_fci_energy.pkl')
    state_path = os.path.join(save_dir, f'{name}_fci_state.pkl')

    with open(energy_path, 'wb') as f:
        pickle.dump(gs_energy, f)
    print(f"FCI Energy saved to {energy_path}")

    with open(state_path, 'wb') as f:
        pickle.dump(gs_eigvec, f)
    print(f"FCI State saved to {state_path}")

    return qubit_hamiltonian, gs_energy, gs_eigvec

if __name__ == '__main__':
    name = 'H2'
    dist = 0.74  # 典型的H2分子键长，单位是Å
    save_dir = '/mnt/nas-new/home/yange/nature_comm/pack3/fidelity'

    os.makedirs(save_dir, exist_ok=True)

    geometry = generate_H2_geometry(dist)
    hamiltonian, fci_energy, fci_state = load_hamiltonian_and_fci_state(geometry, name, save_dir)
