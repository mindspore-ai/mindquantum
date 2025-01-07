"""Utils for VQE."""
from typing import List
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum import InteractionOperator, FermionOperator, Transform


def read_mol_data(file_name: str):
    """
    读取分子坐标文件。

    Examples:
        >>> read_mol_data('data_mol/mol.csv')
    """
    mol = []
    with open(file_name, 'r') as f:
        data = f.readlines()
    for i in data:
        j = i.split(',')
        mol.append([j[0]])
        mol[-1].append([float(k) for k in j[1:]])
    return mol


def generate_molecule(geometry: List[List[float]]) -> MolecularData:
    """
    产生分子文件。

    Args:
        geometry (List[List[float]]): 分子坐标文件。

    Examples:
        >>> dist = 1.5
        >>> geometry = [
        >>>     ['H', [0.0, 0.0, 0.0 * dist]],
        >>>     ['H', [0.0, 0.0, 1.0 * dist]],
        >>>     ['H', [0.0, 0.0, 2.0 * dist]],
        >>>     ['H', [0.0, 0.0, 3.0 * dist]],
        >>> ]
        >>> mol = generate_molecule(geometry)
    """
    basis = "sto3g"
    # basis = "cc-pvtz"
    # basis = "3-21g"
    # basis = "6-31g"
    print('basis:', basis)
    spin = 0
    molecule_of = MolecularData(geometry, basis, multiplicity=2 * spin + 1, data_directory='./')
    molecule_of = run_pyscf(molecule_of, run_scf=1, run_ccsd=1, run_fci=1)
    return molecule_of


def get_molecular_hamiltonian(mol: MolecularData):
    """
    根据分子文件生成哈密顿量。
    Args:
        mol (MolecularData): 分子文件。

    Examples:
        >>> dist = 1.5
        >>> geometry = [
        >>>     ['H', [0.0, 0.0, 0.0 * dist]],
        >>>     ['H', [0.0, 0.0, 1.0 * dist]],
        >>>     ['H', [0.0, 0.0, 2.0 * dist]],
        >>>     ['H', [0.0, 0.0, 3.0 * dist]],
        >>> ]
        >>> mol = generate_molecule(geometry)
        >>> ham = get_molecular_hamiltonian(mol)
    """
    ham_of = mol.get_molecular_hamiltonian()
    inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
    ham_hiq = FermionOperator(inter_ops)
    qubit_ham = Transform(ham_hiq).jordan_wigner().compress()
    return qubit_ham
