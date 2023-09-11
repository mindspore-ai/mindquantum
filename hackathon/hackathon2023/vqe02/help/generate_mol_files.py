# Here provides an example of how to generate molecular file.
# You can use it to generate more kind of molecules to test your algorithm. 

from openfermionpyscf import run_pyscf
from openfermion.chem import MolecularData
from math import sin, cos, pi


def generate_H2_file():
    geometry = [
        ["H", [0.0, 0.0, 0.0]],
        ["H", [0.0, 0.0, 1.4]],
    ]
    basis = "sto3g"
    spin = 0

    mol0 = MolecularData(geometry, basis, multiplicity=2 * spin + 1, filename="H2_1.4")
    mol = run_pyscf(mol0, run_scf=1, run_ccsd=1, run_fci=1, verbose=1)
    mol.save()

def generate_H4_file():
    bond_len_list = [0.5, 1.0, 1.5]
    for bond_len in bond_len_list:
        geometry = [
            ["H", [0.0, 0.0, 0.0]],
            ["H", [0.0, 0.0, bond_len]],
            ["H", [0.0, 0.0, 2 * bond_len]],
            ["H", [0.0, 0.0, 3 * bond_len]],
        ]
        basis = "sto3g"
        spin = 0

        mol0 = MolecularData(geometry, basis, multiplicity=2 * spin + 1, filename=f"H4_{bond_len}")
        mol = run_pyscf(mol0, run_scf=1, run_ccsd=1, run_fci=1, verbose=1)
        mol.save()


def generate_LiH_file():
    bond_len_list = [0.5, 1.5, 2.5]
    for bond_len in bond_len_list:
        geometry = [
            ["Li", [0.0, 0.0, 0.0]],
            ["H", [0.0, 0.0, bond_len]],
        ]
        basis = "sto3g"
        spin = 0

        mol0 = MolecularData(geometry, basis, multiplicity=2 * spin + 1, filename=f"LiH_{bond_len}")
        mol = run_pyscf(mol0, run_scf=1, run_ccsd=1, run_fci=1, verbose=1)
        mol.save()


def generate_BeH2_file():
    geometry = [
        ["Be", [0.0, 0.0, 0.0]],
        ["H", [0.0, 0.0, 1.3]],
        ["H", [0.0, 0.0, -1.3]],
    ]
    basis = "sto3g"
    spin = 0

    mol0 = MolecularData(
        geometry, basis, multiplicity=2 * spin + 1, filename="BeH2_1.3"
    )
    mol = run_pyscf(mol0, run_scf=1, run_ccsd=1, run_fci=1, verbose=1)
    mol.save()

def generate_H2O_file():
    # Define molecular geometry and basis set
    bond_len=1.0
    atom_1 = 'O'
    atom_2 = 'H'
    atom_3 = 'H'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (bond_len, 0.0, 0.0)
    coordinate_3 = (cos(104.45 / 180 * pi) * bond_len, sin(104.45 / 180 * pi) * bond_len, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2), (atom_3, coordinate_3)]
    basis = "sto3g"
    spin = 0

    mol0 = MolecularData(geometry, basis, multiplicity=2 * spin + 1, filename="H2O_1.0")
    mol = run_pyscf(mol0, run_scf=1, run_ccsd=1, run_fci=1, verbose=1)
    mol.save()


if __name__ == "__main__":
    generate_H4_file()
    generate_LiH_file()
    generate_BeH2_file()
    generate_H2O_file()
