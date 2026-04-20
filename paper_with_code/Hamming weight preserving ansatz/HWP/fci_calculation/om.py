
from openfermion.chem import MolecularData
from openfermion.transforms import *
from openfermionpyscf import run_pyscf
import numpy as np
from openfermion.linalg import get_sparse_operator
from openfermion.utils import save_operator

# 1. 计算 Hartree-Fock 态
# geometry=[("H", (0., 0., 0.)), ("H", (0., 0., 0.7414))],
molecule = MolecularData([('O', (0, 0, 0)), ('H',(0,0,0.958*1.3)), ('H',(0,0,-0.958*1.3))],
                         basis='sto-3g',
                         multiplicity=1,
                         charge=0)
molecule = run_pyscf(molecule, run_scf=1, run_fci=1)

# 打印 Hartree-Fock 能量
print(f"HF Energy (Method 1, OpenFermion + PySCF): {molecule.hf_energy}")
print(f"FCI Energy (run_fci): {molecule.fci_energy}")
# 2. 生成分子哈密顿量并进行 Jordan-Wigner 变换
molecular_hamiltonian = molecule.get_molecular_hamiltonian()
fermion_operator = get_fermion_operator(molecular_hamiltonian)
jw_hamiltonian = jordan_wigner(fermion_operator)

# 打印 Jordan-Wigner 变换后的哈密顿量
print(f"JW Hamiltonian (Method 1, OpenFermion + PySCF):\n{jw_hamiltonian}")

# 3. 将哈密顿量转换为矩阵并进行对角化
jw_matrix = get_sparse_operator(jw_hamiltonian).toarray()

# 打印 JW 矩阵
print(f"JW Matrix (Method 1, OpenFermion + PySCF):\n{jw_matrix}")

# 4. 进行本征值求解
eigenvalues, eigenvectors = np.linalg.eigh(jw_matrix)

# 打印最小本征值
print(f"FCI Energy (Method 1, OpenFermion + PySCF): {eigenvalues[0]}")
