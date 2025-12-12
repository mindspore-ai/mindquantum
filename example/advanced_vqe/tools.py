"""便于算法执行的一些基础工具函数"""

from typing import Optional

import numpy as np
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from scipy.optimize import minimize

from mindquantum import Circuit, FermionOperator, InteractionOperator, mqchem


def get_h4_geometry(dist=1.00):
    """H4 分子"""
    geometry = [['H', [0.0, 0.0, i * dist]] for i in range(4)]
    return geometry


def get_hf_geometry(dist=1.00):
    """HF 分子"""
    geometry = [['H', [0.0, 0.0, 0.0 * dist]], ['F', [0.0, 0.0, 1.0 * dist]]]
    return geometry


def get_h2o_geometry(dist=1.00):
    """H2O 水分子"""
    angle = np.radians(104.5)
    geometry = [
        ['O', [0.0, 0.0, 0.0]],
        ['H', [dist, 0.0, 0.0]],
        ['H', [dist * np.cos(angle), dist * np.sin(angle), 0.0]],
    ]
    return geometry


def get_mole_info(geometry: list, fci=True, mp2=False, ccsd=False, scf=False):
    """基于 pyscf 完成经典计算任务, 获取分子的信息, 如 FCI、HF 能量, 系统哈密顿量等"""
    basis = "sto3g"
    spin = 0
    mole = MolecularData(geometry, basis, multiplicity=2 * spin + 1)
    mole = run_pyscf(mole, run_fci=fci, run_mp2=mp2, run_ccsd=ccsd, run_scf=scf)
    ham_of = mole.get_molecular_hamiltonian()
    int_ops = InteractionOperator(*(ham_of.n_body_tensors.values()))
    f_ham = FermionOperator(int_ops).compress().normal_ordered()
    ham = mqchem.CIHamiltonian(f_ham)
    return mole.hf_energy, mole.fci_energy, ham, mole.n_qubits, mole.n_electrons


def fop_to_core(fop: FermionOperator):
    """s_0 [5^ 4] -> (5, 4)"""
    excitation = fop_to_excitation(fop)
    return tuple(op[0] for op in excitation)


def fop_to_excitation(fop: FermionOperator):
    """将费米子形式的激发算符转换为元组形式 5^ 4^ 1 0 ->  (5,4,1,0)"""
    return tuple(next(iter(fop.terms.keys())))


def core_to_excitation(core: tuple):
    """(5, 4, 3, 2) -> ((5,1), (4,1), (3,0), (2,0))"""
    return tuple((core[i], int(i < len(core) / 2)) for i in range(len(core)))


def get_circ(ansatz: FermionOperator):
    """获取UCC线路"""
    return Circuit([mqchem.UCCExcitationGate(fop) for fop in ansatz])


def run_vqe(ham: mqchem.CIHamiltonian, n_qubits: int, n_electrons: int, circ: Circuit, p0: Optional[np.ndarray] = None):
    """根据给定的量子线路运行 VQE 算法, 求基态能级."""
    if p0 is None:
        p0 = np.array([0.0] * len(circ.params_name))

    grad_ops = mqchem.MQChemSimulator(n_qubits, n_electrons).get_expectation_with_grad(ham, circ)
    res = minimize(grad_ops, p0, method='bfgs', jac=True)
    return res.fun, res.x
