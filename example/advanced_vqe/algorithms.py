"""构建多种量子算法"""

import itertools
from typing import Optional

import numpy as np
from pyscf import gto, scf, symm
from tools import core_to_excitation, fop_to_core, fop_to_excitation, get_mole_info, run_vqe

from mindquantum import Circuit, FermionOperator, PRGenerator, mqchem, uccsd_singlet_generator


def get_uccsd(n_qubits: int, n_electrons: int, share_parameter=True):
    """获取 UCCSD 拟设, free_parameter 激发算符时候共享参数"""
    uccsd = uccsd_singlet_generator(n_qubits, n_electrons, False)
    if share_parameter:
        ansatz = uccsd
    else:
        theta = PRGenerator('theta_')
        ansatz = FermionOperator()
        for fop in uccsd:
            ansatz += FermionOperator(fop_to_excitation(fop), theta.new())
    return list(ansatz)


def get_best_core(grads: dict):
    """选择梯度值最大的激发算符"""
    sortd_cores = sorted(grads, key=grads.get, reverse=True)
    best_core = sortd_cores[0]
    norm = np.sqrt(sum(i**2 for i in grads.values()))
    return best_core, norm


def get_init_grads(sim: mqchem.MQChemSimulator, com_mapping: FermionOperator):
    """计算当前状态下，算符池中各算符的初始梯度幅值"""
    init_grads = {}
    for k, v in com_mapping.items():
        init_grads[k] = abs(sim.get_expectation(mqchem.CIHamiltonian(v)))
    return init_grads


def run_adapt_vqe(geometry: list, tol: float = 1e-3):
    """运行 ADAPT-VQE 算法"""
    hf, fci, ham, n_qubits, n_electrons = get_mole_info(geometry, fci=True)
    f_ham = ham.fermion_hamiltonian
    uccsd = uccsd_singlet_generator(n_qubits, n_electrons, False)
    cores_pool = [fop_to_core(fop) for fop in uccsd]
    theta = PRGenerator("theta_")
    pre_res = {}
    sim = mqchem.MQChemSimulator(n_qubits, n_electrons)
    energys, errors = [hf], [abs(hf - fci)]
    com_mapping = {
        core: (
            FermionOperator(core_to_excitation(core)) * f_ham - f_ham * FermionOperator(core_to_excitation(core))
        ).normal_ordered()
        for core in cores_pool
    }
    circ = Circuit()
    while True:
        sim.reset()
        sim.apply_circuit(circ, pre_res)
        init_grads = get_init_grads(sim, com_mapping)
        best_core, norm = get_best_core(init_grads)
        if norm < tol:
            break
        fop = FermionOperator(core_to_excitation(best_core), theta.new())
        circ += mqchem.UCCExcitationGate(fop)
        p0 = np.array([pre_res[k] if k in pre_res else 0.0 for k in circ.params_name])  # warm-start
        energy, params = run_vqe(ham, n_qubits, n_electrons, circ, p0)
        pre_res = dict(zip(circ.params_name, params))
        energys.append(energy)
        errors.append(abs(fci - energy))
    return energys, errors


def get_k_upccgsd(n_qubits: int, k: int, share_parameter=False):
    """获取upccgsd拟设"""
    ansatz = []
    theta = PRGenerator('theta_')
    for _ in range(k):
        for p, q in itertools.product(range(n_qubits // 2), range(n_qubits // 2)):
            if p <= q:
                continue
            # single
            if not share_parameter:
                ansatz.append(FermionOperator(((2 * p, 1), (2 * q, 0)), theta.new()))
                ansatz.append(FermionOperator(((2 * p + 1, 1), (2 * q + 1, 0)), theta.new()))
            else:
                param = theta.new()
                ansatz.append(FermionOperator(((2 * p, 1), (2 * q, 0)), param))
                ansatz.append(FermionOperator(((2 * p + 1, 1), (2 * q + 1, 0)), param))
            # double
            ansatz.append(FermionOperator(((2 * p + 1, 1), (2 * p, 1), (2 * q + 1, 0), (2 * q, 0)), theta.new()))
    return ansatz


def get_state_irrep(group_name: str, mo_irreps: list[str], n_electrons: int, fop: Optional[FermionOperator] = None):
    """获取单行列式量子态的不可约表示"""
    spin_orbs = list(range(n_electrons))

    if fop is not None:
        core = fop_to_core(fop)
        mid = len(core) // 2
        creations = core[:mid]
        annihilations = core[mid:]
        for i, annihilation in enumerate(annihilations):
            spin_orbs[annihilation] = creations[i]

    mole_orbs = np.array(spin_orbs) // 2
    irreps = [mo_irreps[orb] for orb in mole_orbs]
    irreps_id = [symm.irrep_name2id(group_name, irrep) for irrep in irreps]
    state_irrep_id = 0
    for irrep_id in irreps_id:
        state_irrep_id ^= irrep_id

    return symm.irrep_id2name(group_name, state_irrep_id)


def sym_filter(geo, ansatz: list[FermionOperator], n_electrons: int, symmetry=True):
    """过滤掉不符合点群对称性的激发算符"""
    mol = gto.M(atom=geo, basis="sto-3g", symmetry=symmetry, verbose=0)  # 启用对称性分析
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()

    mo_irreps = symm.label_orb_symm(
        mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff  # 输出irrep符号（如'A1'）  # 对称性适配基组  # 分子轨道系数
    )
    print('\n点群名称为\n', mol.groupname)  # #  C2v
    mo_irreps = list(mo_irreps)  # ['A1' 'A1' 'B2' 'A1' 'B1' 'A1' 'B2']
    hf_irrep = get_state_irrep(mol.groupname, mo_irreps, n_electrons)
    # print(' '.join(mo_irreps))

    sym_ansatz = []
    for fop in ansatz:
        state_irrep = get_state_irrep(mol.groupname, mo_irreps, n_electrons, fop)
        if state_irrep == hf_irrep:
            sym_ansatz.append(fop)
    return sym_ansatz
