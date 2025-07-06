"""使用 ADAPT-VQE 算法求解分子基态能量"""

from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum import *
from share_funs import *

def get_mole_energy(geometry:list, run_fci:bool=True):
    """基于 OpenFermion 完成经典计算任务, 获取分子的 FCI、HF 能量, 以及分子"""
    basis = "sto3g"
    spin = 0

    mole = MolecularData(geometry, basis, multiplicity=2 * spin + 1)
    mole = run_pyscf(mole, run_scf=0, run_ccsd=0, run_fci=run_fci)
    return mole.hf_energy, mole.fci_energy, mole

def get_mole_info(geometry:list, run_fci:bool=True):
    """基于 OpenFermion 完成经典计算任务, 获取分子的信息, 如 FCI、HF 能量, 系统哈密顿量等
    
    输入:
        geometry(list): 分子的几何结构
        run_fci(bool): 是否运行 FCI
    返回:
        hf_energy(float): HF 能量
        fci_energy(float): FCI 能量
        ham(Hamiltonian): 哈密顿量
        n_qubits(int): 量子比特数
        n_electrons: 电子数
    """
    hf_energy, fci_energy, mole = get_mole_energy(geometry, run_fci)
    ham_qubit_ops = get_qubit_hamiltonian(mole)
    n_qubits = ham_qubit_ops.count_qubits()
    ham = Hamiltonian(ham_qubit_ops.real)
    return hf_energy, fci_energy, ham, n_qubits, mole.n_electrons


def get_grads(sim:Simulator, ham:Hamiltonian, ops_pool:dict):
    """计算当前状态下，算符池中各算符梯度的绝对幅值"""
    grads = {}
    for k, v in ops_pool.items():
        op = FermionOperator(k, v)
        tmp = Transform(op - op.hermitian()).jordan_wigner()
        circ = TimeEvolution(tmp.imag).circuit
        grad_ops = sim.get_expectation_with_grad(ham, circ)
        _, grad = grad_ops([0.])
        grads.update({k: abs(grad.flatten()[0])})
    return grads


def get_best_op(grads:dict, tol:float=1e-5):
    """选择梯度值最大的激发算符"""
    sortd_ops = sorted(grads, key=grads.get, reverse=True)
    best_op = sortd_ops[0]
    if grads[best_op] < tol:
        return ()
    return best_op


def get_op_circ(op:FermionOperator):
    """获取算符的量子线路"""
    tmp = Transform(op - op.hermitian()).jordan_wigner()
    circ = TimeEvolution(tmp.imag).circuit
    return circ

    
def run_adapt_vqe(geometry:list, tol:float=1e-3):
    """运行 ADAPT-VQE 算法"""
    mole = MolecularData(geometry, "sto3g", multiplicity=1)
    mole = run_pyscf(mole, run_scf=False, run_ccsd=False, run_fci=True)
    ham_qubit_ops = get_qubit_hamiltonian(mole)
    n_qubits = ham_qubit_ops.count_qubits()
    ham = Hamiltonian(ham_qubit_ops.real)
    uccsd_ansatz = uccsd_singlet_generator(n_qubits, mole.n_electrons, False)
    terms_pool = list(uccsd_ansatz.terms.keys()) # 无系数激发算符池
    theta = PRGenerator('theta_')

    # 预先定义一组，防止 0 参数现象
    energy, _, _, _, _ = get_mole_info(geometry, False)
    simpled_ansatz = FermionOperator()
    free_param_names = []
    single_count, double_count = 0, 0

    circ = Circuit([X.on(i) for i in range(mole.n_electrons)]) # HF 线路
    params = {}
    sim = Simulator(Device, n_qubits)
    good_ops = []
    ops_pool = {term: theta.new() for term in terms_pool} # 给每个激发算符都加上独立参数 
    iter_num = len(ops_pool)
    for i in range(iter_num):
        sim.apply_circuit(circ, params)
        grads = get_grads(sim, ham, ops_pool)
        best_op = get_best_op(grads, tol)
        if best_op == ():
            break
        circ += get_op_circ(FermionOperator(best_op, ops_pool[best_op]))
        energy, x = run_vqe(ham, circ)
        # print(i, energy)
        params = {k:v for k, v in zip(circ.params_name, x)}
        ops_pool.pop(best_op)
        good_ops.append(best_op)
        sim.reset()

    single_count, double_count = count_ops(good_ops)
    simpled_ansatz = FermionOperator()
    # 重新加上自旋匹配，降低参数量
    free_param_names = [] # 自由参数，只统计单参数的算符，避免引入无效参数
    for k, v in uccsd_ansatz.terms.items():
        if k in good_ops:
            simpled_ansatz += FermionOperator(k, v)
            if len(v) == 1:
                free_param_names.append(v.params_name[0])
    
    return energy, abs(energy-mole.fci_energy) < chem_acc, simpled_ansatz, set(free_param_names),\
            single_count, double_count, circ.depth(False)
