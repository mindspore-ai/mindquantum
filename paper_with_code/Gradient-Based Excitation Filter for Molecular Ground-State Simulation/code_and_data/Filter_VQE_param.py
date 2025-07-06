## 集成化
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum import *
from share_funs import *

from scipy.optimize import minimize
import numpy as np
import multiprocessing
import time
import traceback


### 正文开始

def _get_new_ham_term(ham_term:tuple, fermion:FermionOperator, n_electrons:int):
    """将哈密顿中的一项根据激发算符和电子数进行压缩。"""
    
    # ham_term : (((0, 'Y'), (1, 'Z'), (2, 'X'), (6, 'Z')), 1.23)
    core = list(fermion.terms.keys())[0] # ((3, 1), (2, 1), (1, 0), (0, 0))
    coeff = ham_term[1] # 1.23

    if core == (): # 常数项
        return QubitOperator(core, coeff)
    
    this, other = [], [] # other 不受激发算符影响的比特
    if len(core) == 2: # 单激发算符
        p, q = core[1][0], core[0][0]
        this_qubits = [p, q] # this 受到激发算符影响的比特

    else: # 双激发算符
        p, q, r, s = core[0][0], core[1][0], core[2][0], core[3][0]
        this_qubits = [p, q, r, s]
        
    this = [(i, v) for i, v in ham_term[0] if i in this_qubits]
    other = [(i, v) for i, v in ham_term[0] if i not in this_qubits]

    other_occ_qubits, other_occ_ops = [], []
    other_un_occ_qubits, other_un_occ_ops = [], []
    occ_qubits = set(range(n_electrons)) # HF 态中,被电子占据的比特
    for i, v in other:
        if i in occ_qubits:
            other_occ_qubits.append(i)
            other_occ_ops.append(v)
        else:
            other_un_occ_qubits.append(i)
            other_un_occ_ops.append(v)
    if set(other_un_occ_ops + other_occ_ops) & set(["X", "Y"]):
        coeff = 0
    else:
        coeff = ham_term[1]
    coeff *= (-1) ** (other_occ_ops.count("Z"))
    
    return QubitOperator(this, coeff)


def _compress_fermion(fermion:FermionOperator):
    """将费米子激发算符进行压缩, 返回压缩后的激发算符和模拟器需要用的量子比特数"""
    core = list(fermion.terms.keys())[0]
    coeff = list(fermion.terms.values())[0]
    if len(core) == 2:
        term = ((1, 1), (0, 0))
    else:
        term = ((3, 1), (2, 1), (1, 0), (0, 0))
    return FermionOperator(term, coeff), len(core)


def _compress_ham_term(ham_term:QubitOperator, fermion:FermionOperator):
    """将哈密顿量的一项进行压缩"""
    core = list(ham_term.terms.keys())[0]
    coeff = list(ham_term.terms.values())[0]
    qubits = [term[0] for term in list(fermion.terms.keys())[0]] # 费米子算符的比特
    qubits.reverse() # 比特由小到大排

    if core == ():
        pass
    else:
        core = tuple([(qubits.index(term[0]), term[1]) for term in core])

    return QubitOperator(core, coeff)


def compress_ham_fermion_qubits(ham: Hamiltonian, fermion:FermionOperator, n_electrons:int):
    """将哈密顿量和激发算符压缩在以作用在 HF 态上很小的子空间内. 

    输入:
        ham(Hamiltonian): 分子的哈密顿量
        fermion(FermionOperator): 激发算符
        n_electrons(int): 分子的电子数

    返回:
        new_ham(Hamiltonian): 压缩后的哈密顿量
        new_fermion(FermionOperator): 压缩后的激发算符
        n_qubits(int): 模拟器需要用的量子比特数

    案例:
    ```python
        >>> fermion = FermionOperator("3^ 1", "a") 
        >>> ham = Hamiltonian(QubitOperator("Z0 Z1 Z2 Z3 Z4 Z5", 1.23))
        >>> new_ham, new_fermion, n_qubits = compress_ham_fermion_qubits(ham, fermion, n_electrons=2)
        >>> print(new_ham)
        >>> print(new_fermion)
        >>> print(n_qubits)
            -1.23 [Z0 Z1]
            a [1^ 0]
            2
    ```
    """
    new_ham = QubitOperator()
    for term in ham.ham_termlist: # [(((0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'Z'), (4, 'Z'), (5, 'Z')), 1.0)]
        new_term = _get_new_ham_term(term, fermion, n_electrons)
        new_op = _compress_ham_term(new_term, fermion)
        new_ham += new_op
    new_fermion, n_qubits = _compress_fermion(fermion)

    return Hamiltonian(new_ham), new_fermion, n_qubits


def get_ansatz_circ(ansatz:FermionOperator):
    """对给定的非幺正拟设生成 JW 变换的量子线路"""
    ansatz -= ansatz.hermitian()
    qubit_ops = Transform(ansatz).jordan_wigner()
    circ = TimeEvolution(qubit_ops.imag, 1.0).circuit
    return circ


def success_check(energy:float, fci_energy:float):
    """判断给定能量是否达到化学精度"""
    return abs(fci_energy - energy) < chem_acc


def get_hf_circ(n_electrons:int):
    """根据电子数生成 HF 线路"""
    return Circuit([X.on(i) for i in range(n_electrons)]) # HF 线路


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


def get_deputy(n_qubits:int, n_electrons:int):
    """获取原始算符池中各代理算符的参数名 (UCCSD 拟设中由自由参数名) 及代理算符
    
    输入:
        n_qubits(int): UCCSD 的比特数
        n_electrons(int): 分子的电子数
    返回:
        deputy_names(list[str]): 代理参数的参数名
        deputy_ops(list[Fermion]): 代理算符, 都是单一参数且权重为 1 的激发算符
        uccsd_ansatz(Fermion): UCCSD 拟设 (非幺正)
    """
    uccsd_ansatz = uccsd_singlet_generator(n_qubits, 
                                        n_electrons, 
                                        anti_hermitian=False) # 非幺正
    
    deputy_pool = {} # dict[str:Fermion] 代理算符池,键为独立参数的 str, 值为使用该参数的一个算符. 这里只统计含单参数的算符. 

    # 挑选出所有算符中的独立参数,并为每一个独立参数配一个算符
    for op in uccsd_ansatz: # op Fermion: -2*a [5^ 1]
        coeff = list(op.terms.values())[0] # PR: -2*a
        if len(coeff.params_name) == 1: # 只考虑含单个参数的算符
            coeff_name = coeff.params_name[0] # string: 'a'
            if coeff_name not in deputy_pool:
                term = list(op.terms.keys())[0] # tuple: ((5, 1), (1, 0))
                new_op = FermionOperator(term, coeff_name) # Fermion: a [5^ 1]
                deputy_pool.update({coeff_name: new_op})

    deputy_names = list(deputy_pool.keys()) # list[str], 各代理算符的参数名
    deputy_ops = list(deputy_pool.values()) # list[Fermion], 幺正化
    return deputy_names, deputy_ops, uccsd_ansatz


def cut_list_by_mag(lst, mag:float=0.5):
    """对列表进行按数量级差进行截取,若前后两个数相差 mag 个数量级, 就从此截断"""    
    # 2. 寻找截断点
    index = len(lst)  # 默认不截断
    for i in range(len(lst)-1):
        if lst[i] / lst[i+1] >= 10**mag:
            index = i + 1
            break
    return lst[:index]


def loop_parallel(names, ops, ham, n_electrons, tol):
    large_grads_dict = {}
    st = time.time()
    p0 = np.array([0.0])
    for name, op in zip(names, ops):
        try:
            new_ham, new_op, n_qubits = compress_ham_fermion_qubits(ham, op, n_electrons)
            hf_circ = Circuit([X.on(i) for i in range(n_qubits // 2)]) # 单激发算符 2 个比特，双激发算符4个，占据数占一半
            circ = hf_circ + get_ansatz_circ(new_op)
            grad_ops = Simulator("mqvector", n_qubits).get_expectation_with_grad(new_ham, circ)
            grad = grad_ops(p0)[1][0][0]
            grad = np.abs(grad[0])
            if grad > tol:
                large_grads_dict[name] = grad 
        except Exception as e:
            print('error')
            traceback.print_exc()
    return large_grads_dict, time.time() - st

def get_good_deputy(deputy_names:list, 
                    deputy_ops:list, 
                    ham:Hamiltonian,
                    n_qubits:int,
                    n_electrons:int, 
                    tol:float=0.,
                    mag:float=0.5, parallel:bool=True):
    """对代理参数及代理算符池进行梯度测试. 
    并返回通过阈值筛选的优秀代理参数名, 及对应的梯度绝对值. 

    输入
        depupy_names(list[str]): 代理参数名列表
        deputy_ops(list[Fermoin]): 代理激发算符列表
        ham(Hamiltonian): 分子的哈密顿量
        n_qubits(int): 量子比特数
        n_electrons(int): 电子数
        tol(float): 梯度过滤阈值的小数位数, 默认0, 即全保留
    返回:
        good_deputy_names(list[str]): 对应梯度绝对值由大到小排列的优秀代理参数名列表
        good_deputy_grads(list[float]): 优秀代理的梯度绝对值
        注意: 通过阈值测试的优秀代理参数名和梯度绝对值. \
        它们的顺序根据测试的梯度绝对值从大到小排序. 
    """
    deputy_grads = []
    p0 = np.array([0.0]) # 算符的初始化参数
    start_time = time.time()
    # print('ops len {}'.format(len(deputy_ops)))
    all_times = []

    batch_size = 20

    if not parallel or len(deputy_ops) <= batch_size:
        for op in deputy_ops:
            st = time.time()
            new_ham, new_op, n_qubits = compress_ham_fermion_qubits(ham, op, n_electrons)
            hf_circ = Circuit([X.on(i) for i in range(n_qubits // 2)]) # 单激发算符 2 个比特，双激发算符4个，占据数占一半
            circ = hf_circ + get_ansatz_circ(new_op)
            grad_ops = Simulator("mqvector", n_qubits).get_expectation_with_grad(new_ham, circ)
            grad = grad_ops(p0)[1][0][0]
            deputy_grads.append(abs(grad[0]))
            all_times.append(time.time() - st)
        large_grads_dict = {i: np.abs(v) for i, v in zip(deputy_names, deputy_grads) if np.abs(v) > abs(tol)}
        # print('inner origin cost {:.2f}, {:.2f}'.format(time.time()-start_time, all_times[0]))
    
    else:
        all_tasks = []
        large_grads_dict = {}
        max_process = min(20, int(len(deputy_ops)/batch_size))
        with multiprocessing.Pool(processes=max_process) as  pool:
            idx = 0
            while idx < len(deputy_ops):
                all_tasks.append(pool.apply_async(loop_parallel, args=(deputy_names[idx:idx+batch_size], deputy_ops[idx:idx+batch_size], ham, n_electrons, abs(tol))))
                idx = idx + batch_size
            for task in all_tasks:
                res = task.get()
                if not res:
                    continue
                all_times.append(res[1])
                large_grads_dict.update(res[0])  
            pool.close()
            pool.join()
            # print('inner parallel cost {:.2f}, {:.2f}'.format(time.time()-start_time, all_times[0]))

    # hf 态下, 超过阈值的参数    
    # 将代理根据梯度从大到小进行排序
    good_deputy_names = sorted(large_grads_dict, key=large_grads_dict.get, reverse=True)
    good_deputy_grads = []
    for k in good_deputy_names:
        good_deputy_grads.append(large_grads_dict[k])
    
    good_deputy_grads = cut_list_by_mag(good_deputy_grads, mag)
    good_deputy_names = good_deputy_names[:len(good_deputy_grads)]

    return good_deputy_names, good_deputy_grads

def get_deputy_ops(ansatz:FermionOperator):
    """从代理拟设中提取出代理算符"""
    deputy_ops = []
    for op in ansatz:
        deputy_ops.append(list(op.terms.keys())[0])
    return deputy_ops


def get_circ(good_deputy_names:list, 
             uccsd_ansatz:FermionOperator, 
             n_electrons:int):
    """根据超过筛选阈值的代理参数,重新构建有效拟设线路

    输入:
        good_deputy_names(list[str]): 优秀代理的参数名列表
        uccsd_ansatz(Fermion): UCCSD 拟设
    返回:
        circ(Circuit): 有效拟设线路
    """
    
    good_names_set = set(good_deputy_names)
    all_names_set = set(uccsd_ansatz.params_name)
    diff_names = list(all_names_set - good_names_set)
    ansatz = uccsd_ansatz.subs({k: 0 for k in diff_names})
    deputy_ops = get_deputy_ops(ansatz)
    # 根据有效拟设构建量子线路,并执行 VQE 获得分子基态能级
    return deputy_ops, ansatz, get_hf_circ(n_electrons) + get_ansatz_circ(ansatz)


def take_ablation_test(deputy_names:list,
                     deputy_grads:list,
                     ansatz:FermionOperator,
                     ham:Hamiltonian,
                     n_qubits:int,
                     n_electrons:int,
                     fci_energy:float,
                     hf_energy:float):
    """融合实验，逐次多删除一个参数后, VQE 的性能. 

    返回: VQE 能量列表, 是否成功列表， 参数梯度列表. 
    """
    hf_circ = get_hf_circ(n_electrons)
    energy_list = []
    success_list = []
    num_ops_list = []
    num_params_list = []
    for name in deputy_names[::-1][:-1]: # 从梯度最小的开始，逐个赋值为 0 直到最后一个
        ansatz = ansatz.subs({name: 0.})
        num_ops_list.append(len(ansatz))
        num_params_list.append(len(ansatz.params_name))
        circ = hf_circ + get_ansatz_circ(ansatz)
        p0 = [0.] * len(ansatz.params_name) # 参数全 0 初始化
        grad_ops = Simulator(Device, n_qubits).get_expectation_with_grad(ham, circ)
        res = minimize(fun, p0, args=(grad_ops), method='bfgs', jac=True)
        energy_list.append(res.fun)
        success_list.append(success_check(res.fun, fci_energy))
    # 最后补上 HF 能量的信息
    energy_list.append(hf_energy)
    success_list.append(success_check(hf_energy, fci_energy))
    num_params_list.append(0)
    num_ops_list.append(0)
    return energy_list, success_list, num_params_list, num_ops_list


def deputy_fast_check(geometry:list, tol:float=0., mag:float=5., show:bool=True):
    """在正式运行前看一下优秀代理参数及其梯度信息"""
    _, _, ham, n_qubits, n_electrons = get_mole_info(geometry, run_fci=False)
    deputy_names, deputy_ops, uccsd_ansatz =  get_deputy(n_qubits, n_electrons)
    good_deputy_names, good_deputy_grads = get_good_deputy(deputy_names, 
                                                           deputy_ops, 
                                                           ham, n_qubits, 
                                                           n_electrons, 
                                                           tol=tol,
                                                           mag=mag)
    if show:
        print("原始参数量: \t", len(uccsd_ansatz.params_name))
        print("代理参数量:\t", len(deputy_names))
        print("优秀代理参数量: \t", len(good_deputy_names))

        for k, v in zip(good_deputy_names, good_deputy_grads):
            print(k, ":", v)
    return good_deputy_names, good_deputy_grads



def study_mole(geometry:list, tol:float=0., mag:float=0., ablation:bool=False):
    """对给定分子执行算法. 
    输入:
        geometry:list 分子结构
        tol:float: 梯度截断阈值
        fusion_test(bool): 是否测试再多删一个参数的效果
    返回:
        hf_energy(float): HF 能量
        vqe_energy(float): VQE 算法能量
        fci_energy(float): FCI 能量
        success(bool): VQE 算法是否达到化学精度
        ablation_energy(float): 再多删除一个算符所能达到的最小能量
        ablation_success(bool): 再多删除一个算符是否能达到化学精度
    """

    hf_energy, fci_energy, ham, n_qubits, n_electrons = get_mole_info(geometry, run_fci=True)
    deputy_names, deputy_ops, uccsd_ansatz = get_deputy(n_qubits, n_electrons)

    good_deputy_names, good_deputy_grads = get_good_deputy(deputy_names, deputy_ops, ham, n_qubits, n_electrons, tol, mag)
    
    # single_count, double_count = count_ops(good_deputy_ops)
    num_params = len(good_deputy_names) # VQE 使用的自由参数的数量
    good_deputy_ops, ansatz, circ = get_circ(good_deputy_names, uccsd_ansatz, n_electrons)
    single_count, double_count = count_ops(good_deputy_ops)

    vqe_energy, _ = run_vqe(ham, circ)
    success = success_check(vqe_energy, fci_energy)
    main_info = [hf_energy, vqe_energy, fci_energy, num_params, success, single_count, double_count, circ.depth(False)]
    tmp = [0] * len(good_deputy_names)
    ablation_info = [tmp, tmp, tmp, tmp]

    if ablation:
        ablation_energies, ablation_succeses, ablation_params, ablation_ops = take_ablation_test(good_deputy_names,
                                                                                good_deputy_grads,
                                                                                ansatz,
                                                                                ham, n_qubits, 
                                                                                n_electrons,
                                                                                fci_energy,
                                                                                hf_energy)
        ablation_info = [ablation_energies, ablation_succeses, ablation_params, ablation_ops]

    return main_info, ablation_info


def study_given_params_num(geometry:list, params_num:int):
    """对给定分子执行算法. 
    输入:
        geometry:list 分子结构
        params_num_list: list 代理参数量
    返回:
        hf_energy(float): HF 能量
        vqe_energy(float): VQE 算法能量
        fci_energy(float): FCI 能量
        success(bool): VQE 算法是否达到化学精度
    """

    hf_energy, fci_energy, ham, n_qubits, n_electrons = get_mole_info(geometry, run_fci=True)
    deputy_names, deputy_ops, uccsd_ansatz = get_deputy(n_qubits, n_electrons)
    if not params_num:
        return hf_energy, hf_energy, fci_energy, 0, success_check(hf_energy, fci_energy), 0, 0, 0
    good_deputy_names, good_deputy_grads = get_good_deputy(deputy_names, deputy_ops, ham, n_qubits, n_electrons, tol=0., mag=5)
    good_deputy_names = good_deputy_names[:params_num]
    good_deputy_grads = good_deputy_grads[:params_num]
    num_params = len(good_deputy_names) # VQE 使用的自由参数的数量
    good_deputy_ops, ansatz, circ = get_circ(good_deputy_names, uccsd_ansatz, n_electrons)
    single_count, double_count = count_ops(good_deputy_ops)
    
    vqe_energy, _ = run_vqe(ham, circ)
    success = success_check(vqe_energy, fci_energy)
    return hf_energy, vqe_energy, fci_energy, num_params, success, single_count, double_count, circ.depth(False)


