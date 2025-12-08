import time
from functools import reduce
import warnings
warnings.filterwarnings('ignore', message='Hamiltonian coefficients must be real numbers.*')
import numpy as np
from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner
from openfermion.utils import hermitian_conjugated
from pyscf.scf.addons import partial_cholesky_orth_
from loky import get_reusable_executor

from mindquantum.core.operators import QubitOperator as QubitOperator_ms
from mindquantum.core.operators import Hamiltonian as Hamiltonian_ms
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import DepolarizingChannelAdder, SequentialAdder, MixerAdder, QubitNumberConstrain, GateSelector
max_workers = 16

class QSE(object):
    """
    Quantum Subspace Expansion is a method for computing the eigenvalues of a Hermitian operator in the subspace spanned by the reference state and it's excited states.

    Args:
        vqeci:
            VQECI object.
    """

    def __init__(self, vqeci):
        self.vqeci = vqeci
        # Convert the quri_parts circuit to MindQuantum circuit once during initialization
        self.adder = SequentialAdder([
            # 等价于 DepolarizingNoise(0.001, [], [])：应用于所有单比特门
            MixerAdder([
                QubitNumberConstrain(1),  # 限制为单比特门
                DepolarizingChannelAdder(p=0.001, n_qubits=1)  # 添加去极化噪声，概率 0.001
            ]),
            # 等价于 DepolarizingNoise(0.009, [], [gate_names.CNOT, gate_names.CZ])：应用于指定的双比特门（CNOT 和 CZ）
            MixerAdder([
                QubitNumberConstrain(2),  # 限制为双比特门
                GateSelector('cx'),  # 指定只针对 CNOT 和 CZ 门（MindQuantum 中门名为小写）
                DepolarizingChannelAdder(p=0.01, n_qubits=2)  # 添加去极化噪声，概率 0.009
            ]),    
            MixerAdder([
                QubitNumberConstrain(2),  # 限制为双比特门
                GateSelector('cz'),  # 指定只针对 CNOT 和 CZ 门（MindQuantum 中门名为小写）
                DepolarizingChannelAdder(p=0.01, n_qubits=2)  # 添加去极化噪声，概率 0.009
            ])
        ])


    def gen_singles_doubles(self):
        singles = []
        doubles = []
        n_electron = self.vqeci.n_electron
        n_qubit = self.vqeci.n_qubit
     
        for i_spatial in range(n_electron // 2):  # 空间轨道循环
            for a_spatial in range(n_electron // 2, n_qubit // 2):
                i_alpha, i_beta = 2 * i_spatial, 2 * i_spatial + 1
                a_alpha, a_beta = 2 * a_spatial, 2 * a_spatial + 1
                op = (1 / np.sqrt(2)) * (
                    FermionOperator(f"{a_alpha}^ {i_alpha}", 1.0) +
                    FermionOperator(f"{a_beta}^ {i_beta}", 1.0)
                )
                singles.append(op)
        doubles = []
        for i_spatial in range(n_electron // 2):
            for j_spatial in range(n_electron // 2):
                for a_spatial in range(n_electron // 2, n_qubit // 2):
                    for b_spatial in range(n_electron // 2, n_qubit // 2):
                        i_alpha, i_beta = 2 * i_spatial, 2 * i_spatial + 1
                        j_alpha, j_beta = 2 * j_spatial, 2 * j_spatial + 1
                        a_alpha, a_beta = 2 * a_spatial, 2 * a_spatial + 1
                        b_alpha, b_beta = 2 * b_spatial, 2 * b_spatial + 1

                        # Singlet-adapted double: sum over spin combinations
                        op = (
                            FermionOperator(f"{a_alpha}^ {i_alpha} {b_alpha}^ {j_alpha}", 1.0) +  # αα → αα
                            FermionOperator(f"{a_beta}^ {i_beta} {b_alpha}^ {j_alpha}", 1.0) +  # βα → βα (mixed)
                            FermionOperator(f"{a_alpha}^ {i_alpha} {b_beta}^ {j_beta}", 1.0) +  # αβ → αβ
                            FermionOperator(f"{a_beta}^ {i_beta} {b_beta}^ {j_beta}", 1.0)     # ββ → ββ
                        )
                        if op != FermionOperator():  # 避免零算符
                            doubles.append(op)


        return singles, doubles

    def gen_excitation_operators(self, types="ee", n_excitations=2):
        self.e_op = []
        if types == "ee":
            if n_excitations == 2:
                singles, doubles = self.gen_singles_doubles()
                self.e_op = singles + doubles
            elif n_excitations == 1:
                singles, doubles = self.gen_singles_doubles()
                self.e_op = singles

    def build_H(self):
        noperators = len(self.e_op)
        print(f"Generating {noperators} excitation operators...")
        self.hamiltonian = np.zeros((noperators + 1, noperators + 1), dtype=np.complex128)
        self.hamiltonian[0, 0] = self.vqeci.e
    
        # 定义工作进程中计算 H 矩阵元素的函数
        def compute_h_element(idx, jdx, e_op, fermionic_hamiltonian, n_qubit, opt_circuit, opt_param)   :
            if jdx == 0:
                myop_i_fermi = hermitian_conjugated(e_op[idx - 1]) * fermionic_hamiltonian
            else:
                myop_i_fermi = hermitian_conjugated(e_op[idx - 1]) * fermionic_hamiltonian * e_op   [jdx - 1]
            
            # Transform to qubit operator using Jordan-Wigner
            qubit_op = jordan_wigner(myop_i_fermi)
            ms_qubit_op = QubitOperator_ms.from_openfermion(qubit_op)
            ham = Hamiltonian_ms(ms_qubit_op)
            
            # Create simulator in worker process
            sim = Simulator('mqvector', n_qubit)
            sim.apply_circuit(opt_circuit, pr=opt_param)  # Add pr here
            return sim.get_expectation(ham).real  # Assuming real value for Hermitian operator
    
        # 创建 loky 进程池
        executor = get_reusable_executor(max_workers=max_workers)  # 可根据需要调整   工作进程数
    
        # 提交任务到进程池，只计算上三角部分（包括对角线）
        futures = {}
        for idx in range(1, noperators + 1):
            for jdx in range(noperators + 1):
                if jdx > idx:  # 利用 Hermitian 对称性跳过下三角
                    continue
                
                future = executor.submit(
                    compute_h_element,
                    idx,
                    jdx,
                    self.e_op,
                    self.vqeci.fermionic_hamiltonian,
                    self.vqeci.n_qubit,
                    self.vqeci.opt_states[0],
                    self.vqeci.opt_param  # Pass opt_param here
                )
                futures[future] = (idx, jdx)
    
        # 收集结果并填充矩阵
        for future in futures:
            idx, jdx = futures[future]
            self.hamiltonian[idx, jdx] = future.result()
            if jdx != 0:  # 利用 Hermitian 对称性填充下三角
                self.hamiltonian[jdx, idx] = np.conj(self.hamiltonian[idx, jdx])
    
    def build_S(self):
        noperators = len(self.e_op)
        self.S = np.zeros((noperators + 1, noperators + 1), dtype=np.complex128)
        self.S[0, 0] = 1.0
    
        # 定义工作进程中计算 S 矩阵元素的函数
        def compute_s_element(idx, jdx, e_op, n_qubit, opt_circuit, opt_param)  :
            myop_i_fermi = hermitian_conjugated(e_op[idx - 1])
            if jdx != 0:
                myop_i_fermi = myop_i_fermi * e_op[jdx - 1]
            
            # Transform to qubit operator using Jordan-Wigner
            qubit_op = jordan_wigner(myop_i_fermi)
            ms_qubit_op = QubitOperator_ms.from_openfermion(qubit_op)
            ham = Hamiltonian_ms(ms_qubit_op)
            
            # Create simulator in worker process
            sim = Simulator('mqvector', n_qubit)
            sim.apply_circuit(opt_circuit, pr=opt_param)  # Add pr here
            return sim.get_expectation(ham).real  # Assuming real value for Hermitian operator
    
        # 创建 loky 进程池
        executor = get_reusable_executor(max_workers=max_workers)  # 可根据需要调整   工作进程数
    
        # 提交任务到进程池，只计算上三角部分（包括对角线）
        futures = {}
        for idx in range(1, noperators + 1):
            for jdx in range(noperators + 1):
                if jdx > idx:  # 利用 Hermitian 对称性跳过下三角
                    continue
                future = executor.submit(
                    compute_s_element,
                    idx,
                    jdx,
                    self.e_op,
                    self.vqeci.n_qubit,
                    self.vqeci.opt_states[0],
                    self.vqeci.opt_param  # Pass opt_param here
                )
                futures[future] = (idx, jdx)
    
        # 收集结果并填充矩阵
        for future in futures:
            idx, jdx = futures[future]
            self.S[idx, jdx] = future.result()
            if jdx != 0:  # 利用 Hermitian 对称性填充下三角
                self.S[jdx, idx] = np.conj(self.S[idx, jdx])

    def solve(self):
        self.build_H()
        self.build_S()

        # Use PySCF's partial cholesky orthogonalization to remove linear dependency
        threshold = 1.0e-7
        cholesky_threshold = 1.0e-9

        def dense_or_not(A):
            nnz = np.count_nonzero(A)
            total_elements = A.size
            sparsity = nnz / total_elements
            threshold = 0.1  # 设定阈值，例如 10%
            if sparsity < threshold:
                return False
            else:
                return True

        def eigh(h, s):
            dense_or_not(h)
            dense_or_not(s)
            x = partial_cholesky_orth_(s, canthr=threshold, cholthr=cholesky_threshold)
            xhx = reduce(np.dot, (x.T.conj(), h, x))
            e, c = np.linalg.eigh(xhx)
            c = np.dot(x, c)
            return e, c

        self.eigenvalues, self.eigenvectors = eigh(self.hamiltonian, self.S)
        # print up to 10 excitations
        eigenvalues = self.eigenvalues[:10]
        for idx, ie in enumerate(eigenvalues):
            print("{} {}".format(idx, ie))