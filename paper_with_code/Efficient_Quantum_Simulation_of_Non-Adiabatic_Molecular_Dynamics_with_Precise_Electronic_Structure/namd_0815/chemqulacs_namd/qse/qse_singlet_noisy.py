import time
from functools import reduce

import numpy as np
from openfermion.ops import FermionOperator
from openfermion.utils import hermitian_conjugated
from pyscf.scf.addons import partial_cholesky_orth_
from loky import get_reusable_executor
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator,create_qulacs_density_matrix_estimator

from loky import ProcessPoolExecutor
from quri_parts.circuit.noise import NoiseModel, DepolarizingNoise

import quri_parts.circuit.gate_names as gate_names
noise_model = [
        DepolarizingNoise(0.001, qubit_indices=[], target_gates=[]),
        DepolarizingNoise(0.009, qubit_indices=[], target_gates=[gate_names.CNOT, gate_names.CZ])
    ]
class QSE(object):
    """
    Quantum Subspace Expansion is a method for computing the eigenvalues of a Hermitian operator in the subspace spanned by the reference state and it's excited states.

    Args:
        vqeci:
            VQECI object.
    """

    def __init__(self, vqeci):
        self.vqeci = vqeci

    def gen_singles_doubles(self):
        singles = []
        doubles = []
        n_electron = self.vqeci.n_electron
        n_qubit = self.vqeci.n_qubit
        # singles (only spin-conserving for singlets)
        # for i in range(n_electron):
        #     for a in range(n_electron, n_qubit):
        #         spin_i = i % 2  # 假设偶数索引为α，奇数为β
        #         spin_a = a % 2
        #         if spin_i == spin_a:  # 仅保留自旋守恒的单激发
        #             singles += [FermionOperator("{}^ {}".format(a, i), 1.0)]
        # for i in range(n_electron):
        #     for j in range(n_electron):
        #         for a in range(n_electron, n_qubit):
        #             for b in range(n_electron, n_qubit):
        #                 spin_i = i % 2
        #                 spin_j = j % 2
        #                 spin_a = a % 2
        #                 spin_b = b % 2
        #                 delta_sz = (spin_a - spin_i) + (spin_b - spin_j)
        #                 if delta_sz == 0:  # 只保留 ΔSz=0 的双激发
        #                     doubles += [
        #                         FermionOperator("{}^ {} {}^ {}".format(a, i, b, j), 1.0)
        #                     ]
     
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
        op_mapper = self.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
            n_spin_orbitals=self.vqeci.n_qubit, n_fermions=self.vqeci.n_electron
        )
        self.hamiltonian = np.zeros((noperators + 1, noperators + 1), dtype=np.complex128)
        self.hamiltonian[0, 0] = self.vqeci.e

        # 定义工作进程中计算 H 矩阵元素的函数
        def compute_h_element(idx, jdx, op_mapper, e_op, fermionic_hamiltonian, opt_state):
            if jdx == 0:
                myop_i_fermi = hermitian_conjugated(e_op[idx - 1]) * fermionic_hamiltonian
                myop_i_qubit = op_mapper(myop_i_fermi)
            else:
                myop_i_fermi = hermitian_conjugated(e_op[idx - 1])
                myop_ij = myop_i_fermi * fermionic_hamiltonian * e_op[jdx - 1]
                myop_i_qubit = op_mapper(myop_ij)
            # 在工作进程中创建 estimator，避免传递 executor

            estimator = create_qulacs_density_matrix_estimator(noise_model)
            return estimator([myop_i_qubit], [opt_state])[0].value

        # 创建 loky 进程池
        executor = get_reusable_executor(max_workers=8)  # 可根据需要调整工作进程数

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
                    op_mapper,
                    self.e_op,
                    self.vqeci.fermionic_hamiltonian,
                    self.vqeci.opt_states[0]
                )
                futures[future] = (idx, jdx)

        # 收集结果并填充矩阵
        for future in futures:
            idx, jdx = futures[future]
            self.hamiltonian[idx, jdx] = future.result()
            if jdx != 0:  # 利用 Hermitian 对称性填充下三角
                self.hamiltonian[jdx, idx] = self.hamiltonian[idx, jdx].conj()

    def build_S(self):
        noperators = len(self.e_op)
        op_mapper = self.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
            n_spin_orbitals=self.vqeci.n_qubit, n_fermions=self.vqeci.n_electron
        )
        self.S = np.zeros((noperators + 1, noperators + 1), dtype=np.complex128)
        self.S[0, 0] = 1.0

        # 定义工作进程中计算 S 矩阵元素的函数
        def compute_s_element(idx, jdx, op_mapper, e_op, opt_state):
            myop_i_fermi = hermitian_conjugated(e_op[idx - 1])
            if jdx == 0:
                myop_i_qubit = op_mapper(myop_i_fermi)
            else:
                myop_ij = myop_i_fermi * e_op[jdx - 1]
                myop_i_qubit = op_mapper(myop_ij)
            # 在工作进程中创建 estimator，避免传递 executor
            estimator = create_qulacs_density_matrix_estimator(noise_model)
            return estimator([myop_i_qubit], [opt_state])[0].value

        # 创建 loky 进程池
        executor = get_reusable_executor(max_workers=8)  # 可根据需要调整工作进程数

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
                    op_mapper,
                    self.e_op,
                    self.vqeci.opt_states[0]
                )
                futures[future] = (idx, jdx)

        # 收集结果并填充矩阵
        for future in futures:
            idx, jdx = futures[future]
            self.S[idx, jdx] = future.result()
            if jdx != 0:  # 利用 Hermitian 对称性填充下三角
                self.S[jdx, idx] = self.S[idx, jdx].conj()

    def solve(self):
        self.build_H()
        self.build_S()

        # Use PySCF's partial cholesky orthogonalization to remove linear dependency
        threshold = 1.0e-8
        cholesky_threshold = 1.0e-08

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
