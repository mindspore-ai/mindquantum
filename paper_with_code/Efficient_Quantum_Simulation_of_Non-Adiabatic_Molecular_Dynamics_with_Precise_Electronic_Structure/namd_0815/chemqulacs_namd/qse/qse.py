# ---------------串行版本-------------


# import time
# from functools import reduce
#
# import numpy as np
# from openfermion.ops import FermionOperator
# from openfermion.utils import hermitian_conjugated
# from pyscf.scf.addons import partial_cholesky_orth_
#
#
# class QSE(object):
#     """
#     Quantum Subspace Expansion is a method for computing the eigenvalues of a Hermitian operator in the subspace spanned by the reference state and it's excited states.
#
#     Args:
#         vqeci:
#             VQECI object.
#
#     """
#
#     def __init__(self, vqeci):
#         self.vqeci = vqeci
#
#     def gen_singles_doubles(self):
#         singles = []
#         doubles = []
#         n_electron = self.vqeci.n_electron
#         n_qubit = self.vqeci.n_qubit
#         # singles
#         for i in range(n_electron):
#             for a in range(n_electron, n_qubit):
#                 singles += [FermionOperator("{}^ {}".format(a, i), 1.0)]
#         # doubles
#         for i in range(n_electron):
#             for j in range(n_electron):
#                 for a in range(n_electron, n_qubit):
#                     for b in range(n_electron, n_qubit):
#                         doubles += [
#                             FermionOperator("{}^ {} {}^ {}".format(a, i, b, j), 1.0)
#                         ]
#         return singles, doubles
#
#     def gen_excitation_operators(self, types="ee", n_excitations=2):
#         self.e_op = []
#         if types == "ee":
#             if n_excitations == 2:
#                 singles, doubles = self.gen_singles_doubles()
#                 self.e_op = singles + doubles
#             elif n_excitations == 1:
#                 singles, doubles = self.gen_singles_doubles()
#                 self.e_op = singles
#
#     def build_H(self):
#         noperators = len(self.e_op)
#         op_mapper = self.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
#             n_spin_orbitals=self.vqeci.n_qubit, n_fermions=self.vqeci.n_electron
#         )
#         self.hamiltonian = np.zeros(
#             (noperators + 1, noperators + 1), dtype=np.complex128
#         )
#         self.hamiltonian[0, 0] = self.vqeci.e
#         for idx, iop in enumerate(self.e_op, 1):
#             myop_i_fermi = hermitian_conjugated(iop) * self.vqeci.fermionic_hamiltonian
#             myop_i_qubit = op_mapper(myop_i_fermi)
#             self.hamiltonian[idx, 0] = self.vqeci.estimator(
#                 [myop_i_qubit], [self.vqeci.opt_states[0]]
#             )[0].value
#             self.hamiltonian[0, idx] = self.hamiltonian[idx, 0].conj()
#             for jdx, jop in enumerate(self.e_op, 1):
#                 myop_ij = myop_i_fermi * jop
#                 myop_ij = op_mapper(myop_ij)
#                 self.hamiltonian[idx, jdx] = self.vqeci.estimator(
#                     [myop_ij], [self.vqeci.opt_states[0]]
#                 )[0].value
#                 self.hamiltonian[jdx, idx] = (self.hamiltonian[idx, jdx]).conj()
#
#     def build_S(self):
#         noperators = len(self.e_op)
#         op_mapper = self.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
#             n_spin_orbitals=self.vqeci.n_qubit, n_fermions=self.vqeci.n_electron
#         )
#         self.S = np.zeros((noperators + 1, noperators + 1), dtype=np.complex128)
#         self.S[0, 0] = 1.0
#         for idx, iop in enumerate(self.e_op, 1):
#             myop_i_fermi = hermitian_conjugated(iop)
#             myop_i_qubit = op_mapper(myop_i_fermi)
#             self.S[idx, 0] = self.vqeci.estimator(
#                 [myop_i_qubit], [self.vqeci.opt_states[0]]
#             )[0].value
#             self.S[0, idx] = (self.S[idx, 0]).conj()
#             for jdx, jop in enumerate(self.e_op, 1):
#                 myop_ij = myop_i_fermi * jop
#                 myop_ij = op_mapper(myop_ij)
#                 self.S[idx, jdx] = self.vqeci.estimator(
#                     [myop_ij], [self.vqeci.opt_states[0]]
#                 )[0].value
#                 self.S[jdx, idx] = (self.S[idx, jdx]).conj()
#
#     def solve(self):
#         time_start = time.time()
#         self.build_H()
#         self.build_S()
#         print(f"build H and S Time: {time.time() - time_start}")
#
#         # Use PySCF's partial cholesky orthogonalization to remove linear dependency
#         threshold = 1.0e-8
#         cholesky_threshold = 1.0e-08
#
#         def eigh(h, s):
#             x = partial_cholesky_orth_(s, canthr=threshold, cholthr=cholesky_threshold)
#             xhx = reduce(np.dot, (x.T.conj(), h, x))
#             e, c = np.linalg.eigh(xhx)
#             c = np.dot(x, c)
#             return e, c
#         time_start=time.time()
#         self.eigenvalues, self.eigenvectors = eigh(self.hamiltonian, self.S)
#         # print up to 10 excitations
#         eigenvalues = self.eigenvalues[:10]
#         for idx, ie in enumerate(eigenvalues):
#             print("{} {}".format(idx, ie))
#         print(f"eigh Time: {time.time() - time_start}")



# --------------loky进程池(由线程池修改而来，添加了executor传参， 但是报错说不可序列化)---------------
# from functools import reduce
#
# import numpy as np
# from openfermion.ops import FermionOperator
# from openfermion.utils import hermitian_conjugated
# from pyscf.scf.addons import partial_cholesky_orth_
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor
# from openfermion.ops import FermionOperator
# from openfermion.utils import hermitian_conjugated
#
#
#
# class QSE(object):
#     """
#     Quantum Subspace Expansion 是一种在参考态及其激发态张成的子空间中计算 Hermitian 算符特征值的方法。
#
#     Args:
#         vqeci: VQECI 对象。
#     """
#     def __init__(self, vqeci):
#         self.vqeci = vqeci
#
#     def gen_singles_doubles(self):
#         """生成单激发和双激发算符"""
#         singles = []
#         doubles = []
#         n_electron = self.vqeci.n_electron
#         n_qubit = self.vqeci.n_qubit
#         for i in range(n_electron):
#             for a in range(n_electron, n_qubit):
#                 singles += [FermionOperator("{}^ {}".format(a, i), 1.0)]
#         for i in range(n_electron):
#             for j in range(n_electron):
#                 for a in range(n_electron, n_qubit):
#                     for b in range(n_electron, n_qubit):
#                         doubles += [FermionOperator("{}^ {} {}^ {}".format(a, i, b, j), 1.0)]
#         return singles, doubles
#
#     def gen_excitation_operators(self, types="ee", n_excitations=2):
#         """生成激发算符"""
#         self.e_op = []
#         if types == "ee":
#             if n_excitations == 2:
#                 singles, doubles = self.gen_singles_doubles()
#                 self.e_op = singles + doubles
#             elif n_excitations == 1:
#                 singles, doubles = self.gen_singles_doubles()
#                 self.e_op = singles
#
#
#     def compute_h_element(self, idx, jdx, op_mapper,executor):
#         """计算哈密顿矩阵 H 的元素"""
#         from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator
#         estimator=create_qulacs_vector_concurrent_estimator(executor,concurrency=1)
#         if jdx == 0:
#             myop_i_fermi = hermitian_conjugated(self.e_op[idx - 1]) * self.vqeci.fermionic_hamiltonian
#             myop_i_qubit = op_mapper(myop_i_fermi)
#             return estimator([myop_i_qubit], [self.vqeci.opt_states[0]])[0].value
#         else:
#             myop_i_fermi = hermitian_conjugated(self.e_op[idx - 1])
#             myop_ij = myop_i_fermi * self.vqeci.fermionic_hamiltonian * self.e_op[jdx - 1]
#             myop_ij = op_mapper(myop_ij)
#             return estimator([myop_ij], [self.vqeci.opt_states[0]])[0].value
#
#     def compute_s_element(self, idx, jdx, op_mapper,executor):
#         """计算重叠矩阵 S 的元素"""
#         from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator
#         estimator=create_qulacs_vector_concurrent_estimator(executor,concurrency=1)
#         if jdx == 0:
#             myop_i_fermi = hermitian_conjugated(self.e_op[idx - 1])
#             myop_i_qubit = op_mapper(myop_i_fermi)
#             return estimator([myop_i_qubit], [self.vqeci.opt_states[0]])[0].value
#         else:
#             myop_i_fermi = hermitian_conjugated(self.e_op[idx - 1])
#             myop_ij = myop_i_fermi * self.e_op[jdx - 1]
#             myop_ij = op_mapper(myop_ij)
#             return estimator([myop_ij], [self.vqeci.opt_states[0]])[0].value
#
#     def build_H(self):
#         """构建并行化的哈密顿矩阵 H"""
#         noperators = len(self.e_op)
#         op_mapper = self.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
#             n_spin_orbitals=self.vqeci.n_qubit, n_fermions=self.vqeci.n_electron
#         )
#         self.hamiltonian = np.zeros((noperators + 1, noperators + 1), dtype=np.complex128)
#         self.hamiltonian[0, 0] = self.vqeci.e
#         from loky import get_reusable_executor
#
#
#         # 使用进程池并行计算所有元素
#         with get_reusable_executor(max_workers=1) as executor:
#             futures = {
#                 executor.submit(self.compute_h_element, idx, jdx, op_mapper,executor): (idx, jdx)
#                 for idx in range(1, noperators + 1)
#                 for jdx in range(noperators + 1)
#             }
#             for future in futures:
#                 idx, jdx = futures[future]
#                 self.hamiltonian[idx, jdx] = future.result()
#
#         # 确保 Hermitian 对称性
#         for idx in range(noperators + 1):
#             for jdx in range(idx + 1, noperators + 1):
#                 self.hamiltonian[idx, jdx] = self.hamiltonian[jdx, idx].conj()
#
#     def build_S(self):
#         """构建并行化的重叠矩阵 S"""
#         noperators = len(self.e_op)
#         op_mapper = self.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
#             n_spin_orbitals=self.vqeci.n_qubit, n_fermions=self.vqeci.n_electron
#         )
#         self.S = np.zeros((noperators + 1, noperators + 1), dtype=np.complex128)
#         self.S[0, 0] = 1.0
#         from loky import get_reusable_executor
#         # 使用进程池并行计算所有元素
#         with get_reusable_executor(max_workers=1) as executor:
#             futures = {
#                 executor.submit(self.compute_s_element, idx, jdx, op_mapper,executor): (idx, jdx)
#                 for idx in range(1, noperators + 1)
#                 for jdx in range(noperators + 1)
#             }
#             for future in futures:
#                 idx, jdx = futures[future]
#                 self.S[idx, jdx] = future.result()
#
#         # 确保 Hermitian 对称性
#         for idx in range(noperators + 1):
#             for jdx in range(idx + 1, noperators + 1):
#                 self.S[idx, jdx] = self.S[jdx, idx].conj()
#
#     def solve(self):
#         """求解特征值问题"""
#
#         start=time.time()
#         self.build_H()
#         self.build_S()
#         end=time.time()
#         print("Time taken for buiding H and S:",str(end-start))
#         # 这里假设已有 partial_cholesky_orth_ 和 reduce 函数
#         def eigh(h, s):
#             x = partial_cholesky_orth_(s, canthr=1.0e-8, cholthr=1.0e-08)
#             xhx = reduce(np.dot, (x.T.conj(), h, x))
#             e, c = np.linalg.eigh(xhx)
#             c = np.dot(x, c)
#             return e, c
#
#         self.eigenvalues, self.eigenvectors = eigh(self.hamiltonian, self.S)
#         for idx, ie in enumerate(self.eigenvalues[:10]):
#             print(f"{idx} {ie}")



#--------------------进程池2---
import time
from functools import reduce

import numpy as np
from openfermion.ops import FermionOperator
from openfermion.utils import hermitian_conjugated
from pyscf.scf.addons import partial_cholesky_orth_
from loky import get_reusable_executor
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator

import dill
from loky import set_loky_pickler

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
        # singles
        for i in range(n_electron):
            for a in range(n_electron, n_qubit):
                singles += [FermionOperator("{}^ {}".format(a, i), 1.0)]
        # doubles
        for i in range(n_electron):
            for j in range(n_electron):
                for a in range(n_electron, n_qubit):
                    for b in range(n_electron, n_qubit):
                        doubles += [
                            FermionOperator("{}^ {} {}^ {}".format(a, i, b, j), 1.0)
                        ]
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
            estimator = create_qulacs_vector_concurrent_estimator(None, concurrency=1)
            return estimator([myop_i_qubit], [opt_state])[0].value

        # 创建 loky 进程池
        set_loky_pickler('cloudpickle')
        executor = get_reusable_executor(max_workers=8)  # 可根据需要调整工作进程数
        set_loky_pickler('cloudpickle')
        # executor=ProcessPoolExecutor(max_workers=8)

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
            estimator = create_qulacs_vector_concurrent_estimator(None, concurrency=1)
            return estimator([myop_i_qubit], [opt_state])[0].value

        # 创建 loky 进程池
        set_loky_pickler('cloudpickle')
        executor = get_reusable_executor(max_workers=8)  # 可根据需要调整工作进程数
        # executor=ProcessPoolExecutor(max_workers=8)
        set_loky_pickler('cloudpickle')
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

        def eigh(h, s):
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
