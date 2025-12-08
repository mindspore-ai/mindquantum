import time
from functools import reduce

import numpy as np
from openfermion.ops import FermionOperator
from openfermion.transforms import normal_ordered
from openfermion.utils import hermitian_conjugated
from pyscf.scf.addons import partial_cholesky_orth_
from loky import get_reusable_executor
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator
from ..vqe.vqeci import VQECI
from quri_parts.qulacs.estimator import create_qulacs_density_matrix_estimator
concurrency=1
maxworker=16

from quri_parts.circuit.noise import NoiseModel, DepolarizingNoise

import quri_parts.circuit.gate_names as gate_names
noise_model = [
        DepolarizingNoise(0.001, qubit_indices=[], target_gates=[]),
        DepolarizingNoise(0.009, qubit_indices=[], target_gates=[gate_names.CNOT, gate_names.CZ])
    ]



def is_zero_operator(op, tol=1e-6):
    """Check if a FermionOperator is effectively zero within a tolerance.

    Args:
        op (FermionOperator): The operator to check.
        tol (float): Absolute tolerance for coefficients. Default: 1e-6.

    Returns:
        bool: True if all coefficients are zero or below tolerance, False otherwise.
    """
    if not op.terms:
        return True
    max_coeff = max(abs(coeff) for coeff in op.terms.values())
    # if max_coeff >= tol:
    #     print(f"Operator not zero: max coefficient = {max_coeff:.2e}")
    return max_coeff < tol

class QSE_exact(object):
    """
    Quantum Subspace Expansion is a method for computing the eigenvalues of a Hermitian operator in the subspace spanned by the reference state and it's excited states.

    Args:
        vqeci:
            VQECI object.
    """

    def __init__(self, vqeci: VQECI):
        self.vqeci = vqeci

    def get_s2_fermi(self):
        """Construct the fermionic S^2 operator.

        Returns:
            FermionOperator: The S^2 operator for the system.
        """
        n_spatial = self.vqeci.n_qubit // 2
        S_plus = FermionOperator()
        S_minus = FermionOperator()
        S_z = FermionOperator()
        for p in range(n_spatial):
            up = 2 * p
            down = 2 * p + 1
            S_plus += FermionOperator(f"{up}^ {down}")
            S_minus += FermionOperator(f"{down}^ {up}")
            S_z += 0.5 * (FermionOperator(f"{up}^ {up}") - FermionOperator(f"{down}^ {down}"))
        S2_fermi = S_z * S_z + 0.5 * (S_plus * S_minus + S_minus * S_plus)
        return S2_fermi

    def gen_singles_doubles(self):
        singles_exc = []
        doubles_exc = []
        n_electron = self.vqeci.n_electron
        n_qubit = self.vqeci.n_qubit
        n_occ = n_electron // 2
        n_spatial = n_qubit // 2

        # Singles excitations (singlet-adapted)
        for i_spatial in range(n_occ):
            for a_spatial in range(n_occ, n_spatial):
                i_alpha, i_beta = 2 * i_spatial, 2 * i_spatial + 1
                a_alpha, a_beta = 2 * a_spatial, 2 * a_spatial + 1
                op = (1 / np.sqrt(2)) * (
                    FermionOperator(f"{a_alpha}^ {i_alpha}", 1.0) +
                    FermionOperator(f"{a_beta}^ {i_beta}", 1.0)
                )
                if op != FermionOperator():
                    singles_exc.append(op)

        # Doubles excitations (singlet-adapted, with optional symmetrization to reduce count)
        for i_spatial in range(n_occ):
            for j_spatial in range(i_spatial, n_occ):  # i <= j to symmetrize
                for a_spatial in range(n_occ, n_spatial):
                    for b_spatial in range(a_spatial, n_spatial):  # a <= b to symmetrize
                        i_alpha, i_beta = 2 * i_spatial, 2 * i_spatial + 1
                        j_alpha, j_beta = 2 * j_spatial, 2 * j_spatial + 1
                        a_alpha, a_beta = 2 * a_spatial, 2 * a_spatial + 1
                        b_alpha, b_beta = 2 * b_spatial, 2 * b_spatial + 1

                        op = (
                            FermionOperator(f"{a_alpha}^ {i_alpha} {b_alpha}^ {j_alpha}", 1.0) +  # αα
                            FermionOperator(f"{a_beta}^ {i_beta} {b_alpha}^ {j_alpha}", 1.0) +  # βα
                            FermionOperator(f"{a_alpha}^ {i_alpha} {b_beta}^ {j_beta}", 1.0) +  # αβ
                            FermionOperator(f"{a_beta}^ {i_beta} {b_beta}^ {j_beta}", 1.0)      # ββ
                        )

                        # If i < j and a < b, add symmetrized ji ba term and normalize
                        if i_spatial < j_spatial and a_spatial < b_spatial:
                            op += (
                                FermionOperator(f"{b_alpha}^ {j_alpha} {a_alpha}^ {i_alpha}", 1.0) +
                                FermionOperator(f"{b_beta}^ {j_beta} {a_alpha}^ {i_alpha}", 1.0) +
                                FermionOperator(f"{b_alpha}^ {j_alpha} {a_beta}^ {i_beta}", 1.0) +
                                FermionOperator(f"{b_beta}^ {j_beta} {a_beta}^ {i_beta}", 1.0)
                            )
                            op *= (1 / np.sqrt(2))  # Normalize for symmetrized operator

                        if op != FermionOperator():
                            doubles_exc.append(op)
        doubles_sf=[]
            # Doubles spin-flip (singlet-adapted with opposite flips)
        for i_spatial in range(n_occ):
            for j_spatial in range(i_spatial, n_occ):  # i <= j to symmetrize
                for a_spatial in range(n_occ, n_spatial):
                    for b_spatial in range(a_spatial, n_spatial):  # a <= b to symmetrize
                        i_alpha, i_beta = 2 * i_spatial, 2 * i_spatial + 1
                        j_alpha, j_beta = 2 * j_spatial, 2 * j_spatial + 1
                        a_alpha, a_beta = 2 * a_spatial, 2 * a_spatial + 1
                        b_alpha, b_beta = 2 * b_spatial, 2 * b_spatial + 1

                        op = (
                            FermionOperator(f"{a_alpha}^ {i_beta} {b_beta}^ {j_alpha}", 1.0) +
                            FermionOperator(f"{a_beta}^ {i_alpha} {b_alpha}^ {j_beta}", 1.0)
                        )

                        # If i < j and a < b, add symmetrized ji ba term and normalize
                        if i_spatial < j_spatial and a_spatial < b_spatial:
                            op += (
                                FermionOperator(f"{b_alpha}^ {j_beta} {a_beta}^ {i_alpha}", 1.0) +
                                FermionOperator(f"{b_beta}^ {j_alpha} {a_alpha}^ {i_beta}", 1.0)
                            )
                            op *= (1 / np.sqrt(2))  # Normalize for symmetrized operator

                        if op != FermionOperator():
                            doubles_sf.append(op)

        orbital_rot_ops = []
        for p_spatial in range(n_spatial):  # 遍历所有空间轨道
            for q_spatial in range(p_spatial + 1, n_spatial):  # p < q 以避免冗余
                p_alpha, p_beta = 2 * p_spatial, 2 * p_spatial + 1
                q_alpha, q_beta = 2 * q_spatial, 2 * q_spatial + 1
                op = (
                    FermionOperator(f"{p_alpha}^ {q_alpha}") - FermionOperator(f"{q_alpha}^ {p_alpha}") +  # α自    旋
                    FermionOperator(f"{p_beta}^ {q_beta}") - FermionOperator(f"{q_beta}^ {p_beta}")        # β自    旋
                )
                if op != FermionOperator():
                    orbital_rot_ops.append(op)

   
      
 
        mixed_ops = []
        for i_spatial in range(n_occ):
            for p_spatial in range(n_occ):
                for a_spatial in range(n_occ, n_spatial):
                    for q_spatial in range(n_occ, n_spatial):
                        i_alpha, i_beta = 2 * i_spatial, 2 * i_spatial + 1
                        p_alpha, p_beta = 2 * p_spatial, 2 * p_spatial + 1
                        a_alpha, a_beta = 2 * a_spatial, 2 * a_spatial + 1
                        q_alpha, q_beta = 2 * q_spatial, 2 * q_spatial + 1
                        op = (1 / np.sqrt(2)) * (
                            FermionOperator(f"{a_alpha}^ {i_alpha} {q_alpha} {p_alpha}", 1.0) +
                            FermionOperator(f"{a_beta}^ {i_beta} {q_beta} {p_beta}", 1.0)
                        )
                        if op != FermionOperator():
                            mixed_ops.append(op)
        # Introduce electron-electron interaction (correlation) operators from two-body terms
        correlation_ops = []
        for term, coeff in self.vqeci.fermionic_hamiltonian.terms.items():
            if len(term) == 4 and abs(coeff) > 1e-6:  # Select non-trivial two-body terms, ignore small coefficients
                op = normal_ordered(FermionOperator(term, 1.0))  # Normalize coefficient to 1.0 for subspace spanning
                if op != FermionOperator():
                    correlation_ops.append(op)

        de_cor = [hermitian_conjugated(op) for op in correlation_ops]

        # Generate de-excitations as Hermitian conjugates
        de_double = [hermitian_conjugated(op) for op in doubles_exc]

        de_mixed=[hermitian_conjugated(op) for op in mixed_ops]

        de_sf=[hermitian_conjugated(op) for op in doubles_sf]

        de_singles = [hermitian_conjugated(op) for op in singles_exc]

        de_orb_rot=[hermitian_conjugated(op) for op in orbital_rot_ops]

        all_ops=singles_exc+doubles_exc+correlation_ops
        print("all_ops length:",len(all_ops))
        # Strengthen singlet-adapted excitations by ensuring they commute with S^2
        S2 = self.get_s2_fermi()
        strengthened_exc = []
        for idx, op in enumerate(all_ops):
            comm = op * S2 - S2 * op
            comm = normal_ordered(comm)
            if is_zero_operator(comm, tol=0.001):
                strengthened_exc.append(op)


        print(f"Retained {len(strengthened_exc)} operators after S^2 commutator check")
        if not strengthened_exc:
            print("Warning: No operators commute with S^2. Consider increasing tolerance or checking operator construction.")
        all_ops = strengthened_exc

        return all_ops  # Return list for self.e_op

    def gen_excitation_operators(self, types="ee", n_excitations=2):
        self.e_op = []
        if types == "ee":
            if n_excitations == 2:
                self.e_op = self.gen_singles_doubles()  # Now returns exc + deexc
            elif n_excitations == 1:
                singles_exc, _ = self.gen_singles_doubles()  # Adjusted to avoid doubles computation
                self.e_op = singles_exc + [hermitian_conjugated(op) for op in singles_exc]

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
            # estimator = create_qulacs_vector_concurrent_estimator(None, concurrency=concurrency)
            estimator = create_qulacs_density_matrix_estimator(noise_model)
            return estimator([myop_i_qubit], [opt_state])[0].value

        # 创建 loky 进程池
        executor = get_reusable_executor(max_workers=maxworker)  # 可根据需要调整工作进程数

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
        executor.shutdown(wait=True)

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
        executor = get_reusable_executor(max_workers=maxworker)  # 可根据需要调整工作进程数

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
        executor.shutdown(wait=True)

        # 收集结果并填充矩阵
        for future in futures:
            idx, jdx = futures[future]
            self.S[idx, jdx] = future.result()
            if jdx != 0:  # 利用 Hermitian 对称性填充下三角
                self.S[jdx, idx] = self.S[idx, jdx].conj()

    def build_S2(self):
        noperators = len(self.e_op)
        op_mapper = self.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
            n_spin_orbitals=self.vqeci.n_qubit, n_fermions=self.vqeci.n_electron
        )
        # Construct fermionic S^2 operator
        n_spatial = self.vqeci.n_qubit // 2
        S_plus = FermionOperator()
        S_minus = FermionOperator()
        S_z = FermionOperator()
        for p in range(n_spatial):
            up = 2 * p
            down = 2 * p + 1
            S_plus += FermionOperator(f"{up}^ {down}")
            S_minus += FermionOperator(f"{down}^ {up}")
            S_z += 0.5 * (FermionOperator(f"{up}^ {up}") - FermionOperator(f"{down}^ {down}"))
        S2_fermi = S_z * S_z + 0.5 * (S_plus * S_minus + S_minus * S_plus)
        self.S2_matrix = np.zeros((noperators + 1, noperators + 1), dtype=np.complex128)
        # self.S2_matrix[0, 0] = 0.0  # Assume reference is singlet, but compute properly

        # 定义工作进程中计算 S2 矩阵元素的函数
        def compute_s2_element(idx, jdx, op_mapper, e_op, S2_fermi, opt_state):
            if jdx == 0:
                myop_i_fermi = hermitian_conjugated(e_op[idx - 1]) * S2_fermi
                myop_i_qubit = op_mapper(myop_i_fermi)
            else:
                myop_i_fermi = hermitian_conjugated(e_op[idx - 1])
                myop_ij = myop_i_fermi * S2_fermi * e_op[jdx - 1]
                myop_i_qubit = op_mapper(myop_ij)
            estimator = create_qulacs_vector_concurrent_estimator(None, concurrency=concurrency)
            return estimator([myop_i_qubit], [opt_state])[0].value

        # 创建 loky 进程池
        executor = get_reusable_executor(max_workers=maxworker)

        # 提交任务到进程池，只计算上三角部分（包括对角线）
        futures = {}
        for idx in range(1, noperators + 1):
            for jdx in range(noperators + 1):
                if jdx > idx:
                    continue
                future = executor.submit(
                    compute_s2_element,
                    idx,
                    jdx,
                    op_mapper,
                    self.e_op,
                    S2_fermi,
                    self.vqeci.opt_states[0]
                )
                futures[future] = (idx, jdx)

        # 收集结果并填充矩阵
        for future in futures:
            idx, jdx = futures[future]
            self.S2_matrix[idx, jdx] = future.result()
            if jdx != idx:
                self.S2_matrix[jdx, idx] = self.S2_matrix[idx, jdx].conj()


    def solve(self):
        self.build_H()
        self.build_S()
        self.build_S2()

        
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

        # # Spin projection: Diagonalize S2_matrix in the subspace to find singlet subspace
        # s2_evals, s2_evecs = eigh(self.S2_matrix, self.S)
        # s2_tolerance = 1e-4
        # low_spin_indices = np.where(np.abs(s2_evals) < s2_tolerance)[0]
        # print(f"Number of singlet states in subspace: {len(low_spin_indices)}")
        # if len(low_spin_indices) == 0:
        #     raise ValueError("No singlet states found in subspace")
        # C = s2_evecs[:, low_spin_indices]

        # # Project Hamiltonian onto the singlet subspace
        # H_proj = C.conj().T @ self.hamiltonian @ C

        # # Diagonalize the projected Hamiltonian (S_proj is identity)
        # self.eigenvalues, evecs_proj = np.linalg.eigh(H_proj)
        # self.eigenvectors = C @ evecs_proj


        self.eigenvalues, self.eigenvectors = eigh(self.hamiltonian, self.S)
        print('qse get'+str(len(self.eigenvalues))+' eigenvalues')
        # Compute reference weights |eigenvectors[0, idx]|^2 for each state
        num_states = len(self.eigenvalues)
        weights = np.abs(self.eigenvectors[0, :])**2

        # Identify ground-like state: max weight
        ground_idx = np.argmax(weights)
        E_ground = self.vqeci.energies[0]  # Use QSE projected ground energy
   

        # Collect all states, sort by energy, exclude the ground
        excited_indices = [i for i in range(num_states) if i != ground_idx]
        excited_energies = self.eigenvalues[excited_indices]
        excited_weights = weights[excited_indices]
        sort_order = np.argsort(excited_energies)
        excited_energies = excited_energies[sort_order]
        excited_weights = excited_weights[sort_order]

        # Compute <S^2> for ground and all excited (should be ~0 due to projection)

        s2_values = []
        for k in [ground_idx] + [excited_indices[i] for i in sort_order]:
            vec = self.eigenvectors[:, k]
            s2 = np.real(np.dot(vec.conj().T, np.dot(self.S2_matrix, vec)))
            s2_values.append(s2)

        ground_s2 = s2_values[0]
        excited_s2 = s2_values[1:]

        # Since projected, all states are valid singlets; apply minimal energy filter if needed
        min_energy_threshold = E_ground 
        valid_excited_energies = []
        valid_excited_weights = []
        valid_excited_s2 = []
        for en, wt, s2 in zip(excited_energies, excited_weights, excited_s2):
            if en >= min_energy_threshold and abs(s2) < 1e-1:
                valid_excited_energies.append(en)
                valid_excited_weights.append(wt)
                valid_excited_s2.append(s2)

        # min_energy_threshold = E_ground 
        # valid_excited_energies = []
        # valid_excited_weights = []

        # for en, wt in zip(excited_energies, excited_weights):
        #     if en >= min_energy_threshold:
        #         valid_excited_energies.append(en)
        #         valid_excited_weights.append(wt)


        # Further filtering only for Exc1
        # if valid_excited_energies:
        #     exc_energy = valid_excited_energies[0] - E_ground
        #     wt = valid_excited_weights[0]
        #     if exc_energy < 0.01 or wt >0.1:
        #         valid_excited_energies = valid_excited_energies[1:]
        #         valid_excited_weights = valid_excited_weights[1:]
        #         valid_excited_s2 = valid_excited_s2[1:]


        # Print up to 10 valid states (ground + excitations)
        print("Selected states (ground + low-lying excitations, sorted by energy):")
        print(f"Ground: {E_ground} (<S^2> {ground_s2:.9f})")
        for idx, (en, wt, s2) in enumerate(zip(valid_excited_energies[:4], valid_excited_weights[:4], valid_excited_s2[:4])):
            print(f"Exc {idx+1}: {en} (ref weight {wt:.9f}, excitation energy {en - E_ground}, <S^2> {s2:.9f})")



        # print("Selected states (ground + low-lying excitations, sorted by energy):")
        # print(f"Ground: {E_ground} ")
        # for idx, (en, wt) in enumerate(zip(valid_excited_energies[:8], valid_excited_weights[:8])):
        #     print(f"Exc {idx+1}: {en} (ref weight {wt:.9f}, excitation energy {en - E_ground})")
        self.eigenvalues[0]=E_ground
        self.eigenvalues[1]=valid_excited_energies[0]
        self.eigenvalues[2]=valid_excited_energies[1]
        # self.eigenvalues[3]=valid_excited_energies[2]
        # self.eigenvalues[4]=valid_excited_energies[3]