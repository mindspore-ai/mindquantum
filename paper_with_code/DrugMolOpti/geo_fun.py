import os
import json
import time
from typing import Dict, List, Tuple, Callable

import numpy as np
import scipy
from scipy.optimize import minimize
import warnings
import itertools

from tangelo import SecondQuantizedMolecule
from tangelo.problem_decomposition import DMETProblemDecomposition
from tangelo.problem_decomposition.dmet import Localization  # noqa: F401
from tangelo.algorithms import VQESolver, FCISolver, CCSDSolver
from tangelo.toolboxes.operators import count_qubits, FermionOperator, QubitOperator  # noqa: F401
from tangelo.toolboxes.molecular_computation.rdms import pad_rdms_with_frozen_orbitals_restricted
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping

from mindquantum.core.operators import QubitOperator as mq_QubitOperator
from mindquantum.core.gates import X
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian, TimeEvolution
from mindquantum.simulator import Simulator

from mindquantum.algorithm.nisq import Transform, get_qubit_hamiltonian  # noqa: F401
from mindquantum.algorithm.nisq import (
    uccsd_singlet_generator,
    uccsd_singlet_get_packed_amplitudes,
)

# ---------------------- Environment variables ---------------------- #
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class DMETVQEGeometryOptimizer:
    """
    A class that encapsulates the DMET + VQE + geometry optimization workflow:
      - Self-consistent solution of DMET chemical potential
      - VQE (UCCSD) on each fragment
      - L-BFGS-B optimization of geometry parameters para
      - Saving / restoring parameters to/from para_var.npy
    """

    def __init__(
        self,
        para_file: str = "./para_var.npy",
        geometry_fn: Callable[[List[float]], str] | None = None,
    ):
        # Initial geometry parameters (bond_1, bond_2)
        self.init_para: List[float] = [1.0, 1.0]
        self.n_para: int = len(self.init_para)

        # Geometry bounds
        self.special_bounds: List[Tuple[float, float]] = [
            (0.5, 2.0),  # bond_1
            (0.5, 3.0),  # bond_2
        ]

        # DMET fragment settings: 4 atoms, each in its own fragment
        self.fragment_atoms: List[int] = [1] * 4

        # Path to save parameters
        self.para_file: str = para_file

        # If user provides a custom geometry function, use it;
        # otherwise use the default H4 geometry.
        self.geometry_fn: Callable[[List[float]], str] = (
            geometry_fn if geometry_fn is not None else self.default_geometry_to_xyz
        )

        # DMET outer iteration counter, to decide if this is the first call
        self.n_iter: List[float] = []

        # For logging / printing
        self.para_list: List[List[float]] = []
        self.toggle: bool = True
        self.t_time: float = 0.0

    # ---------- default geometry (H4) ----------
    def default_geometry_to_xyz(self, para: List[float]) -> str:
        """Default H4 geometry: para = [bond_1, bond_2]."""
        theta_1 = 0.0
        theta_2 = 0.0

        bond_1, bond_2 = para

        x_H3 = bond_2 * np.cos(theta_1)
        y_H3 = bond_2 * np.sin(theta_1)
        x_H4 = x_H3 + bond_1 * np.cos(theta_2)
        y_H4 = y_H3 + bond_1 * np.sin(theta_2)

        mo_all = f"""
        H          {-1 * bond_1}   0.0    0.0
        H          0.0    0.0    0.0
        H          {x_H3}    {y_H3}    0.0 
        H          {x_H4}    {y_H4}    0.0 
        """
        return mo_all

    def build_dmet(
        self,
        para: List[float],
        chemical_potential: float,
        verbose: bool = False,
    ) -> DMETProblemDecomposition:
        """Given geometry and μ, build a DMETProblemDecomposition instance."""
        mol_mo = SecondQuantizedMolecule(
            self.geometry_fn(para), q=0, spin=0, basis="sto-3g"
        )

        options_mo_dmet = {
            "molecule": mol_mo,
            "fragment_atoms": self.fragment_atoms,
            "fragment_solvers": "vqe",
            "initial_chemical_potential": chemical_potential,
            "verbose": verbose,
        }
        dmet = DMETProblemDecomposition(options_mo_dmet)
        dmet.build()
        return dmet

    def get_current_core_energy(self, para: List[float], chemical_potential: float) -> float:
        """Given geometry & μ, compute the current core_constant_energy."""
        dmet_mo = self.build_dmet(para, chemical_potential, verbose=False)
        return dmet_mo.orbitals.core_constant_energy

    def get_fragment(
        self,
        para: List[float],
        chemical_potential: float,
    ) -> Tuple[List[VQESolver], List[SecondQuantizedMolecule], DMETProblemDecomposition]:
        """
        Given geometry & μ, call DMET.get_resources to obtain the fragment solvers and dummy molecules.
        """
        dmet_mo = self.build_dmet(para, chemical_potential, verbose=False)
        _, solver_fragment_list, dummy_mol_list = dmet_mo.get_resources(chemical_potential)
        return solver_fragment_list, dummy_mol_list, dmet_mo

    @staticmethod
    def _sort_ucc_parameters(a_list: List[float], str_list: List[str]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extracted version of original mind_get_total_para:
        Sort parameters by name (singles first, doubles later), returning the sorted array and a dict.
        """

        def custom_sort(item):
            prefix, rest = item[0].split("_")
            if prefix.startswith("s"):
                return (0, int(rest))
            elif prefix.startswith("d"):
                rest_parts = rest.split("_")
                i = int(rest_parts[0][1:]) if rest_parts[0][1:] else 0
                j = int(rest_parts[1]) if len(rest_parts) > 1 else 0
                return (1, i, j)

        sorted_result = sorted(zip(str_list, a_list), key=custom_sort)
        sorted_ansatz_name, sorted_para = zip(*sorted_result)
        sorted_para = np.array(sorted_para)
        total_para = dict(zip(str_list, a_list))
        return sorted_para, total_para

    def _split_theta_by_fragment(
        self,
        all_theta: List[float],
        total_circuit_list: Dict[int, Circuit],
    ) -> Dict[int, np.ndarray]:
        """Split the concatenated theta vector back into p0_list[i] for each fragment."""
        p0_list: Dict[int, np.ndarray] = {}
        idx = 0
        for i in range(len(total_circuit_list)):
            n_s = len(total_circuit_list[i].params_name)
            p0_list[i] = np.array(all_theta[idx: idx + n_s])
            idx += n_s
        return p0_list

    # ==========================================================
    #              Build UCCSD ansatz
    # ==========================================================
    def build_fragment_ansatz_from_ccsd(
        self,
        solver_fragment: VQESolver,
        dummy_mol: SecondQuantizedMolecule,
    ) -> Tuple[List[str], List[float], Circuit]:
        """
        Build HF + UCCSD ansatz on a single fragment, and initialize UCC parameters.
        Here we use a small constant initial amplitude for simplicity.
        """
        n_qubits = solver_fragment.get_resources()["circuit_width"]
        n_electrons = dummy_mol.n_active_electrons

        hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(n_electrons)])

        ucc_fermion_ops = uccsd_singlet_generator(
            n_qubits, n_electrons, anti_hermitian=True
        )
        ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()
        ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag, 1.0).circuit
        ansatz_parameter_names = ansatz_circuit.params_name

        total_circuit = hartreefock_wfn_circuit + ansatz_circuit

        # Simple small initial amplitudes
        init_amplitudes = [0.01] * len(ansatz_parameter_names)

        return ansatz_parameter_names, init_amplitudes, total_circuit

    def build_total_ansatz(
        self,
        para: List[float],
        chemical_potential: float,
    ) -> Tuple[
        List[VQESolver],
        Dict[int, List[str]],
        Dict[int, List[float]],
        Dict[int, Circuit],
    ]:
        """
        Given geometry para, build UCCSD ansatz initial parameters for all fragments.
        """
        solver_fragment_list, dummy_mol_list, dmet_mo = self.get_fragment(para, chemical_potential)

        ansatz_parameter_names_list: Dict[int, List[str]] = {}
        init_amplitudes_ccsd_list: Dict[int, List[float]] = {}
        total_circuit_list: Dict[int, Circuit] = {}

        for i in range(len(solver_fragment_list)):
            (
                ansatz_parameter_names,
                init_amplitudes_ccsd,
                total_circuit,
            ) = self.build_fragment_ansatz_from_ccsd(solver_fragment_list[i], dummy_mol_list[i])

            ansatz_parameter_names_list[i] = ansatz_parameter_names
            init_amplitudes_ccsd_list[i] = init_amplitudes_ccsd
            total_circuit_list[i] = total_circuit

        return solver_fragment_list, ansatz_parameter_names_list, init_amplitudes_ccsd_list, total_circuit_list

    # ==========================================================
    #    Geometry RDM computation for fixed θ
    # ==========================================================
    def _get_rdm_list_for_geometry(
        self,
        para: List[float],
        p0_list: Dict[int, np.ndarray],
        total_circuit_list: Dict[int, Circuit],
        chemical_potential: float,
    ):
        """
        For current geometry para and given p0_list, compute 1-RDM and 2-RDM
        for each fragment. This is expensive and should be called only when θ changes.
        """
        solver_fragment_list, dummy_mol_list, dmet_mo = self.get_fragment(para, chemical_potential)

        onerdm_temp_list: Dict[int, np.ndarray] = {}
        twordm_temp_list: Dict[int, np.ndarray] = {}

        for i in range(len(total_circuit_list)):
            p0 = p0_list[i]
            total_circuit = total_circuit_list[i]
            ansatz_parameter_names = total_circuit.params_name
            solver_fragment = solver_fragment_list[i]

            sorted_para_temp, total_para_temp = self._sort_ucc_parameters(
                p0, ansatz_parameter_names
            )
            onerdm_temp, twordm_temp = self.get_rdm(
                solver_fragment, total_para_temp, total_circuit, sorted_para_temp
            )
            onerdm_temp_list[i] = onerdm_temp
            twordm_temp_list[i] = twordm_temp

        return onerdm_temp_list, twordm_temp_list

    def get_rdm(self, vqe_solver, total_para, total_circuit, var_params, sum_spin=True):
        """
        Compute the 1- and 2- RDM matrices using the VQE wavefunction.
        This uses an explicit measurement-style evaluation via fermion_to_qubit_mapping.
        """
        # print("get_rdm called")

        sim = Simulator('mqvector', total_circuit.n_qubits)
        sim.apply_circuit(total_circuit, pr=total_para)

        vqe_solver.ansatz.update_var_params(var_params)

        n_mol_orbitals = vqe_solver.molecule.n_active_mos
        n_spin_orbitals = vqe_solver.molecule.n_active_sos
        rdm1_spin = np.zeros((n_spin_orbitals,) * 2, dtype=complex)
        rdm2_spin = np.zeros((n_spin_orbitals,) * 4, dtype=complex)

        qb_freq_dict = dict()
        qb_expect_dict = dict()

        for key in vqe_solver.molecule.fermionic_hamiltonian.terms:

            if not key:
                continue

            length = len(key)
            if length == 2:
                iele, jele = (int(ele[0]) for ele in tuple(key[0:2]))
            elif length == 4:
                iele, jele, kele, lele = (int(ele[0]) for ele in tuple(key[0:4]))
            else:
                continue

            hamiltonian_temp = FermionOperator(key)

            # Obtain qubit Hamiltonian for this operator
            qubit_hamiltonian2 = fermion_to_qubit_mapping(
                fermion_operator=hamiltonian_temp,
                mapping=vqe_solver.qubit_mapping,
                n_spinorbitals=vqe_solver.molecule.n_active_sos,
                n_electrons=vqe_solver.molecule.n_active_electrons,
                up_then_down=vqe_solver.up_then_down,
                spin=vqe_solver.molecule.spin,
            )
            qubit_hamiltonian2.compress()

            opt_energy2 = 0.0

            for qb_term, qb_coef in qubit_hamiltonian2.terms.items():
                if qb_term:
                    if qb_term not in qb_freq_dict:
                        result_str = ' '.join(
                            '{}{}'.format(term[1], term[0]) for term in qb_term
                        )
                        ham_qb_term = mq_QubitOperator(f'{result_str}')

                    if qb_term not in qb_expect_dict:
                        ham = Hamiltonian(ham_qb_term)
                        expect = sim.get_expectation(ham).real
                        qb_expect_dict[qb_term] = expect
                    expectation = qb_expect_dict[qb_term]
                    opt_energy2 += qb_coef * expectation
                else:
                    opt_energy2 += qb_coef

            if length == 2:
                rdm1_spin[iele, jele] += opt_energy2
            elif length == 4:
                rdm2_spin[iele, lele, jele, kele] += opt_energy2

        vqe_solver.rdm_freq_dict = qb_freq_dict

        if sum_spin:
            rdm1_np = np.zeros((n_mol_orbitals,) * 2, dtype=np.complex128)
            rdm2_np = np.zeros((n_mol_orbitals,) * 4, dtype=np.complex128)

            # Construct spin-summed 1-RDM
            for i, j in itertools.product(range(n_spin_orbitals), repeat=2):
                rdm1_np[i // 2, j // 2] += rdm1_spin[i, j]

            # Construct spin-summed 2-RDM
            for i, j, k, l in itertools.product(range(n_spin_orbitals), repeat=4):
                rdm2_np[i // 2, j // 2, k // 2, l // 2] += rdm2_spin[i, j, k, l]

            return rdm1_np, rdm2_np

        return rdm1_spin, rdm2_spin

    # ==========================================================
    #    Embedded energy given fixed RDMs
    # ==========================================================
    def _get_rdm_x_emb_energy(
        self,
        para: List[float],
        onerdm_temp_list: Dict[int, np.ndarray],
        twordm_temp_list: Dict[int, np.ndarray],
        chemical_potential: float,
    ) -> float:
        """
        Given fixed RDMs, compute the embedded energy for the current geometry para.
        """
        fragment_energy_x = 0.0
        solver_fragment_list, dummy_mol_list, dmet_mo = self.get_fragment(para, chemical_potential)
        core_energy = np.real(self.get_current_core_energy(para, chemical_potential))

        for i in range(len(dummy_mol_list)):
            dummy_mol_temp = dummy_mol_list[i]
            onerdm_temp = onerdm_temp_list[i].copy()
            twordm_temp = twordm_temp_list[i].copy()

            (
                onerdm_padded_temp,
                twordm_padded_temp,
            ) = pad_rdms_with_frozen_orbitals_restricted(
                dummy_mol_temp, onerdm_temp, twordm_temp
            )

            fragment_energy_temp, temp_onerdm = dmet_mo._compute_energy_restricted(
                dummy_mol_temp,
                onerdm_padded_temp,
                twordm_padded_temp,
            )
            fragment_energy_x += fragment_energy_temp

        fragment_energy_x += core_energy
        return float(np.real(fragment_energy_x))

    def get_frag_x_gradient(
        self,
        para: List[float],
        onerdm_temp_list: Dict[int, np.ndarray],
        twordm_temp_list: Dict[int, np.ndarray],
        chemical_potential: float,
        delta: float = 1e-7,
    ) -> Tuple[float, List[float], float]:
        """
        Compute the gradient of embedded energy with respect to geometry parameters
        via finite difference, using fixed RDMs from the initial geometry.
        Returns: (rdm_energy, grad_para_list, core_energy)
        """
        n = len(para)
        x_gradient_list: List[float] = []

        for i in range(n):
            x_plus = para.copy()
            x_minus = para.copy()
            x_plus[i] += delta
            x_minus[i] -= delta

            f_i_plus = self._get_rdm_x_emb_energy(
                x_plus, onerdm_temp_list, twordm_temp_list, chemical_potential
            )
            f_i_minus = self._get_rdm_x_emb_energy(
                x_minus, onerdm_temp_list, twordm_temp_list, chemical_potential
            )

            temp = (f_i_plus - f_i_minus) / (2.0 * delta)
            x_gradient_list.append(float(np.real(temp)))

        rdm_energy = self._get_rdm_x_emb_energy(
            para, onerdm_temp_list, twordm_temp_list, chemical_potential
        )
        core_energy = self.get_current_core_energy(para, chemical_potential)
        return float(np.real(rdm_energy)), x_gradient_list, float(np.real(core_energy))

    def get_x_energy_grad(
        self,
        para: List[float],
        onerdm_temp_list: Dict[int, np.ndarray],
        twordm_temp_list: Dict[int, np.ndarray],
        chemical_potential: float,
    ) -> Tuple[float, float, List[float]]:
        """
        Wrapper that returns (rdm_energy, core_energy, gradient with respect to geometry para),
        using fixed RDMs.
        """
        rdm_energy, x_gradient_list, core_energy = self.get_frag_x_gradient(
            para, onerdm_temp_list, twordm_temp_list, chemical_potential
        )
        return rdm_energy, core_energy, x_gradient_list

    # ==========================================================
    #   Compute electron number & embedded energy (shared by θ/para)
    # ==========================================================
    def get_rdm_and_energy(
        self,
        para: List[float],
        all_x: List[float],
        total_circuit_list: Dict[int, Circuit],
        chemical_potential: float,
    ) -> Tuple[float, float, float]:
        """
        Given geometry para and all optimizer parameters all_x (theta + para),
        compute embedded energy, total electron number, and reference active electron count.
        """
        # Extract all fragment UCC parameters
        theta_all = all_x[:-self.n_para]
        p0_list = self._split_theta_by_fragment(theta_all, total_circuit_list)

        solver_fragment_list, dummy_mol_list, dmet_mo = self.get_fragment(para, chemical_potential)

        onerdm_temp_list: Dict[int, np.ndarray] = {}
        twordm_temp_list: Dict[int, np.ndarray] = {}

        number_of_electron = 0.0
        fragment_energy_e_x = 0.0

        for i in range(len(total_circuit_list)):
            p0 = p0_list[i]
            total_circuit = total_circuit_list[i]
            ansatz_parameter_names = total_circuit.params_name
            solver_fragment = solver_fragment_list[i]

            sorted_para_temp, total_para_temp = self._sort_ucc_parameters(
                p0, ansatz_parameter_names
            )
            # For θ-optimization we can still use solver_fragment.get_rdm
            onerdm_temp, twordm_temp = self.get_rdm(
                solver_fragment, total_para_temp, total_circuit, sorted_para_temp
            )
            onerdm_temp_list[i] = onerdm_temp
            twordm_temp_list[i] = twordm_temp

        for i in range(len(dummy_mol_list)):
            dummy_mol_temp = dummy_mol_list[i]
            onerdm_temp = onerdm_temp_list[i]
            twordm_temp = twordm_temp_list[i]

            (
                onerdm_padded_temp,
                twordm_padded_temp,
            ) = pad_rdms_with_frozen_orbitals_restricted(
                dummy_mol_temp, onerdm_temp, twordm_temp
            )

            fragment_energy_temp, temp_onerdm = dmet_mo._compute_energy_restricted(
                dummy_mol_temp,
                onerdm_padded_temp,
                twordm_padded_temp,
            )

            n_electron_frag = np.trace(
                temp_onerdm[
                    : dummy_mol_list[i].t_list[0],
                    : dummy_mol_list[i].t_list[0],
                ]
            )

            fragment_energy_e_x += np.real(fragment_energy_temp)
            number_of_electron += np.real(n_electron_frag)
            print("number_of_electron ", number_of_electron)

        core_self_energy = self.get_current_core_energy(para, chemical_potential)
        fragment_energy_e_x += np.real(core_self_energy)
        ref_number_active_electrons = dmet_mo.orbitals.number_active_electrons

        return (
            float(np.real(fragment_energy_e_x)),
            float(np.real(number_of_electron)),
            float(np.real(ref_number_active_electrons)),
        )

    # ==========================================================
    #         Optimize θ (VQE on each fragment)
    # ==========================================================
    def optimize_theta_for_all_fragments(
        self,
        para: List[float],
        all_x: List[float],
        maxiter_theta: int,
        chemical_potential: float,
        total_circuit_list: Dict[int, Circuit],
    ) -> List[float]:
        """
        Fix the geometry para, perform VQE + BFGS optimization of all fragment UCC parameters,
        and return updated all_x (theta_all_fragments + para).
        """
        solver_fragment_list, dummy_mol_list, dmet_mo = self.get_fragment(para, chemical_potential)

        # Split the leading part of all_x (theta) into fragment-wise p0_list
        theta_all = all_x[:-self.n_para]
        p0_list = self._split_theta_by_fragment(theta_all, total_circuit_list)

        new_p0: List[float] = []

        for f in range(len(solver_fragment_list)):
            print(f"Optimize fragment-{f} energy")

            hamiltonian_QubitOp = mq_QubitOperator.from_openfermion(
                QubitOperator.to_openfermion(solver_fragment_list[f].qubit_hamiltonian)
            )

            n_qubits = solver_fragment_list[f].get_resources()["circuit_width"]
            n_electrons = dummy_mol_list[f].n_active_electrons

            hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(n_electrons)])

            ucc_fermion_ops = uccsd_singlet_generator(
                n_qubits, n_electrons, anti_hermitian=True
            )
            ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()
            ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag, 1.0).circuit
            total_circuit = hartreefock_wfn_circuit + ansatz_circuit

            print(total_circuit.summary())

            solver_fragment_ccsd = CCSDSolver(dummy_mol_list[f])
            total_energy, _, _ = solver_fragment_ccsd.simulate()
            print("The CCSD energy is:", total_energy)
            solver_fragment_fci = FCISolver(dummy_mol_list[f])
            print("The FCI energy is:", solver_fragment_fci.simulate())

            # Use the old theta of this fragment as initial guess
            init_para = p0_list[f].copy()

            grad_ops = Simulator("mqvector", total_circuit.n_qubits).get_expectation_with_grad(
                Hamiltonian(hamiltonian_QubitOp.real),
                total_circuit,
            )

            t0 = time.time()

            def fun(p0, molecule_pqc, energy_list=None):
                f_val, g_val = molecule_pqc(p0)
                f_val = np.real(f_val)[0, 0]
                g_val = np.real(g_val)[0, 0]
                if energy_list is not None:
                    energy_list.append(f_val)
                    if len(energy_list) % 5 == 0:
                        print(
                            f"Step: {len(energy_list)},\tenergy: {f_val},\titerative time: {time.time() - t0}"
                        )
                return f_val, g_val

            energy_list_theta: List[float] = []
            res = minimize(
                fun,
                init_para,
                args=(grad_ops, energy_list_theta),
                method="bfgs",
                jac=True,
                options={"maxiter": maxiter_theta},
            )

            optimal_para = res.x
            new_p0 += np.array(optimal_para).tolist()

        new_all_x = new_p0 + para
        return new_all_x

    # ==========================================================
    #            Geometry objective for L-BFGS-B
    # ==========================================================
    def geometry_objective(
        self,
        para: np.ndarray,
        all_x_no_para: List[float],
        total_circuit_list: Dict[int, Circuit],
        energy_list: List[float],
        chemical_potential: float,
        onerdm_temp_list: Dict[int, np.ndarray],
        twordm_temp_list: Dict[int, np.ndarray],
    ) -> Tuple[float, np.ndarray]:
        """
        Objective function for L-BFGS-B optimization of geometry para:
          - Input current para and fixed theta (all_x_no_para),
          - Use precomputed RDMs (onerdm_temp_list, twordm_temp_list),
          - Call get_x_energy_grad to compute finite-difference geometry gradients,
          - Return (energy, gradient w.r.t. para).
        """
        para = np.array(para, dtype=float)
        para_list = para.tolist()
        all_x = list(all_x_no_para) + para_list
        print("The para is (all lost grad)", para_list)

        rdm_energy, core_energy, grad_para = self.get_x_energy_grad(
            para_list, onerdm_temp_list, twordm_temp_list, chemical_potential
        )

        self.para_list.append(para_list.copy())

        para_grad_norm = np.linalg.norm(grad_para)
        penalty_energy = rdm_energy
        theta_grad_norm = 0.0

        if energy_list is not None:
            energy_list.append(rdm_energy)
            print(
                f"Step: {len(energy_list)},\tenergy: {penalty_energy},\tcore_energy: {core_energy},"
                f"\ttheta_grad_norm: {theta_grad_norm},\ttoggle: {self.toggle}"
            )
            print(
                f"Step: {len(energy_list)},\trdm_energy: {rdm_energy},\tpara_grad_norm: {para_grad_norm}"
            )
            print("iterative time:", time.time() - self.t_time)

        new_grad_list = np.array(grad_para, dtype=float)
        return penalty_energy, new_grad_list

    # ==========================================================
    #         DMET self-consistency residual
    # ==========================================================
    def dmet_residual(self, chemical_potential: float) -> float:
        """
        Given chemical_potential μ, run one nested optimization of (theta, para)
        and return N(μ) - N_ref for use in scipy.optimize.newton.
        """
        print(f"The DMET loop is {len(self.n_iter)}")
        print("-------------------------")

        self.t_time = time.time()
        self.para_list = []
        self.toggle = True

        # ===== Initialize all_x (theta + para) =====
        if len(self.n_iter) == 0 or (not os.path.exists(self.para_file)):
            para = self.init_para.copy()
            all_x: List[float] = []
            (
                solver_fragment_list,
                ansatz_parameter_names_list,
                init_amplitudes_ccsd_list,
                total_circuit_list,
            ) = self.build_total_ansatz(para, chemical_potential)

            for i in range(len(solver_fragment_list)):
                all_x += init_amplitudes_ccsd_list[i]
            all_x += para
        else:
            all_x = np.load(self.para_file).tolist()
            para = all_x[-self.n_para :].copy()
            (
                solver_fragment_list,
                ansatz_parameter_names_list,
                init_amplitudes_ccsd_list,
                total_circuit_list,
            ) = self.build_total_ansatz(para, chemical_potential)

        print("The all_x is:", all_x)

        energy_list: List[float] = []

        temp_theta = 2
        temp_para = 2
        all_iter = 5
        bounds = self.special_bounds

        number_of_electron = 0.0
        ref_number_active_electrons = 0.0

        for i in range(all_iter):
            iter_id = i + 1

            if iter_id % 2 == 0:
                # ---- Optimize geometry para (with fixed RDMs) ---- #
                if iter_id == all_iter - 1:
                    temp_para = 500
                self.toggle = False
                print("The iter is ", iter_id, "toggle :", self.toggle)
                print("optimize para", "maxiter :", temp_para)

                all_x_no_para = all_x[:-self.n_para]

                # Compute RDMs ONCE for current θ and current geometry
                theta_all = all_x_no_para
                p0_list = self._split_theta_by_fragment(theta_all, total_circuit_list)
                onerdm_temp_list, twordm_temp_list = self._get_rdm_list_for_geometry(
                    para, p0_list, total_circuit_list, chemical_potential
                )

                def obj(para_inner):
                    return self.geometry_objective(
                        para_inner,
                        all_x_no_para,
                        total_circuit_list,
                        energy_list,
                        chemical_potential,
                        onerdm_temp_list,
                        twordm_temp_list,
                    )

                res = minimize(
                    obj,
                    para,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    options={"maxiter": temp_para},
                )

                para = res.x.tolist()
                all_x = all_x_no_para + para
            else:
                # ---- Optimize theta ---- #
                self.toggle = True
                if iter_id == all_iter or iter_id == all_iter - 2:
                    temp_theta = 500
                print("The iter is ", iter_id, "toggle :", self.toggle)
                print("optimize theta", "maxiter :", temp_theta)

                all_x = self.optimize_theta_for_all_fragments(
                    para,
                    all_x,
                    temp_theta,
                    chemical_potential,
                    total_circuit_list,
                )
                rdm_energy, number_of_electron, ref_number_active_electrons = self.get_rdm_and_energy(
                    para, all_x, total_circuit_list, chemical_potential
                )
                para = all_x[-self.n_para :].copy()

                print("new_all_x is", all_x)
                print("rdm_energy is", rdm_energy)

            optimal_all_x = all_x
            np.save(self.para_file, np.array(optimal_all_x))
            print("The para_list is", self.para_list)

        # Record number of outer DMET iterations
        self.n_iter.append(1.0)
        optimal_all_x = all_x
        np.save(self.para_file, np.array(optimal_all_x))

        residual = float(np.real(number_of_electron - ref_number_active_electrons))
        print("DMET residual N(μ) - N_ref =", residual)
        return residual

    # ==========================================================
    #                 Public interface: solve μ
    # ==========================================================
    def solve_chemical_potential(
        self,
        mu0: float = 0.0,
        tol: float = 1e-5,
    ) -> float:
        """
        Use SciPy's newton method on dmet_residual(μ) to perform 1D root-finding,
        and return the converged chemical potential.
        """
        result = scipy.optimize.newton(
            self.dmet_residual,
            mu0,
            tol=tol,
        )
        print("The optimized chemical_potential is", result.real)
        return float(np.real(result))


# ==========================================================
#                         Main entry
# ==========================================================
if __name__ == "__main__":
    def geometry_to_xyz(para: List[float]) -> str:
        """User-defined H4 geometry: para = [bond_1, bond_2]."""
        theta_1 = 0.0
        theta_2 = 0.0

        bond_1, bond_2 = para

        x_H3 = bond_2 * np.cos(theta_1)
        y_H3 = bond_2 * np.sin(theta_1)
        x_H4 = x_H3 + bond_1 * np.cos(theta_2)
        y_H4 = y_H3 + bond_1 * np.sin(theta_2)

        mo_all = f"""
        H          {-1 * bond_1}   0.0    0.0
        H          0.0    0.0    0.0
        H          {x_H3}    {y_H3}    0.0 
        H          {x_H4}    {y_H4}    0.0 
        """
        return mo_all

    optimizer = DMETVQEGeometryOptimizer(
        para_file="./para_var.npy",
        geometry_fn=geometry_to_xyz,
    )
    optimizer.init_para = [1.0, 1.0]
    optimizer.special_bounds = [
        (0.5, 2.0),
        (0.5, 3.0),
    ]
    optimizer.fragment_atoms = [1] * 4

    optimizer.solve_chemical_potential(mu0=0.0, tol=1e-5)
