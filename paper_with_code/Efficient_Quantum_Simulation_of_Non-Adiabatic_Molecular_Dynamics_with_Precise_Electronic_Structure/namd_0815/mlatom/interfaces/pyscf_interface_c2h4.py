#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! Interface_PySCF: interface to the PySCF program                           ! 
  ! Implementations by: Yuxinxin Chen                                         !
  !---------------------------------------------------------------------------! 
'''

# TODO: check convergence criterion for gradients and energy

import os
import numpy as np
from multiprocessing import cpu_count
import sys
from quri_parts.circuit.noise import NoiseModel, DepolarizingNoise
from quri_parts.qulacs.estimator import create_qulacs_density_matrix_estimator
from pyscf.scf.addons import partial_cholesky_orth_
from functools import reduce
from openfermion.utils import hermitian_conjugated
from loky import get_reusable_executor
from mlatom import constants, stopper, models
from ..decorators import doc_inherit

import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion import get_fermion_operator, get_sparse_operator
from quri_parts.core.operator.sparse import get_sparse_matrix
from quri_parts.qulacs.simulator import evaluate_state_to_vector
from chemqulacs_namd.qse import force_utility
import time
from chemqulacs_namd.vqe.vqemcscf_noisy import VQECASCI as VQECASCI_noisy
from chemqulacs_namd.vqe.vqeci_noisy import Ansatz as Ansatz_noisy
from chemqulacs_namd.qse.qse_singlet import QSE
from chemqulacs_namd.qse.qse_singlet_commute_electric_no_proj import QSE_exact
from pyscf.fci import addons
from pyscf import scf
nstates=2
max_worker_number=32
excitation_numb=1
def generate_excited_state(qse, k: int) -> np.ndarray:
    if k == 0:
        return evaluate_state_to_vector(qse.vqeci.opt_states[0]).vector
    else:
        ref_state = qse.vqeci.opt_states[0]
        n_qubits = qse.vqeci.n_qubit
        ref_vector = evaluate_state_to_vector(ref_state).vector
        eigenvector = qse.eigenvectors[:, k]
        e_ops = qse.e_op
        excited_vector = eigenvector[0] * ref_vector
        op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
            n_spin_orbitals=qse.vqeci.n_qubit, n_fermions=qse.vqeci.n_electron
        )
        for i, e_op in enumerate(e_ops, start=1):
            qubit_op = op_mapper(e_op)
            sparse_matrix = get_sparse_matrix(qubit_op, n_qubits)
            excited_part = sparse_matrix.dot(ref_vector)
            excited_vector += eigenvector[i] * excited_part
        norm = np.linalg.norm(excited_vector)
        if norm < 1e-10:
            raise ValueError(f"激发态 {k} 的向量范数过小，无法归一化")
        excited_vector /= norm
        return excited_vector

def create_noise_model(noise_prob,b) -> NoiseModel:
    from quri_parts.circuit.noise import NoiseModel, BitFlipNoise,BitPhaseFlipNoise
    noises = []
    import quri_parts.circuit.gate_names as gate_names
    noises.append(DepolarizingNoise(0.001,qubit_indices=[],target_gates=[]))
    noises.append(DepolarizingNoise(0.009,qubit_indices=[],target_gates=[gate_names.CNOT,gate_names.CZ]))
    # noises.append(BitFlipNoise(0.02,qubit_indices=[],target_gates=[]))
    noise_model=NoiseModel(noises)
    return noise_model

def compute_matrix_element(idx, jdx, fermionic_hamiltonian, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths, is_ham=True, force_ferm_op=None):
    noise_model = create_noise_model(noise_types, noise_strengths)
    estimator = create_qulacs_density_matrix_estimator(noise_model)
    if is_ham:
        if idx == 0:
            myop_fermi = fermionic_hamiltonian * e_ops[jdx - 1]
        elif jdx == 0:
            myop_fermi = hermitian_conjugated(e_ops[idx - 1]) * fermionic_hamiltonian
        else:
            myop_fermi = hermitian_conjugated(e_ops[idx - 1]) * fermionic_hamiltonian * e_ops[jdx - 1]
    else:
        if idx == 0:
            myop_fermi = force_ferm_op * e_ops[jdx - 1]
        elif jdx == 0:
            myop_fermi = hermitian_conjugated(e_ops[idx - 1]) * force_ferm_op
        else:
            myop_fermi = hermitian_conjugated(e_ops[idx - 1]) * force_ferm_op * e_ops[jdx - 1]
    myop = op_mapper(myop_fermi)
    val = estimator(myop, bound_circuit).value
    return idx, jdx, val

def compute_s_matrix_element(idx, jdx, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths):
    noise_model = create_noise_model(noise_types, noise_strengths)
    estimator = create_qulacs_density_matrix_estimator(noise_model)
    if idx == 0:
        myop_fermi = e_ops[jdx - 1]
    elif jdx == 0:
        myop_fermi = hermitian_conjugated(e_ops[idx - 1])
    else:
        myop_fermi = hermitian_conjugated(e_ops[idx - 1]) * e_ops[jdx - 1]
    myop = op_mapper(myop_fermi)
    val = estimator(myop, bound_circuit).value
    return idx, jdx, val
def compute_noisy_energies(mol,noise_types,noise_strengths):
    mf = scf.ROHF(mol).run()
    start_time = time.time()
    noise_kwargs = {
        "noise_type": noise_types,
        "noise_prob": [noise_strengths['depol_1q'], noise_strengths['depol_2q']]
    }
    vqe_casci = VQECASCI_noisy(mf, 2, 2, singlet_excitation=True, ansatz=Ansatz_noisy.UCCSD, use_singles=False,  noise_kwargs=noise_kwargs, excitation_number=excitation_numb)
    vqe_casci.kernel()
    qse = QSE(vqe_casci.fcisolver)
    qse.gen_excitation_operators("ee", 2)

    cas_state = vqe_casci.fcisolver.opt_states[0]
    cas_hamiltonian = vqe_casci.fcisolver.qubit_hamiltonian
    bound_circuit = cas_state

    noperators = len(qse.e_op)
    e_ops = qse.e_op
    fermionic_hamiltonian = qse.vqeci.fermionic_hamiltonian
    n = noperators + 1
    executor = get_reusable_executor()
    print('executor worker number:'+str(executor._max_workers))
    hamiltonian = np.zeros((n, n), dtype=np.complex128)
    hamiltonian[0, 0] = vqe_casci.fcisolver.energies[0]
    tasks = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
    op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
        n_spin_orbitals=qse.vqeci.n_qubit, n_fermions=qse.vqeci.n_electron
    )
    futures = [executor.submit(compute_matrix_element, idx, jdx, fermionic_hamiltonian, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths, is_ham=True) for idx, jdx in tasks]
    for future in futures:
        idx, jdx, val = future.result()
        hamiltonian[idx, jdx] = val
        hamiltonian[jdx, idx] = np.conj(val)
    S = np.zeros((n, n), dtype=np.complex128)
    S[0, 0] = 1.0
    tasks_s = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
    futures_s = [executor.submit(compute_s_matrix_element, idx, jdx, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths) for idx, jdx in tasks_s]
    for future in futures_s:
        idx, jdx, val = future.result()
        S[idx, jdx] = val
        S[jdx, idx] = np.conj(val)
    threshold = 1.0e-7
    cholesky_threshold = 1.0e-9
    x = partial_cholesky_orth_(S, canthr=threshold, cholthr=cholesky_threshold)
    xhx = reduce(np.dot, (x.T.conj(), hamiltonian, x))
    e_tot_noisy, eigenvectors_noisy = np.linalg.eigh(xhx)
    e_tot_noisy[0] = vqe_casci.fcisolver.energies[0]
    eigenvectors_noisy = np.dot(x, eigenvectors_noisy)
    return e_tot_noisy[:nstates], eigenvectors_noisy

class OMP_pyscf(models.model):
    def set_num_threads(self, nthreads=0):
        super().set_num_threads(nthreads)
        if self.nthreads:
            import pyscf
            #这里改过！！！！2025 0913
            # os.environ["OMP_NUM_THREADS"] = str(self.nthreads)
            pyscf.lib.misc.num_threads(n=self.nthreads)


class pyscf_methods(OMP_pyscf, metaclass=models.meta_method, available_methods=["HF", 'MP2', "FCI", "CISD" "CCSD", "CCSD(T)","NOISY","FDM","CASCI","CASCI-VQE","CASCI-VQE-EX"]):
    '''
    PySCF interface

    Arguments:
        method (str): Method to use
        nthreads (int): Set the number of OMP threads

    .. note::

        Methods supported:

        Energy: HF, MP2, DFT, CISD, FCI, CCSD/CCSD(T), TD-DFT/TD-HF

        Gradients: HF, MP2, DFT, CISD, CCSD, RCCSD(T), TD-DFT/TD-HF

        Hessian: HF, DFT
        
    '''

    def __init__(self, method='B3LYP/6-31g', nthreads=None, **kwargs):
        
        self.method = method.split('/')[0]
        if not 'DM21' in self.method.upper():
            if 'PYSCF_PATH' in os.environ:
                sys.path.insert(0,os.environ['PYSCF_PATH'])
       
        self.basis = method.split('/')[1]
        if nthreads is None:
            self.nthreads = cpu_count()
        else:
            self.nthreads = nthreads
        if 'infrared' in kwargs:
            self.infrared = kwargs['infrared']
        else:
            self.infrared = False
        if 'density_fitting' in kwargs:
            self.density_fitting = kwargs['density_fitting']
        else:
            self.density_fitting = False
            
    @classmethod
    def is_available_method(cls, method):
        # methods can be used without `method` keywords:
        # DFT, HF, MP2, FCI, CISD, CCSD, CCSD(T) / [basis set]
        if not 'DM21' in method.upper():
            if 'PYSCF_PATH' in os.environ:
                sys.path.insert(0,os.environ['PYSCF_PATH'])
        from pyscf.dft.libxc import parse_xc
        method = method.split('/')[0]
        if method.casefold() in [m.casefold() for m in cls.available_methods]:
            return True
        try:
            parse_xc(method)
            return True
        except:
            return False


    def predict_for_molecule(self, molecule = None, calculate_energy=True,calculate_energy_gradients=False, calculate_hessian=False,calculate_dipole_derivatives=False, nstates=nstates, **kwargs):
            from pyscf import gto, scf, fci,mcscf
            from pyscf.tdscf import TDHF
            pyscf_mol = gto.Mole()
            pyscf_mol.atom = [
                [a.element_symbol, tuple(a.xyz_coordinates)]
                for a in molecule.atoms]
            pyscf_mol.basis = self.basis

            pyscf_mol.charge = molecule.charge

            pyscf_mol.spin= molecule.multiplicity - 1

            pyscf_mol.verbose = 0
            pyscf_mol.unit= 'Angstrom'
            pyscf_mol.symmetry=False
            pyscf_mol.build()


            if kwargs.get('current_state', None) is None:
                current_state=0
            else:
                current_state=kwargs.get('current_state')


            for atom in molecule.atoms:
                if not atom.atomic_number in [1, 6, 7, 8]:
                    errmsg = ' * Warning * Molecule contains elements other than CHNO, no calculations performed'
                    # print(errmsg)
                    raise ValueError(errmsg)

            if nstates > 1:
                mol_copy = molecule.copy()
                mol_copy.electronic_states = []
                for _ in range(nstates - len(molecule.electronic_states)):
                    molecule.electronic_states.append(mol_copy.copy())
            # 这里append是整个对象

            # for molecule in molecules:
            if nstates > 1 and isinstance(calculate_energy_gradients, list):
                if any(calculate_energy_gradients):
                    calculate_energy_gradients = [True] * nstates

            calculate_energy_gradients = bool(np.array(calculate_energy_gradients).any())

            calculate_energy_gradients = [True] * nstates



            if calculate_energy:


                if 'HF' == self.method.upper():
                    pyscf_method = scf.RHF(pyscf_mol).run()
                    if pyscf_method.converged:
                        pyscf_method.kernel()
                        molecule.energy = pyscf_method.e_tot
                        print('E HF = ' + str(molecule.energy))

                if 'TD-HF' == self.method.upper():
                    pyscf_method = scf.RHF(pyscf_mol).run()
                    pyscf_method = TDHF(pyscf_method).run(nrooots=nstates)
                    # if pyscf_method.converged.any():
                    pyscf_method.kernel()
                    state_idx = 0
                    # molecule.energy = pyscf_method.e[0]
                    for mol_el_st in molecule.electronic_states:
                        mol_el_st.energy = pyscf_method.e[state_idx]
                        print('electronic_state' + str(state_idx) + ' energy is' + str(mol_el_st.energy))
                        state_idx += 1

                if 'CASCI' == self.method.upper():

                    mf = scf.RHF(pyscf_mol).run()
                    mc = mcscf.CASCI(mf, 2, 2)
                    mc.fcisolver.nstates = nstates
                    mc.kernel()
                    e_tot = mc.e_tot
                    state_idx = 0
                    for mol_el_st in molecule.electronic_states:
                        mol_el_st.energy = e_tot[state_idx]
                        print('electronic_state' + str(state_idx) + ' energy is' + str(mol_el_st.energy))
                        state_idx += 1

                    molecule.energy = e_tot[0]


                if 'CASCI-SINGLET'== self.method.upper():
                    mf = scf.RHF(pyscf_mol).run()
                    mf.scf()
                    mc = mcscf.CASCI(mf, 2, 2)
                    mc.fcisolver = fci.direct_spin0.FCI(pyscf_mol)
                    addons.fix_spin_(mc.fcisolver, ss=0)
                    mc.fcisolver.nroots = nstates
                    mc.kernel()
                    e_tot = mc.e_tot
                    state_idx = 0
                    for mol_el_st in molecule.electronic_states:
                        mol_el_st.energy = e_tot[state_idx]
                        print('electronic_state' + str(state_idx) + ' energy is' + str(mol_el_st.energy))
                        state_idx += 1

                    molecule.energy = e_tot[0]
                if 'CASCI-VQE-EX'== self.method.upper():
                    import time
                    from chemqulacs_namd.vqe.vqemcscf import VQECASCI
                    from chemqulacs_namd.vqe.vqeci import Ansatz


                    state_idx=0
                    mf=scf.ROHF(pyscf_mol).run()

                    mf.scf()
                    start_time=time.time()

                    vqe_casci = VQECASCI(mf, 2, 2,singlet_excitation=True,ansatz=Ansatz.UCCSD, use_singles=False,excitation_number=excitation_numb)
                    vqe_casci.kernel()
                    # print(f"VQE-CASCI Energy: {vqe_casci.e_tot}")
                    e_tot =[]
                    endtime=time.time()
                    print(f"CASCI-VQE Time: {endtime-start_time} ")
                    start_time = time.time()
                    qse=QSE_exact(vqe_casci.fcisolver)
                    qse.gen_excitation_operators("ee", 2)
                    qse.solve()
                    e_tot=qse.eigenvalues
                    e_tot[0]=vqe_casci.fcisolver.energies[0]
                    endtime = time.time()
                    # print('qse energies:'+str(e_tot))
                    print(f"QSE-ex Time: {endtime-start_time} ")

                    for mol_el_st in molecule.electronic_states:
                        mol_el_st.energy = e_tot[state_idx]
                        # print('electronic_state' + str(state_idx) + ' energy is' + str(mol_el_st.energy))
                        state_idx += 1

                    molecule.energy = e_tot[0]

                if 'CASCI-VQE'== self.method.upper():


                    import time
                    from chemqulacs_namd.vqe.vqemcscf import VQECASCI
                    from chemqulacs_namd.vqe.vqeci import Ansatz


                    state_idx=0
                    mf=scf.ROHF(pyscf_mol).run()

                    mf.scf()
                    start_time=time.time()

                    vqe_casci = VQECASCI(mf, 2, 2,singlet_excitation=True,ansatz=Ansatz.UCCSD, use_singles=False,excitation_number=excitation_numb)
                    vqe_casci.kernel()
                    # print(f"VQE-CASCI Energy: {vqe_casci.e_tot}")
                    e_tot =[]
                    endtime=time.time()
                    print(f"CAS-CI-VQE Time: {endtime-start_time} ")
                    start_time = time.time()
                    qse=QSE(vqe_casci.fcisolver)
                    qse.gen_excitation_operators("ee", 2)
                    qse.solve()
                    e_tot=qse.eigenvalues
                    e_tot[0]=vqe_casci.fcisolver.energies[0]
                    endtime = time.time()
                    # print('qse energies:'+str(e_tot))
                    print(f"QSE Time: {endtime-start_time} ")

                    for mol_el_st in molecule.electronic_states:
                        mol_el_st.energy = e_tot[state_idx]
                        # print('electronic_state' + str(state_idx) + ' energy is' + str(mol_el_st.energy))
                        state_idx += 1

                    molecule.energy = e_tot[0]

                if 'NOISY' == self.method.upper():
                    import time
                    noise_types = ['depol']
                    noise_strengths = {'depol_1q': 0.0001, 'depol_2q': 0.002}
                    state_idx = 0
                    mf = scf.ROHF(pyscf_mol).run()
                    mf.scf()
                    start_time = time.time()
                    noise_kwargs = {
                        "noise_type": noise_types,
                        "noise_prob": [noise_strengths['depol_1q'], noise_strengths['depol_2q']]
                    }
                    vqe_casci = VQECASCI_noisy(mf, 2, 2, singlet_excitation=True, ansatz=Ansatz_noisy.UCCSD, noise_kwargs=noise_kwargs,use_singles=False,excitation_number=excitation_numb)
                    vqe_casci.kernel()
                    qse = QSE(vqe_casci.fcisolver)
                    qse.gen_excitation_operators("ee", 2)

                    cas_state = vqe_casci.fcisolver.opt_states[0]

                    bound_circuit = cas_state


                    noperators = len(qse.e_op)
                    e_ops = qse.e_op
                    fermionic_hamiltonian = qse.vqeci.fermionic_hamiltonian
                    n = noperators + 1
                    executor = get_reusable_executor()
                    print('executor worker number:'+str(executor._max_workers))
                    hamiltonian = np.zeros((n, n), dtype=np.complex128)
                    hamiltonian[0, 0] = vqe_casci.fcisolver.energies[0]
                    tasks = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
                    op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
                        n_spin_orbitals=qse.vqeci.n_qubit, n_fermions=qse.vqeci.n_electron
                    )
                    futures = [executor.submit(compute_matrix_element, idx, jdx, fermionic_hamiltonian, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths, is_ham=True) for idx, jdx in tasks]
                    for future in futures:
                        idx, jdx, val = future.result()
                        hamiltonian[idx, jdx] = val
                        hamiltonian[jdx, idx] = np.conj(val)
                    S = np.zeros((n, n), dtype=np.complex128)
                    S[0, 0] = 1.0
                    tasks_s = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
                    futures_s = [executor.submit(compute_s_matrix_element, idx, jdx, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths) for idx, jdx in tasks_s]
                    for future in futures_s:
                        idx, jdx, val = future.result()
                        S[idx, jdx] = val
                        S[jdx, idx] = np.conj(val)
                    threshold = 1.0e-7
                    cholesky_threshold = 1.0e-9
                    x = partial_cholesky_orth_(S, canthr=threshold, cholthr=cholesky_threshold)
                    xhx = reduce(np.dot, (x.T.conj(), hamiltonian, x))
                    e_tot_noisy, eigenvectors_noisy = np.linalg.eigh(xhx)
                    e_tot_noisy[0]=vqe_casci.fcisolver.energies[0]
                    eigenvectors_noisy = np.dot(x, eigenvectors_noisy)
                    endtime = time.time()
                    print(f"CASCI-VQE-NOISY Time: {endtime - start_time} ")
                    for mol_el_st in molecule.electronic_states:
                        mol_el_st.energy = e_tot_noisy[state_idx]
                        state_idx += 1
                    molecule.energy = e_tot_noisy[0]
                if 'FDM'== self.method.upper():
                    import time
                    from chemqulacs_namd.vqe.vqemcscf import VQECASCI
                    from chemqulacs_namd.vqe.vqeci import Ansatz
                    state_idx = 0
                    mf = scf.ROHF(pyscf_mol).run()
                    mf.scf()
                    vqe_casci = VQECASCI(mf, 2, 2, singlet_excitation=True, ansatz=Ansatz.UCCSD,  use_singles=False,excitation_number=excitation_numb)
                    vqe_casci.kernel()
                    qse = QSE(vqe_casci.fcisolver)
                    qse.gen_excitation_operators("ee", 2)
                    qse.solve()
                    e_tot = qse.eigenvalues
                    e_tot[0] = vqe_casci.fcisolver.energies[0]
                    for mol_el_st in molecule.electronic_states:
                        mol_el_st.energy = e_tot[state_idx]
                        state_idx += 1
                    molecule.energy = e_tot[0]


                if 'NOISY-FDM' == self.method.upper():
                    import time
                    noise_types = ['depol']
                    noise_strengths = {'depol_1q': 0.0001, 'depol_2q': 0.002}
                    state_idx = 0
                    mf = scf.ROHF(pyscf_mol)
                    mf.scf()
                    start_time = time.time()
                    noise_kwargs = {
                        "noise_type": noise_types,
                        "noise_prob": [noise_strengths['depol_1q'], noise_strengths['depol_2q']]
                    }
                    vqe_casci = VQECASCI_noisy(mf, 2, 2, singlet_excitation=True, ansatz=Ansatz_noisy.UCCSD,   noise_kwargs=noise_kwargs,use_singles=False,excitation_number=excitation_numb)
                    vqe_casci.kernel()
                    qse = QSE(vqe_casci.fcisolver)
                    qse.gen_excitation_operators("ee", 2)

                    cas_state = vqe_casci.fcisolver.opt_states[0]
                    cas_hamiltonian = vqe_casci.fcisolver.qubit_hamiltonian
                    bound_circuit = cas_state


                    noperators = len(qse.e_op)
                    e_ops = qse.e_op
                    fermionic_hamiltonian = qse.vqeci.fermionic_hamiltonian
                    n = noperators + 1
                    executor = get_reusable_executor()
                    print('executor worker number:'+str(executor._max_workers))
                    hamiltonian = np.zeros((n, n), dtype=np.complex128)
                    hamiltonian[0, 0] = vqe_casci.fcisolver.energies[0]
                    tasks = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
                    op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
                        n_spin_orbitals=qse.vqeci.n_qubit, n_fermions=qse.vqeci.n_electron
                    )
                    futures = [executor.submit(compute_matrix_element, idx, jdx, fermionic_hamiltonian, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths, is_ham=True) for idx, jdx in tasks]
                    for future in futures:
                        idx, jdx, val = future.result()
                        hamiltonian[idx, jdx] = val
                        hamiltonian[jdx, idx] = np.conj(val)

                    S = np.zeros((n, n), dtype=np.complex128)
                    S[0, 0] = 1.0
                    tasks_s = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
                    futures_s = [executor.submit(compute_s_matrix_element, idx, jdx, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths) for idx, jdx in tasks_s]
                    for future in futures_s:
                        idx, jdx, val = future.result()
                        S[idx, jdx] = val
                        S[jdx, idx] = np.conj(val)
      
                    threshold = 1.0e-7
                    cholesky_threshold = 1.0e-9

                    x = partial_cholesky_orth_(S, canthr=threshold, cholthr=cholesky_threshold)
                    xhx = reduce(np.dot, (x.T.conj(), hamiltonian, x))
                    e_tot_noisy, eigenvectors_noisy = np.linalg.eigh(xhx)
  
                    e_tot_noisy[0]=vqe_casci.fcisolver.energies[0]
                    eigenvectors_noisy = np.dot(x, eigenvectors_noisy)
                    endtime = time.time()
                    print(f"NOISY ENERGY Time: {endtime - start_time} ")
                    for mol_el_st in molecule.electronic_states:
                        mol_el_st.energy = e_tot_noisy[state_idx]
                        state_idx += 1
                    molecule.energy = e_tot_noisy[0]

            if calculate_energy_gradients:
                if 'NOISY' == self.method.upper():
                    time1 = time.time()
                    noise_types = ['depol']
                    noise_strengths = {'depol_1q': 0.0001, 'depol_2q': 0.002}
                    hcore_mo = np.einsum('pi,pq,qj->ij', mf.mo_coeff.conj(), mf.get_hcore(), mf.mo_coeff)
                    tei_ao = pyscf_mol.intor('int2e')
                    tei_mo = of.general_basis_change(tei_ao, mf.mo_coeff, key=(1, 0, 1, 0)).transpose(0, 2, 3, 1)
                    force_ops = force_utility.gradient_mo_operator(pyscf_mol, mf.mo_coeff, hcore_mo, tei_mo, with_pulay=True)
                    grad_hellmann = np.zeros((3, 3, 3))  # states, atoms, coords
                    n_qubits = qse.vqeci.n_qubit
                    op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
                        n_spin_orbitals=n_qubits, n_fermions=qse.vqeci.n_electron
                    )
                    noise_model=create_noise_model(noise_types, noise_strengths)
                    estimator = create_qulacs_density_matrix_estimator(noise_model)
                    g_matrices = {}
                    tasks_g_base = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
                    for f_idx, force_op in enumerate(force_ops):
                        atom = f_idx // 3
                        xyz = f_idx % 3
                        if xyz == 2:
                            grad_hellmann[:, atom, xyz] = 0.0
                            continue
                        force_ferm_op = get_fermion_operator(force_op)
                        qubit_op_00 = op_mapper(force_ferm_op)
                        g_00 = estimator(qubit_op_00, bound_circuit).value
                        G = np.zeros((n, n), dtype=np.complex128)
                        G[0, 0] = g_00
                        futures_g = [executor.submit(compute_matrix_element, idx, jdx, fermionic_hamiltonian, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths, is_ham=False, force_ferm_op=force_ferm_op) for idx, jdx in tasks_g_base]
                        for future in futures_g:
                            idx, jdx, val = future.result()
                            G[idx, jdx] = val
                            G[jdx, idx] = np.conj(val)
                        g_matrices[f_idx] = G
                    state_idx = 0
                    for mol_el_st in molecule.electronic_states:
                        eigenvector = eigenvectors_noisy[:, state_idx]
                        for f_idx, force_op in enumerate(force_ops):
                            atom = f_idx // 3
                            xyz = f_idx % 3
                            if xyz == 2:
                                grad_hellmann[state_idx, atom, xyz] = 0.0
                                continue
                            G = g_matrices[f_idx]
                            exp_val = np.real(eigenvector.conj().T @ G @ eigenvector)
                            grad_hellmann[state_idx, atom, xyz] = exp_val
                        mol_el_st.energy_gradients = grad_hellmann[state_idx] / constants.Bohr2Angstrom
                        for ii in range(len(mol_el_st.atoms)):
                            mol_el_st.atoms[ii].energy_gradients = mol_el_st.energy_gradients[ii]
                        if state_idx == 0:
                            molecule.energy_gradients = mol_el_st.energy_gradients
                        state_idx += 1
                    time2 = time.time()
                    print(f"Noisy HFM Gradient Time: {time2 - time1}")


                if 'FDM' == self.method.upper():
                    import time
                    from chemqulacs_namd.vqe.vqemcscf import VQECASCI
                    from chemqulacs_namd.vqe.vqeci import Ansatz
                    time1 = time.time()
                    delta = 0.01  # Step size in Bohr
                    natom = pyscf_mol.natm  # Get the number of atoms dynamically
                    grad_fdm = np.zeros((nstates, natom, 3))  # states, atoms, coords
                    coords = pyscf_mol.atom_coords(unit='Angstrom')
                    for state_idx in range(nstates):  # Iterate over the number of states requested
                        for a in range(natom):  # Iterate over all atoms
                            for c in range(3):  # Iterate over x, y, z coordinates
                                coords_fwd = coords.copy()
                                coords_fwd[a, c] += delta
                                mol_fwd = pyscf_mol.copy()
                                mol_fwd.set_geom_(coords_fwd, unit='Angstrom')
                                mol_fwd.build()
                                mf_fwd = scf.ROHF(mol_fwd).run()
                                vqe_casci_fwd = VQECASCI(mf_fwd, 2, 2, singlet_excitation=True, ansatz=Ansatz.UCCSD,  use_singles=False,excitation_number=excitation_numb)
                                vqe_casci_fwd.kernel()
                                qse_fwd = QSE(vqe_casci_fwd.fcisolver)
                                qse_fwd.gen_excitation_operators("ee", 2)
                                qse_fwd.solve()
                                e_fwd = qse_fwd.eigenvalues
                                e_fwd[0] = vqe_casci_fwd.fcisolver.energies[0]
                                coords_bwd = coords.copy()
                                coords_bwd[a, c] -= delta
                                mol_bwd = pyscf_mol.copy()
                                mol_bwd.set_geom_(coords_bwd, unit='Angstrom')
                                mol_bwd.build()
                                mf_bwd = scf.ROHF(mol_bwd).run()
                                vqe_casci_bwd = VQECASCI(mf_bwd, 2, 2, singlet_excitation=True, ansatz=Ansatz.UCCSD,  use_singles=False,excitation_number=excitation_numb)
                                vqe_casci_bwd.kernel()
                                qse_bwd = QSE(vqe_casci_bwd.fcisolver)
                                qse_bwd.gen_excitation_operators("ee", 2)
                                qse_bwd.solve()
                                e_bwd = qse_bwd.eigenvalues
                                e_bwd[0] = vqe_casci_bwd.fcisolver.energies[0]
                                grad_fdm[state_idx, a, c] = (e_fwd[state_idx] - e_bwd[state_idx]) / (2 * delta)
                    state_idx = 0
                    for mol_el_st in molecule.electronic_states:
                        mol_el_st.energy_gradients = grad_fdm[state_idx] 
                        for ii in range(len(mol_el_st.atoms)):
                            mol_el_st.atoms[ii].energy_gradients = mol_el_st.energy_gradients[ii]
                        if state_idx == 0:
                            molecule.energy_gradients = mol_el_st.energy_gradients
                        state_idx += 1
                    time2 = time.time()
                    print(f"FDM Gradient Time: {time2 - time1}")                    



                if 'HF' == self.method.upper():
                    molecule.energy_gradients = pyscf_method.nuc_grad_method().kernel()
                    molecule.energy_gradients = molecule.energy_gradients / constants.Bohr2Angstrom
                    for ii in range(len(molecule.atoms)):
                        molecule.atoms[ii].energy_gradients = molecule.energy_gradients[ii]
                    # else:
                    #     print("PySCF doesn't converge and energy gradients will not be stored in molecule")

                if 'TD-HF' == self.method.upper():

                    # if pyscf_method.converged.any():
                    state_idx = 0

                    for mol_el_st in molecule.electronic_states:
                        molecule_gradients = pyscf_method.nuc_grad_method().kernel(state=state_idx)
                        molecule_gradients = molecule_gradients / constants.Bohr2Angstrom
                        mol_el_st.energy_gradients = molecule_gradients
                        if state_idx is 0:
                            molecule.energy_gradients = molecule_gradients
                        for ii in range(len(mol_el_st.atoms)):
                            mol_el_st.atoms[ii].energy_gradients = molecule_gradients[ii]

                        if pyscf_method.converged[state_idx]:
                            print('electronic_state' + str(state_idx) + ' gradients is' + 'converged')
                        else:
                            print('electronic_state' + str(state_idx) + ' gradients is' + 'not converged')

                        state_idx += 1

                if 'CASCI' == self.method.upper():

                    state_idx=0

                    for mol_el_st in molecule.electronic_states:
                        molecule_gradients = mc.Gradients().kernel(state=state_idx)
                        molecule_gradients = molecule_gradients / constants.Bohr2Angstrom
                        if state_idx is 0:
                            molecule.energy_gradients = molecule_gradients

                        
                        mol_el_st.energy_gradients = molecule_gradients
                        for ii in range(len(mol_el_st.atoms)):
                            mol_el_st.atoms[ii].energy_gradients = molecule_gradients[ii]
                        state_idx += 1
                if 'CASCI-SINGLET' == self.method.upper():


                    state_idx=0
                    from pyscf import grad
                    for mol_el_st in molecule.electronic_states:

                        g = grad.casci.Gradients(mc)  
                        molecule_gradients = g.kernel(state=state_idx)

                        if state_idx==0:
                            molecule.energy_gradients = molecule_gradients/constants.Bohr2Angstrom
                        molecule_gradients = molecule_gradients / constants.Bohr2Angstrom
                        mol_el_st.energy_gradients = molecule_gradients
                        for ii in range(len(mol_el_st.atoms)):
                            mol_el_st.atoms[ii].energy_gradients = molecule_gradients[ii]
                        state_idx += 1

                    
                if 'CASCI-VQE-EX'== self.method.upper():
                    def generate_excited_state(qse, k: int) -> np.ndarray:
                        if k == 0:
                            return evaluate_state_to_vector(qse.vqeci.opt_states[0]).vector
                        else:
                            ref_state = qse.vqeci.opt_states[0]
                            n_qubits = qse.vqeci.n_qubit
                            ref_vector = evaluate_state_to_vector(ref_state).vector
                            eigenvector = qse.eigenvectors[:, k]
                            e_ops = qse.e_op
                            excited_vector = eigenvector[0] * ref_vector
                            op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
                                n_spin_orbitals=qse.vqeci.n_qubit, n_fermions=qse.vqeci.n_electron
                            )
                            for i, e_op in enumerate(e_ops, start=1):
                                qubit_op = op_mapper(e_op)
                                sparse_matrix = get_sparse_matrix(qubit_op, n_qubits)
                                excited_part = sparse_matrix.dot(ref_vector)
                                excited_vector += eigenvector[i] * excited_part
                            norm = np.linalg.norm(excited_vector)
                            if norm < 1e-10:
                                raise ValueError(f"激发态 {k} 的向量范数过小，无法归一化")
                            excited_vector /= norm
                            return excited_vector

                    # Hellmann-Feynman gradient calculation
                    time1 = time.time()
                    hcore_mo = np.einsum('pi,pq,qj->ij', mf.mo_coeff.conj(), mf.get_hcore(), mf.mo_coeff)
                    tei_ao = pyscf_mol.intor('int2e')
                    tei_mo = of.general_basis_change(tei_ao, mf.mo_coeff, key=(1, 0, 1, 0)).transpose(0, 2, 3, 1)
                    force_ops = force_utility.gradient_mo_operator(pyscf_mol, mf.mo_coeff, hcore_mo, tei_mo, with_pulay=True)
                    grad_hellmann = np.zeros((3, 3, 3))  # states, atoms, coords
                    n_qubits = qse.vqeci.n_qubit
                    op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
                        n_spin_orbitals=n_qubits, n_fermions=qse.vqeci.n_electron
                    )
                    state_idx = 0
                    for mol_el_st in molecule.electronic_states:
                        state_vector = generate_excited_state(qse, state_idx)
                        for f_idx, force_op in enumerate(force_ops):
                            fermion_op = get_fermion_operator(force_op)
                            qubit_op = op_mapper(fermion_op)
                            sparse = get_sparse_matrix(qubit_op, n_qubits)
                            exp_val = np.real(np.dot(state_vector.conj().T, sparse @ state_vector))
                            atom = f_idx // 3
                            xyz = f_idx % 3
                            grad_hellmann[state_idx, atom, xyz] = exp_val
                        mol_el_st.energy_gradients = grad_hellmann[state_idx] / constants.Bohr2Angstrom
                        for ii in range(len(mol_el_st.atoms)):
                            mol_el_st.atoms[ii].energy_gradients = mol_el_st.energy_gradients[ii]
                        if state_idx == 0:
                            molecule.energy_gradients = mol_el_st.energy_gradients
                        state_idx += 1
                    time2 = time.time()
                    print('CASCI-VQE-EX HFM Gradient Time 2D:' + str(time2 - time1))

                if 'CASCI-VQE'== self.method.upper():
                    def generate_excited_state(qse, k: int) -> np.ndarray:
                        if k == 0:
                            return evaluate_state_to_vector(qse.vqeci.opt_states[0]).vector
                        else:
                            ref_state = qse.vqeci.opt_states[0]
                            n_qubits = qse.vqeci.n_qubit
                            ref_vector = evaluate_state_to_vector(ref_state).vector
                            eigenvector = qse.eigenvectors[:, k]
                            e_ops = qse.e_op
                            excited_vector = eigenvector[0] * ref_vector
                            op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
                                n_spin_orbitals=qse.vqeci.n_qubit, n_fermions=qse.vqeci.n_electron
                            )
                            for i, e_op in enumerate(e_ops, start=1):
                                qubit_op = op_mapper(e_op)
                                sparse_matrix = get_sparse_matrix(qubit_op, n_qubits)
                                excited_part = sparse_matrix.dot(ref_vector)
                                excited_vector += eigenvector[i] * excited_part
                            norm = np.linalg.norm(excited_vector)
                            if norm < 1e-10:
                                raise ValueError(f"激发态 {k} 的向量范数过小，无法归一化")
                            excited_vector /= norm
                            return excited_vector
                                        # Hellmann-Feynman gradient calculation
                    time1 = time.time()
                    hcore_mo = np.einsum('pi,pq,qj->ij', mf.mo_coeff.conj(), mf.get_hcore(), mf.mo_coeff)
                    tei_ao = pyscf_mol.intor('int2e')
                    tei_mo = of.general_basis_change(tei_ao, mf.mo_coeff, key=(1, 0, 1, 0)).transpose(0, 2, 3, 1)
                    force_ops = force_utility.gradient_mo_operator(pyscf_mol, mf.mo_coeff, hcore_mo, tei_mo, with_pulay=True)
                    grad_hellmann = np.zeros((3, 3, 3))  # states, atoms, coords
                    n_qubits = qse.vqeci.n_qubit
                    op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
                        n_spin_orbitals=n_qubits, n_fermions=qse.vqeci.n_electron
                    )
                    state_idx = 0
                    for mol_el_st in molecule.electronic_states:
                        state_vector = generate_excited_state(qse, state_idx)
                        for f_idx, force_op in enumerate(force_ops):
                            fermion_op = get_fermion_operator(force_op)
                            qubit_op = op_mapper(fermion_op)
                            sparse = get_sparse_matrix(qubit_op, n_qubits)
                            exp_val = np.real(np.dot(state_vector.conj().T, sparse @ state_vector))
                            atom = f_idx // 3
                            xyz = f_idx % 3
                            grad_hellmann[state_idx, atom, xyz] = exp_val
                        mol_el_st.energy_gradients = grad_hellmann[state_idx] / constants.Bohr2Angstrom
                        for ii in range(len(mol_el_st.atoms)):
                            mol_el_st.atoms[ii].energy_gradients = mol_el_st.energy_gradients[ii]
                        if state_idx == 0:
                            molecule.energy_gradients = mol_el_st.energy_gradients
                        state_idx += 1
                    time2 = time.time()
                    print('CASCI-VQE  HFM Gradient Time 2D:' + str(time2 - time1))


                if 'NOISY-FDM' == self.method.upper():
                    time1 = time.time()
                    noise_types = ['depol', 'bitflip']
                    noise_strengths = {'depol_1q': 0.0001, 'depol_2q': 0.002, 'bitflip': 0.02}


                    delta = 0.01  # Step size in Angstrom, as in noisy_grad_compare_final.py
                    natom = pyscf_mol.natm
                    grad_fdm = np.zeros((nstates, natom, 3))
                    coords = pyscf_mol.atom_coords(unit='Angstrom')
                    for state_idx in range(nstates):
                        for a in range(natom):
                            for c in range(3):
                                coords_fwd = coords.copy()
                                coords_fwd[a, c] += delta
                                mol_fwd = pyscf_mol.copy()
                                mol_fwd.set_geom_(coords_fwd, unit='Angstrom')
                                mol_fwd.build()
                                e_fwd = compute_noisy_energies(mol_fwd, noise_types,noise_strengths)[0][state_idx]
                                coords_bwd = coords.copy()
                                coords_bwd[a, c] -= delta
                                mol_bwd = pyscf_mol.copy()
                                mol_bwd.set_geom_(coords_bwd, unit='Angstrom')
                                mol_bwd.build()
                                e_bwd = compute_noisy_energies(mol_bwd, noise_types,noise_strengths)[0][state_idx]
                                grad_fdm[state_idx, a, c] = (e_fwd - e_bwd) / (2 * delta)
                    state_idx = 0
                    for mol_el_st in molecule.electronic_states:
                        mol_el_st.energy_gradients = grad_fdm[state_idx]
                        for ii in range(len(mol_el_st.atoms)):
                            mol_el_st.atoms[ii].energy_gradients = mol_el_st.energy_gradients[ii]
                        if state_idx == 0:
                            molecule.energy_gradients = mol_el_st.energy_gradients
                        state_idx += 1
                    time2 = time.time()
                    print(f"Noisy FDM Gradient Time: {time2 - time1}")

            if calculate_hessian:
                hessian_step_length=0.01
                if 'NOISY-FDM' == self.method.upper():
                    def compute_hessian(pyscf_mol, epsilon, state_idx=0):
                        """
                        Compute the nuclear Hessian matrix using central finite differences of analytical gradients.

                        Parameters:
                            pyscf_mol: PySCF CASCI object
                            epsilon: Step size for finite differences (in Bohr)
                            state_idx: Electronic state index (default: 0)

                        Returns:
                            hessian: (3N, 3N) Hessian matrix in Hartree/Bohr^2, where N is the number of atoms
                        """
                        mol = pyscf_mol
                        N = mol.natm  # Number of atoms
                        hessian = np.zeros((3 * N, 3 * N))
                        coords=mol.atom_coords(unit='Angstrom')
                        coords = coords.flatten()  # Flatten to (3N,) vector

                        for j in range(3 * N):
                            # Positive displacement along coordinate j
                            coords_pos = coords.copy()
                            coords_pos[j] += epsilon
                            mol_pos = mol.copy()
                            mol_pos.set_geom_(coords_pos.reshape(N, 3), unit='Angstrom')
                            mol_pos.build()
                            mf_pos= scf.RHF(mol_pos).run()
                            mf_pos.kernel()
                            mc_pos = mcscf.CASCI(mf_pos, 2, 2).run()

                            # mc_pos.fcisolver = fci.direct_spin0.FCI(mol_pos)
                            # addons.fix_spin_(mc_pos.fcisolver, ss=0)
                            mc_pos.fix_spin(ss=0)
                            mc_pos.fcisolver.nroots = nstates
                            mc_pos.kernel()
                            g_pos = grad.casci.Gradients(mc_pos).kernel(state=state_idx).flatten()

                            # Negative displacement along coordinate j
                            coords_neg = coords.copy()
                            coords_neg[j] -= epsilon
                            mol_neg = mol.copy()
                            mol_neg.set_geom_(coords_neg.reshape(N, 3), unit='Angstrom')
                            mol_neg.build()
                            mf_neg= scf.RHF(mol_neg).run()
                            mf_neg.kernel()
                            mc_neg = mcscf.CASCI(mf_neg, 2, 2).run()

                            # mc_neg.fcisolver = fci.direct_spin0.FCI(mol_neg)
                            # addons.fix_spin_(mc_neg.fcisolver, ss=0)
                            mc_neg.fix_spin(ss=0)
                            mc_neg.fcisolver.nroots = nstates
                            mc_neg.kernel()
                            g_neg = grad.casci.Gradients(mc_neg).kernel(state=state_idx).flatten()

                            # Compute Hessian column j using central difference
                            hessian[:, j] = (g_pos - g_neg) / (2 * epsilon)

                        return hessian


                    eps_hess = hessian_step_length

                    hess = compute_hessian(pyscf_mol, eps_hess,0)
                    molecule.hessian = hess/ constants.Bohr2Angstrom
                if 'HF'== self.method.upper():
                    pyscf_method_hf = scf.RHF(pyscf_mol).run()
                    hess = pyscf_method_hf.Hessian().kernel()

                    # if pyscf_method.converged.any():
                    natom = len(molecule.atoms)
                    h = np.zeros((3 * natom, 3 * natom))
                    for ii in range(natom):
                        for jj in range(natom):
                            h[ii * 3:(ii + 1) * 3, jj * 3:(jj + 1) * 3] = hess[ii][jj]
                    molecule.hessian = h / constants.Bohr2Angstrom ** 2
                if 'CASCI-VQE-EX'== self.method.upper():
                    def generate_excited_state(qse, k: int) -> np.ndarray:
                        """
                        根据 QSE 的特征向量和激发算符，生成第 k 个近似的激发态波函数向量。

                        Args:
                            qse: QSE 或 QSE_serial 对象，包含激发算符 self.e_op 和特征向量 self.eigenvectors
                            k: 激发态的索引

                        Returns:
                            excited_state_vector: 第 k 个近似的激发态向量（numpy 数组）
                        """
                        # 获取参考态向量
                        ref_state = qse.vqeci.opt_states[0]  # CircuitQuantumState
                        n_qubits = qse.vqeci.n_qubit
                        ref_vector = evaluate_state_to_vector(ref_state).vector  # Correct: pass state, not circuit

                        # 获取第 k 个特征向量
                        eigenvector = qse.eigenvectors[:, k]
                        
                        # 获取激发算符列表
                        e_ops = qse.e_op

                        # 初始化激发态向量：基态部分
                        excited_vector = eigenvector[0] * ref_vector

                        # 获取算符映射器（与 QSE 一致）
                        op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
                            n_spin_orbitals=qse.vqeci.n_qubit, n_fermions=qse.vqeci.n_electron
                        )

                        # 计算每个激发态贡献
                        for i, e_op in enumerate(e_ops, start=1):
                            # 将费米子算符映射到 quri-parts Operator（qubit 空间）
                            qubit_op = op_mapper(e_op)  # Correct: use mapper to handle fermion-to-qubit conversion
                    
                            # 将 Operator 转换为稀疏矩阵
                            sparse_matrix = get_sparse_matrix(qubit_op, n_qubits)
                            
                            # 应用激发算符到参考态向量
                            excited_part = sparse_matrix.dot(ref_vector)
                            
                            # 根据特征向量系数累加
                            excited_vector += eigenvector[i] * excited_part

                        # 归一化激发态向量
                        norm = np.linalg.norm(excited_vector)
                        if norm < 1e-10:  # 避免除以零
                            raise ValueError(f"激发态 {k} 的向量范数过小，无法归一化")
                        excited_vector /= norm

                        return excited_vector
                    
                    def compute_gradient(pyscf_molecule, state_idx=0):
                        """
                        使用 Hellmann-Feynman 计算分子核坐标的梯度。

                        参数:
                            pyscf_molecule: PySCF 分子对象
                            state_idx: 电子态索引，默认为 0

                        返回:
                            gradient: (3N,) 的梯度向量，单位为 Hartree / Angstrom
                        """
                        mol = pyscf_molecule
                        mf = scf.RHF(mol).run()
                        vqe_casci = VQECASCI(mf, 2, 2, singlet_excitation=True, ansatz=Ansatz.UCCSD, use_singles=False,excitation_number=excitation_numb)
                        vqe_casci.kernel()
                        qse = QSE(vqe_casci.fcisolver)
                        qse.gen_excitation_operators("ee",2)
                        qse.solve()
                        hcore_mo = np.einsum('pi,pq,qj->ij', mf.mo_coeff.conj(), mf.get_hcore(), mf.mo_coeff)
                        tei_ao = mol.intor('int2e')
                        tei_mo = of.general_basis_change(tei_ao, mf.mo_coeff, key=(1, 0, 1, 0)).transpose(0, 2, 3, 1)
                        force_ops = force_utility.gradient_mo_operator(mol, mf.mo_coeff, hcore_mo, tei_mo, with_pulay=True)
                        N = mol.natm
                        gradient = np.zeros(3 * N)
                        n_qubits = qse.vqeci.n_qubit
                        op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
                            n_spin_orbitals=n_qubits, n_fermions=qse.vqeci.n_electron
                        )
                        state_vector = generate_excited_state(qse, state_idx)
                        for f_idx, force_op in enumerate(force_ops):
                            fermion_op = get_fermion_operator(force_op)
                            qubit_op = op_mapper(fermion_op)
                            sparse = get_sparse_matrix(qubit_op, n_qubits)
                            exp_val = np.real(np.dot(state_vector.conj().T, sparse @ state_vector))
                            atom = f_idx // 3
                            xyz = f_idx % 3
                            gradient[atom*3 + xyz] = exp_val

                        return gradient

                    def compute_hessian(pyscf_molecule, epsilon, state_idx=0):
                        """
                        计算分子核坐标的 Hessian 矩阵。

                        参数:
                            pyscf_molecule: PySCF 分子对象
                            epsilon: 用于计算 Hessian 的有限差分步长
                            state_idx: 电子态索引，默认为 0

                        返回:
                            hessian: (3N, 3N) 的 Hessian 矩阵，单位为 Hartree / Angstrom^2
                        """
                        mol = pyscf_molecule
                        N = mol.natm  # 原子数量
                        atom_coords = mol.atom_coords()  # 原始坐标，形状 (N, 3)
                        hessian = np.zeros((3 * N, 3 * N))

                        def get_displaced_mol(displacements):
                            coords = atom_coords.copy()
                            for i, sign in displacements:
                                a = i // 3  # 原子索引
                                d = i % 3  # 方向索引 (0=x, 1=y, 2=z)
                                coords[a, d] += sign * epsilon
                            mol_new = mol.copy()
                            mol_new.set_geom_(coords, unit='Bohr')
                            mol_new.build()
                            return mol_new

                        # 计算 Hessian
                        for j in range(3 * N):
                            # q_j 正向位移 (epsilon)
                            mol_pos_j = get_displaced_mol([(j, +1)])
                            G_pos_j = compute_gradient(mol_pos_j, state_idx)

                            # q_j 负向位移 (epsilon)
                            mol_neg_j = get_displaced_mol([(j, -1)])
                            G_neg_j = compute_gradient(mol_neg_j, state_idx)

                            # Hessian 第 j 列
                            hessian[:, j] = (G_pos_j - G_neg_j) / (2 * epsilon)

                        return hessian

                    eps_hess = hessian_step_length

                    hess = compute_hessian(pyscf_mol, eps_hess, state_idx=0)

                    molecule.hessian = hess/ constants.Bohr2Angstrom ** 2

                    return hess
                if 'CASCI-VQE'== self.method.upper():
                    def generate_excited_state(qse, k: int) -> np.ndarray:
                        """
                        根据 QSE 的特征向量和激发算符，生成第 k 个近似的激发态波函数向量。

                        Args:
                            qse: QSE 或 QSE_serial 对象，包含激发算符 self.e_op 和特征向量 self.eigenvectors
                            k: 激发态的索引

                        Returns:
                            excited_state_vector: 第 k 个近似的激发态向量（numpy 数组）
                        """
                        # 获取参考态向量
                        ref_state = qse.vqeci.opt_states[0]  # CircuitQuantumState
                        n_qubits = qse.vqeci.n_qubit
                        ref_vector = evaluate_state_to_vector(ref_state).vector  # Correct: pass state, not circuit

                        # 获取第 k 个特征向量
                        eigenvector = qse.eigenvectors[:, k]
                        
                        # 获取激发算符列表
                        e_ops = qse.e_op

                        # 初始化激发态向量：基态部分
                        excited_vector = eigenvector[0] * ref_vector

                        # 获取算符映射器（与 QSE 一致）
                        op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
                            n_spin_orbitals=qse.vqeci.n_qubit, n_fermions=qse.vqeci.n_electron
                        )

                        # 计算每个激发态贡献
                        for i, e_op in enumerate(e_ops, start=1):
                            # 将费米子算符映射到 quri-parts Operator（qubit 空间）
                            qubit_op = op_mapper(e_op)  # Correct: use mapper to handle fermion-to-qubit conversion
                    
                            # 将 Operator 转换为稀疏矩阵
                            sparse_matrix = get_sparse_matrix(qubit_op, n_qubits)
                            
                            # 应用激发算符到参考态向量
                            excited_part = sparse_matrix.dot(ref_vector)
                            
                            # 根据特征向量系数累加
                            excited_vector += eigenvector[i] * excited_part

                        # 归一化激发态向量
                        norm = np.linalg.norm(excited_vector)
                        if norm < 1e-10:  # 避免除以零
                            raise ValueError(f"激发态 {k} 的向量范数过小，无法归一化")
                        excited_vector /= norm

                        return excited_vector
                    
                    def compute_gradient(pyscf_molecule, state_idx=0):
                        """
                        使用 Hellmann-Feynman 计算分子核坐标的梯度。

                        参数:
                            pyscf_molecule: PySCF 分子对象
                            state_idx: 电子态索引，默认为 0

                        返回:
                            gradient: (3N,) 的梯度向量，单位为 Hartree / Angstrom
                        """
                        mol = pyscf_molecule
                        mf = scf.RHF(mol).run()
                        vqe_casci = VQECASCI(mf, 2, 2, singlet_excitation=True, ansatz=Ansatz.UCCSD, use_singles=False,excitation_number=excitation_numb)
                        vqe_casci.kernel()
                        qse = QSE(vqe_casci.fcisolver)
                        qse.gen_excitation_operators("ee",2)
                        qse.solve()
                        hcore_mo = np.einsum('pi,pq,qj->ij', mf.mo_coeff.conj(), mf.get_hcore(), mf.mo_coeff)
                        tei_ao = mol.intor('int2e')
                        tei_mo = of.general_basis_change(tei_ao, mf.mo_coeff, key=(1, 0, 1, 0)).transpose(0, 2, 3, 1)
                        force_ops = force_utility.gradient_mo_operator(mol, mf.mo_coeff, hcore_mo, tei_mo, with_pulay=True)
                        N = mol.natm
                        gradient = np.zeros(3 * N)
                        n_qubits = qse.vqeci.n_qubit
                        op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
                            n_spin_orbitals=n_qubits, n_fermions=qse.vqeci.n_electron
                        )
                        state_vector = generate_excited_state(qse, state_idx)
                        for f_idx, force_op in enumerate(force_ops):
                            fermion_op = get_fermion_operator(force_op)
                            qubit_op = op_mapper(fermion_op)
                            sparse = get_sparse_matrix(qubit_op, n_qubits)
                            exp_val = np.real(np.dot(state_vector.conj().T, sparse @ state_vector))
                            atom = f_idx // 3
                            xyz = f_idx % 3
                            gradient[atom*3 + xyz] = exp_val

                        return gradient

                    def compute_hessian(pyscf_molecule, epsilon, state_idx=0):
                        """
                        计算分子核坐标的 Hessian 矩阵。

                        参数:
                            pyscf_molecule: PySCF 分子对象
                            epsilon: 用于计算 Hessian 的有限差分步长
                            state_idx: 电子态索引，默认为 0

                        返回:
                            hessian: (3N, 3N) 的 Hessian 矩阵，单位为 Hartree / Angstrom^2
                        """
                        mol = pyscf_molecule
                        N = mol.natm  # 原子数量
                        atom_coords = mol.atom_coords()  # 原始坐标，形状 (N, 3)
                        hessian = np.zeros((3 * N, 3 * N))

                        def get_displaced_mol(displacements):
                            coords = atom_coords.copy()
                            for i, sign in displacements:
                                a = i // 3  # 原子索引
                                d = i % 3  # 方向索引 (0=x, 1=y, 2=z)
                                coords[a, d] += sign * epsilon
                            mol_new = mol.copy()
                            mol_new.set_geom_(coords, unit='Bohr')
                            mol_new.build()
                            return mol_new

                        # 计算 Hessian
                        for j in range(3 * N):
                            # q_j 正向位移 (epsilon)
                            mol_pos_j = get_displaced_mol([(j, +1)])
                            G_pos_j = compute_gradient(mol_pos_j, state_idx)

                            # q_j 负向位移 (epsilon)
                            mol_neg_j = get_displaced_mol([(j, -1)])
                            G_neg_j = compute_gradient(mol_neg_j, state_idx)

                            # Hessian 第 j 列
                            hessian[:, j] = (G_pos_j - G_neg_j) / (2 * epsilon)

                        return hessian

                    eps_hess = hessian_step_length

                    hess = compute_hessian(pyscf_mol, eps_hess, state_idx=0)

                    molecule.hessian = hess/ constants.Bohr2Angstrom ** 2

                    return hess

                if 'CASCI-SINGLET'== self.method.upper():

                    def compute_hessian(pyscf_mol, epsilon, state_idx=0):
                        """
                        Compute the nuclear Hessian matrix using central finite differences of analytical gradients.

                        Parameters:
                            pyscf_mol: PySCF CASCI object
                            epsilon: Step size for finite differences (in Bohr)
                            state_idx: Electronic state index (default: 0)

                        Returns:
                            hessian: (3N, 3N) Hessian matrix in Hartree/Bohr^2, where N is the number of atoms
                        """
                        mol = pyscf_mol
                        N = mol.natm  # Number of atoms
                        hessian = np.zeros((3 * N, 3 * N))
                        coords=mol.atom_coords(unit='Angstrom')
                        coords = coords.flatten()  # Flatten to (3N,) vector

                        for j in range(3 * N):
                            # Positive displacement along coordinate j
                            coords_pos = coords.copy()
                            coords_pos[j] += epsilon
                            mol_pos = mol.copy()
                            mol_pos.set_geom_(coords_pos.reshape(N, 3), unit='Angstrom')
                            mol_pos.build()
                            mf_pos= scf.RHF(mol_pos).run()
                            mf_pos.kernel()
                            mc_pos = mcscf.CASCI(mf_pos, 2, 2).run()

                            # mc_pos.fcisolver = fci.direct_spin0.FCI(mol_pos)
                            # addons.fix_spin_(mc_pos.fcisolver, ss=0)
                            mc_pos.fix_spin(ss=0)
                            mc_pos.fcisolver.nroots = nstates
                            mc_pos.kernel()
                            g_pos = grad.casci.Gradients(mc_pos).kernel(state=state_idx).flatten()

                            # Negative displacement along coordinate j
                            coords_neg = coords.copy()
                            coords_neg[j] -= epsilon
                            mol_neg = mol.copy()
                            mol_neg.set_geom_(coords_neg.reshape(N, 3), unit='Angstrom')
                            mol_neg.build()
                            mf_neg= scf.RHF(mol_neg).run()
                            mf_neg.kernel()
                            mc_neg = mcscf.CASCI(mf_neg, 2, 2).run()

                            # mc_neg.fcisolver = fci.direct_spin0.FCI(mol_neg)
                            # addons.fix_spin_(mc_neg.fcisolver, ss=0)
                            mc_neg.fix_spin(ss=0)
                            mc_neg.fcisolver.nroots = nstates
                            mc_neg.kernel()
                            g_neg = grad.casci.Gradients(mc_neg).kernel(state=state_idx).flatten()

                            # Compute Hessian column j using central difference
                            hessian[:, j] = (g_pos - g_neg) / (2 * epsilon)

                        return hessian


                    eps_hess = hessian_step_length

                    hess = compute_hessian(pyscf_mol, eps_hess,0)
                    molecule.hessian = hess/ constants.Bohr2Angstrom

                    return hess
                if 'NOISY'== self.method.upper():
                    def compute_gradient(pyscf_molecule, state_idx=0):
                        mol = pyscf_molecule
                        mf = scf.RHF(mol).run()
                        noise_types = ['depol']
                        noise_strengths = {'depol_1q': 0.0001, 'depol_2q': 0.002}
                        noise_kwargs = {
                            "noise_type": noise_types,
                            "noise_prob": [noise_strengths['depol_1q'], noise_strengths['depol_2q']]
                        }
                        vqe_casci = VQECASCI_noisy(mf, 2, 2, singlet_excitation=True, ansatz=Ansatz_noisy.UCCSD,   noise_kwargs=noise_kwargs,use_singles=False,excitation_number=excitation_numb)
                        vqe_casci.kernel()
                        qse = QSE(vqe_casci.fcisolver)
                        qse.gen_excitation_operators("ee", 2)
                        qse.solve()
                        cas_state = vqe_casci.fcisolver.opt_states[0]
                        cas_hamiltonian = vqe_casci.fcisolver.qubit_hamiltonian
                        bound_circuit = cas_state
                        noperators = len(qse.e_op)
                        e_ops = qse.e_op
                        fermionic_hamiltonian = qse.vqeci.fermionic_hamiltonian
                        n = noperators + 1
                        executor = get_reusable_executor()
                        print('executor worker number:'+str(executor._max_workers))
                        noise_model = create_noise_model(noise_types, noise_strengths)
                        estimator = create_qulacs_density_matrix_estimator(noise_model)
                        noisy_energy = estimator(cas_hamiltonian, bound_circuit).value
                        hamiltonian = np.zeros((n, n), dtype=np.complex128)
                        hamiltonian[0, 0] = noisy_energy
                        tasks = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
                        op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
                            n_spin_orbitals=qse.vqeci.n_qubit, n_fermions=qse.vqeci.n_electron
                        )
                        futures = [executor.submit(compute_matrix_element, idx, jdx, fermionic_hamiltonian, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths, is_ham=True) for idx, jdx in tasks]
                        for future in futures:
                            idx, jdx, val = future.result()
                            hamiltonian[idx, jdx] = val
                            hamiltonian[jdx, idx] = np.conj(val)
                        S = np.zeros((n, n), dtype=np.complex128)
                        S[0, 0] = 1.0
                        tasks_s = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
                        futures_s = [executor.submit(compute_s_matrix_element, idx, jdx, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths) for idx, jdx in tasks_s]
                        for future in futures_s:
                            idx, jdx, val = future.result()
                            S[idx, jdx] = val
                            S[jdx, idx] = np.conj(val)
                        threshold = 1.0e-7
                        cholesky_threshold = 1.0e-9
                        x = partial_cholesky_orth_(S, canthr=threshold, cholthr=cholesky_threshold)
                        xhx = reduce(np.dot, (x.T.conj(), hamiltonian, x))
                        _, eigenvectors_noisy = np.linalg.eigh(xhx)
                        eigenvectors_noisy = np.dot(x, eigenvectors_noisy)
                        hcore_mo = np.einsum('pi,pq,qj->ij', mf.mo_coeff.conj(), mf.get_hcore(), mf.mo_coeff)
                        tei_ao = mol.intor('int2e')
                        tei_mo = of.general_basis_change(tei_ao, mf.mo_coeff, key=(1, 0, 1, 0)).transpose(0, 2, 3, 1)
                        force_ops = force_utility.gradient_mo_operator(mol, mf.mo_coeff, hcore_mo, tei_mo, with_pulay=True)
                        N = mol.natm
                        gradient = np.zeros(3 * N)
                        g_matrices = {}
                        tasks_g_base = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
                        for f_idx, force_op in enumerate(force_ops):
                            atom = f_idx // 3
                            xyz = f_idx % 3
                            if xyz == 2:
                                gradient[atom * 3 + xyz] = 0.0
                                continue
                            force_ferm_op = get_fermion_operator(force_op)
                            qubit_op_00 = op_mapper(force_ferm_op)
                            g_00 = estimator(qubit_op_00, bound_circuit).value
                            G = np.zeros((n, n), dtype=np.complex128)
                            G[0, 0] = g_00
                            futures_g = [executor.submit(compute_matrix_element, idx, jdx, fermionic_hamiltonian, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths, is_ham=False, force_ferm_op=force_ferm_op) for idx, jdx in tasks_g_base]
                            for future in futures_g:
                                idx, jdx, val = future.result()
                                G[idx, jdx] = val
                                G[jdx, idx] = np.conj(val)
                            g_matrices[f_idx] = G
                        eigenvector = eigenvectors_noisy[:, state_idx]
                        for f_idx, force_op in enumerate(force_ops):
                            atom = f_idx // 3
                            xyz = f_idx % 3
                            if xyz == 2:
                                gradient[atom * 3 + xyz] = 0.0
                                continue
                            G = g_matrices[f_idx]
                            exp_val = np.real(eigenvector.conj().T @ G @ eigenvector)
                            gradient[atom * 3 + xyz] = exp_val
                        return gradient

                    def compute_hessian(pyscf_molecule, epsilon, state_idx=0):
                        mol = pyscf_molecule
                        N = mol.natm
                        atom_coords = mol.atom_coords()
                        hessian = np.zeros((3 * N, 3 * N))
                        def get_displaced_mol(displacements):
                            coords = atom_coords.copy()
                            for i, sign in displacements:
                                a = i // 3
                                d = i % 3
                                coords[a, d] += sign * epsilon
                            mol_new = mol.copy()
                            mol_new.set_geom_(coords, unit='Bohr')
                            mol_new.build()
                            return mol_new
                        for j in range(3 * N):
                            atom = j // 3
                            xyz = j % 3
                            if xyz == 2:  # Skip z-direction
                                hessian[:, j] = 0.0
                                continue
                            mol_pos_j = get_displaced_mol([(j, +1)])
                            G_pos_j = compute_gradient(mol_pos_j, state_idx)
                            mol_neg_j = get_displaced_mol([(j, -1)])
                            G_neg_j = compute_gradient(mol_neg_j, state_idx)
                            hessian[:, j] = (G_pos_j - G_neg_j) / (2 * epsilon)
                        return hessian
                    eps_hess = hessian_step_length
                    hess = compute_hessian(pyscf_mol, eps_hess, state_idx=0)
                    molecule.hessian = hess / constants.Bohr2Angstrom ** 2
                    return hess                    


                if 'FDM'== self.method.upper():
                    import time
                    from chemqulacs_namd.vqe.vqemcscf import VQECASCI
                    from chemqulacs_namd.vqe.vqeci import Ansatz
                    def compute_gradient(pyscf_molecule, state_idx=0):
                        mol = pyscf_molecule
                        delta = 0.02
                        N = mol.natm
                        gradient = np.zeros(3 * N)
                        coords = mol.atom_coords()
                        for a in range(N):
                            for c in range(3):
                                if c == 2:  # Skip z-direction
                                    gradient[a * 3 + c] = 0.0
                                    continue
                                coords_fwd = coords.copy()
                                coords_fwd[a, c] += delta
                                mol_fwd = mol.copy()
                                mol_fwd.set_geom_(coords_fwd, unit='Bohr')
                                mol_fwd.build()
                                mf_fwd = scf.RHF(mol_fwd).run()
                                vqe_casci_fwd = VQECASCI(mf_fwd, 2, 2, singlet_excitation=True, ansatz=Ansatz.UCCSD, use_singles=False,excitation_number=excitation_numb)
                                vqe_casci_fwd.kernel()
                                qse_fwd = QSE(vqe_casci_fwd.fcisolver)
                                qse_fwd.gen_excitation_operators("ee", 2)
                                qse_fwd.solve()
                                e_fwd = qse_fwd.eigenvalues
                                e_fwd[0] = vqe_casci_fwd.fcisolver.energies[0]
                                coords_bwd = coords.copy()
                                coords_bwd[a, c] -= delta
                                mol_bwd = mol.copy()
                                mol_bwd.set_geom_(coords_bwd, unit='Bohr')
                                mol_bwd.build()
                                mf_bwd = scf.RHF(mol_bwd).run()
                                vqe_casci_bwd = VQECASCI(mf_bwd, 2, 2, singlet_excitation=True, ansatz=Ansatz.UCCSD, use_singles=False,excitation_number=excitation_numb)
                                vqe_casci_bwd.kernel()
                                qse_bwd = QSE(vqe_casci_bwd.fcisolver)
                                qse_bwd.gen_excitation_operators("ee", 2)
                                qse_bwd.solve()
                                e_bwd = qse_bwd.eigenvalues
                                e_bwd[0] = vqe_casci_bwd.fcisolver.energies[0]
                                gradient[a * 3 + c] = (e_fwd[state_idx] - e_bwd[state_idx]) / (2 * delta)
                        return gradient
                
                    def compute_hessian(pyscf_molecule, epsilon, state_idx=0):
                        mol = pyscf_molecule
                        N = mol.natm
                        atom_coords = mol.atom_coords()
                        hessian = np.zeros((3 * N, 3 * N))
                        def get_displaced_mol(displacements):
                            coords = atom_coords.copy()
                            for i, sign in displacements:
                                a = i // 3
                                d = i % 3
                                coords[a, d] += sign * epsilon
                            mol_new = mol.copy()
                            mol_new.set_geom_(coords, unit='Bohr')
                            mol_new.build()
                            return mol_new
                        for j in range(3 * N):
                            atom = j // 3
                            xyz = j % 3
                            if xyz == 2:  # Skip z-direction
                                hessian[:, j] = 0.0
                                continue
                            mol_pos_j = get_displaced_mol([(j, +1)])
                            G_pos_j = compute_gradient(mol_pos_j, state_idx)
                            mol_neg_j = get_displaced_mol([(j, -1)])
                            G_neg_j = compute_gradient(mol_neg_j, state_idx)
                            hessian[:, j] = (G_pos_j - G_neg_j) / (2 * epsilon)
                        return hessian
                    eps_hess = hessian_step_length
                    hess = compute_hessian(pyscf_mol, eps_hess, state_idx=0)
                    molecule.hessian = hess / constants.Bohr2Angstrom ** 2
                    return hess                  

    @doc_inherit
    def predict(self, molecule=None, molecular_database=None, calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, **kwargs):
        '''
            **kwargs: ``# needs to be documented``.
        '''
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        for mol in molDB.molecules:
            self.predict_for_molecule(molecule=mol, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, **kwargs)


def thermo_calculation(molecule):
# construct pyscf molecule object
    from pyscf.hessian.thermo import harmonic_analysis
    from pyscf import gto
    import numpy
    pyscf_mol = gto.Mole()
    pyscf_mol.atom = [[a.element_symbol, tuple(a.xyz_coordinates)] for a in molecule.atoms]
    #pyscf_mol.basis = method.split('/')[1]
    pyscf_mol.charge = molecule.charge
    # print('freq build mole charge = '+str(pyscf_mol.charge))
    pyscf_mol.spin = molecule.multiplicity-1
    pyscf_mol.symmetry=False
    # print('freq build mole spin = '+str(pyscf_mol.spin))
    # pyscf_mol.verbose = 0
    pyscf_mol.build()
    pyscf_mol.energy = molecule.energy
    # pyscf_mol.symmetry=False
    #method = method.split('/')[0]

    # reconstruct hessian
    natom = len(molecule.atoms)
    h = molecule.hessian
    hess = np.zeros((natom, natom, 3, 3))
    for ii in range(natom):
        for jj in range(natom):
            hess[ii][jj] = h[ii*3:(ii+1)*3, jj*3:(jj+1)*3]
    hess = hess / constants.Angstrom2Bohr**2
    # harmonic analysis
    
    thermo_results = harmonic_analysis(pyscf_mol, hess, imaginary_freq=True)
    freq_wn = thermo_results['freq_wavenumber']
    if numpy.iscomplexobj(freq_wn):
        freq_wn = freq_wn.real - abs(freq_wn.imag)
    molecule.frequencies = freq_wn
    molecule.force_constants = thermo_results['force_const_dyne'].real
    molecule.reduced_masses = thermo_results['reduced_mass'].real
    # normal modes
    mode = thermo_results['norm_mode'].real
    # Pyscf provides mass deweighted unnormalized normal modes. Normalize them here.
    for imode in range(mode.shape[0]):
        mode /= np.sqrt(np.sum(mode[imode]**2))
    for iatom in range(len(molecule.atoms)):
        molecule.atoms[iatom].normal_modes = []
        for ii in range(mode.shape[0]):
            xyzs = mode[ii, iatom]
            molecule.atoms[iatom].normal_modes.append(xyzs)
        molecule.atoms[iatom].normal_modes = np.array(molecule.atoms[iatom].normal_modes)
    thermo_results = thermo_modified_from_pyscf(pyscf_mol, thermo_results['freq_au'], 298.15, 101325)
    molecule.ZPE = thermo_results['ZPE'][0]
    #molecule.DeltaE2U = 
    #molecule.DeltaE2H = 
    #molecule.DeltaE2G =
    molecule.U0 = thermo_results['E_0K'][0]
    molecule.H0 = molecule.U0
    molecule.U = thermo_results['E_tot'][0]
    molecule.H = thermo_results['H_tot'][0]
    molecule.G = thermo_results['G_tot'][0]
    molecule.S = thermo_results['S_tot'][0]

    return True 


def thermo_modified_from_pyscf(molecule, freq, temperature=298.15, pressure=101325):
# https://github.com/pyscf/pyscf/blob/master/pyscf/hessian/thermo.py

    import numpy
    from pyscf.data import nist
    from pyscf.hessian.thermo import _get_rotor_type, rotation_const, rotational_symmetry_number

    mol = molecule
    atom_coords = mol.atom_coords()
    mass = mol.atom_mass_list(isotope_avg=True)
    mass_center = numpy.einsum('z,zx->x', mass, atom_coords) / mass.sum()
    atom_coords = atom_coords - mass_center

    kB = nist.BOLTZMANN
    h = nist.PLANCK
    # c = nist.LIGHT_SPEED_SI
    # beta = 1. / (kB * temperature)
    R_Eh = kB*nist.AVOGADRO / (nist.HARTREE2J * nist.AVOGADRO)

    results = {}
    results['temperature'] = (temperature, 'K')
    results['pressure'] = (pressure, 'Pa')

    E0 = mol.energy
    results['E0'] = (E0, 'Eh')

    # Electronic part
    results['S_elec' ] = (R_Eh * numpy.log(mol.multiplicity), 'Eh/K')
    results['Cv_elec'] = results['Cp_elec'] = (0, 'Eh/K')
    results['E_elec' ] = results['H_elec' ] = (E0, 'Eh')

    # Translational part. See also https://cccbdb.nist.gov/thermo.asp for the
    # partition function q_trans
    mass_tot = mass.sum() * nist.ATOMIC_MASS
    q_trans = ((2.0 * numpy.pi * mass_tot * kB * temperature / h**2)**1.5
               * kB * temperature / pressure)
    results['S_trans' ] = (R_Eh * (2.5 + numpy.log(q_trans)), 'Eh/K')
    results['Cv_trans'] = (1.5 * R_Eh, 'Eh/K')
    results['Cp_trans'] = (2.5 * R_Eh, 'Eh/K')
    results['E_trans' ] = (1.5 * R_Eh * temperature, 'Eh')
    results['H_trans' ] = (2.5 * R_Eh * temperature, 'Eh')

    # Rotational part
    rot_const = rotation_const(mass, atom_coords, 'GHz')
    results['rot_const'] = (rot_const, 'GHz')
    rotor_type = _get_rotor_type(rot_const)

    sym_number = rotational_symmetry_number(mol)
    results['sym_number'] = (sym_number, '')

    # partition function q_rot (https://cccbdb.nist.gov/thermo.asp)
    if rotor_type == 'ATOM':
        results['S_rot' ] = (0, 'Eh/K')
        results['Cv_rot'] = results['Cp_rot'] = (0, 'Eh/K')
        results['E_rot' ] = results['H_rot' ] = (0, 'Eh')
    elif rotor_type == 'LINEAR':
        B = rot_const[1] * 1e9
        q_rot = kB * temperature / (sym_number * h * B)
        results['S_rot' ] = (R_Eh * (1 + numpy.log(q_rot)), 'Eh/K')
        results['Cv_rot'] = results['Cp_rot'] = (R_Eh, 'Eh/K')
        results['E_rot' ] = results['H_rot' ] = (R_Eh * temperature, 'Eh')
    else:
        ABC = rot_const * 1e9
        q_rot = ((kB*temperature/h)**1.5 * numpy.pi**.5
                 / (sym_number * numpy.prod(ABC)**.5))
        results['S_rot' ] = (R_Eh * (1.5 + numpy.log(q_rot)), 'Eh/K')
        results['Cv_rot'] = results['Cp_rot'] = (1.5 * R_Eh, 'Eh/K')
        results['E_rot' ] = results['H_rot' ] = (1.5 * R_Eh * temperature, 'Eh')

    # Vibrational part.
    au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * numpy.pi)
    idx = freq.real > 0
    vib_temperature = freq.real[idx] * au2hz * h / kB
    # reduced_temperature
    rt = vib_temperature / max(1e-14, temperature)
    e = numpy.exp(-rt)

    ZPE = R_Eh * .5 * vib_temperature.sum()
    results['ZPE'] = (ZPE, 'Eh')

    results['S_vib' ] = (R_Eh * (rt*e/(1-e) - numpy.log(1-e)).sum(), 'Eh/K')
    results['Cv_vib'] = results['Cp_vib'] = (R_Eh * (e * rt**2/(1-e)**2).sum(), 'Eh/K')
    results['E_vib' ] = results['H_vib' ] = \
            (ZPE + R_Eh * temperature * (rt * e / (1-e)).sum(), 'Eh')

    results['G_elec' ] = (results['H_elec' ][0] - temperature * results['S_elec' ][0], 'Eh')
    results['G_trans'] = (results['H_trans'][0] - temperature * results['S_trans'][0], 'Eh')
    results['G_rot'  ] = (results['H_rot'  ][0] - temperature * results['S_rot'  ][0], 'Eh')
    results['G_vib'  ] = (results['H_vib'  ][0] - temperature * results['S_vib'  ][0], 'Eh')

    def _sum(f):
        keys = ('elec', 'trans', 'rot', 'vib')
        return sum(results.get(f+'_'+key, (0,))[0] for key in keys)
    results['S_tot' ] = (_sum('S' ), 'Eh/K')
    results['Cv_tot'] = (_sum('Cv'), 'Eh/K')
    results['Cp_tot'] = (_sum('Cp'), 'Eh/K')
    results['E_0K' ]  = (E0 + ZPE, 'Eh')
    results['E_tot' ] = (_sum('E'), 'Eh')
    results['H_tot' ] = (_sum('H'), 'Eh')
    results['G_tot' ] = (_sum('G'), 'Eh')

    return results

if __name__ == '__main__':
    pass
                
