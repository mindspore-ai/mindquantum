# import numpy as np
# import matplotlib.pyplot as plt
# from pyscf import gto, scf, fci, mcscf, grad

# from chemqulacs_namd.qse.qse_singlet import QSE
# from chemqulacs_namd.qse import force_utility
# import time
# from quri_parts.core.operator.sparse import get_sparse_matrix
# from quri_parts.qulacs.simulator import evaluate_state_to_vector
# from quri_parts.circuit.noise import NoiseModel, DepolarizingNoise, ThermalRelaxationNoise, BitFlipNoise,GeneralDepolarizingNoise
# from quri_parts.qulacs.estimator import create_qulacs_density_matrix_estimator
# import openfermion as of
# from openfermion import get_fermion_operator
# from pyscf.scf.addons import partial_cholesky_orth_
# from functools import reduce
# from openfermion.utils import hermitian_conjugated
# import os
# from loky import get_reusable_executor
# from mpi4py import MPI

# def creat_H3_plus_ion(r):
#     h1 = np.array([-0.988019/2, 0.0, 0.0])
#     h2 = np.array([0.988019/2, 0.0, 0.0])
#     mid_point = (h1 + h2) / 2
#     angle = np.radians(90)
#     x = r * np.cos(angle)
#     y = r * np.sin(angle)
#     h3 = mid_point + np.array([x, y, 0.0])
#     mol = gto.M(
#         atom=[['H', h1], ['H', h2], ['H', h3]],
#         charge=1,
#         basis='sto-3g',
#         unit='Angstrom'
#     )
#     return mol

# def generate_excited_state(qse, k: int) -> np.ndarray:
#     ref_state = qse.vqeci.opt_states[0]
#     n_qubits = qse.vqeci.n_qubit
#     ref_vector = evaluate_state_to_vector(ref_state).vector
#     eigenvector = qse.eigenvectors[:, k]
#     e_ops = qse.e_op
#     excited_vector = eigenvector[0] * ref_vector
#     op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
#         n_spin_orbitals=qse.vqeci.n_qubit, n_fermions=qse.vqeci.n_electron
#     )
#     for i, e_op in enumerate(e_ops, start=1):
#         qubit_op = op_mapper(e_op)
#         sparse_matrix = get_sparse_matrix(qubit_op, n_qubits)
#         excited_part = sparse_matrix.dot(ref_vector)
#         excited_vector += eigenvector[i] * excited_part
#     norm = np.linalg.norm(excited_vector)
#     if norm < 1e-10:
#         raise ValueError(f"激发态 {k} 的向量范数过小，无法归一化")
#     excited_vector /= norm
#     return excited_vector

# def create_noise_model(noise_types, noise_strengths) -> NoiseModel:
#     noises = []
#     import quri_parts.circuit.gate_names as gate_names
#     noises.append(DepolarizingNoise(0.001,qubit_indices=[],target_gates=[]))
#     noises.append(DepolarizingNoise(0.009,qubit_indices=[],target_gates=[gate_names.CNOT,gate_names.CZ]))
#     noises.append(BitFlipNoise(0.02,qubit_indices=[],target_gates=[]))


#     return NoiseModel(noises)

# def compute_matrix_element(idx, jdx, fermionic_hamiltonian, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths, is_ham=True, force_ferm_op=None):
#     noise_model = create_noise_model(noise_types, noise_strengths)
#     estimator = create_qulacs_density_matrix_estimator(noise_model)
#     if is_ham:
#         if idx == 0:
#             myop_fermi = fermionic_hamiltonian * e_ops[jdx - 1]
#         elif jdx == 0:
#             myop_fermi = hermitian_conjugated(e_ops[idx - 1]) * fermionic_hamiltonian
#         else:
#             myop_fermi = hermitian_conjugated(e_ops[idx - 1]) * fermionic_hamiltonian * e_ops[jdx - 1]
#     else:
#         if idx == 0:
#             myop_fermi = force_ferm_op * e_ops[jdx - 1]
#         elif jdx == 0:
#             myop_fermi = hermitian_conjugated(e_ops[idx - 1]) * force_ferm_op
#         else:
#             myop_fermi = hermitian_conjugated(e_ops[idx - 1]) * force_ferm_op * e_ops[jdx - 1]
#     myop = op_mapper(myop_fermi)
#     val = estimator(myop, bound_circuit).value
#     return idx, jdx, val

# def compute_s_matrix_element(idx, jdx, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths):
#     noise_model = create_noise_model(noise_types, noise_strengths)
#     estimator = create_qulacs_density_matrix_estimator(noise_model)
#     if idx == 0:
#         myop_fermi = e_ops[jdx - 1]
#     elif jdx == 0:
#         myop_fermi = hermitian_conjugated(e_ops[idx - 1])
#     else:
#         myop_fermi = hermitian_conjugated(e_ops[idx - 1]) * e_ops[jdx - 1]
#     myop = op_mapper(myop_fermi)
#     val = estimator(myop, bound_circuit).value
#     return idx, jdx, val

# def compute_vqe_both(mol, noise_types, noise_strengths):
#     start_time = time.time()
#     mf = scf.RHF(mol)
#     mf.run()
#     print(f"RHF computation time: {time.time() - start_time:.2f} seconds")

#     start_time = time.time()
#     from chemqulacs_namd.vqe.vqemcscf import VQECASCI
#     from chemqulacs_namd.vqe.vqeci import Ansatz
#     vqe_casci = VQECASCI(mf, 3, 2, singlet_excitation=False, ansatz=Ansatz.UCCSD, use_singles=False)
#     vqe_casci.kernel()
#     print(f"VQE-CASCI kernel time: {time.time() - start_time:.2f} seconds")

#     start_time = time.time()
#     qse = QSE(vqe_casci.fcisolver)
#     qse.gen_excitation_operators("ee", 2)
#     qse.solve()
#     print(f"QSE solve time (no noise): {time.time() - start_time:.2f} seconds")

#     e_tot = qse.eigenvalues.copy()
#     e_tot[0] = vqe_casci.fcisolver.energies[0]

#     hcore_mo = np.einsum('pi,pq,qj->ij', mf.mo_coeff.conj(), mf.get_hcore(), mf.mo_coeff)
#     tei_ao = mol.intor('int2e')
#     tei_mo = of.general_basis_change(tei_ao, mf.mo_coeff, (1, 0, 1, 0)).transpose(0, 2, 3, 1)
#     force_ops = force_utility.gradient_mo_operator(mol, mf.mo_coeff, hcore_mo, tei_mo, with_pulay=True)

#     start_time = time.time()
#     grad_hellmann = np.zeros((3, 3, 3))
#     n_qubits = qse.vqeci.n_qubit
#     op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
#         n_spin_orbitals=n_qubits, n_fermions=qse.vqeci.n_electron
#     )
#     for s in range(3):
#         state_vector = generate_excited_state(qse, s)
#         for f_idx, force_op in enumerate(force_ops):
#             atom = f_idx // 3
#             xyz = f_idx % 3
#             if xyz == 2:  # z-axis, set to zero for 2D system
#                 grad_hellmann[s, atom, xyz] = 0.0
#                 continue
#             fermion_op = get_fermion_operator(force_op)
#             qubit_op = op_mapper(fermion_op)
#             sparse = get_sparse_matrix(qubit_op, n_qubits)
#             exp_val = np.real(np.dot(state_vector.conj().T, sparse @ state_vector))
#             grad_hellmann[s, atom, xyz] = exp_val
#     print(f"No noise Hellmann-Feynman gradient computation time: {time.time() - start_time:.2f} seconds")

#     # Noisy part
#     start_time = time.time()
#     import chemqulacs_namd.vqe.vqemcscf_noisy as vqemcscf_noisy
#     from chemqulacs_namd.vqe.vqeci_noisy import Ansatz
#     noise_kwargs = {
#     "noise_type":['depol','bitflip']
#     ,
#     "noise_prob":[0.000827,0.925]
#     }
#     vqe_casci_noisy=vqemcscf_noisy.VQECASCI(mf, 3, 2, singlet_excitation=False, ansatz=Ansatz.UCCSD, use_singles=False,noise_kwargs=noise_kwargs)
#     vqe_casci_noisy.kernel()
#     cas_state = vqe_casci_noisy.fcisolver.opt_states[0]
#     cas_hamiltonian = vqe_casci_noisy.fcisolver.qubit_hamiltonian
#     bound_circuit = cas_state
#     noise_model = create_noise_model(noise_types, noise_strengths)
#     print('noisy qse'+str(noise_model))
#     estimator = create_qulacs_density_matrix_estimator(noise_model)
#     noisy_energy = estimator(cas_hamiltonian, bound_circuit).value
#     print(f"Noisy energy estimation time: {time.time() - start_time:.2f} seconds")

#     noperators = len(qse.e_op)
#     e_ops = qse.e_op
#     fermionic_hamiltonian = qse.vqeci.fermionic_hamiltonian
#     n = noperators + 1

#     executor = get_reusable_executor(max_workers=8)

#     # Parallelize Hamiltonian matrix construction
#     start_time = time.time()
#     hamiltonian = np.zeros((n, n), dtype=np.complex128)
#     hamiltonian[0, 0] = noisy_energy
#     tasks = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
#     futures = [executor.submit(compute_matrix_element, idx, jdx, fermionic_hamiltonian, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths, is_ham=True) for idx, jdx in tasks]
#     for future in futures:
#         idx, jdx, val = future.result()
#         hamiltonian[idx, jdx] = val
#         hamiltonian[jdx, idx] = np.conj(val)
#     print(f"Noisy Hamiltonian matrix construction time: {time.time() - start_time:.2f} seconds")

#     # Parallelize S matrix construction
#     start_time = time.time()
#     S = np.zeros((n, n), dtype=np.complex128)
#     S[0, 0] = 1.0
#     tasks_s = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
#     futures_s = [executor.submit(compute_s_matrix_element, idx, jdx, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths) for idx, jdx in tasks_s]
#     for future in futures_s:
#         idx, jdx, val = future.result()
#         S[idx, jdx] = val
#         S[jdx, idx] = np.conj(val)
#     print(f"Noisy S matrix construction time: {time.time() - start_time:.2f} seconds")

#     start_time = time.time()
#     threshold = 1.0e-8
#     cholesky_threshold = 1.0e-08
#     x = partial_cholesky_orth_(S, canthr=threshold, cholthr=cholesky_threshold)
#     xhx = reduce(np.dot, (x.T.conj(), hamiltonian, x))
#     e_tot_noisy, eigenvectors_noisy = np.linalg.eigh(xhx)
#     eigenvectors_noisy = np.dot(x, eigenvectors_noisy)
#     print(f"Noisy eigenvalue solving time: {time.time() - start_time:.2f} seconds")

#     # Parallelize noisy Hellmann-Feynman gradient computation
#     start_time = time.time()
#     grad_hellmann_noisy = np.zeros((3, 3, 3))

#     # Precompute all G matrices for each relevant force_op
#     g_matrices = {}
#     tasks_g_base = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
#     for f_idx, force_op in enumerate(force_ops):
#         atom = f_idx // 3
#         xyz = f_idx % 3
#         if xyz == 2:
#             continue  # Skip z-component
#         force_ferm_op = get_fermion_operator(force_op)
#         qubit_op_00 = op_mapper(force_ferm_op)
#         g_00 = estimator(qubit_op_00, bound_circuit).value

#         G = np.zeros((n, n), dtype=np.complex128)
#         G[0, 0] = g_00

#         futures_g = [executor.submit(compute_matrix_element, idx, jdx, fermionic_hamiltonian, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths, is_ham=False, force_ferm_op=force_ferm_op) for idx, jdx in tasks_g_base]
#         for future in futures_g:
#             idx, jdx, val = future.result()
#             G[idx, jdx] = val
#             G[jdx, idx] = np.conj(val)

#         g_matrices[f_idx] = G

#     # Now for each state s, compute the expectation values using precomputed G
#     for s in range(3):
#         eigenvector = eigenvectors_noisy[:, s]
#         for f_idx, force_op in enumerate(force_ops):
#             atom = f_idx // 3
#             xyz = f_idx % 3
#             if xyz == 2:
#                 grad_hellmann_noisy[s, atom, xyz] = 0.0
#                 continue
#             G = g_matrices[f_idx]
#             exp_val = np.real(eigenvector.conj().T @ G @ eigenvector)
#             grad_hellmann_noisy[s, atom, xyz] = exp_val
#     print(f"Noisy Hellmann-Feynman gradient computation time: {time.time() - start_time:.2f} seconds")

#     return e_tot[:3], grad_hellmann, e_tot_noisy[:3], grad_hellmann_noisy

# if __name__ == "__main__":
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()

#     r_values_all = np.arange(0, 2.0, 0.08)
#     r_splits = np.array_split(r_values_all, size)
#     local_r_values = r_splits[rank]

#     local_no_noise_energies = []
#     local_noisy_energies = []
#     local_casci_energies = []
#     local_no_noise_grad_hellmanns = []
#     local_noisy_grad_hellmanns = []
#     local_casci_grads = []

#     noise_types = ['depol','bitflip']
#     noise_strengths = {'depol_1q': 0.0001, 'depol_2q': 0.002,'p00':0.925,'p11':0.925}

#     for r in local_r_values:
#         mol = creat_H3_plus_ion(r)
#         print(f"Rank {rank}: Computing for r = {r:.2f} Å")
#         no_noise_energy, no_noise_grad_hellmann, noisy_energy, noisy_grad_hellmann = compute_vqe_both(mol, noise_types, noise_strengths)
#         local_no_noise_energies.append((r, no_noise_energy))
#         local_no_noise_grad_hellmanns.append((r, no_noise_grad_hellmann))
#         local_noisy_energies.append((r, noisy_energy))
#         local_noisy_grad_hellmanns.append((r, noisy_grad_hellmann))
#         mf = scf.RHF(mol).run()
#         mc = mcscf.CASCI(mf, 3, 2)
#         mc.fcisolver = fci.direct_spin0.FCI(mol)
#         mc.fcisolver.nroots = 3
#         mc.kernel()
#         casci_energy = mc.e_tot
#         casci_grad = np.zeros((3, 3, 3))
#         for i in range(3):
#             g = grad.casci.Gradients(mc)
#             cas_grad = g.kernel(state=i)
#             cas_grad[:, 2] = 0.0
#             casci_grad[i] = cas_grad
#         local_casci_energies.append((r, casci_energy))
#         local_casci_grads.append((r, casci_grad))

#         # Save gradients locally (filenames are unique per r)
#         os.makedirs('h3plus_grad_noisy_depol_bitf', exist_ok=True)
#         np.save(f'h3plus_grad_noisy_depol_bitf/casci_grad_r_{r:.2f}.npy', casci_grad)
#         np.save(f'h3plus_grad_noisy_depol_bitf/no_noise_grad_hellmann_r_{r:.2f}.npy', no_noise_grad_hellmann)
#         np.save(f'h3plus_grad_noisy_depol_bitf/noisy_grad_hellmann_r_{r:.2f}.npy', noisy_grad_hellmann)

#     # Gather all data to rank 0
#     all_no_noise_energies = comm.gather(local_no_noise_energies, root=0)
#     all_noisy_energies = comm.gather(local_noisy_energies, root=0)
#     all_casci_energies = comm.gather(local_casci_energies, root=0)
#     all_no_noise_grad_hellmanns = comm.gather(local_no_noise_grad_hellmanns, root=0)
#     all_noisy_grad_hellmanns = comm.gather(local_noisy_grad_hellmanns, root=0)
#     all_casci_grads = comm.gather(local_casci_grads, root=0)

#     if rank == 0:
#         # Flatten and sort by r
#         no_noise_energies = sorted([item for sublist in all_no_noise_energies for item in sublist], key=lambda x: x[0])
#         noisy_energies = sorted([item for sublist in all_noisy_energies for item in sublist], key=lambda x: x[0])
#         casci_energies = sorted([item for sublist in all_casci_energies for item in sublist], key=lambda x: x[0])
#         no_noise_grad_hellmanns = sorted([item for sublist in all_no_noise_grad_hellmanns for item in sublist], key=lambda x: x[0])
#         noisy_grad_hellmanns = sorted([item for sublist in all_noisy_grad_hellmanns for item in sublist], key=lambda x: x[0])
#         casci_grads = sorted([item for sublist in all_casci_grads for item in sublist], key=lambda x: x[0])

#         # Extract values
#         no_noise_energies = [ens for r, ens in no_noise_energies]
#         noisy_energies = [ens for r, ens in noisy_energies]
#         casci_energies = [ens for r, ens in casci_energies]
#         no_noise_grad_hellmanns = [g for r, g in no_noise_grad_hellmanns]
#         noisy_grad_hellmanns = [g for r, g in noisy_grad_hellmanns]
#         casci_grads = [g for r, g in casci_grads]

#         with open('no_noise_e_data.txt', 'w') as f:
#             f.write('r energies\n')
#             for r, ens in zip(r_values_all, no_noise_energies):
#                 f.write(f"{r:.4f} {' '.join(f'{e:.8f}' for e in ens)}\n")

#         with open('noisy_e_data.txt', 'w') as f:
#             f.write('r energies\n')
#             for r, ens in zip(r_values_all, noisy_energies):
#                 f.write(f"{r:.4f} {' '.join(f'{e:.8f}' for e in ens)}\n")

#         with open('casci_data.txt', 'w') as f:
#             f.write('r energies\n')
#             for r, ens in zip(r_values_all, casci_energies):
#                 f.write(f"{r:.4f} {' '.join(f'{e:.8f}' for e in ens)}\n")

#         plt.figure(figsize=(12, 8))
#         colors = ['blue', 'green', 'red']
#         for i in range(3):
#             no_noise_y = [ens[i] for ens in no_noise_energies]
#             noisy_y = [ens[i] for ens in noisy_energies]
#             casci_y = [ens[i] for ens in casci_energies]
#             plt.plot(r_values_all, no_noise_y, color=colors[i], linestyle='--', marker='x', label=f'No Noise state {i}')
#             plt.plot(r_values_all, noisy_y, color=colors[i], linestyle=':', marker='o', label=f'Noisy state {i}')
#             plt.plot(r_values_all, casci_y, color=colors[i], linestyle='-', marker=None, label=f'CASCI state {i}')

#         plt.xlabel('r (Å)')
#         plt.ylabel('Energy (Hartree)')
#         plt.title('PES Comparison: No Noise vs Noisy vs CASCI for H3+')
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig('pes_comparison_noisy_bitf.svg')

#         plt.figure(figsize=(12, 8))
#         for i in range(3):
#             no_noise_grad_y = [g[i, 2, 1] for g in no_noise_grad_hellmanns]
#             noisy_grad_y = [g[i, 2, 1] for g in noisy_grad_hellmanns]
#             casci_grad_y = [g[i, 2, 1] for g in casci_grads]
#             plt.plot(r_values_all, no_noise_grad_y, color=colors[i], linestyle='', marker='x', label=f'No Noise Hellmann state {i}')
#             plt.plot(r_values_all, noisy_grad_y, color=colors[i], linestyle=':', marker='o', label=f'Noisy Hellmann state {i}')
#             plt.plot(r_values_all, casci_grad_y, color=colors[i], linestyle='-', marker=None, label=f'CASCI state {i}')

#         plt.xlabel('r (Å)')
#         plt.ylabel('Gradient on third H atom (y-component, Hartree/Å)')
#         plt.title('Hellmann-Feynman Gradient Comparison: No Noise vs Noisy vs CASCI for H3+')
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig('gradient_comparison_noisy_bitf.svg')


import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci, mcscf, grad

from chemqulacs_namd.qse.qse_singlet import QSE
from chemqulacs_namd.qse import force_utility
import time
from quri_parts.core.operator.sparse import get_sparse_matrix
from quri_parts.qulacs.simulator import evaluate_state_to_vector
from quri_parts.circuit.noise import NoiseModel, DepolarizingNoise, ThermalRelaxationNoise, BitFlipNoise,GeneralDepolarizingNoise
from quri_parts.qulacs.estimator import create_qulacs_density_matrix_estimator
import openfermion as of
from openfermion import get_fermion_operator
from pyscf.scf.addons import partial_cholesky_orth_
from functools import reduce
from openfermion.utils import hermitian_conjugated
import os
from loky import get_reusable_executor
from mpi4py import MPI

def creat_H3_plus_ion(r):
    h1 = np.array([-0.988019/2, 0.0, 0.0])
    h2 = np.array([0.988019/2, 0.0, 0.0])
    mid_point = (h1 + h2) / 2
    angle = np.radians(90)
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    h3 = mid_point + np.array([x, y, 0.0])
    mol = gto.M(
        atom=[['H', h1], ['H', h2], ['H', h3]],
        charge=1,
        basis='sto-3g',
        unit='Angstrom'
    )
    return mol

def generate_excited_state(qse, k: int) -> np.ndarray:
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

def create_noise_model(noise_types, noise_strengths) -> NoiseModel:
    noises = []
    import quri_parts.circuit.gate_names as gate_names
    noises.append(DepolarizingNoise(0.001,qubit_indices=[],target_gates=[]))
    noises.append(DepolarizingNoise(0.009,qubit_indices=[],target_gates=[gate_names.CNOT,gate_names.CZ]))
    # noises.append(BitFlipNoise(0.02,qubit_indices=[],target_gates=[]))


    return NoiseModel(noises)

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
def compute_energies(mol, noise_types=None, noise_strengths=None, noisy=False):
    mf = scf.RHF(mol).run()
    start_time_energy = time.time()
    if noisy:
        noise_kwargs = {
            "noise_type": noise_types,
            "noise_prob": [noise_strengths['depol_1q'], noise_strengths['depol_2q']]
        }
        from chemqulacs_namd.vqe.vqemcscf_noisy import VQECASCI as VQECASCI_noisy
        from chemqulacs_namd.vqe.vqeci_noisy import Ansatz as Ansatz_noisy
        vqe_casci = VQECASCI_noisy(mf, 3, 2, singlet_excitation=True, ansatz=Ansatz_noisy.UCCSD, use_singles=False, noise_kwargs=noise_kwargs)
        vqe_casci.kernel()
        qse = QSE(vqe_casci.fcisolver)
        qse.gen_excitation_operators("ee", 2)
        qse.solve()
        # Noisy computation
        cas_state = vqe_casci.fcisolver.opt_states[0]
        cas_hamiltonian = vqe_casci.fcisolver.qubit_hamiltonian
        bound_circuit = cas_state
        noise_model = create_noise_model(noise_types, noise_strengths)
        estimator = create_qulacs_density_matrix_estimator(noise_model)
        noisy_energy = estimator(cas_hamiltonian, bound_circuit).value
        noperators = len(qse.e_op)
        e_ops = qse.e_op
        fermionic_hamiltonian = qse.vqeci.fermionic_hamiltonian
        n = noperators + 1
        executor = get_reusable_executor(max_workers=4)
        # Hamiltonian matrix
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
        # S matrix
        S = np.zeros((n, n), dtype=np.complex128)
        S[0, 0] = 1.0
        tasks_s = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
        futures_s = [executor.submit(compute_s_matrix_element, idx, jdx, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths) for idx, jdx in tasks_s]
        for future in futures_s:
            idx, jdx, val = future.result()
            S[idx, jdx] = val
            S[jdx, idx] = np.conj(val)
        threshold = 1.0e-8
        cholesky_threshold = 1.0e-08
        x = partial_cholesky_orth_(S, canthr=threshold, cholthr=cholesky_threshold)
        xhx = reduce(np.dot, (x.T.conj(), hamiltonian, x))
        e_tot_noisy, eigenvectors_noisy = np.linalg.eigh(xhx)
        eigenvectors_noisy = np.dot(x, eigenvectors_noisy)
        print(f"Time for 有噪声vqeci+qse计算激发态能量: {time.time() - start_time_energy:.2f} seconds")
        return e_tot_noisy[:3], eigenvectors_noisy
    else:
        from chemqulacs_namd.vqe.vqemcscf import VQECASCI
        from chemqulacs_namd.vqe.vqeci import Ansatz
        vqe_casci = VQECASCI(mf, 3, 2, singlet_excitation=True, ansatz=Ansatz.UCCSD, use_singles=False)
        vqe_casci.kernel()
        qse = QSE(vqe_casci.fcisolver)
        qse.gen_excitation_operators("ee", 2)
        qse.solve()
        e_tot = qse.eigenvalues
        e_tot[0] = vqe_casci.fcisolver.energies[0]
        print(f"Time for 无噪声vqeci+qse计算激发态能量: {time.time() - start_time_energy:.2f} seconds")
        return e_tot[:3], None

def compute_vqe_both(mol, noise_types, noise_strengths):
    start_time = time.time()
    mf = scf.RHF(mol)
    mf.run()
    print(f"RHF computation time: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    from chemqulacs_namd.vqe.vqemcscf import VQECASCI
    from chemqulacs_namd.vqe.vqeci import Ansatz
    vqe_casci = VQECASCI(mf, 3, 2, singlet_excitation=True, ansatz=Ansatz.UCCSD, use_singles=False)
    vqe_casci.kernel()
    print(f"VQE-CASCI kernel time: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    qse = QSE(vqe_casci.fcisolver)
    qse.gen_excitation_operators("ee", 2)
    qse.solve()
    print(f"QSE solve time (no noise): {time.time() - start_time:.2f} seconds")

    e_tot = qse.eigenvalues.copy()
    e_tot[0] = vqe_casci.fcisolver.energies[0]

    hcore_mo = np.einsum('pi,pq,qj->ij', mf.mo_coeff.conj(), mf.get_hcore(), mf.mo_coeff)
    tei_ao = mol.intor('int2e')
    tei_mo = of.general_basis_change(tei_ao, mf.mo_coeff, (1, 0, 1, 0)).transpose(0, 2, 3, 1)
    force_ops = force_utility.gradient_mo_operator(mol, mf.mo_coeff, hcore_mo, tei_mo, with_pulay=True)

    start_time = time.time()
    grad_hellmann = np.zeros((3, 3, 3))
    n_qubits = qse.vqeci.n_qubit
    op_mapper = qse.vqeci.fermion_qubit_mapping.get_of_operator_mapper(
        n_spin_orbitals=n_qubits, n_fermions=qse.vqeci.n_electron
    )
    for s in range(3):
        state_vector = generate_excited_state(qse, s)
        for f_idx, force_op in enumerate(force_ops):
            atom = f_idx // 3
            xyz = f_idx % 3
            if xyz == 2:  # z-axis, set to zero for 2D system
                grad_hellmann[s, atom, xyz] = 0.0
                continue
            fermion_op = get_fermion_operator(force_op)
            qubit_op = op_mapper(fermion_op)
            sparse = get_sparse_matrix(qubit_op, n_qubits)
            exp_val = np.real(np.dot(state_vector.conj().T, sparse @ state_vector))
            grad_hellmann[s, atom, xyz] = exp_val
    print(f"No noise Hellmann-Feynman gradient computation time: {time.time() - start_time:.2f} seconds")

    # Noisy part
    start_time = time.time()
    import chemqulacs_namd.vqe.vqemcscf_noisy as vqemcscf_noisy
    from chemqulacs_namd.vqe.vqeci_noisy import Ansatz
    noise_kwargs = {
    "noise_type":['depol','bitflip']
    ,
    "noise_prob":[0.000827,0.925]
    }
    vqe_casci_noisy=vqemcscf_noisy.VQECASCI(mf, 3, 2, singlet_excitation=True, ansatz=Ansatz.UCCSD, use_singles=False,noise_kwargs=noise_kwargs)
    vqe_casci_noisy.kernel()
    cas_state = vqe_casci_noisy.fcisolver.opt_states[0]
    cas_hamiltonian = vqe_casci_noisy.fcisolver.qubit_hamiltonian
    bound_circuit = cas_state
    noise_model = create_noise_model(noise_types, noise_strengths)
    print('noisy qse'+str(noise_model))
    estimator = create_qulacs_density_matrix_estimator(noise_model)
    noisy_energy = estimator(cas_hamiltonian, bound_circuit).value
    print(f"Noisy energy estimation time: {time.time() - start_time:.2f} seconds")

    noperators = len(qse.e_op)
    e_ops = qse.e_op
    fermionic_hamiltonian = qse.vqeci.fermionic_hamiltonian
    n = noperators + 1

    executor = get_reusable_executor(max_workers=8)

    # Parallelize Hamiltonian matrix construction
    start_time = time.time()
    hamiltonian = np.zeros((n, n), dtype=np.complex128)
    hamiltonian[0, 0] = noisy_energy
    tasks = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
    futures = [executor.submit(compute_matrix_element, idx, jdx, fermionic_hamiltonian, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths, is_ham=True) for idx, jdx in tasks]
    for future in futures:
        idx, jdx, val = future.result()
        hamiltonian[idx, jdx] = val
        hamiltonian[jdx, idx] = np.conj(val)
    print(f"Noisy Hamiltonian matrix construction time: {time.time() - start_time:.2f} seconds")

    # Parallelize S matrix construction
    start_time = time.time()
    S = np.zeros((n, n), dtype=np.complex128)
    S[0, 0] = 1.0
    tasks_s = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
    futures_s = [executor.submit(compute_s_matrix_element, idx, jdx, e_ops, op_mapper, bound_circuit, noise_types, noise_strengths) for idx, jdx in tasks_s]
    for future in futures_s:
        idx, jdx, val = future.result()
        S[idx, jdx] = val
        S[jdx, idx] = np.conj(val)
    print(f"Noisy S matrix construction time: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    threshold = 1.0e-8
    cholesky_threshold = 1.0e-08
    x = partial_cholesky_orth_(S, canthr=threshold, cholthr=cholesky_threshold)
    xhx = reduce(np.dot, (x.T.conj(), hamiltonian, x))
    e_tot_noisy, eigenvectors_noisy = np.linalg.eigh(xhx)
    e_tot_noisy[0]=vqe_casci.fcisolver.energies[0]
    eigenvectors_noisy = np.dot(x, eigenvectors_noisy)
    print(f"Noisy eigenvalue solving time: {time.time() - start_time:.2f} seconds")

    # Parallelize noisy Hellmann-Feynman gradient computation
    start_time = time.time()
    grad_hellmann_noisy = np.zeros((3, 3, 3))

    # Precompute all G matrices for each relevant force_op
    g_matrices = {}
    tasks_g_base = [(idx, jdx) for idx in range(n) for jdx in range(idx, n) if not (idx == 0 and jdx == 0)]
    for f_idx, force_op in enumerate(force_ops):
        atom = f_idx // 3
        xyz = f_idx % 3
        if xyz == 2:
            continue  # Skip z-component
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

    # Now for each state s, compute the expectation values using precomputed G
    for s in range(3):
        eigenvector = eigenvectors_noisy[:, s]
        for f_idx, force_op in enumerate(force_ops):
            atom = f_idx // 3
            xyz = f_idx % 3
            if xyz == 2:
                grad_hellmann_noisy[s, atom, xyz] = 0.0
                continue
            G = g_matrices[f_idx]
            exp_val = np.real(eigenvector.conj().T @ G @ eigenvector)
            grad_hellmann_noisy[s, atom, xyz] = exp_val
    print(f"Noisy Hellmann-Feynman gradient computation time: {time.time() - start_time:.2f} seconds")

    return e_tot[:3], grad_hellmann, e_tot_noisy[:3], grad_hellmann_noisy

def compute_fdms(mol, noise_types, noise_strengths, noisy=False):
    deltas = [0.1]  # Fixed delta as per request
    mf = scf.RHF(mol).run()
    coords = mol.atom_coords()
    grad_fdms = np.zeros((3, 3, 3))  # 3 states, 3 atoms, 3 coords
    start_time = time.time()
    for a in range(3):
        for c in range(3):
            if c == 2:
                continue  # Skip z-component
            coords_fwd = coords.copy()
            coords_fwd[a, c] += deltas[0]
            mol_fwd = mol.copy()
            mol_fwd.set_geom_(coords_fwd, unit='Ang')
            mol_fwd.build()
            e_fwd = compute_energies(mol_fwd, noise_types, noise_strengths, noisy)[0]

            coords_bwd = coords.copy()
            coords_bwd[a, c] -= deltas[0]
            mol_bwd = mol.copy()
            mol_bwd.set_geom_(coords_bwd, unit='Ang')
            mol_bwd.build()
            e_bwd = compute_energies(mol_bwd, noise_types, noise_strengths, noisy)[0]

            grad_fdms[:, a, c] = (e_fwd - e_bwd) / (2 * deltas[0])
    print(f"{'Noisy' if noisy else 'No noise'} FDM gradient computation time: {time.time() - start_time:.2f} seconds")
    return grad_fdms

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    r_values_all = np.arange(0, 2.0, 0.08)
    r_splits = np.array_split(r_values_all, size)
    local_r_values = r_splits[rank]

    local_no_noise_energies = []
    local_noisy_energies = []
    local_casci_energies = []
    local_no_noise_grad_hellmanns = []
    local_noisy_grad_hellmanns = []
    local_no_noise_grad_fdms = []
    local_noisy_grad_fdms = []
    local_casci_grads = []

    noise_types = ['depol','bitflip']
    noise_strengths = {'depol_1q': 0.001, 'depol_2q': 0.009,'bitfip':0.02}

    for r in local_r_values:
        mol = creat_H3_plus_ion(r)
        print(f"Rank {rank}: Computing for r = {r:.2f} Å")
        no_noise_energy, no_noise_grad_hellmann, noisy_energy, noisy_grad_hellmann = compute_vqe_both(mol, noise_types, noise_strengths)
        local_no_noise_energies.append((r, no_noise_energy))
        local_no_noise_grad_hellmanns.append((r, no_noise_grad_hellmann))
        local_noisy_energies.append((r, noisy_energy))
        local_noisy_grad_hellmanns.append((r, noisy_grad_hellmann))
        mf = scf.RHF(mol).run()
        mc = mcscf.CASCI(mf, 3, 2)
        mc.fcisolver = fci.direct_spin0.FCI(mol)
        mc.fcisolver.nroots = 3
        mc.kernel()
        casci_energy = mc.e_tot
        casci_grad = np.zeros((3, 3, 3))
        for i in range(3):
            g = grad.casci.Gradients(mc)
            cas_grad = g.kernel(state=i)
            cas_grad[:, 2] = 0.0
            casci_grad[i] = cas_grad
        local_casci_energies.append((r, casci_energy))
        local_casci_grads.append((r, casci_grad))

        # Compute FDM gradients
        no_noise_grad_fdm = compute_fdms(mol, noise_types, noise_strengths, noisy=False)
        noisy_grad_fdm = compute_fdms(mol, noise_types, noise_strengths, noisy=True)
        local_no_noise_grad_fdms.append((r, no_noise_grad_fdm))
        local_noisy_grad_fdms.append((r, noisy_grad_fdm))

        # Save gradients locally (filenames are unique per r)
        os.makedirs('h3plus_grad_noisy_depol_bitf', exist_ok=True)
        np.save(f'h3plus_grad_noisy_depol_bitf/casci_grad_r_{r:.2f}.npy', casci_grad)
        np.save(f'h3plus_grad_noisy_depol_bitf/no_noise_grad_hellmann_r_{r:.2f}.npy', no_noise_grad_hellmann)
        np.save(f'h3plus_grad_noisy_depol_bitf/noisy_grad_hellmann_r_{r:.2f}.npy', noisy_grad_hellmann)
        np.save(f'h3plus_grad_noisy_depol_bitf/no_noise_grad_fdm_delta_0.1_r_{r:.2f}.npy', no_noise_grad_fdm)
        np.save(f'h3plus_grad_noisy_depol_bitf/noisy_grad_fdm_delta_0.1_r_{r:.2f}.npy', noisy_grad_fdm)

    # Gather all data to rank 0
    all_no_noise_energies = comm.gather(local_no_noise_energies, root=0)
    all_noisy_energies = comm.gather(local_noisy_energies, root=0)
    all_casci_energies = comm.gather(local_casci_energies, root=0)
    all_no_noise_grad_hellmanns = comm.gather(local_no_noise_grad_hellmanns, root=0)
    all_noisy_grad_hellmanns = comm.gather(local_noisy_grad_hellmanns, root=0)
    all_no_noise_grad_fdms = comm.gather(local_no_noise_grad_fdms, root=0)
    all_noisy_grad_fdms = comm.gather(local_noisy_grad_fdms, root=0)
    all_casci_grads = comm.gather(local_casci_grads, root=0)

    if rank == 0:
        # Flatten and sort by r
        no_noise_energies = sorted([item for sublist in all_no_noise_energies for item in sublist], key=lambda x: x[0])
        noisy_energies = sorted([item for sublist in all_noisy_energies for item in sublist], key=lambda x: x[0])
        casci_energies = sorted([item for sublist in all_casci_energies for item in sublist], key=lambda x: x[0])
        no_noise_grad_hellmanns = sorted([item for sublist in all_no_noise_grad_hellmanns for item in sublist], key=lambda x: x[0])
        noisy_grad_hellmanns = sorted([item for sublist in all_noisy_grad_hellmanns for item in sublist], key=lambda x: x[0])
        no_noise_grad_fdms = sorted([item for sublist in all_no_noise_grad_fdms for item in sublist], key=lambda x: x[0])
        noisy_grad_fdms = sorted([item for sublist in all_noisy_grad_fdms for item in sublist], key=lambda x: x[0])
        casci_grads = sorted([item for sublist in all_casci_grads for item in sublist], key=lambda x: x[0])

        # Extract values
        no_noise_energies = [ens for r, ens in no_noise_energies]
        noisy_energies = [ens for r, ens in noisy_energies]
        casci_energies = [ens for r, ens in casci_energies]
        no_noise_grad_hellmanns = [g for r, g in no_noise_grad_hellmanns]
        noisy_grad_hellmanns = [g for r, g in noisy_grad_hellmanns]
        no_noise_grad_fdms = [g for r, g in no_noise_grad_fdms]
        noisy_grad_fdms = [g for r, g in noisy_grad_fdms]
        casci_grads = [g for r, g in casci_grads]

        with open('no_noise_e_data.txt', 'w') as f:
            f.write('r energies\n')
            for r, ens in zip(r_values_all, no_noise_energies):
                f.write(f"{r:.4f} {' '.join(f'{e:.8f}' for e in ens)}\n")

        with open('noisy_e_data.txt', 'w') as f:
            f.write('r energies\n')
            for r, ens in zip(r_values_all, noisy_energies):
                f.write(f"{r:.4f} {' '.join(f'{e:.8f}' for e in ens)}\n")

        with open('casci_data.txt', 'w') as f:
            f.write('r energies\n')
            for r, ens in zip(r_values_all, casci_energies):
                f.write(f"{r:.4f} {' '.join(f'{e:.8f}' for e in ens)}\n")

        plt.figure(figsize=(12, 8))
        colors = ['blue', 'green', 'red']
        for i in range(3):
            no_noise_y = [ens[i] for ens in no_noise_energies]
            noisy_y = [ens[i] for ens in noisy_energies]
            casci_y = [ens[i] for ens in casci_energies]
            plt.plot(r_values_all, no_noise_y, color=colors[i], linestyle='--', marker='x', label=f'No Noise state {i}')
            plt.plot(r_values_all, noisy_y, color=colors[i], linestyle=':', marker='o', label=f'Noisy state {i}')
            plt.plot(r_values_all, casci_y, color=colors[i], linestyle='-', marker=None, label=f'CASCI state {i}')

        plt.xlabel('r (Å)')
        plt.ylabel('Energy (Hartree)')
        plt.title('PES Comparison: No Noise vs Noisy vs CASCI for H3+')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('pes_comparison_noisy_bitflip.svg')

        plt.figure(figsize=(12, 8))
        for i in range(3):
            no_noise_grad_hellmann_y = [g[i, 2, 1] for g in no_noise_grad_hellmanns]
            noisy_grad_hellmann_y = [g[i, 2, 1] for g in noisy_grad_hellmanns]
            no_noise_grad_fdm_y = [g[i, 2, 1] for g in no_noise_grad_fdms]
            noisy_grad_fdm_y = [g[i, 2, 1] for g in noisy_grad_fdms]
            casci_grad_y = [g[i, 2, 1] for g in casci_grads]
            plt.plot(r_values_all, no_noise_grad_hellmann_y, color=colors[i], linestyle='', marker='x', label=f'No Noise Hellmann state {i}')
            plt.plot(r_values_all, noisy_grad_hellmann_y, color=colors[i], linestyle=':', marker='o', label=f'Noisy Hellmann state {i}')
            plt.plot(r_values_all, no_noise_grad_fdm_y, color=colors[i], linestyle='', marker='^', label=f'No Noise FDM state {i}')
            plt.plot(r_values_all, noisy_grad_fdm_y, color=colors[i], linestyle=':', marker='^', label=f'Noisy FDM state {i}')
            plt.plot(r_values_all, casci_grad_y, color=colors[i], linestyle='-', marker=None, label=f'CASCI state {i}')

        plt.xlabel('r (Å)')
        plt.ylabel('Gradient on third H atom (y-component, Hartree/Å)')
        plt.title('Gradient Comparison: No Noise vs Noisy (Hellmann & FDM) vs CASCI for H3+')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('gradient_comparison_noisy_with_fdm_bitflip.svg')