import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci
from pyscf.scf.addons import project_mo_nr2nr
from chemqulacs_namd.vqe.vqemcscf import VQECASCI
from chemqulacs_namd.vqe.vqeci import Ansatz
from chemqulacs_namd.qse.qse_singlet_ee import QSE_exact
import time
import os

def filter_triplets(energies, threshold=0.0016):
    """
    Filter an energy array to identify triplets and keep singlets and middle triplet values.
    
    Args:
        energies (list): List of energy values (typically 15 values).
        threshold (float): Threshold for considering values equal (default: 0.0016).
    
    Returns:
        list: Filtered array with singlets and middle values of triplets.
    """
    n = len(energies)
    triplet_indices = set()  # Track indices that belong to triplets
    triplets_middle = []  # Store middle values of triplets with their indices
    
    # Scan for triplets
    for i in range(n - 2):  # Need i+2, so stop at n-3
        if i not in triplet_indices:
            if abs(energies[i] - energies[i + 1]) < threshold:
                # Check for Case 1: two equal, next different
                if i + 2 < n and abs(energies[i + 1] - energies[i + 2]) > threshold:
                    if i > 0:  # Ensure i-1 exists
                        triplet_indices.add(i - 1)
                        triplet_indices.add(i)
                        triplet_indices.add(i + 1)
                        triplets_middle.append((i, energies[i]))  # Middle value is E_i
                # Check for Case 2: three equal
                elif i + 2 < n and abs(energies[i + 1] - energies[i + 2]) < threshold:
                    triplet_indices.add(i)
                    triplet_indices.add(i + 1)
                    triplet_indices.add(i + 2)
                    triplets_middle.append((i + 1, energies[i + 1]))  # Middle value is E_{i+1}
    
    # Build filtered array: include singlets and middle triplet values
    filtered = []
    for i in range(n):
        if i not in triplet_indices:
            filtered.append(energies[i])
        elif any(t[0] == i for t in triplets_middle):
            filtered.append(energies[i])  # Include middle value of triplet
    
    return filtered

def creat_H3_plus_ion(r):
    h1 = np.array([-0.6, 0.0, 0.0])
    h2 = np.array([0.6, 0.0, 0.0])
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

def creat_H3_plus_ion_v2(r):
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
    mol.symmetry=False
    return mol

def compute_fci(mol):
    start = time.time()
    mf = scf.RHF(mol)
    mf.conv_tol=1e-15
    mf.kernel()
    from pyscf import mcscf
    mc=mcscf.CASCI(mf, 3, 2)
    mc.fcisolver=fci.direct_spin0.FCISolver(mol)
    mc.fcisolver.nroots=3
    mc.fix_spin_(ss=0)
    mc.kernel()
    
    e =[]
    for i in range(num_states_fci):
        e.append(mc.e_tot[i])
    return e

# Define the range of r values
r_values = list(np.arange(0.0, 2.0, 1.1))

# Data lists
qse_energies = []
fci_energies = []

# Number of states
num_states_fci =3
num_states_qse = 3

# Create directory
os.makedirs('h3plus_vec_singlet_qse_quri', exist_ok=True)

# Initialize previous mol and mo_coeff
prev_mol = None
prev_vqe_mo = None

# Compute energies for each r
for r in r_values:
    mol = creat_H3_plus_ion_v2(r)
    print(f"Computing for r = {r:.3f} Å")
    
    # Compute QSE energy
    mf = scf.ROHF(mol)
    mf.run()
    vqe_casci = VQECASCI(mf, 3, 2, singlet_excitation=True, ansatz=Ansatz.KUpCCGSD, use_singles=False,k=2)
    
    if prev_mol is not None:
        # projected_mo = project_mo_nr2nr(prev_mol, prev_vqe_mo, mol)
        # vqe_casci.kernel(mo_coeff=projected_mo)
        vqe_casci.kernel()
    else:
        vqe_casci.kernel()
    
    qse = QSE_exact(vqe_casci.fcisolver)
    qse.gen_excitation_operators("ee", 2)
    qse.solve()
    e_tot = qse.eigenvalues
    e_tot[0] = vqe_casci.fcisolver.energies[0]
    v = qse.eigenvectors
    qse_energy = e_tot[:num_states_qse]
    qse_energies.append((r, qse_energy))
    np.save(f'h3plus_vec_singlet_qse_quri/qse_vector_r_{r:.3f}.npy', v)
    
    # Update previous
    prev_mol = mol
    prev_vqe_mo = vqe_casci.mo_coeff
    
    # Compute FCI energy
    fci_energy = compute_fci(mol)
    fci_energies.append((r, fci_energy))

# Sort by r (though already sorted)
r_values_sorted = [x[0] for x in qse_energies]
qse_energies_sorted = [x[1] for x in qse_energies]
fci_energies_sorted = [x[1] for x in fci_energies]

# Save QSE data to txt
with open('h3ion_qse_e_data_singlet_quri.txt', 'w') as f:
    f.write('r energies\n')
    for r, ens in zip(r_values_sorted, qse_energies_sorted):
        f.write(f"{r:.4f} {' '.join(f'{e:.8f}' for e in ens)}\n")

# Save FCI data to txt
with open('h3ion_fci_data.txt', 'w') as f:
    f.write('r energies\n')
    for r, ens in zip(r_values_sorted, fci_energies_sorted):
        f.write(f"{r:.4f} {' '.join(f'{e:.8f}' for e in ens)}\n")

# Plot line graph comparing results, including excited states
for i in range(num_states_qse):
    qse_state_energies = [sorted(ens)[i] for ens in qse_energies_sorted]
    plt.plot(r_values_sorted, qse_state_energies, color='limegreen', linestyle='-', marker='x', label='QSE' if i == 0 else None)
    if i < num_states_fci:
        fci_state_energies = [sorted(ens)[i] for ens in fci_energies_sorted]
        plt.plot(r_values_sorted, fci_state_energies, color='deepskyblue', linestyle='--', marker=None, label='FCI' if i == 0 else None)

plt.xlabel('r (Å)')
plt.ylabel('Energy (Hartree)')
plt.title('Energy Comparison: QSE vs FCI for H3+ (Ground and Excited States)')
plt.legend()
plt.grid(False)
plt.savefig('h3ion_singlet.svg')
plt.close()