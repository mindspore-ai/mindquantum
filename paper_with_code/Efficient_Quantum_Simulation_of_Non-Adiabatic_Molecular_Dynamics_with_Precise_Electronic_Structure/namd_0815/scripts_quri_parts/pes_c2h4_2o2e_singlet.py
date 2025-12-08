import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci, mcscf
from pyscf.scf.addons import project_mo_nr2nr
from chemqulacs_namd.vqe.vqemcscf import VQECASCI
from chemqulacs_namd.vqe.vqeci import Ansatz
from chemqulacs_namd.qse.qse_singlet import QSE
import time
import os

def filter_outliers(energies):
    """
    Filter outliers in energy values: replace any energy lower than the ground state energy with the ground state energy.
    
    Args:
        energies (list or np.ndarray): Energy values where energies[0] is the ground state.
    
    Returns:
        np.ndarray: Filtered energy array.
    """
    energies = np.asarray(energies)
    if len(energies) == 0:
        return energies
    ground_state = energies[0]
    energies[energies < ground_state] = ground_state
    return energies

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

def get_ethylene_mol_twisted_and_folding_according_to_theta(theta):
    """
    Returns a PySCF Mole object for a twisted and folded ethylene molecule (C2H4),
    where the two CH2 planes are perpendicular, starting from the twisted geometry,
    and folding the CH2 at C1 towards the C2 side by the angle theta (in degrees).
    Bond lengths and H-C-H angles are kept fixed.
    """
    from pyscf import gto
    import math

    cc_half = 0.6695
    h_y = 0.9289
    h_z_rel = 1.2321 - cc_half  # 0.5626

    sin_theta = math.sin(math.radians(theta))
    cos_theta = math.cos(math.radians(theta))

    h1_x = h_z_rel * sin_theta
    h1_y = h_y
    h1_z = cc_half + h_z_rel * cos_theta

    h2_x = h1_x
    h2_y = -h_y
    h2_z = h1_z

    h3_x = -h_y
    h3_y = 0.0000
    h3_z = -cc_half - h_z_rel  # -1.2321

    h4_x = h_y
    h4_y = 0.0000
    h4_z = h3_z

    c1_x = 0.0000
    c1_y = 0.0000
    c1_z = cc_half

    c2_x = 0.0000
    c2_y = 0.0000
    c2_z = -cc_half

    mol = gto.Mole()
    mol.atom = f"""
    C  {c1_x:7.4f}  {c1_y:7.4f}  {c1_z:7.4f}
    C  {c2_x:7.4f}  {c2_y:7.4f}  {c2_z:7.4f}
    H  {h1_x:7.4f}  {h1_y:7.4f}  {h1_z:7.4f}
    H  {h2_x:7.4f}  {h2_y:7.4f}  {h2_z:7.4f}
    H  {h3_x:7.4f}  {h3_y:7.4f}  {h3_z:7.4f}
    H  {h4_x:7.4f}  {h4_y:7.4f}  {h4_z:7.4f}
    """
    mol.basis = 'sto-3g'
    mol.unit = 'Angstrom'
    mol.build()
    return mol

def compute_casci(mol):

    mf = scf.RHF(mol)
    mf.kernel()
    cisolver = mcscf.CASCI(mf, ncas=2, nelecas=2)
    cisolver.fcisolver.nroots=2
    cisolver.fix_spin(ss=0)
    cisolver.kernel()
    e_tot=cisolver.e_tot
    e=[]
    for i in range(len(e_tot)):
        e.append(e_tot[i])
    
    
    return e

# Define the range of r values
r_values = list(np.arange(90,100, 1))

# Data lists
qse_energies = []
fci_energies = []

# Number of states
num_states_fci = 2
num_states_qse = 2

# Create directory
os.makedirs('c2h4_vec_2o2e', exist_ok=True)

# Initialize previous mol and mo_coeff
prev_mol = None
prev_vqe_mo = None

# Compute energies for each r
for r in r_values:
    mol = get_ethylene_mol_twisted_and_folding_according_to_theta(r)
    print(f"Computing for r = {r:.3f} Å")
    
    # Compute QSE energy
    mf = scf.RHF(mol)

    
    if prev_mol is not None:
        # projected_mo = project_mo_nr2nr(prev_mol, prev_vqe_mo, mol)
        # mf.mo_coeff=projected_mo
        mf.run()
        vqe_casci = VQECASCI(mf, 2, 2, singlet_excitation=True, ansatz=Ansatz.UCCSD, use_singles=False)
        vqe_casci.kernel()
    else:
        mf.run()
        vqe_casci = VQECASCI(mf, 2, 2, singlet_excitation=True, ansatz=Ansatz.UCCSD, use_singles=False)
        vqe_casci.kernel()
    
    qse = QSE(vqe_casci.fcisolver)
    qse.gen_excitation_operators("ee", 2)
    qse.solve()
    e_tot = qse.eigenvalues
    e_tot[0] = vqe_casci.fcisolver.energies[0]
    v = qse.eigenvectors
    qse_energy = e_tot[:2]
    qse_energies.append((r, qse_energy))
    np.save(f'c2h4_vec_2o2e/qse_vector_r_{r:.3f}.npy', v)
    
    # Update previous
    prev_mol = mol
    prev_vqe_mo = vqe_casci.mo_coeff
    
    # Compute FCI energy
    fci_energy = compute_casci(mol)
    fci_energies.append((r, fci_energy))

# Sort by r
r_values_sorted = [x[0] for x in qse_energies]
qse_energies_sorted = [x[1] for x in qse_energies]
fci_energies_sorted = [x[1] for x in fci_energies]

# Save QSE data to txt
with open('qse_e_data_c2h4_2o2e.txt', 'w') as f:
    f.write('r energies\n')
    for r, ens in zip(r_values_sorted, qse_energies_sorted):
        f.write(f"{r:.4f} {' '.join(f'{e:.8f}' for e in ens)}\n")

# Save FCI data to txt
with open('fci_data_c2h4_2o2e.txt', 'w') as f:
    f.write('r energies\n')
    for r, ens in zip(r_values_sorted, fci_energies_sorted):
        f.write(f"{r:.4f} {' '.join(f'{e:.8f}' for e in ens)}\n")

# Plot line graph comparing results, including excited states
for i in range(num_states_qse):
    qse_state_energies = [sorted(ens)[i] for ens in qse_energies_sorted]
    plt.plot(r_values_sorted, qse_state_energies, color='limegreen', linestyle='-', marker='x', label='QSE' if i == 0 else None)
    if i < num_states_fci:
        fci_state_energies = [sorted(ens)[i] for ens in fci_energies_sorted]
        plt.plot(r_values_sorted, fci_state_energies, color='deepskyblue', linestyle='--', marker=None, label='CASCI' if i == 0 else None)

plt.xlabel('r (Å)')
plt.ylabel('Energy (Hartree)')
plt.title('Energy Comparison: QSE vs CASCI for c2h4 (Ground and Excited States)')
plt.legend()
plt.grid(True)
plt.savefig('c2h4_energy_comparison_2o2e.svg')
plt.close()