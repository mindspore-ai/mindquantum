import warnings
warnings.filterwarnings('ignore', message='Hamiltonian coefficients must be real numbers.*')
import numpy as np
from pyscf import gto,scf,mcscf,fci
import time,os
from chemqulacs_namd.vqe.vqemcscf_mindspore import VQECASCI
from chemqulacs_namd.vqe.vqeci_mindspore_ss import Ansatz
from chemqulacs_namd.qse.qse_singlet_ee_mindspore import QSE_exact



# from chemqulacs_namd.vqe.vqemcscf import VQECASCI
# from chemqulacs_namd.vqe.vqeci import Ansatz
num_states_fci = 3
num_states_qse = 3

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

    mf = scf.RHF(mol)
    mf.conv_tol=1e-15
    mf.kernel()
    mc=mcscf.CASCI(mf, 3, 2)
    mc.fcisolver=fci.direct_spin0.FCI(mol)
    mc.fix_spin(ss=0)
    mc.fcisolver.nroots=num_states_fci
    mc.kernel()
    e =[]
    for i in range(num_states_fci):
        e.append(mc.e_tot[i])

    return e

def compute_vqe(mol):
    start=time.time()
    mf=scf.ROHF(mol)
    mf.conv_tol=1e-15
    mf.run()
    vqeci = VQECASCI(
        mf,
        ncas=3,
        nelecas=2,singlet_excitation=True,ansatz=Ansatz.KUpCCGSD,
        use_singles=False,k=7
    )
    vqeci.kernel()
    e=vqeci.fcisolver.energies[0]
    # print(f'vqe results 1:{vqeci.fcisolver.energies[1]}')

    qse = QSE_exact(vqeci.fcisolver)
    # qse = QSE(vqeci.fcisolver)
    qse.gen_excitation_operators("ee", 2)
    qse.solve()
    e_tot = qse.eigenvalues
    e_tot[0] = vqeci.fcisolver.energies[0]
    print(f"Time for 噪声vqeci+qse计算激发态能量: {time.time() - start:.2f} seconds")
    print(f'vqeci+qse results:{e_tot}')

    return e_tot


r_values = list(np.arange(0.0, 2.0, 0.1))

mol=creat_H3_plus_ion_v2(1.0)

qse_energies = []
fci_energies = []

for r in r_values:
    mol=creat_H3_plus_ion_v2(r)
    fci_energy = compute_fci(mol)
    vqe_energy = compute_vqe(mol)
    print(f'r={r}, fci_energy={fci_energy}, vqe_energy={vqe_energy}')
    fci_energies.append([r, fci_energy])
    qse_energies.append([r, vqe_energy])

import matplotlib.pyplot as plt

r_values_sorted = [x[0] for x in qse_energies]
qse_energies_sorted = [x[1] for x in qse_energies]
fci_energies_sorted = [x[1] for x in fci_energies]

# Create subdirectory for results
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Save QSE data to txt
with open(os.path.join(results_dir, 'h3ion_qse_e_data_singlet_mindspore.txt'), 'w') as f:
    f.write('r energies\n')
    for r, ens in zip(r_values_sorted, qse_energies_sorted):
        f.write(f"{r:.4f} {' '.join(f'{e:.8f}' for e in ens)}\n")

# Save FCI data to txt
with open(os.path.join(results_dir, 'h3ion_fci_data_mindspore.txt'), 'w') as f:
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
plt.title('Energy Comparison: QSE* vs FCI for H3+ (Ground and Excited States)')
plt.legend()
plt.grid(False)
plt.savefig(os.path.join(results_dir, 'h3ion_singlet_mindspore.svg'))
plt.close()

# os.makedirs('h3plus_vec', exist_ok=True)




