import warnings

from yarg import get
warnings.filterwarnings('ignore', message='Hamiltonian coefficients must be real numbers.*')
import numpy as np
from pyscf import gto,scf,mcscf,fci
import time,os
from chemqulacs_namd.vqe.vqemcscf_mindspore import VQECASCI
from chemqulacs_namd.vqe.vqeci_mindspore_ss import Ansatz
from chemqulacs_namd.qse.qse_singlet_dex_mindspore import QSE_exact

# from chemqulacs_namd.vqe.vqemcscf import VQECASCI
# from chemqulacs_namd.vqe.vqeci import Ansatz
num_states_fci = 2
num_states_qse = 2

def get_ethylene_mol_twisted_and_folding_according_to_theta(theta):
    """
    Returns a PySCF Mole object for a twisted and folded ethylene molecule (C2H4),
    where the two CH2 planes are perpendicular, starting from the twisted geometry,
    and folding the CH2 at C1 towards the C2 side by the angle theta (in degrees).
    Bond lengths and H-C-H angles are kept fixed.
    """
    from pyscf import gto
    import math

    cc_half = 1.340679/2.0
    h_y = 0.918168
    h_z_rel = 0.57122  # 0.5626

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
    # mol.verbose=10
    # mol.max_memory=8000
    mol.symmetry=  False
    mol.build()
    
    return mol



def compute_fci(mol):

    mf = scf.ROHF(mol)
    mf.kernel()
    mc=mcscf.CASCI(mf, 2, 2)
    mc.fcisolver=fci.direct_spin0.FCI(mol)
    mc.fix_spin(ss=0)
    mc.fcisolver.nroots=num_states_fci
    mc.kernel()
    e=mc.e_tot

    return e

def compute_vqe(mol):
    start=time.time()
    mf=scf.ROHF(mol)
    mf.run()
    mf.conv_tol = 1e-13
    vqeci = VQECASCI(
        mf,
        ncas=2,
        nelecas=2,singlet_excitation=True,ansatz=Ansatz.UCCSD,
        use_singles=False,excitation_number=1,weight_policy="exponential"
    )
    
    vqeci.kernel()
    e=[]


    qse = QSE_exact(vqeci.fcisolver)
    qse.gen_excitation_operators("ee", 2)
    qse.solve()

    print(f"Time for 无噪声vqeci+qse计算激发态能量: {time.time() - start:.2f} seconds")

    e.append(vqeci.fcisolver.energies[0])
    if qse.eigenvalues[1]-vqeci.fcisolver.energies[0] > 1e-6:
        e.append(qse.eigenvalues[1])
    else:
        e.append(qse.eigenvalues[0])
    #给e从大到小排序
    e.sort(reverse=True)
    return e


r_values = list(np.arange(1, 180, 10))

qse_energies = []
fci_energies = []

for r in r_values:
    mol=get_ethylene_mol_twisted_and_folding_according_to_theta(r)
    fci_energy = compute_fci(mol)
    vqe_energy = compute_vqe(mol)
    print(f'r={r}, fci_energy={fci_energy}, vqe_energy={vqe_energy}')
    fci_energies.append(fci_energy)
    qse_energies.append(vqe_energy)

import matplotlib.pyplot as plt

# Create subdirectory for results
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Save QSE data to txt
with open(os.path.join(results_dir, 'ethylene_qse_e_data_singlet_dex_mindspore.txt'), 'w') as f:
    f.write('theta energies\n')
    for theta, ens in zip(r_values, qse_energies):
        f.write(f"{theta:.4f} {' '.join(f'{e:.8f}' for e in ens)}\n")

# Save FCI data to txt
with open(os.path.join(results_dir, 'ethylene_fci_data_mindspore.txt'), 'w') as f:
    f.write('theta energies\n')
    for theta, ens in zip(r_values, fci_energies):
        f.write(f"{theta:.4f} {' '.join(f'{e:.8f}' for e in ens)}\n")

# Plot line graph comparing results, including excited states
for i in range(num_states_qse):
    qse_state_energies = [sorted(ens)[i] for ens in qse_energies]
    plt.plot(r_values, qse_state_energies, color='limegreen', linestyle='-', marker='x', label='QSE' if i == 0 else None)
    if i < num_states_fci:
        fci_state_energies = [sorted(ens)[i] for ens in fci_energies]
        plt.plot(r_values, fci_state_energies, color='deepskyblue', linestyle='--', marker=None, label='FCI' if i == 0 else None)

plt.xlabel('theta (degrees)')
plt.ylabel('Energy (Hartree)')
plt.title('Energy Comparison: QSE vs FCI for Twisted Ethylene (Ground and Excited States)')
plt.legend()
plt.grid(False)
plt.savefig(os.path.join(results_dir, 'ethylene_singlet_dex_mindspore.svg'))
plt.close()


# os.makedirs('h3plus_vec', exist_ok=True)




