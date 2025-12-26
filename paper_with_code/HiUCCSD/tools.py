"""Basic utility functions for algorithm execution"""

from mindquantum import *
from scipy.optimize import minimize
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
import numpy as np


def get_HF_geometry(dist=1.00):
    """HF molecule"""
    geometry = [['H', [0.0, 0.0, 0.0 * dist]],
                ['F', [0.0, 0.0, 1.0 * dist]]]
    return geometry


def get_LiH_geometry(dist=1.55):
    """LiH molecule"""
    geometry = [['H', [0.0, 0.0, 0.0 * dist]],
                ['Li', [0.0, 0.0, 1.0 * dist]]]
    return geometry


def get_NaH_geometry(dist=1.64):
    """NaH molecule"""  # 修正原注释错误（原写为LiH molecule）
    geometry = [['H', [0.0, 0.0, 0.0 * dist]],
                ['Na', [0.0, 0.0, 1.0 * dist]]]
    return geometry


def get_H2O_geometry(dist=1.02):
    """H2O (water) molecule"""
    angle = np.radians(104.5)
    geometry = [['O', [0.0, 0.0, 0.0]],  # Oxygen atom at the origin
                ['H', [dist, 0.0, 0.0]],  # First hydrogen atom on the x-axis
                ['H', [dist * np.cos(angle), dist * np.sin(angle), 0.0]]]
    return geometry


def get_N2_geometry(dist=1.2):
    """N2 molecule"""  # 补充缺失的注释
    geometry = [['N', [0.0, 0.0, 0.0]],
                ['N', [0.0, 0.0, dist]]]
    return geometry


def get_BeH2_geometry(dist=1.32):
    """BeH2 molecule"""
    geometry = [['Be', [0.0, 0.0, 0.0]],  # Beryllium atom at the origin
                ['H', [0.0, 0.0, dist]],  # First hydrogen atom on the z-axis
                ['H', [0.0, 0.0, -dist]]]  # Second hydrogen atom on the negative z-axis
    return geometry


def get_CO_geometry(dist=1.2):
    """Generate the Cartesian coordinate configuration of CO molecule (unit: Å), default distribution along the z-axis"""
    # Carbon atom at the origin, oxygen atom on the z-axis
    geometry = [
        ['C', [0.0, 0.0, 0.0]],
        ['O', [0.0, 0.0, dist]]
    ]
    return geometry


def get_NH3_geometry(dist=1.09, angle=107.3):
    """
    Generate the geometric configuration of NH3 molecule (triangular pyramidal structure)
    """
    # Nitrogen atom at the coordinate origin (C₃ axis along the z-axis)
    N = ['N', [0.0, 0.0, 0.0]]
    
    # Calculate H atom coordinates: triangular pyramidal geometry based on bond length and bond angle
    theta = np.radians(angle)  # Convert to radians
    # Projection radius of H atoms on the xy-plane & z-axis coordinate (height of the triangular pyramid)
    r_xy = dist * np.sin(theta / 2)
    z_H = -dist * np.cos(theta / 2)  # H atoms are located below the N atom
    
    # Three H atoms are evenly distributed in the xy-plane (120° apart, satisfying C₃ symmetry)
    H1 = ['H', [r_xy, 0.0, z_H]]  # Along the x-axis
    H2 = ['H', [r_xy * np.cos(2*np.pi/3), r_xy * np.sin(2*np.pi/3), z_H]]  # 120°
    H3 = ['H', [r_xy * np.cos(4*np.pi/3), r_xy * np.sin(4*np.pi/3), z_H]]  # 240°
    
    geometry = [N, H1, H2, H3]
    return geometry


def get_CH4_geometry(dist=1.09):
    """Generate the coordinates of the regular tetrahedral geometric structure of CH4 molecule"""
    # Carbon atom at the origin
    C = np.array([0.0, 0.0, 0.0])
    
    # Coordinates of H atoms in regular tetrahedron (bond angle ~109.5°)
    # Calculate regular tetrahedron vertex coordinates: based on cube vertices
    h1 = np.array([1, 1, 1])
    h2 = np.array([1, -1, -1])
    h3 = np.array([-1, 1, -1])
    h4 = np.array([-1, -1, 1])
    
    # Normalize and multiply by bond length
    h1 = dist * h1 / np.linalg.norm(h1)
    h2 = dist * h2 / np.linalg.norm(h2)
    h3 = dist * h3 / np.linalg.norm(h3)
    h4 = dist * h4 / np.linalg.norm(h4)
    
    return [['C', C], ['H', h1], ['H', h2], ['H', h3], ['H', h4]]


def get_C2H4_geometry(d1=1.34, d2=1.08, a=120.0):
    """Generate the geometric configuration of C2H4 (ethylene) molecule"""  # 补充缺失的注释
    rad = a / 180 * np.pi
    x, y = d2 * np.cos(rad), d2 * np.sin(rad)
    hc = d1 / 2
    return [
        ['C', [-hc, 0, 0]], ['C', [hc, 0, 0]],
        ['H', [-hc - x, y, 0]], ['H', [-hc - x, -y, 0]],
        ['H', [hc + x, y, 0]], ['H', [hc + x, -y, 0]]
    ]


def get_bound_list(bound_range: list, num_points: int = 21):
    """Get the list of bond lengths"""
    return [bound_range[0] + i * (bound_range[1] - bound_range[0]) / (num_points - 1) for i in range(num_points)]


def get_mole_info(geometry: list, fci=True, mp2=False, ccsd=False, scf=False):
    """Complete classical calculation tasks based on PySCF to obtain molecular information, 
       such as FCI energy, HF energy, system Hamiltonian, etc."""
    basis = "sto3g"
    spin = 0
    mole = MolecularData(geometry, basis, multiplicity=2 * spin + 1)
    mole = run_pyscf(mole, run_fci=fci, run_mp2=mp2, run_ccsd=ccsd, run_scf=scf)
    ham_of = mole.get_molecular_hamiltonian()
    int_ops = InteractionOperator(*(ham_of.n_body_tensors.values()))
    f_ham = FermionOperator(int_ops).compress().normal_ordered()
    ham = mqchem.CIHamiltonian(f_ham)
    return mole.hf_energy, mole.fci_energy, ham, mole.n_qubits, mole.n_electrons


def fop_to_core(fop: FermionOperator):
    """Convert fermion operator to core indices: s_0 [5^ 4] -> (5, 4)"""
    excitation = fop_to_excitation(fop)
    return tuple([op[0] for op in excitation])


def fop_to_excitation(fop: FermionOperator):
    """Convert fermion excitation operator to tuple format: 5^ 4^ 1 0 -> (5,4,1,0)"""
    return tuple([op for op in next(iter(fop.terms.keys()))])


def core_to_excitation(core: tuple):
    """Convert core indices to excitation tuple: (5, 4, 3, 2) -> ((5,1), (4,1), (3,0), (2,0))"""
    return tuple([(core[i], int(i < len(core)/2)) for i in range(len(core))])


def get_circ(ansatz: FermionOperator):
    """Generate UCC quantum circuit"""
    return Circuit([mqchem.UCCExcitationGate(fop) for fop in ansatz])


def run_vqe(ham: mqchem.CIHamiltonian,
            n_qubits: int,
            n_electrons: int,
            circ: Circuit,
            p0: np.ndarray | None = None):
    """Run the VQE algorithm with the given quantum circuit to find the ground state energy."""
    if p0 is None:
        p0 = np.array([0.] * len(circ.params_name))

    grad_ops = mqchem.MQChemSimulator(n_qubits, n_electrons).get_expectation_with_grad(ham, circ)
    res = minimize(grad_ops, p0, method='bfgs', jac=True)
    return res.fun, res.x