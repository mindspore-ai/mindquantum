"""Construct various quantum algorithms"""

import numpy as np
from mindquantum import *
from pyscf import gto, scf, symm
from tools import *
import itertools

def get_uccsd(n_qubits:int, n_electrons:int, share_parameter=True):
    """Obtain the UCCSD ansatz, where share_parameter determines if parameters are shared among excitation operators"""
    uccsd = uccsd_singlet_generator(n_qubits, n_electrons, False)
    if share_parameter:
        ansatz = uccsd
    else:
        theta = PRGenerator('theta_')
        ansatz = FermionOperator()
        for fop in uccsd:
            ansatz += FermionOperator(fop_to_excitation(fop), theta.new())
    return [fop for fop in ansatz]


def get_k_upccgsd(n_qubits:int, k:int, share_parameter=False):
    """Obtain the k-UpCCGSD ansatz"""
    ansatz = []
    theta = PRGenerator('theta_')
    for _ in range(k):
        for p, q in itertools.product(range(n_qubits//2), range(n_qubits//2)):
            if p <= q:
                continue
            # Single excitations
            if not share_parameter:
                ansatz.append(FermionOperator(((2*p, 1), (2*q, 0)), theta.new()))
                ansatz.append(FermionOperator(((2*p+1, 1), (2*q+1, 0)), theta.new()))
            else:
                param = theta.new()
                ansatz.append(FermionOperator(((2*p, 1), (2*q, 0)), param))
                ansatz.append(FermionOperator(((2*p+1, 1), (2*q+1, 0)), param))
            # Double excitations
            ansatz.append(FermionOperator(((2*p+1, 1), (2*p, 1), (2*q+1, 0), (2*q, 0)), theta.new()))
    return ansatz


def get_state_irrep(group_name:str, mo_irreps:list[str], n_electrons:int, fop:FermionOperator|None=None):
    """Obtain the irreducible representation of a single-determinant quantum state"""
    spin_orbs = list(range(n_electrons))

    if fop is not None:
        core = fop_to_core(fop)
        mid = len(core) // 2
        creations = core[:mid]
        annihilations = core[mid:]
        for i in range(len(annihilations)):
            spin_orbs[annihilations[i]] = creations[i]

    mole_orbs = np.array(spin_orbs) // 2 
    irreps = [mo_irreps[orb] for orb in mole_orbs]
    irreps_id = [symm.irrep_name2id(group_name, irrep) for irrep in irreps]
    state_irrep_id = 0
    for id in irreps_id:
        state_irrep_id ^= id

    return symm.irrep_id2name(group_name, state_irrep_id)
    

def get_symuccsd(geo, ansatz:list[FermionOperator], n_electrons:int, symmetry=True):
    """Filter out excitation operators that do not satisfy point group symmetry"""
    mol = gto.M(atom=geo, basis="sto-3g", symmetry=symmetry, verbose=0) # Enable symmetry analysis
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()

    mo_irreps = symm.label_orb_symm(
        mol, 
        mol.irrep_name,  # Output irrep symbols (e.g., 'A1')
        mol.symm_orb,    # Symmetry-adapted basis set
        mf.mo_coeff      # Molecular orbital coefficients
    )
    # print('\nPoint group name:\n', mol.groupname) #  C2v
    mo_irreps = list(mo_irreps) # ['A1' 'A1' 'B2' 'A1' 'B1' 'A1' 'B2']
    hf_irrep = get_state_irrep(mol.groupname, mo_irreps, n_electrons)
    # print(' '.join(mo_irreps))
    
    sym_ansatz = []
    for fop in ansatz:
        state_irrep = get_state_irrep(mol.groupname, mo_irreps, n_electrons, fop)
        if state_irrep == hf_irrep:
            sym_ansatz.append(fop)
    return sym_ansatz


def get_hiuccsd(ham:mqchem.CIHamiltonian, ansatz:FermionOperator, tol=1e-8):
    """Obtain the symmetry-adapted UpCCGSD ansatz"""
    # Select excitation operators included in the Hamiltonian (coefficients of symmetry-violating terms are zero)
    f_ham = ham.fermion_hamiltonian.compress(tol)
    ham_terms = set(f_ham.terms.keys())
    sym_ansatz = []
    for fop in ansatz:
        if fop_to_excitation(fop) in ham_terms:
            sym_ansatz.append(fop)
    return sym_ansatz


def get_init_grads(sim:mqchem.MQChemSimulator, 
                    com_mapping:FermionOperator):
    """Calculate the initial gradient magnitudes of each operator in the operator pool for the current state"""
    init_grads = {}
    for k, v in com_mapping.items():
        init_grads[k] = abs(sim.get_expectation(mqchem.CIHamiltonian(v)))
    return init_grads


def get_best_core(grads:dict):
    """Select the excitation operator with the largest gradient value"""
    sortd_cores = sorted(grads, key=grads.get, reverse=True)
    best_core = sortd_cores[0]
    norm = np.sqrt(sum([i**2 for i in list(grads.values())]))
    return best_core, norm


def run_vqe(sim:mqchem.MQChemSimulator, ham:mqchem.CIHamiltonian, circ:Circuit, p0=None):
    """Run the VQE algorithm"""
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    if p0 is None:
        p0 = np.array([0.]*len(circ.params_name))
    res = minimize(grad_ops, p0, method='bfgs', jac=True)
    return res.fun, res.x


def run_adapt_vqe(geometry:list, pool:list[FermionOperator], max_iters:int, tol:float=1e-3):
    """Run the ADAPT-VQE algorithm"""
    hf, fci, ham, n_qubits, n_electrons = get_mole_info(geometry, fci=True)
    f_ham = ham.fermion_hamiltonian
    cores_pool = [fop_to_core(fop) for fop in pool]
    theta = PRGenerator("theta_")
    pre_res = {}
    sim = mqchem.MQChemSimulator(n_qubits, n_electrons)
    energys, errors = [hf], [abs(hf - fci)]
    com_mapping = {core: (FermionOperator(core_to_excitation(core))*f_ham - f_ham*FermionOperator(core_to_excitation(core))).normal_ordered() for core in cores_pool}
    circ = Circuit()
    iter = 0
    while True:
        iter += 1
        if iter > max_iters:
            break

        sim.reset()
        sim.apply_circuit(circ, pre_res)
        init_grads = get_init_grads(sim, com_mapping)
        best_core, norm = get_best_core(init_grads)
        if norm < tol:
            break
        fop = FermionOperator(core_to_excitation(best_core), theta.new())
        circ += mqchem.UCCExcitationGate(fop)
        p0 = np.array([pre_res[k] if k in pre_res else 0. for k in circ.params_name]) # warm-start
        sim.reset()
        energy, params = run_vqe(sim, ham, circ, p0)
        pre_res = dict(zip(circ.params_name, params))
        energys.append(energy)
        errors.append(abs(fci - energy))
    return energys, errors