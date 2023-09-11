# Example program. (No meaningful result)
# The judging program "eval.py" will call `Main.run()` function from "src/main.py",
# and receive an energy value. So please put the executing part of your algorithm
# in `Main.run()`, return an energy value.
# All of your code should be placed in "src/" folder.

# Author: Rongge Xu, Hui Dai (Durga Team)

import numpy as np
import sys
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import InteractionOperator, normal_ordered, FermionOperator
from qupack.vqe import ESConserveHam, ExpmPQRSFermionGate, ESConservation
from scipy.optimize import minimize
import itertools
from mindquantum.core.parameterresolver import ParameterResolver


def excited_state_VQD_solver(mol, depth = 1):
    print('Qubits: %s, Electrons: %s'%(mol.n_qubits, mol.n_electrons))
    ftol = 1e-3
    stol = 1e-3
    fiter = 500
    siter = 1000
    weight = 5
    lr = 1
    weight1 = 10
    lr1 = 1
    if mol.n_qubits <= 8:
        depth = 3
        opt_it = 1
        ftol = 1e-4
        fiter = 1000
    elif mol.n_qubits <= 12:
        depth = 3
        opt_it = 1  
    elif mol.n_qubits <= 16:
        depth = 3
        opt_it = 1
        lr1 = 1
        weight = 10
        fiter = 1500
    else:
        depth = 3 + int(1*(mol.n_qubits-8)//4)
        opt_it = 1
        siter = 2000
    
    print('Set depth to: %s, optimizer iter: %s'%(depth, opt_it))

    ham_of = mol.get_molecular_hamiltonian()
    inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
    ham_hiq = FermionOperator(inter_ops)
    ham_fo = normal_ordered(ham_hiq).real
    ham = ESConserveHam(ham_fo)
    
    gs_circ = Circuit()
    for d in range(depth):
        ucc_fermion_ops = uccsd_singlet_generator_with_pr(mol.n_qubits, mol.n_electrons, anti_hermitian=False, depth=d+1)
        for term in ucc_fermion_ops:
            gs_circ += ExpmPQRSFermionGate(term)
    gs_sim = ESConservation(mol.n_qubits, mol.n_electrons)
    gs_grad_ops = gs_sim.get_expectation_with_grad(ham, gs_circ)
    def gs_func(x, grad_ops):
        f, g = grad_ops(x)
        return np.real(np.squeeze(f)), np.real(np.squeeze(g))
    # Initialize amplitudes
    init_amplitudes = np.random.uniform(size=len(gs_circ.params_name)) * 0.01

    # Get Optimized result
    gs_res = minimize(gs_func, init_amplitudes, args=(gs_grad_ops), method="bfgs", jac=True,tol=stol)

    # Construct parameter resolver of the ground state circuit
    gs_pr = dict(zip(gs_circ.params_name, gs_res.x))

    for d in range(depth):
        gs_circ_pr = Circuit()
        ucc_fermion_ops_pr = uccsd_singlet_generator_with_pr(mol.n_qubits, mol.n_electrons, anti_hermitian=False, pr=gs_pr, depth=d+1)

        for term in ucc_fermion_ops_pr:
            gs_circ_pr += ExpmPQRSFermionGate(term)

    es_circ = Circuit()
    for d in range(depth):
        ucc_fermion_ops = uccsd_singlet_generator_with_pr(mol.n_qubits, mol.n_electrons, anti_hermitian=False, depth=d+1)
 
        for term in ucc_fermion_ops:
            es_circ += ExpmPQRSFermionGate(term)
    es_sim = ESConservation(mol.n_qubits, mol.n_electrons)
    es_grad_ops = es_sim.get_expectation_with_grad(ham, es_circ)
    ip_grad_ops = es_sim.get_expectation_with_grad(ESConserveHam(FermionOperator('')), es_circ, gs_circ_pr, simulator_left=gs_sim)    

    def func(x, es_grad_ops, ip_grad_ops, weight, lr):
        f0, g0 = es_grad_ops(x)
        f1, g1 = ip_grad_ops(x)
        f0, g0, f1, g1 = (np.squeeze(f0), np.squeeze(g0), np.squeeze(f1), np.squeeze(g1))
        cost = np.real(f0) + weight * np.abs(f1) ** 2  # Calculate cost function
        punish_g = np.conj(g1) * f1 + g1 * np.conj(f1)  # Calculate punishment term gradient
        total_g = (g0 + weight * punish_g)*lr
        return cost, total_g.real
    
    def func_opt(x, es_grad_ops, ip_grad_ops, weight, lr):
        f0, g0 = es_grad_ops(x)
        f1, g1 = ip_grad_ops(x)
        f0, g0, f1, g1 = (np.squeeze(f0), np.squeeze(g0), np.squeeze(f1), np.squeeze(g1))
        cost = np.real(f0) + weight * np.abs(f1) ** 2  # Calculate cost function
        ratio = np.abs(np.real(f0)/np.real(f1))
        punish_g = np.conj(g1) * f1 + g1 * np.conj(f1)  # Calculate punishment term gradient
        g0_m = np.mean(np.abs(g0))
        pg_m = np.mean(np.abs(punish_g))
        if ratio < 0.5:
            weight = 1.0/ratio
            total_g = (2*g0 + weight * punish_g)*lr
        elif ratio > 50:
            if g0_m <0.01:
                g0 = 0.05*g0/g0_m
            if pg_m < 0.01:
                punish_g = 0.01*punish_g/pg_m
            total_g = (g0 + weight * punish_g)*lr/10.0
        else:
            total_g = (g0 + weight * punish_g)*lr
        return cost, total_g.real
    
    res = None
    min_es = 100
    min_x = None
    for it in range(opt_it):
        print('Iter: %s'%it)
        p0 = np.random.uniform(size=len(es_circ.params_name)) * 0.06
        if res is None:
            if opt_it == 1:
                res = minimize(func, p0, args=(es_grad_ops, ip_grad_ops, weight1, lr1), method="bfgs", jac=True, tol=ftol, options={"maxiter": fiter*(1.2**it)})
            else:
                res = minimize(func_opt, p0, args=(es_grad_ops, ip_grad_ops, weight1, lr1), method="bfgs", jac=True, tol=ftol, options={"maxiter": fiter*(1.2**it)})
        else:
            res = minimize(func, min_x +p0, args=(es_grad_ops, ip_grad_ops, weight, lr), method="bfgs", jac=True, tol=stol, options={"maxiter": siter*(1.2**it)})
        if res.fun < min_es:
            min_x = res.x
            min_es = res.fun
        print(res.fun,min_es, res.x, res.nfev)
    es_pr = dict(zip(es_circ.params_name, min_x))

    # Calculate energy of excited state
    es_en = es_sim.get_expectation(ham, es_circ, es_pr).real

    # print("Ground state energy: ", gs_en)
    print("First excited state energy: ", es_en)
    return es_en

def uccsd_singlet_generator_with_pr(n_qubits, n_electrons, anti_hermitian=True, pr=None, depth=1):
    """
    Create a spin-extended singlet UCCSD generator for a system with n_electrons.

    This function generates a FermionOperator for a UCCSD generator designed
    to act on a single reference state consisting of n_qubits spin orbitals
    and n_electrons electrons, that is a spin singlet operator, meaning it
    conserves spin.

    Args:
        n_qubits(int): Number of spin-orbitals used to represent the system,
            which also corresponds to number of qubits in a non-compact map.
        n_electrons(int): Number of electrons in the physical system.
        anti_hermitian(bool): Flag to generate only normal CCSD operator
            rather than unitary variant, primarily for testing.
        pr(dict): Initial coeff dict of FermionOperator.
        depth(int): Depth label of coeff name. 

    Returns:
        FermionOperator, Generator of the UCCSD operator that
        builds the UCCSD wavefunction.

    Examples:
        >>> from mindquantum.algorithm.nisq.chem import uccsd_singlet_generator
        >>> uccsd_singlet_generator(4, 2)
        -s_0 [0^ 2] +
        -d1_0 [0^ 2 1^ 3] +
        -s_0 [1^ 3] +
        -d1_0 [1^ 3 0^ 2] +
        s_0 [2^ 0] +
        d1_0 [2^ 0 3^ 1] +
        s_0 [3^ 1] +
        d1_0 [3^ 1 2^ 0]
    """
    # pylint: disable=import-outside-toplevel
    from mindquantum.core.operators import (  # pylint: disable=import-outside-toplevel
        FermionOperator,
    )
    from mindquantum.core.operators.utils import down_index, up_index

    if n_qubits % 2 != 0:
        raise ValueError('The total number of spin-orbitals should be even.')

    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(np.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    # Initialize operator
    generator = FermionOperator()

    # Generate excitations
    spin_index_functions = [up_index, down_index]

    # pylint: disable=invalid-name
    # Generate all spin-conserving single and double excitations derived from one spatial occupied-virtual pair
    d = depth
    for i, (p, q) in enumerate(itertools.product(range(n_virtual), range(n_occupied))):
        # Get indices of spatial orbitals
        virtual_spatial = n_occupied + p
        occupied_spatial = q
        for spin in range(2):
            # Get the functions which map a spatial orbital index to a
            # spin orbital index
            this_index = spin_index_functions[spin]
            other_index = spin_index_functions[1 - spin]

            # Get indices of spin orbitals
            virtual_this = this_index(virtual_spatial)
            virtual_other = other_index(virtual_spatial)
            occupied_this = this_index(occupied_spatial)
            occupied_other = other_index(occupied_spatial)

            # Generate single excitations
            if pr is not None:
                coeff = pr[f's_{i}_{spin}_{d}']
            else:
                coeff = ParameterResolver({f's_{i}_{spin}_{d}': 1})
            generator += FermionOperator(((virtual_this, 1), (occupied_this, 0)), coeff)
            if anti_hermitian:
                generator += FermionOperator(((occupied_this, 1), (virtual_this, 0)), -1 * coeff)

            # Generate double excitation
            if p <= 1:
                if pr is not None:
                    coeff = pr[f'd1_{i}_{spin}_{d}']
                else:
                    coeff = ParameterResolver({f'd1_{i}_{spin}_{d}': 1})
            else:
                if pr is not None:
                    coeff = pr[f'd1_{i}_{d}']
                else:
                    coeff = ParameterResolver({f'd1_{i}_{d}': 1})
            generator += FermionOperator(
                ((virtual_this, 1), (occupied_this, 0), (virtual_other, 1), (occupied_other, 0)), coeff
            )
            if anti_hermitian:
                generator += FermionOperator(
                    ((occupied_other, 1), (virtual_other, 0), (occupied_this, 1), (virtual_this, 0)), -1 * coeff
                )

    # Generate all spin-conserving double excitations derived from two spatial occupied-virtual pairs
    for i, ((p, q), (r, s)) in enumerate(
        itertools.combinations(itertools.product(range(n_virtual), range(n_occupied)), 2)
    ):
        # Get indices of spatial orbitals
        virtual_spatial_1 = n_occupied + p
        occupied_spatial_1 = q
        virtual_spatial_2 = n_occupied + r
        occupied_spatial_2 = s

        # Generate double excitations
        for (spin_a, spin_b) in itertools.product(range(2), repeat=2):
            if pr is not None:
                coeff = pr[f'd2_{i}_{d}']
            else:
                coeff = ParameterResolver({f'd2_{i}_{d}': 1})
            # Get the functions which map a spatial orbital index to a
            # spin orbital index
            index_a = spin_index_functions[spin_a]
            index_b = spin_index_functions[spin_b]

            # Get indices of spin orbitals
            virtual_1_a = index_a(virtual_spatial_1)
            occupied_1_a = index_a(occupied_spatial_1)
            virtual_2_b = index_b(virtual_spatial_2)
            occupied_2_b = index_b(occupied_spatial_2)

            if virtual_1_a == virtual_2_b or occupied_1_a == occupied_2_b:
                continue
            generator += FermionOperator(
                ((virtual_1_a, 1), (occupied_1_a, 0), (virtual_2_b, 1), (occupied_2_b, 0)), coeff
            )
            if anti_hermitian:
                generator += FermionOperator(
                    ((occupied_2_b, 1), (virtual_2_b, 0), (occupied_1_a, 1), (virtual_1_a, 0)), -1 * coeff
                )
        
    return generator


class Main:
    def run(self, mol):
        return excited_state_VQD_solver(mol, depth=5)


if __name__ == "__main__":
    geometry = [
        # ["H", [0.0, 0.0, 0.0]],
        # ["H", [0.0, 0.0, 1.4]],
        ["H", [-0.7652202948444247, 0.06003946218716467, -0.027446108666039762]],
    ["H", [-0.02854009401695, 0.05916010250720479, -0.01009669093305215]],
    ["H", [2.9063332212230697, 0.06599913214823182, 0.03663327695324018]],
    ["H", [3.640338623941231, 0.03715886151989883, 0.05092879742461919]],
    ]
    basis = "sto3g"
    spin = 0
    print("Geometry: \n", geometry)

    # Initialize molcular data
    # mol = MolecularData(geometry, basis, multiplicity=2 * spin + 1)
    mol = MolecularData(geometry, basis, multiplicity=2 * spin + 1, data_directory=sys.path[0])
    mol = run_pyscf(mol)
    es_en = excited_state_VQD_solver(mol, depth=3)
    