# Author: Rongge Xu, Hui Dai (Durga Team)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-mol", help="input molecular data", type=str, default="h4.csv")
parser.add_argument("-x", "--output-mol", help="output molecular data", type=str, default="h4_best.csv")
args = parser.parse_args()

# Your solution is below
# Example:
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "4"
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.core.operators import InteractionOperator, normal_ordered
from qupack.vqe import ESConserveHam, ExpmPQRSFermionGate, ESConservation
from mindquantum.algorithm.nisq import uccsd_singlet_generator
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import FermionOperator
from mindquantum.core.parameterresolver import ParameterResolver
from scipy.optimize import minimize
import time
import itertools

def read_csv(file_name):
    with open(file_name, 'r') as f:
        data = f.readlines()
    mol_name = []
    mol_poi = []
    for i in data:
        tmp = i.split(',')
        mol_name.append(tmp[0])
        mol_poi.extend([float(i) for i in tmp[1:]])
    return mol_name, np.array(mol_poi)

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

def gene_uccsd(mol):
    geometry = mol[1].reshape(len(mol[0]), -1)
    geometry = [[mol[0][i], list(j)] for i, j in enumerate(geometry)]
    basis = "sto3g"
    molecule_of = MolecularData(geometry, basis, multiplicity=1, data_directory='./')
    mol = run_pyscf(
        molecule_of,
        run_fci=1,
    )
    ham_of = mol.get_molecular_hamiltonian()
    inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
    ham_hiq = FermionOperator(inter_ops)
    ham_fo = normal_ordered(ham_hiq).real
    ham = ESConserveHam(ham_fo)
    circ = Circuit()
    if mol.n_qubits <= 8:
        depth = 1
    elif mol.n_qubits <= 16:
        depth = 1
    else:
        depth = 1
    for d in range(depth):
        ucc_fermion_ops = uccsd_singlet_generator_with_pr(mol.n_qubits, mol.n_electrons, anti_hermitian=False, depth=d+1)
        for term in ucc_fermion_ops:
            circ += ExpmPQRSFermionGate(term)
    return ham, circ, mol.n_qubits, mol.n_electrons


def run_uccsd(ham, circ, nq, ne, tol=1e-5):
    sim = ESConservation(nq, ne)
    grad_ops = sim.get_expectation_with_grad(ham, circ)

    def fun(p0, grad_ops):
        f, g = grad_ops(p0)
        return f.real, g.real

    p0 = np.random.uniform(size=len(circ.params_name)) * 0.1

    res = minimize(fun, p0, args=(grad_ops, ), jac=True, method='bfgs', tol=tol, options={'maxiter':200})
    return res.fun

def opti_geo(geo, mol_name):
    ham, circ, nq, ne = gene_uccsd([mol_name, geo])
    res = run_uccsd(ham, circ, nq, ne)
    return res


name, geo = read_csv(args.input_mol)
t0 = time.time()
res = minimize(opti_geo, geo, args=(name, ), method='BFGS', tol=1e-4, options={'maxiter':200})
t1 = time.time()
print("energy: ", res.fun)
print("time: ", t1 - t0)
print("steps: ", res.nfev)
best_x = res.x.reshape(len(name), -1)

out = []
for idx, n in enumerate(name):
    tmp = [n]
    tmp.extend([str(i) for i in best_x[idx]])
    out.append(', '.join(tmp) + '\n')

with open(args.output_mol, 'w') as f:
    f.writelines(out)

