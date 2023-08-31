import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-mol", help="input molecular data", type=str, default="h4.csv")
parser.add_argument("-x", "--output-mol", help="output molecular data", type=str, default="h4_best.csv")
args = parser.parse_args()

# Your solution is below
# Example:
import numpy as np
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.core.operators import InteractionOperator, normal_ordered
from qupack.vqe import ESConserveHam, ExpmPQRSFermionGate, ESConservation
from qupack.vqe.pr_converter import pr_converter
from mindquantum.algorithm.nisq import uccsd_singlet_generator
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X, RY
from mindquantum.core.operators import FermionOperator
from mindquantum.simulator import Simulator 
from mindquantum.core.parameterresolver import ParameterResolver
from scipy.optimize import minimize
import time

# UCCSD but only CAS orbitals are involved
# ncas : label helps to tell the different layers
# noe : number of orbitals / electrons in CAS
def casci_generator(n_qubits, n_electrons, anti_hermitian=True,  ncas="lay1", noe=2):
    import itertools
    from openfermion.utils.indexing import down_index, up_index
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
    #for i, (p, q) in enumerate(itertools.product(range(n_virtual), range(n_occupied))):
    for i, (p, q) in enumerate(itertools.product(range(noe), range(n_occupied-noe,n_occupied))):
        # Get indices of spatial orbitals
        virtual_spatial = n_occupied + p
        occupied_spatial = q
        if virtual_spatial>=n_spatial_orbitals:
            continue

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
            coeff = ParameterResolver({f'{ncas}_s_{i}': 1})
            generator += FermionOperator(((virtual_this, 1), (occupied_this, 0)), coeff)
            if anti_hermitian:
                generator += FermionOperator(((occupied_this, 1), (virtual_this, 0)), -1 * coeff)

            # Generate double excitation
            coeff = ParameterResolver({f'{ncas}_d1_{i}': 1})
            generator += FermionOperator(
                ((virtual_this, 1), (occupied_this, 0), (virtual_other, 1), (occupied_other, 0)), coeff
            )
            if anti_hermitian:
                generator += FermionOperator(
                    ((occupied_other, 1), (virtual_other, 0), (occupied_this, 1), (virtual_this, 0)), -1 * coeff
                )
    # Generate all spin-conserving double excitations derived from two spatial occupied-virtual pairs
    for i, ((p, q), (r, s)) in enumerate(
        itertools.combinations(itertools.product(range(noe), range(n_occupied-noe,n_occupied)), 2)
    ):
        # Get indices of spatial orbitals
        virtual_spatial_1 = n_occupied + p
        occupied_spatial_1 = q
        virtual_spatial_2 = n_occupied + r
        occupied_spatial_2 = s
        if virtual_spatial_1>=n_spatial_orbitals or virtual_spatial_2>=n_spatial_orbitals:
            continue

        # Generate double excitations
        coeff = ParameterResolver({f'{ncas}_d2_{i}': 1})
        for (spin_a, spin_b) in itertools.product(range(2), repeat=2):
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


def gene_uccsd(mol):
    geometry = mol[1].reshape(len(mol[0]), -1)
    geometry = [[mol[0][i], list(j)] for i, j in enumerate(geometry)]
    basis = "sto3g"
    molecule_of = MolecularData(geometry, basis, multiplicity=1, data_directory='./')
    mol = run_pyscf(
        molecule_of,
        #run_fci=1,
    )
    ham_of = mol.get_molecular_hamiltonian()
    inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
    ham_hiq = FermionOperator(inter_ops)
    ham_fo = normal_ordered(ham_hiq).real
    ham = ESConserveHam(ham_fo)
    ucc_fermion_ops = casci_generator(mol.n_qubits, mol.n_electrons, anti_hermitian=False,noe=2)
    #ucc_fermion_ops += casci_generator(mol.n_qubits, mol.n_electrons, anti_hermitian=False,ncas="lay2",noe=3)
    circ = Circuit()
    for term in ucc_fermion_ops:
        circ += ExpmPQRSFermionGate(term)
    return ham, circ, mol.n_qubits, mol.n_electrons


def run_uccsd(ham, circ, nq, ne):
    sim = ESConservation(nq, ne)

    grad_ops = sim.get_expectation_with_grad(ham, circ)

    def fun(p0, grad_ops):
        f, g = grad_ops(p0)
        return f.real, g.real

    #p0 = np.random.uniform(size=len(circ.params_name)) * 0.01
    p0 = np.ones((len(circ.params_name),)) * 0.1
    res = minimize(fun, p0, args=(grad_ops, ), jac=True, method='bfgs')

    return res.fun

def opti_geo(geo, mol_name):
    ham, circ, nq, ne = gene_uccsd([mol_name, geo])
    res = run_uccsd(ham, circ, nq, ne)
    print(res,'\t', time.ctime())
    best_x = geo.reshape(len(mol_name), -1)
    out = []
    for idx, n in enumerate(mol_name):
        tmp = [n]
        tmp.extend([str(i) for i in best_x[idx]])
        out.append(', '.join(tmp) + '\n')

    with open(args.output_mol, 'w') as f:
        f.writelines(out)

    return res

Start = time.time()
name, p0 = read_csv(args.input_mol)
#p0 = np.random.uniform(size=len(p0)) 
res = minimize(opti_geo, p0, args=(name, ), method='BFGS',options={"disp":True})
best_x = res.x.reshape(len(name), -1)
print("Total time : ", time.time()-Start)

out = []
for idx, n in enumerate(name):
    tmp = [n]
    tmp.extend([str(i) for i in best_x[idx]])
    out.append(', '.join(tmp) + '\n')

with open(args.output_mol, 'w') as f:
    f.writelines(out)
