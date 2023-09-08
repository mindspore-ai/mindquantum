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
from mindquantum.algorithm.nisq import uccsd_singlet_generator
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import FermionOperator
from scipy.optimize import minimize
import time

class Timer:
    def __init__(self, t0=0.0):
        self.start_time = time.time()
        self.t0 = t0

    def runtime(self):
        return time.time() - self.start_time + self.t0

    def resetime(self):
        self.start_time = time.time()

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
        run_fci=1,
    )
    #print("FCI energy: %20.16f Ha" % (mol.fci_energy))
    ham_of = mol.get_molecular_hamiltonian()
    inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
    ham_hiq = FermionOperator(inter_ops)
    ham_fo = normal_ordered(ham_hiq).real
    ham = ESConserveHam(ham_fo)
    ucc_fermion_ops = uccsd_singlet_generator(mol.n_qubits, mol.n_electrons, anti_hermitian=False)
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

    p0 = np.random.uniform(size=len(circ.params_name)) * 0.01
    res = minimize(fun, p0, args=(grad_ops, ), jac=True, method='bfgs')
    return res.fun

def opti_geo(geo, mol_name):
    global diedaicishu
    diedaicishu+=1
    if timer.runtime()>1200:
        return res
    ham, circ, nq, ne = gene_uccsd([mol_name, geo])
    res = run_uccsd(ham, circ, nq, ne)
    print(res,'\t', time.ctime())
    return res

diedaicishu = 0
timer = Timer()
name, p0 = read_csv(args.input_mol)
p0=np.random.uniform(size=len(p0))
res = minimize(opti_geo, p0, args=(name, ), method='BFGS')
best_x = res.x.reshape(len(name), -1)

print(f"Ground state: \n{res.fun}\n")

out = []
for idx, n in enumerate(name):
    tmp = [n]
    tmp.extend([str(i) for i in best_x[idx]])
    out.append(', '.join(tmp) + '\n')

with open(args.output_mol, 'w') as f:
    f.writelines(out)
