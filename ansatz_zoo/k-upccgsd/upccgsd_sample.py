import os


os.environ['OMP_NUM_THREADS'] = '4'
import time
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool as ThreadPool

from openfermionpyscf import run_pyscf
from openfermion.chem import MolecularData
from mindquantum import Circuit, X, RX, Hamiltonian
from mindquantum.simulator.simulator import Simulator
import mole
from data import initdata, savedata
from ucc import generate_upccgsd


def energy_obj(n_paras, mol_pqc):

    ansatz_data = np.array(n_paras)
    e, grad = mol_pqc(ansatz_data)
    return np.real(e[0, 0]), np.real(grad[0, 0])


mole_name = 'H4'
basis = 'sto3g'
charge = 0
multiplicity = 1
transform = 'jordan_wigner'
method = "2-UpCCGSD"

bond_lengths = [0.1 * i + 0.6 for i in range(15)]
bbond_lengths = [0.2 * i + 2.2 for i in range(5)]
bond_lengths.extend(bbond_lengths)


def process(bond_len):
    geometry = getattr(mole, 'get_{}_geo'.format(mole_name))(bond_len)

    molecule_of = MolecularData(geometry, basis, multiplicity)
    molecule_of = run_pyscf(molecule_of, run_scf=1, run_ccsd=1, run_fci=1)

    start = time.time()
    hartreefock_wfn_circuit = Circuit(
        [X.on(i) for i in range(molecule_of.n_electrons)])

    k_upccgsd_circuit, parameters_name, qubit_hamiltonian, \
    n_qubits, n_electrons = generate_upccgsd(molecule_of, k=1)

    hamiltonian_QubitOp = qubit_hamiltonian.real
    sparsed_ham = Hamiltonian(hamiltonian_QubitOp)
    # sparsed_ham.sparse(n_qubits)
    total_circuit = hartreefock_wfn_circuit + k_upccgsd_circuit
    total_pqc = Simulator('projectq', n_qubits).get_expectation_with_grad(
        sparsed_ham, total_circuit)

    #para_only_energy_obj = partial(energy_obj, hea_circuit, q_ham)

    n_paras = len(total_circuit.params_name)
    paras = [1e-6 for i in range(n_paras)]

    res = minimize(energy_obj,
                   paras,
                   args=(total_pqc, ),
                   method='bfgs',
                   jac=True,
                   tol=1e-6)
    energy = float(res.fun)

    # time cost
    t_cost = time.time() - start
    return [bond_len, energy, t_cost, n_paras]


pool = ThreadPool()
results = pool.map(process, bond_lengths)
pool.close()
pool.join()
print(results)
''' Save the data to json file
'''
# Only run the next line at the first time for each molecule
#initdata(mole_name)
# Save data
#savedata(mole_name, results, method, init=True)