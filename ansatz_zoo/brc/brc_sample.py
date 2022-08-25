import json
import os
import time
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool as ThreadPool

# from mindspore.scipy.optimize import minimize
from mindquantum import Hamiltonian
from mindquantum.simulator.simulator import Simulator
from q_ham import q_ham_producer
from data import initdata, savedata, rounddata
import mole
import brc


os.environ['OMP_NUM_THREADS'] = '4'

method = "BRC"

mole_name = 'H4'
# mole_name = 'LiH'
# mole_name = 'BeH2'
# mole_name = 'H2O'
# mole_name = 'CH4'
# mole_name = 'N2'
basis = 'sto3g'
charge = 0
multiplicity = 1
transform = 'jordan_wigner'

# bond_lens = [1.7, 1.8, 2.4, 2.6, 2.8, 3.0]


def bond_lengths(mole):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           r'../data/mindquantum_energies_{}.json'.format(mole)), 'r+', newline='') as f:
        data = f.read()
        energies = json.loads(data)
    return energies["bond_lengths"]


def energy_obj(n_paras, mol_pqc):
    ansatz_data = np.array(n_paras)
    e, grad = mol_pqc(ansatz_data)
    # print(f'energy: {e[0][0]}')
    return np.real(e[0, 0]), np.real(grad[0, 0])


def process(bond_len):
    geometry = getattr(mole, 'get_{}_geo'.format(mole_name))(bond_len)

    n_qubits, n_electrons, \
    hf_energy, ccsd_energy, \
    fci_energy, q_ham = q_ham_producer(geometry, basis, charge, multiplicity, transform)

    sparsed_q_ham = Hamiltonian(q_ham)
    # print(sparsed_q_ham)
    # sparsed_q_ham.sparse(n_qubits)

    # step 2: construct the brc ansatz circuit
    start = time.time()

    brc_ansatz_circuit = brc.ansatz_circuit(n_qubits, n_electrons)
    # print(brc_ansatz_circuit)
    # brc_ansatz_circuit.summary()

    # step 3: objective function
    total_pqc = Simulator('projectq', n_qubits).get_expectation_with_grad(
        sparsed_q_ham, brc_ansatz_circuit)

    # step 4: optimization step.
    n_params = len(brc_ansatz_circuit.params_name)
    energys = []
    roll_times = 5
    for epoch in range(roll_times):  #
        params = [np.random.uniform(-np.pi, np.pi) for i in range(n_params)]
        res = minimize(energy_obj,
                       params,
                       args=(total_pqc,),
                       method='bfgs',
                       jac=True,
                       tol=1e-6)
        energys.append(float(res.fun))
    # params = res.x.tolist()
    # print("optimal parameters:", params)

    energy = min(energys)
    t_cost = time.time() - start  # time cost

    print(f'n_qubits: {n_qubits}')
    print(f'n_electrons: {n_electrons}')
    print(f'bond_length: {bond_len}')
    print("Hartree-Fock energy: %20.16f Ha" % (hf_energy))
    print("CCSD energy: %20.16f Ha" % (ccsd_energy))
    print("FCI energy: %20.16f Ha" % (fci_energy))
    print("BRC energy: %20.16f Ha" % (energy))
    print("---------------------------------------------")
    return [bond_len, energy, t_cost, n_params]


pool = ThreadPool()
results = pool.map(process, bond_lengths(mole_name))
pool.close()
pool.join()

''' Save the data to json file'''
savedata(mole_name, results, method, init=True)

rounddata(mole_name)
