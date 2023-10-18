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
from data import initdata, savedata
import mole

from ldca import generate_ldca
from numpy.random import rand


def energy_obj(n_paras, mol_pqc):

    ansatz_data = np.array(n_paras)
    e, grad = mol_pqc(ansatz_data)
    # print(f"e = {e}, grad = {grad}")
    # print(f"e={e}")
    return np.real(e[0, 0]), np.real(grad[0, 0])


L = 2
method = f"LDCA"

mole_name = 'LiH'
basis = 'sto3g'
charge = 0
multiplicity = 1
transform = 'jordan_wigner'

bond_lengths = [
    0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2,
    2.4, 2.6, 2.8, 3.0
]  # LiH
# bond_lengths = [1.1]


def process(bond_len):
    geometry = getattr(mole, 'get_{}_geo'.format(mole_name))(bond_len)

    molecule_of = MolecularData(geometry, basis, multiplicity)
    molecule_of = run_pyscf(molecule_of, run_scf=1, run_ccsd=1, run_fci=1)

    start = time.time()

    hartreefock_wfn_circuit = Circuit(
        [X.on(i) for i in range(molecule_of.n_electrons)])

    ansatz_circuit, \
    ansatz_parameter_names, \
    hamiltonian_QubitOp, \
    n_qubits, \
    n_electrons = generate_ldca(molecule_of, L=L)

    hamiltonian_QubitOp = hamiltonian_QubitOp.real
    sparsed_ham = Hamiltonian(hamiltonian_QubitOp)
    sparsed_ham.sparse(n_qubits)

    total_circuit = hartreefock_wfn_circuit + ansatz_circuit
    # total_circuit = Circuit(
    #     [X.on(i) for i in range(n_qubits)]
    # )
    # total_circuit += ansatz_circuit
    # total_circuit = ansatz_circuit

    # total_circuit.summary()

    total_pqc = Simulator('mqvector', n_qubits).get_expectation_with_grad(
        sparsed_ham, total_circuit)

    #para_only_energy_obj = partial(energy_obj, hea_circuit, q_ham)

    n_paras = len(total_circuit.params_name)

    roll_times = 5  # random times for parameter
    paras_list = rand(roll_times * n_paras) * np.pi  # [0,pi]
    energys = []
    for epoch in range(roll_times):
        paras = paras_list[epoch * n_paras:(epoch + 1) * n_paras]
        res = minimize(energy_obj,
                       paras,
                       args=(total_pqc, ),
                       method='bfgs',
                       jac=True,
                       tol=1e-6)

        energys.append(float(res.fun))

    print(f"bond = {bond_len}\nenergys = {energys}")
    energy = min(energys)
    # energy = molecule_of.hf_energy

    # time cost
    t_cost = time.time() - start
    return [bond_len, energy, t_cost, n_paras]


pool = ThreadPool()
results = pool.map(process, bond_lengths)
pool.close()
pool.join()
''' Save the data to json file
'''
print(results)
# Only run the next line at the first time for each molecule
# initdata(mole_name)

# Save data
savedata(mole_name, results, method, init=True)

# LiH
# |1> as initial state
# [[1.4, -7.86056424232997, 1686.7389607429504, 108]]

# hartree fork
# [[1.4, -7.861914113905647, 7852.700343608856, 108]]
# [[0.7, -7.486563265556718, 2104.8185338974, 108]]
# [[0.8, -7.616296443057112, 2699.7702906131744, 108]]
# [[0.9, -7.705866802821873, 949.3966319561005, 108]]
# [[1.0, -7.768063780721664, 1627.3265280723572, 108]]
# [[1.1, -7.811726051920602, 3068.978880405426, 108]]