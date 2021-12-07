import os
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import json
from multiprocessing import Pool as ThreadPool
from mindspore import Tensor
import mindspore as ms
from mindquantum.ansatz import HardwareEfficientAnsatz
from mindquantum.gate import Hamiltonian, RX, RY, RZ, X
from mindquantum.nn import generate_pqc_operator
from scipy.optimize import minimize
from q_ham import q_ham_producer
from data import savedata
import mole
import time

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="CPU")

mole_name = 'H2O'
basis = 'sto3g'
charge = 0
multiplicity = 1
transform = 'jordan_wigner'
method = "Hartree-Fock"

single_rot_gate_seq, entangle_gate, entangle_mapping = [RY, RZ], X, 'linear'

bond_lengths = [0.1*i+0.7 for i in range(14)]
bbond_lengths = [0.2*i+2.2 for i in range(5)]
bond_lengths.extend(bbond_lengths)


def process(bond_len):
    geometry = getattr(mole, 'get_{}_geo'.format(mole_name))(bond_len)
    n_qubits, n_electrons, \
    hf_energy, ccsd_energy, \
    fci_energy, q_ham = q_ham_producer(geometry, basis, charge, multiplicity, transform)

    count = 0
    depth = 1
    # start time
    start = time.time()
    hea = HardwareEfficientAnsatz(n_qubits, single_rot_gate_seq,
                                  entangle_gate, entangle_mapping, depth)

    def energy_obj(n_paras, mol_pqc):
        nonlocal count
        count += 1

        encoder_data = Tensor(np.array([[0]]).astype(np.float32))
        ansatz_data = Tensor(np.array(n_paras).astype(np.float32))
        e, _, grad = mol_pqc(encoder_data, ansatz_data)
        return e.asnumpy()[0, 0], grad.asnumpy()[0, 0]

    while (count < 5e4):
        energy = 0
        for i in range(10):
            hea_circuit = hea.circuit

            mol_pqc = generate_pqc_operator(
                ["null"], hea_circuit.para_name, RX("null").on(0) + hea_circuit, Hamiltonian(q_ham)
            )

            n_paras = len(hea_circuit.para_name)
            paras = [np.random.uniform(-np.pi, np.pi) for i in range(n_paras)]

            res = minimize(energy_obj, paras, args=(mol_pqc, ),
                           method='bfgs', jac=True, tol=1e-6)

            energy = float(res.fun)
            print(bond_len, count, n_paras, energy, fci_energy)
            if abs(energy-fci_energy) < 1.6e-3:
                break
        if abs(energy-fci_energy) < 1.6e-3:
            break
        depth += 1
        hea = HardwareEfficientAnsatz(n_qubits, single_rot_gate_seq, 
                                      entangle_gate, entangle_mapping, depth)

    # time cost
    t_cost = time.time()-start
    return [bond_len, energy, t_cost, n_paras]


pool = ThreadPool()
results = pool.map(process, bond_lengths)
pool.close()
pool.join()
print(results)

''' Save the data to json file
'''
# Only run the next line at the first time for each molecule
# initdata(mole_name)
# Save data
savedata(mole_name, results, method, init=True)
