import os
os.environ['OMP_NUM_THREADS'] = '4'
import mindspore as ms
from mindspore import Tensor
from openfermion.utils import count_qubits
from mindquantum.circuit import Circuit
from mindquantum.gate import Hamiltonian, RX, RY, RZ, X
from mindquantum.nn import generate_pqc_operator
from mindquantum.ansatz import QubitUCCAnsatz
from q_ham import q_ham_producer
import numpy as np
from scipy.optimize import minimize
from data import initdata
import mole
import time
import json
from multiprocessing import Pool as ThreadPool
from data import initdata, savedata

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="CPU")


def energy_obj(n_paras, mol_pqc):
    encoder_data = Tensor(np.array([[0]]).astype(np.float32))
    ansatz_data = Tensor(np.array(n_paras).astype(np.float32))
    e, _, grad = mol_pqc(encoder_data, ansatz_data)
    return e.asnumpy()[0, 0], grad.asnumpy()[0, 0]



bond_lengths = [ 0.1*i+0.6 for i in range(15)]
bbond_lengths = [ 0.2*i+2.2 for i in range(5)]
bond_lengths.extend(bbond_lengths)

mole_name = 'N2'
basis = 'sto3g'
charge = 0
multiplicity = 1
transform = 'jordan_wigner'
method = 'QUCC'

# Only run the next line at the first time for each molecule
#initdata(mole_name)

bond_lengths = [ 0.1*i+0.6 for i in range(15)]
bbond_lengths = [ 0.2*i+2.2 for i in range(5)]
bond_lengths.extend(bbond_lengths)

def process(bond_len): 
    geometry = getattr(mole, 'get_{}_geo'.format(mole_name))(bond_len)

    n_qubits,n_electrons, \
    hf_energy, ccsd_energy, \
    fci_energy, q_ham = q_ham_producer(geometry, basis, charge, multiplicity, transform)

    start = time.time()
    # step 2: constructe qUCC ansatz circuit

    # Create such a qUCC circuit

    initial_circuit = Circuit([X.on(i) for i in range(n_electrons)])

    qucc = QubitUCCAnsatz(n_qubits, n_electrons)

    qucc_circuit = qucc.circuit

    total_circuit = initial_circuit + qucc_circuit

    # step 3: objective function
    # generate a circuit that have right number of qubits
    total_pqc = generate_pqc_operator(["null"], total_circuit.para_name, \
                                    RX("null").on(n_qubits-1) + total_circuit, \
                                    Hamiltonian(q_ham))

    # step 4: optimization step.
    n_paras = [0.0 for j in range(len(total_circuit.para_name))]
    res = minimize(energy_obj,
                n_paras,
                args=(total_pqc, ),
                method='bfgs',
                jac=True,
                tol=1e-6)

    n_p = res.x.tolist()
    n_paras= len(total_circuit.para_name)
    energy = float(res.fun)
    t_cost = time.time() - start

    print(energy)
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
savedata(mole_name, results, method, init=True)
        