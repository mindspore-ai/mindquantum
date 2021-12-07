import os
os.environ['OMP_NUM_THREADS'] = '4'
import time
import json
from multiprocessing import Pool as ThreadPool
import numpy as np
from mindspore import Tensor
import mindspore as ms
from mindquantum import Circuit, X, RX, Hamiltonian
from mindquantum.circuit import generate_uccsd
from mindquantum.nn import generate_pqc_operator
from scipy.optimize import minimize
from openfermionpyscf import run_pyscf
from openfermion.chem import MolecularData
from data import initdata, savedata
import mole


ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="CPU")

def energy_obj(n_paras, mol_pqc):
    encoder_data = Tensor(np.array([[0]]).astype(np.float32))
    ansatz_data = Tensor(np.array(n_paras).astype(np.float32))
    e, _, grad = mol_pqc(encoder_data, ansatz_data)
    return e.asnumpy()[0, 0], grad.asnumpy()[0, 0]

method = "UCCSD"

mole_name = 'H2'
basis = 'sto3g'
charge = 0
multiplicity = 1
transform = 'jordan_wigner'

bond_lengths = [0.1*i+0.3 for i in range(20)]

def process(bond_len):
    geometry = getattr(mole, 'get_{}_geo'.format(mole_name))(bond_len)

    molecule_of = MolecularData(
    geometry,
    basis,
    multiplicity)
    molecule_of = run_pyscf(
    molecule_of,
    run_scf=1,
    run_ccsd=1,
    run_fci=1)

    start = time.time()
    hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(molecule_of.n_electrons)])

    ansatz_circuit, \
    init_amplitudes, \
    ansatz_parameter_names, \
    hamiltonian_QubitOp, \
    n_qubits, n_electrons = generate_uccsd(molecule_of, th=-1)
    hamiltonian_QubitOp = hamiltonian_QubitOp.real

    total_circuit = hartreefock_wfn_circuit + ansatz_circuit

    total_pqc = generate_pqc_operator(["null"], ansatz_parameter_names, \
                                RX("null").on(n_qubits-1) + total_circuit, \
                                Hamiltonian(hamiltonian_QubitOp))

    #para_only_energy_obj = partial(energy_obj, hea_circuit, q_ham)
    
    n_paras = len(total_circuit.para_name)
    paras = init_amplitudes

    res = minimize(energy_obj,
            paras,
            args=(total_pqc, ),
            method='bfgs',
            jac=True,
            tol=1e-6)
    energy = float(res.fun)
    energy = molecule_of.hf_energy
    # time cost
    t_cost = time.time()-start 
    return [bond_len, energy, t_cost, n_paras]

pool = ThreadPool()
results = pool.map(process, bond_lengths)
pool.close()
pool.join()

''' Save the data to json file
'''
print(results)
# Only run the next line at the first time for each molecule
#initdata(mole_name)

# Save data
#savedata(mole_name, results, method)
