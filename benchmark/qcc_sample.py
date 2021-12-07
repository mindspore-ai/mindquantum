import os

os.environ['OMP_NUM_THREADS'] = '4'
import time
import numpy as np
from multiprocessing import Pool as ThreadPool

from mindquantum import QubitOperator
import mole
from qcc import generate_pauli_pool, IterQCC
from data import savedata


def energy_obj(n_paras, mol_pqc):

    ansatz_data = np.array(n_paras)
    e, grad = mol_pqc(ansatz_data)
    return np.real(e[0, 0]), np.real(grad[0, 0])


"""
    n_orb: total spatial orbital
    n_occ: occupied spatial orbital
    n_vir: virtual spatial orbital
"""
method = "QCC"

mole_name = 'H4'
basis = 'sto3g'
charge = 0
multiplicity = 1
transform = 'jordan_wigner'

# for molecule lih, num spatial orbital 6, virtual orbital 4 and occupied orbital 2.
n_orb, n_occ, n_vir = 4, 2, 2
ops_pool = []
for paulistring in generate_pauli_pool(2 * n_orb, 4):
    ops_pool.append(1.0 * QubitOperator(tuple(paulistring)))

print('Num of operators in the pool:{}'.format(len(ops_pool)))

bond_lengths = [0.1 * i + 3.0 for i in range(1)]
#bbond_lengths = [ 0.2*i+2.2 for i in range(5)]
#bond_lengths.extend(bbond_lengths)


def process(bond_len):
    geometry = getattr(mole, 'get_{}_geo'.format(mole_name))(bond_len)

    start = time.time()

    qa = IterQCC(mole_name, geometry, basis, charge, multiplicity, ops_pool)

    qa.process()

    energy = float(qa.step_energies[-1])
    n_paras = len(qa.step_energies)
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
#initdata(mole_name)

# Save data
savedata(mole_name, results, method, init=False)
