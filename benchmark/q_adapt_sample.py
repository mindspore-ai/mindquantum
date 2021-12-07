import os
os.environ['OMP_NUM_THREADS'] = '4'
import time
from pool import singlet_SD, pauli_pool
from qubit_adaptive import QubitAdaptive
import mole
from multiprocessing import Pool as ThreadPool

"""
    n_orb: total spatial orbital
    n_occ: occupied spatial orbital
    n_vir: virtual spatial orbital
"""
method = "qubit-ADAPT"

mole_name = 'LiH'
basis = 'sto3g'
charge = 0
multiplicity = 1
transform = 'jordan_wigner'

# for molecule lih, num spatial orbital 6, virtual orbital 4 and occupied orbital 2.
n_orb, n_occ, n_vir = 6,2,4
pool = singlet_SD(n_orb=n_orb, n_occ=n_occ, n_vir=n_vir)
pool.generate_operators()
collection_of_ferops = pool.fermi_ops
collection_of_pauli_strings_without_z = pauli_pool(collection_of_ferops)

print('-------------generate fermion pool-------------')
print('Num of fermionic operator in the pool:{}'.format(len(collection_of_ferops)))

#bond_lengths = [ 0.1*i+1.7 for i in range(4)]
bond_lengths = [ 0.2*i+2.2 for i in range(1)]
#bond_lengths.extend(bbond_lengths)
#results = []
def process(bond_len):
    geometry = getattr(mole, 'get_{}_geo'.format(mole_name))(bond_len)

    start = time.time()

    qa = QubitAdaptive(mole_name, geometry, basis, charge, multiplicity, collection_of_pauli_strings_without_z)

    qa.process()

    energy = float(qa.step_energies[-1])
    n_paras = len(qa.step_energies)
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