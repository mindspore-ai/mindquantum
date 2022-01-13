import os

os.environ['OMP_NUM_THREADS'] = '4'
import gc
import time
from multiprocessing import Pool as ThreadPool

import mole
from adapt_vqe import AdaptVqe
from data import initdata, savedata
from pool import singlet_SD, pauli_pool, singlet_GSD
"""
    n_orb: total spatial orbital
    n_occ: occupied spatial orbital
    n_vir: virtual spatial orbital
"""
method = "ADAPT"

mole_name = 'H4'
basis = 'sto3g'
charge = 0
multiplicity = 1
transform = 'jordan_wigner'

# for molecule lih, num spatial orbital 6, virtual orbital 4 and occupied orbital 2.
n_orb, n_occ, n_vir = 4, 2, 2
pool = singlet_SD(n_orb=n_orb, n_occ=n_occ, n_vir=n_vir)
pool.generate_operators()
collection_of_ferops = pool.fermi_ops

print('-------------generate fermion pool-------------')
print('Num of fermionic operator in the pool:{}'.format(
    len(collection_of_ferops)))

#bond_lengths = [ 0.1*i+1.7 for i in range(4)]
bond_lengths = [3.0]
#bond_lengths.extend(bbond_lengths)
results = []
for bond_len in bond_lengths:
    geometry = getattr(mole, 'get_{}_geo'.format(mole_name))(bond_len)

    start = time.time()
    adapt = AdaptVqe(mole_name, geometry, basis, charge, multiplicity,
                     collection_of_ferops)

    adapt.opti_process(
        maxiter=100, adapt_thresh=1e-2
    )  # Set optimizaiton max iteration and gradient threshold

    energy = float(adapt.step_energies[-1])
    n_paras = len(adapt.step_energies)
    # time cost
    t_cost = time.time() - start
    results.append([bond_len, energy, t_cost, n_paras])
    del adapt
    gc.collect()
    print(results)

''' Save the data to json file
'''

# Only run the next line at the first time for each molecule
#initdata(mole_name)

# Save data
#savedata(mole_name, results, method)