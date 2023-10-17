import os
from numpy.core.arrayprint import _extendLine
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from multiprocessing import Pool as ThreadPool
import mindspore as ms
from q_ham import q_ham_producer
from functools import partial
from data import initdata
import mole
import json

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="CPU")

mole_name = 'H2O'
basis = 'sto3g'
charge = 0
multiplicity = 1
transform = 'jordan_wigner'

# Only run the next line at the first time for each molecule
#initdata(mole_name)

with open(os.path.join(os.getcwd(), r'data/mindquantum_energies_{}.json'.format(mole_name)), 'r+', newline='') as f:
    data = f.read()
    energies = json.loads(data)

with open(os.path.join(os.getcwd(), r'data/mindquantum_times_{}.json'.format(mole_name)), 'r+', newline='') as f:
    data = f.read()
    times = json.loads(data)

with open(os.path.join(os.getcwd(), r'data/mindquantum_parameters_{}.json'.format(mole_name)), 'r+', newline='') as f:
    data = f.read()
    parameters = json.loads(data)

bond_lengths = [0.1*i+0.7 for i in range(14)]
bbond_lengths = [0.2*i+2.2 for i in range(5)]
bond_lengths.extend(bbond_lengths)

def process(bond_len):
    geometry = getattr(mole, 'get_{}_geo'.format(mole_name))(bond_len)
    n_qubits,n_electrons, \
    hf_energy, ccsd_energy, \
    fci_energy, q_ham = q_ham_producer(geometry, basis, charge, multiplicity, transform)

    return [bond_len, hf_energy, ccsd_energy, fci_energy]

pool = ThreadPool()
results = pool.map(process, bond_lengths)
pool.close()
pool.join()
print(results)

energies["energies"]["Hartree-Fock"] = []
energies["energies"]["full-CI"] = []
energies["energies"]["CCSD"] = []

# Append the calculated energy
for i in range(len(results)):
    #energies["bond_lengths"].append(results[i][0])
    energies["energies"]["Hartree-Fock"].append(results[i][1])
    energies["energies"]["full-CI"].append(results[i][3])
    energies["energies"]["CCSD"].append(results[i][2])
    #times["bond_lengths"].append(results[i][0])
    #parameters["bond_lengths"].append(results[i][0])

with open(os.path.join(os.getcwd(), r'data/mindquantum_energies_{}.json'.format(mole_name)), 'w+', newline='') as f:
    b = json.dumps(energies)
    f.write(b)

with open(os.path.join(os.getcwd(), r'data/mindquantum_times_{}.json'.format(mole_name)), 'w+', newline='') as f:
    b = json.dumps(times)
    f.write(b)

with open(os.path.join(os.getcwd(), r'data/mindquantum_parameters_{}.json'.format(mole_name)), 'w+', newline='') as f:
    b = json.dumps(parameters)
    f.write(b)