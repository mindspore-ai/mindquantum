#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/13 

# 查看测量采样的噪声方差，讨论为何结果很难稳定 :(

from time import time
from solution import *
import simulator ; simulator.init_shots_counter()

''' Config '''
REPEAT = 100    # 812s, ~15min
SHOTS = 1000
DEPTH = 3

''' Data '''
molecule = [
  ['H', [0, 0, 0.0]],
  ['H', [0, 0, 1.0]],
  ['H', [0, 0, 2.0]],
  ['H', [0, 0, 3.0]],
]
mol = generate_molecule(molecule)
ham = get_molecular_hamiltonian(mol)
const, split_ham = split_hamiltonian(ham)
split_ham = prune_hamiltonian(split_ham)

''' Circuit '''
circ = get_hae_ry_circit(mol, DEPTH)
init = get_hae_ry_circit_pretrained_params(DEPTH)
pr = ParameterResolver(dict(zip(circ.params_name, init)))

''' Simulator '''
sim = Simulator('mqvector', mol.n_qubits)
exp_truth = sim.get_expectation(Hamiltonian(ham), circ, pr=pr).real
print('exp_truth:', exp_truth)

''' Hardware '''
sim = HKSSimulator('mqvector', mol.n_qubits)
ts_start = time()
result_list = []
for _ in range(REPEAT):
  res = const + get_exp(sim, circ, pr, split_ham, shots=SHOTS, debug_log=False)
  result_list.append(res)
results = np.asarray(result_list)
ts_end = time()
print('time cost:', ts_end - ts_start)

'''
exp_truth: -2.1381799552114513
max: -1.7705811633777586
min: -1.867114322637767
avg: -1.813848575991954
std: 0.019782057462345747
min score: 2.720357145385876
max score: 3.6891434391933475
'''
print('max:', results.max())
print('min:', results.min())
print('avg:', results.mean())
print('std:', results.std())
error: ndarray = np.abs(results - exp_truth)
print('min score:', 1 / error.max())
print('max score:', 1 / error.min())
