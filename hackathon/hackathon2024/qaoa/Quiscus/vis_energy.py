#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/07 

# 查看各样例在指定查找表上的初始能量

import numpy as np
from numpy import ndarray
from mindquantum.simulator import Simulator
from mindquantum.core.operators import Hamiltonian
from tqdm import tqdm

from utils.path import LOG_PATH
from utils.qcirc import qaoa_hubo, build_ham_high
from score import load_data
from main import main_baseline, main


def make_score_mat() -> ndarray:
  Nq = 12
  sim = Simulator('mqvector', n_qubits=Nq)
  # [prop, k, coef, r, depth(p)]
  score_mat = np.zeros([3, 4, 3, 10, 2], dtype=np.float32)

  total = score_mat.size
  pbar = tqdm(total=total)
  for pid, propotion in enumerate([0.3, 0.6, 0.9]):
    for kid, k in enumerate(range(2, 6)):
      for cid, coef in enumerate(['std', 'uni', 'bimodal']):
        for r in range(10):
          fp = f"data/k{k}/{coef}_p{propotion}_{r}.json"
          Jc_dict = load_data(fp)
          ham = Hamiltonian(build_ham_high(Jc_dict))

          for did, depth in enumerate([4, 8]):
            # NOTE: change this to what lookup_table you want to use!!
            gamma_List, beta_List = main_baseline(Jc_dict, depth, Nq)
            #gamma_List, beta_List = main(Jc_dict, depth, Nq)
            circ = qaoa_hubo(Jc_dict, Nq, gamma_List, beta_List, p=depth)
            E = sim.get_expectation(ham, circ).real
            score_mat[pid, kid, cid, r, did] = E

            pbar.update()

  return score_mat


save_fp = LOG_PATH / 'score_mat.npy'
#save_fp = LOG_PATH / 'score_mat_ft-ada-decay_T=9400.npy'
if not save_fp.exists():
  score_mat = make_score_mat()
  np.save(save_fp, score_mat)
score_mat = np.load(save_fp)


if 'stats':
  print([score_mat[i, ...]         .mean() for i in range(3)])
  print([score_mat[i, ...]         .std()  for i in range(3)])
  print([score_mat[:, i, ...]      .mean() for i in range(4)])
  print([score_mat[:, i, ...]      .std()  for i in range(4)])
  print([score_mat[:, :, i, ...]   .mean() for i in range(3)])
  print([score_mat[:, :, i, ...]   .std()  for i in range(3)])
  print([score_mat[:, :, :, i, ...].mean() for i in range(10)])
  print([score_mat[:, :, :, i, ...].std()  for i in range(10)])
  print([score_mat[..., i]         .mean() for i in range(2)])
  print([score_mat[..., i]         .std()  for i in range(2)])

### baseline
# [prop] edge density
#   avg: [-127.36577, -157.32594, -135.90237]
#   std: [71.151474, 88.70964, 119.21811]
# [k] order of ising-model
#   avg: [-68.222496, -125.817276, -138.61066, -228.14171]
#   std: [29.53053, 50.32695, 109.61016, 91.023994]
# [coef] edge weight probdist
#   avg: [-89.83214, -114.52684, -216.2351]
#   std: [78.56908, 56.96428, 96.05193]
# [r] sample index
#   avg: [-139.10056, -139.65471, -140.61537, -140.91692, -141.76175, -138.6221, -138.42047, -141.0248, -141.98668, -139.87698]
#   std: [96.0376, 97.350685, 93.64047, 96.520065, 95.76261, 93.9899, 96.85584, 96.792404, 99.28371, 93.06068]
# [p] circuit depth
#   avg: [-126.21123, -154.18483]
#   std: [82.35353, 106.02304]
#
### ft-ada-decay_T=9400
# [prop] edge density (0, -7, -35 of each; 边越密集增益越大)
#   avg: [-127.6368, -164.26228, -158.56534]
#   std: [73.07497, 92.10416, 103.769066]
# [k] order of ising-model (0, -2, -31, -7 of each; 阶数越高增益越大)
#   avg: [-68.5941, -127.13828, -169.03491, -235.85196]
#   std: [30.184612, 50.04047, 77.59191, 97.498344]
# [coef] edge weight probdist (-22, +2, -9 of each; 分布为uni的样例有退化，TODO: 可能要按不同分布分别建模)
#   avg: [-111.96218, -112.613014, -225.8892]
#   std: [53.582024, 56.71622, 103.32686]
# [r] sample index (-10 on avg)
#   avg: [-149.15585, -150.24004, -150.1911, -150.52188, -152.05824, -148.99612, -148.21541, -150.36801, -150.84392, -150.95749]
#   std: [91.202065, 94.491295, 91.501434, 90.81646, 90.03252, 91.7499, 92.028015, 93.54402, 92.99735, 91.050766]
# [p] circuit depth (-7, -12 for each)
#   avg: [-133.74776, -166.56184]
#   std: [79.67872, 100.124405]

from code import interact
interact(local=globals())
