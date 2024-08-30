#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/13 

# 量子门数量增加，噪声线性提升
# - ZNE 方案应当是可靠的，但如何扩大两比特门呢

from pathlib import Path
from tqdm import tqdm
from scipy.optimize import curve_fit

from solution import *
import simulator ; simulator.init_shots_counter()
from benchmark_ansatz import IMG_PATH
save_img_fp = IMG_PATH / f'{Path(__file__).stem}.png'

REPEAT = 100
SHOTS = 1000

n_gates = range(0, 30)
exp_avg = []
exp_std = []
for n_gate in tqdm(n_gates):
  circ = Circuit()
  for _ in range(n_gate):
    circ += RY(2 * np.pi / n_gate).on(0)

  sim = HKSSimulator('mqvector', 1)
  ops = QubitOperator('Z0')
  result_list = []
  for _ in range(REPEAT):
    res = measure_single_ham(sim, circ, None, ops, SHOTS)
    result_list.append(res)
  results = np.asarray(result_list)
  exp_avg.append(results.mean())
  exp_std.append(results.std())

print('mean:', exp_avg)
print('std:', exp_std)

f = lambda x, k, b: k * x + b
(k, b), pcov = curve_fit(f, list(n_gates), exp_avg)
# [RY] y = -0.001295813123288742 * x + 0.8997612902876868
# [RX] y = -0.001304845391673250 * x + 0.9000922581792621
# [RZ] y = -0.000834656285802189 * x + 0.8992831828107984
print(f'curve_fit(mean): y = {k} * x + {b}')

import matplotlib.pyplot as plt
plt.plot(n_gates, exp_avg, label='mean', c='orange')
plt.legend()
ax = plt.twinx()
ax.plot(n_gates, exp_std, label='std', c='dodgerblue')
plt.legend()
plt.suptitle('Measure |0> with noise scaling up')
plt.tight_layout()
plt.savefig(save_img_fp, dpi=600)
