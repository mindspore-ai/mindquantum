#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/25

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.absolute()
sys.path.append(str(BASE_PATH))
from solver import *
import solver as S

from copy import deepcopy
import matplotlib.pyplot as plt

# 导出 PPT 用对比图：两阶段优化


def go(args):
  name, init_x = read_csv(args.input_mol)
  ITER_RATIO = 0.01

  # run simple
  args.optim = 'BFGS'
  args.maxiter = int(1000 * ITER_RATIO)
  run(args, name, init_x)

  # collect data
  s_ene = deepcopy(S.track_ene)
  s_fci = [get_mol(name, geo.reshape(len(name), -1)).fci_energy for geo in S.track_geo]

  # reset globals
  S.steps = 0
  S.circ = None
  S.sim = None
  S.track_ene.clear()
  S.track_geo.clear()

  # run compoud
  configs = [
    (200, 'COBYLA'),         # this is fast
    (800, 'BFGS'),           # this is precise
    #(800, 'trust-constr'),   # this is precise (but overfit?)
  ]
  
  best_x = init_x
  for i, (maxiter, optim) in enumerate(configs):
    print(f'>> round {i}: optim use {optim}')
    args.optim = optim
    args.maxiter = int(maxiter * ITER_RATIO)
    if i > 0: args.init = 'orig'
    name, best_x = run(args, name, best_x)    # make successive

  c_ene = deepcopy(S.track_ene)
  c_fci = [get_mol(name, geo.reshape(len(name), -1)).fci_energy for geo in S.track_geo]

  # make plot
  # https://matplotlib.org/stable/tutorials/colors/colors.html
  fig, ax = plt.subplots()
  ax.plot(s_fci, c='blue', label='simple fci')
  ax.plot(s_ene, c='blue', label='simple ene', alpha=0.5)
  ax.plot(c_fci, c='red', label='compound fci')
  ax.plot(c_ene, c='red', label='compound ene', alpha=0.5)
  fig.legend()
  fig.tight_layout()
  fig.savefig('abl_two_stage_opt.png')


if __name__ == '__main__':
  args = get_args()
  
  expname = 'log.solver_H4_cmp_opt'
  log_dp = os.path.abspath(os.path.join(os.getcwd(), expname))
  os.makedirs(log_dp, exist_ok=True)

  args.input_mol = str(BASE_PATH / 'h4.csv')
  args.objective = 'uccsd'
  args.init      = 'linear'
  args.track     = True
  args.log_dp    = log_dp

  globals()['args'] = args
  S.args = args
  go(args)
