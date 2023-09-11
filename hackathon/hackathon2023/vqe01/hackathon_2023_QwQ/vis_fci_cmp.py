#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/14

import os
import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

OPTIM_METH = [
  'BFGS',
  'L-BFGS-B',
  'SLSQP',
  'CG',
  'TNC',
  'COBYLA',
  'Powell',
  'Nelder-Mead',
  'trust-constr',
  # need jac, can NOT run
  #'Newton-CG',
  #'dogleg',
  #'trust-ncg',
  #'trust-exact',
  #'trust-krylov',
]
INIT_METH = [
  'randu', 
  'randn', 
  'linear', 
  'eq-2d',
  'eq-3d',
  'orig',
]


def run(args):
  len_x = len(OPTIM_METH)
  len_y = len(INIT_METH)
  fci = np.zeros([len_x, len_y])
  ts  = np.ones ([len_x, len_y]) * -1

  for i, optim in enumerate(OPTIM_METH):
    for j, init in enumerate(INIT_METH):
      expname = f'O={optim}_i={init}'
      fp = os.path.join(args.log_path, expname, 'stats.json')
      if not os.path.exists(fp): continue

      with open(fp, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
      fci[i, j] = data['final_fci']
      ts [i, j] = data['ts']
  
  print('FCI:', fci.shape)
  print(fci)

  def draw_heatmap(ax:Axes, data:np.ndarray, is_time:bool=False):
    ax.imshow(data.T)
    ax.set_xticks(range(len_x), OPTIM_METH)
    ax.set_yticks(range(len_y), INIT_METH)
    for i in range(len(OPTIM_METH)):
      for j in range(len(INIT_METH)):
        if is_time:
          ax.text(i, j, f'{data[i, j]:.2f}', ha='center', va='center', color='w')
        else:
          ax.text(i, j, f'{data[i, j]:.7f}', ha='center', va='center', color='w')
  
  objective = None
  name: str = os.path.basename(str(args.log_path))
  if '.' in name: objective = name[1+name.find('.'):]

  plt.clf()
  fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8, 12))
  draw_heatmap(ax1, fci)      ; ax1.set_title('FCI')
  draw_heatmap(ax2, ts, True) ; ax2.set_title('run time cost')
  if objective: plt.suptitle(f'{objective}')
  plt.tight_layout()
  fp = os.path.join(args.log_path, 'fci_cmp.png')
  plt.savefig(fp, dpi=600)
  print(f'>> savefig to {fp}')

  fp = os.path.join(args.log_path, 'fci_cmp.npy')
  np.save(fp, fci)
  print(f'>> save dump to {fp}')
  fp = os.path.join(args.log_path, 'fci_cmp_ts.npy')
  np.save(fp, ts)
  print(f'>> save dump to {fp}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-D', '--log_path', default='log', type=Path, help='log file folder')
  args = parser.parse_args()

  assert args.log_path.is_dir()
  run(args)
