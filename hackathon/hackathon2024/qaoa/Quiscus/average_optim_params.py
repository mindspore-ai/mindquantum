#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/30 

# 直接把各样例最优参数暴力平均

import json
from typing import List, Dict

import numpy as np
from numpy import ndarray

from utils.path import DATA_OPT_PATH, LOG_PATH
from utils.lookup_table import plot_lookup_table

assert DATA_OPT_PATH.exists(), 'need run "python train_optimal_params.py" first'


def run():
  ''' Ckpt '''
  opt_params: Dict[int, Dict[int, List[ndarray]]] = {}
  for propotion in [0.3, 0.6, 0.9]:
    for k in range(2, 6):
      for coef in ['std', 'uni', 'bimodal']:
        for r in range(10):
          fp = DATA_OPT_PATH / f"k{k}/{coef}_p{propotion}_{r}.json"
          with open(fp, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
          for p, v in data.items():
            if p not in opt_params: opt_params[p] = {}
            if k not in opt_params[p]: opt_params[p][k] = []
            opt_params[p][k].append(np.asarray(v))

  ''' Average '''
  for p, d1 in opt_params.items():
    for k, d2 in d1.items():
      d1[k] = np.stack(d2, axis=0).mean(axis=0).tolist()

  plot_lookup_table(opt_params, subfolder='optimal-average')

  save_fp = LOG_PATH / 'lookup_table-optimal-avg.json'
  print(f'>> save to {save_fp}')
  with open(save_fp, 'w', encoding='utf-8') as fh:
    json.dump(opt_params, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
  run()
