#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/07 

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from utils.path import LOOKUP_FILE, IMG_PATH
IMG_PATH.mkdir(exist_ok=True)

LookupTable = Dict[int, Dict[int, ndarray]]   # p (depth) -> k (order) -> params
LookupTableEx = Dict[str, LookupTable]        # w (equal-weight) -> -> p (depth) -> k (order) -> params

# the contest focus on depth 4 and 8 :)
DEPTH_LIMIT = [4, 8]

SIM_EQ = 'simeq'  # i.e. unweighted-problem
NON_EQ = 'noneq'  # i.e. weighted-problem


def load_lookup_table_original() -> LookupTable:
  print(f'>> load original lookup table from {LOOKUP_FILE}')
  lookup_table = {}
  with open(LOOKUP_FILE, 'r') as csv_file:
    reader = csv.reader(csv_file)
    rows = [row for row in reader][2:]   # ignore header
    for row in rows:
      q, p, opt_nu, *params = [e for e in row if e]
      k = int(q)    # order
      p = int(p)    # depth
      if p not in lookup_table:
        lookup_table[p] = {}
      lookup_table[p][k] = np.asarray(params, dtype=np.float32)
  return lookup_table


def load_lookup_table(fp:Path) -> LookupTable:
  print(f'>> load lookup table from {fp}')
  with open(fp, 'r', encoding='utf-8') as fh:
    lookup_table: LookupTable = json.load(fh)
  lookup_table = {int(p): {int(k): np.asarray(params) for k, params in data.items()} for p, data in lookup_table.items()}
  return lookup_table


def dump_lookup_table(lookup_table:LookupTable, fp:Path):
  def _cvt_lookup_table(lookup_table:LookupTable) -> LookupTable:
    ret = deepcopy(lookup_table)
    for p, data in ret.items():
      for k, v in data.items():
        ret[p][k] = v.tolist() if isinstance(v, ndarray) else v
    return ret

  print(f'>> save lookup table to {fp}')
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(_cvt_lookup_table(lookup_table), fh, indent=2, ensure_ascii=False)


def plot_lookup_table(lookup_table:LookupTable, subfolder:str='original'):
  save_dp = IMG_PATH / subfolder
  save_dp.mkdir(exist_ok=True)

  for p in sorted(lookup_table):
    if p not in DEPTH_LIMIT: continue
    save_fp = save_dp / f'p={p}.png'
    if save_fp.exists(): continue

    plt.clf()
    for k in sorted(lookup_table[p]):
      plt.plot(lookup_table[p][k], label=f'{k}')
    plt.legend()
    plt.suptitle(f'p={p}')
    plt.tight_layout()
    plt.savefig(save_fp, dpi=400)
    plt.close()


def load_lookup_table_ex(fp:Path) -> LookupTableEx:
  print(f'>> load multi lookup table from {fp}')
  with open(fp, 'r', encoding='utf-8') as fh:
    lookup_table_ex: LookupTableEx = json.load(fh)
  lookup_table_ex = {w: {int(p): {int(k): np.asarray(params) for k, params in p_data.items()} for p, p_data in w_data.items()} for w, w_data in lookup_table_ex.items()}
  return lookup_table_ex


def dump_lookup_table_ex(lookup_table_ex:LookupTableEx, fp:Path):
  def _cvt_lookup_table_ex(lookup_table_ex:LookupTableEx) -> LookupTableEx:
    ret = deepcopy(lookup_table_ex)
    for w, w_data in ret.items():
      for p, p_data in w_data.items():
        for k, v in p_data.items():
          ret[w][p][k] = v.tolist() if isinstance(v, ndarray) else v
    return ret

  print(f'>> save multi lookup table to {fp}')
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(_cvt_lookup_table_ex(lookup_table_ex), fh, indent=2, ensure_ascii=False)


def plot_lookup_table_ex(lookup_table_ex:LookupTableEx, subfolder:str):
  save_dp = IMG_PATH / subfolder
  save_dp.mkdir(exist_ok=True)

  for p in DEPTH_LIMIT:
    save_fp = save_dp / f'p={p}.png'
    if save_fp.exists(): continue

    plt.clf()
    plt.subplot(211)
    w = NON_EQ
    for k in sorted(lookup_table_ex[w][p]):
      plt.plot(lookup_table_ex[w][p][k], label=f'{k}')
    plt.title('NON_EQ')
    plt.legend()
    plt.subplot(212)
    w = SIM_EQ
    for k in sorted(lookup_table_ex[w][p]):
      plt.plot(lookup_table_ex[w][p][k], label=f'{k}')
    plt.title('SIM_EQ')
    plt.legend()
    plt.suptitle(f'p={p}')
    plt.tight_layout()
    plt.savefig(save_fp, dpi=400)
    plt.close()


if __name__ == '__main__':
  lookup_table = load_lookup_table_original()
  plot_lookup_table(lookup_table)
