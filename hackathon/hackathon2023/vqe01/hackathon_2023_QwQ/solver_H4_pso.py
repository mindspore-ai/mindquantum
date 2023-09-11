#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/23

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.absolute()
sys.path.append(str(BASE_PATH))
from solver_H4 import *

from sko.PSO import PSO


def optim_fn(geo:Geo, name:Name) -> float:
  n_mol = len(name)
  geo = geo.reshape(n_mol, -1)
  mol = get_mol(name, geo, run_fci=True)
  fci = mol.fci_energy
  if 'debug': print(f'fci: {fci}')
  return fci


@timer
def run_pso(args):
  name, init_x = read_csv(args.input_mol)
  n_mol = len(name)
  func = lambda x: optim_fn(x, name)
  pso = PSO(func=func, n_dim=3*n_mol, pop=args.pop, max_iter=args.maxiter, lb=-3, ub=+3, w=0.8, c1=0.5, c2=0.5)
  pso.run()
  print(f'best_x is {pso.gbest_x}, best_y is {pso.gbest_y}')
  best_geo = pso.gbest_x
  
  tmp_fp = Path(args.output_mol)
  write_csv(tmp_fp, name, best_geo)
  fci = get_fci_from_csv(args)
  print(f'fci: {fci}')


if __name__ == '__main__':
  args = get_args()
  
  args.input_mol  = BASE_PATH / 'h4.csv'
  args.output_mol = BASE_PATH / 'h4_best_tmp.csv'
  args.objective  = 'pyscf'
  args.maxiter    = 100
  args.pop        = 40

  run_pso(args)
