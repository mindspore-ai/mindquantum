# Judging program. You can run it to test your algorithm.

import os ; os.environ['OMP_NUM_THREADS'] = '4'
from time import time

import numpy as np
from openfermion.chem import MolecularData

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_PATH, 'src')
import sys ; sys.path.append(SRC_PATH)
from src.main import Main


class Timer:

  def __init__(self, t0=0.0):
    self.start_time = time()
    self.t0 = t0

  def runtime(self):
    return time() - self.start_time + self.t0

  def resetime(self):
    self.start_time = time()


# molecules and their FCI ground/excited state energy
molecules = [   # (name, E0, E1)
  ## stage1
  # ('H2_1.4',    -1.015468249,  -0.87542794),      # this is even OK on HEA, but too toy
  # ('H4_0.5',    -1.653116952,  -0.90886229),
  # ('H4_1.0',    -2.166387449,  -1.93375723),
  # ('H4_1.5',    -1.996150326,  -1.92555851),
  # ('LiH_0.5',   -7.050225035,  -7.00849729),
  # ('LiH_1.5',   -7.882362287,  -7.76060920),      # hard case
  # ('LiH_2.5',   -7.823723883,  -7.77815121),
  # ('BeH2_1.3', -15.595047081, -15.329769349),     # hard case
  ## stage2
  ('H2O_1.0', -74.88230198918458, -74.71606158706648),
]
# error threshold
thresh = 0.0016


if __name__ == '__main__':
  S = time()
  main = Main()
  timer = Timer()
  E1_list, time_list = [], []
  err_flag = np.ones(len(molecules))
  with open('./results.txt', 'a') as f:
    for idx, (name, E0_gt, E1_gt) in enumerate(molecules):
      print(f'[{name}]', file=f)
      print(f'[{name}]')

      mol = MolecularData(filename=os.path.join(BASE_PATH, f'./molecule_files/{name}'))
      mol.load()

      print('E0 truth:', E0_gt, file=f)
      print('E0 truth:', E0_gt)

      t = timer.runtime()
      E1_hat = main.run(mol)
      time_list.append(timer.runtime() - t)

      print('E1 truth:', E1_gt, file=f)
      print('E1 truth:', E1_gt)
      err = abs(E1_hat - E1_gt)
      print('E1 error:', err, f'({"OK" if err <= thresh else "FAIL"})', file=f)
      print('E1 error:', err, f'({"OK" if err <= thresh else "FAIL"})')

      E1_list.append(E1_hat)
      if abs(E1_list[-1] - E1_gt) <= thresh:
        err_flag[idx] = 0

      if len(E1_list) != len(molecules):
        print('The length of en_list is not equal to that of molecules!', file=f)

      total_time = np.sum(time_list)
      if (err_flag == 1).any():
        score = err_flag.sum() * 10000
      else:
        score = total_time

    print('Molecules: ',  molecules,  file=f)
    print('Result E1: ',  E1_list,    file=f)
    print('Errors: ',     err_flag,   file=f)
    print('Time: ',       time_list,  file=f)
    print('Total time: ', total_time, file=f)
    print('Score: ',      score,      file=f)
    print('=' * 42, file=f)

    print('Score: ', score)

    T = time()
    print(f'[Timer] {T - S:.3f}s', file=f)
    print(f'[Timer] {T - S:.3f}s')
