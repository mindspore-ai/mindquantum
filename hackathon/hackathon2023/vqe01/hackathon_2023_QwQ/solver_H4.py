import json
from pathlib import Path
from typing import List, Dict, Any
from traceback import print_exc

BASE_PATH = Path(__file__).parent.absolute()
DB_FILE = BASE_PATH / 'h4_best.json'

import sys
sys.path.append(str(BASE_PATH))
import solver; solver.TIMEOUT_LIMIT = 2**30
from solver import *

Record = {
  'fci': float,
  'geo': Geo,
}
DB = {
  'best': 'Record',
  'hist': List['Record'],
}

INIT_BEST = {
  'fci': -2.2746177087120563,
  'geo': [
    -1.5827458758373452, -0.15612578617978118, -1.8075932510918267,
    -1.584725393549196, 0.563197278734018, -1.9579606621652792,
    0.5173960942724174, 0.8591851473019563, 1.329958317164121,
    0.9135797608000218, 0.9808158786539857, 1.9368199513109006,
  ],
}

OPTIM_METH = [
  'BFGS',
  'SLSQP',
  'COBYLA',
  'trust-constr',
]
INIT_METH = [
  'randu', 
  'randn', 
  'linear', 
  'eq-2d',
  'eq-3d',
]


def load_db(fp:Path) -> DB:
  if not fp.exists():
    return {
      'best': INIT_BEST,
      'hist': [],
    }

  with open(fp, 'r', encoding='utf-8') as fh:
    return json.load(fh)

def save_db(data:DB, fp:Path):
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False)
  
def update_db(db:DB, fci:float, geo:List[float]) -> bool:
  rec = {
    'fci': fci,
    'geo': geo,
  }
  db['hist'].append(rec)
  new_best = False
  if fci < db['best']['fci']:
    db['best'] = rec
    new_best = True
    print(f'>> new best found: fci={fci}')
  return new_best


@timer
def exhaustive_search(args):
  name, init_x = read_csv(args.input_mol)
  db = load_db(DB_FILE)
  tmp_fp = Path(args.output_mol)

  try:
    for optim in OPTIM_METH:
      args.optim = optim
      for init in INIT_METH:
        args.init = init

        for i in range(args.n_times):
          print(f'>> run optim={optim}, init={init} [{i}/{args.n_times}]')
          try:
            name, best_x = run(args, name, init_x)
            best_geo = best_x.reshape(len(name), -1)
            write_csv(tmp_fp, name, best_geo)
            fci = get_fci_from_csv(args)
            new_best = update_db(db, fci, best_geo.tolist())
            if new_best: save_db(db, DB_FILE)
          except KeyboardInterrupt:
            raise
          except:
            print_exc()

  finally:
    save_db(db, DB_FILE)
    if tmp_fp.exists(): tmp_fp.unlink()


if __name__ == '__main__':
  args = get_args()
  
  args.input_mol  = BASE_PATH / 'h4.csv'
  args.output_mol = BASE_PATH / 'h4_best_tmp.csv'
  args.objective  = 'pyscf'
  args.track      = False
  args.maxiter    = 1000
  args.n_times    = 3

  exhaustive_search(args)
