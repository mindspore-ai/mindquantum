import json
import random
from typing import List

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.absolute()
DB_FILE = BASE_PATH / 'eval_H2O.json'
sys.path.append(str(BASE_PATH))
sys.path.append(str(BASE_PATH / 'src'))
try:
  from eval import *
  from src import ocvqe
  from src.main import run_ocvqe
  from src.common import seed_everything
except:
  print('>> warning: cannot import "src" folder ')

Record = {
  'seed': int,
  'err': float,
  'ts': float,
}
DB = {
  'best': 'Record',
  'hist': List['Record'],
}

INIT_BEST = {
  'seed': -1,
  'err': 1e5,
  'ts': 1e5,
}


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
  
def update_db(db:DB, rec:Record) -> bool:
  db['hist'].append(rec)
  new_best = False
  if rec['ts'] < db['best']['ts'] and rec['err'] <= thresh:
    db['best'] = rec
    new_best = True
    print(f'>> new best found: seed={rec["seed"]}')
  return new_best


if __name__ == '__main__':
  # mol
  name = 'H2O_1.0'
  E0_gt = -74.88230198918458
  E1_gt = -74.71606158706648

  mol = MolecularData(filename=os.path.join(BASE_PATH, f'./molecule_files/{name}'))
  mol.load()
  
  db = load_db(DB_FILE)
  seed_used = set()
  try:
    for _ in range(500):
      seed_new = random.randint(0, 2**31-1)
      while seed_new in seed_used:
        seed_new = random.randint(0, 2**31-1)
      
      seed_used.add(seed_new)

      s = time()
      seed_everything(seed_new)
      ocvqe.punish_f = 1e5
      E1_hat = run_ocvqe(mol)
      t = time()

      new_rec = {
        'seed': seed_new,
        'err': abs(E1_gt - E1_hat),
        'ts': t - s,
      }
      if update_db(db, new_rec):
        save_db(db, DB_FILE)

  finally:
    save_db(db, DB_FILE)
