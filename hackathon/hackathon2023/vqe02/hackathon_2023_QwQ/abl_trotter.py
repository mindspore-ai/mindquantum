from eval import *
from src.ocvqe import ocvqe_solver, seed_everything
from src import ocvqe

# 导出 PPT 用对比表：QP-uccsd 各种 trotter 深度


def run_ocvqe_(mol:MolecularData, trotter:int) -> float:
  config1 = {
    'ansatz':  'UCCSD-QP-hijack',
    'trotter': 1,
    'optim':   'BFGS',
    'tol':     1e-4,
    'maxiter': 100,
    'dump':    False,
  }
  config2 = {
    'ansatz':  'UCCSD-QP-hijack',
    'trotter': trotter,
    'optim':   'BFGS',
    'tol':     1e-8,
    'beta':    4,
    'eps':     1e-5,
    'maxiter': 1000,
    'cont_evolve': False,
  }
  return ocvqe_solver(mol, config1, config2)


for idx, (name, E0_gt, E1_gt) in enumerate(molecules):
  print(f'[{name}]')

  mol = MolecularData(filename=os.path.join(BASE_PATH, f'./molecule_files/{name}'))
  mol.load()

  for i in range(1, 6):
    # reset global state
    ocvqe.punish_f = 1e5
    seed_everything(42)

    t = time()
    E1_hat = run_ocvqe_(mol, i)
    print(f'[trotter_{i}] error: {abs(E1_hat - E1_gt)}, time: {time() - t}')
