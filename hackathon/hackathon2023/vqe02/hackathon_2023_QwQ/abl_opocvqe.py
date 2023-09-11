from eval import *
from src.ocvqe import ocvqe_solver, seed_everything
from src.opocvqe import opocvqe_solver

# 导出 PPT 用对比表：OC-VQE vs OPOC-VQE


def run_ocvqe(mol:MolecularData) -> float:
  config1 = {
    'ansatz':  'QUCC',
    'trotter': 1,
    'optim':   'BFGS',
    'tol':     1e-3,
    'maxiter': 100,
    'dump':    False,
  }
  config2 = {
    'ansatz':  'QUCC',
    'trotter': 1,
    'optim':   'BFGS',
    'tol':     1e-5,
    'beta':    2,
    'eps':     2e-6,
    'maxiter': 400,
    'cont_evolve': False,
  }
  return ocvqe_solver(mol, config1, config2)


def run_opocvqe(mol:MolecularData) -> float:
  config = {
    'ansatz':  'QUCC',
    'trotter': 1,
    'optim':   'BFGS',  # this is accurate
    'tol':     1e-5,
    'beta':    10,
    'w':       0.2,
    'maxiter': 500,
  }
  return opocvqe_solver(mol, config)


for idx, (name, E0_gt, E1_gt) in enumerate(molecules):
  print(f'[{name}]')

  mol = MolecularData(filename=os.path.join(BASE_PATH, f'./molecule_files/{name}'))
  mol.load()

  t = time()
  seed_everything(42)
  E1_hat = run_ocvqe(mol)
  print(f'[ocvqe] error: {abs(E1_hat - E1_gt)}, time: {time() - t}')

  t = time()
  seed_everything(42)
  E1_hat = run_opocvqe(mol)
  print(f'[opocvqe] error: {abs(E1_hat - E1_gt)}, time: {time() - t}')
