import os
import sys
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(BASE_PATH))
sys.path.append(os.path.abspath(os.path.dirname(BASE_PATH)))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(BASE_PATH))))

from openfermion.chem import MolecularData

from .vqe import vqe_solver, fsm_solver
from .ocvqe import ocvqe_solver
from .opocvqe import opocvqe_solver
from .ssvqe import ssvqe_solver
from .wssvqe import wssvqe_solver
from .common import seed_everything

ANSATZS = [
  # mindquantum
  'QUCC',           # nice~
  'UCC',            # error very large ~0.1
  'UCCSD',          # error large ~0.01
  'HEA',            # FIXME: this does not work, really...?!!
  # qupack
  'UCCSD-QP',       # error very large ~0.1
]
OPTIMS = [
  'trust-constr',   # nice~
  'BFGS',           # error large ~0.01
  'CG',             # error large ~0.01
  'COBYLA',         # error very large ~1
]


def run_fsm(mol:MolecularData) -> float:
  config1 = {
    'ansatz':  'QUCC',
    'trotter': 1,
    'optim':   'BFGS',
    'tol':     1e-6,
    'maxiter': 200,
    'dump':    False,
  }
  config2 = {
    'ansatz':  'QUCC',
    'trotter': 2,
    'optim':   'BFGS',
    'tol':     1e-8,
    'maxiter': 300,
  }
  return fsm_solver(mol, config1, config2)


def run_ocvqe(mol:MolecularData) -> float:
  hijack = True
  lib = 'qp'

  sfx = '-hijack' if hijack else ''
  if lib == 'mq':
    config1 = {
      # circ
      'ansatz':  f'QUCC{sfx}',
      'trotter': 1,
      # optim
      'optim':   'BFGS',
      'tol':     1e-3,
      'maxiter': 100,
      'dump':    False,
      # ham
      'round_one': 6,
      'round_two': 6,
      'trunc_one': 0.001,
      'trunc_two': 0.002,
      'compress':  4e-3,
    }
    config2 = {
      'ansatz':  f'QUCC{sfx}',
      'trotter': 2,
      'optim':   'BFGS',
      'tol':     1e-3,
      'beta':    2,
      'eps':     2e-6,
      'maxiter': 300,
      'cont_evolve': False,
    }
  elif lib == 'qp':
    config1 = {
      'ansatz':  f'UCCSD-QP{sfx}',
      'trotter': 1,
      'optim':   'BFGS',
      'tol':     1e-4,
      'maxiter': 100,
      'dump':    False,
    }
    config2 = {
      'ansatz':  f'UCCSD-QP{sfx}',
      'trotter': 3,
      'optim':   'BFGS',
      'tol':     1e-4,
      'beta':    4,
      'eps':     1e-5,
      'maxiter': 1000,
      'cont_evolve': False,
    }
  return ocvqe_solver(mol, config1, config2)


def run_opocvqe(mol:MolecularData) -> float:
  opt = 'sp'
  if opt == 'sp':
    config = {
      'ansatz':  'QUCC',
      'trotter': 1,
      'opt':     'scipy',
      'optims': [
        {
          'optim':   'CG',    # this is fast on converge 
          'tol':     1e-3,
          'beta':    10,
          'w':       0.2,
          'maxiter': 100,
        },
        {
          'optim':   'BFGS',  # this is accurate
          'tol':     1e-5,
          'beta':    100,
          'w':       0.1,
          'maxiter': 1000,
        },
      ]
    }
  elif opt == 'mq':
    config = {
      'ansatz':  'QUCC',
      'trotter': 2,
      'opt':     'mindspore',
      'optim':   'Adagrad',
      'lr':      0.15,
      'beta':    100,
      'w':       0.1,
      'maxiter': 1000,
    }
  return opocvqe_solver(mol, config)


def run_ssvqe(mol:MolecularData) -> float:
  config1 = {
    'ansatz':     'HEA',
    'rot_gates':  ['RX', 'RY', 'RX'],
    'entgl_gate': 'X',
    'depth':      12,
    'optim':      'BFGS',
    'tol':        1e-8,
    'maxiter':    500,
  }
  config2 = {
    'ansatz':     'HEA',
    'rot_gates':  ['RX', 'RY', 'RX'],
    'entgl_gate': 'X',
    'depth':      12,
    'optim':      'BFGS',
    'tol':        1e-8,
    'maxiter':    500,
  }
  return ssvqe_solver(mol, config1, config2)


def run_wssvqe(mol:MolecularData) -> float:
  config = {
    'ansatz':     'HEA',
    'rot_gates':  ['RX', 'RY', 'RX'],
    'entgl_gate': 'X',
    'depth':      12,
    'optim':      'BFGS',
    'tol':        1e-8,
    'w':          0.1,
    'maxiter':    1000,
  }
  return wssvqe_solver(mol, config)


def excited_state_solver(mol:MolecularData) -> float:
  seed_everything(os.environ.get('SEED', None))

  #algo = 'fsm'
  algo = 'ocvqe'
  #algo = 'opocvqe'
  #algo = 'ssvqe'
  #algo = 'wssvqe'
  return globals()[f'run_{algo}'](mol)


class Main:
  def run(self, mol:MolecularData):
    return excited_state_solver(mol)
