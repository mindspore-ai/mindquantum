from time import time ; global_T = time()
import os ; os.environ['OMP_NUM_THREADS'] = '4'
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
from openfermion.chem import MolecularData
from mindquantum.core.operators import InteractionOperator, FermionOperator, normal_ordered
from mindquantum.core.circuit import Circuit
from mindquantum.algorithm.nisq.chem import uccsd_singlet_generator
from qupack.vqe import ESConservation, ESConserveHam, ExpmPQRSFermionGate
ESConservation._check_qubits_max = lambda *args, **kwargs: None   # monkey-patching avoid FileNotFoundError

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

if 'typing':
  DTYPE = np.float64

  Name  = List[str]
  Geo   = ndarray     # [N*3], flattened

if 'globals':
  steps: int = 0
  circ: Circuit = None
  sim: ESConservation = None
  track_ene: List[float] = []
  track_geo: List[Geo] = []

  # contest time limit: 1h
  TIMEOUT_LIMIT = int(3600 * 0.95)


def timer(fn):
  def wrapper(*args, **kwargs):
    from time import time
    start = time()
    r = fn(*args, **kwargs)
    end = time()
    print(f'[Timer]: {fn.__name__} took {end - start:.3f}s')
    return r
  return wrapper

def read_csv(fp:str) -> Tuple[Name, Geo]:
  with open(fp, 'r', encoding='utf-8') as fh:
    data = fh.readlines()

  name, geo = [], []
  for line in data:
    n, *v3 = line.split(',')
    pt = [DTYPE(e) for e in v3]
    name.append(n)
    geo.extend(pt)
  return name, np.ascontiguousarray(geo, dtype=DTYPE)

def write_csv(fp:str, name:Name, geo:Geo):
  with open(fp, 'w', encoding='utf-8') as fh:
    for i, n in enumerate(name):
      fh.write(f'{n}, ')
      fh.write(', '.join([str(e) for e in geo[i].tolist()]))
      fh.write('\n')


def get_fci_from_csv(args) -> float:
  # due to *.csv read/write precision error for judger of this contest
  # we do NOT use direct output in the program as the final FCI value
  # we parse the csv file and reconstruct to get it :(

  name, geo = read_csv(args.output_mol)
  mol = get_mol(name, geo)
  fci = mol.fci_energy
  print('final fci:', fci)
  return fci


# ↓↓↓ openfermionpyscf/_run_pyscf.py ↓↓↓

'''
  - calc hf_energy, prepare integrals 
  - calc fci_energy (optional)
'''

from pyscf import fci
from pyscf.scf import hf ; hf.MUTE_CHKFILE = True   # monkey-patching avoid tempfile IO
from openfermionpyscf import PyscfMolecularData
from openfermionpyscf._run_pyscf import prepare_pyscf_molecule, compute_scf, compute_integrals

def run_pyscf_hijack(molecule:MolecularData, run_fci:bool=False) -> PyscfMolecularData:
  # Prepare pyscf molecule.
  pyscf_molecule = prepare_pyscf_molecule(molecule)
  molecule.n_orbitals = int(pyscf_molecule.nao_nr())
  molecule.n_qubits = 2 * molecule.n_orbitals
  molecule.nuclear_repulsion = float(pyscf_molecule.energy_nuc())

  # Run SCF.
  pyscf_scf = compute_scf(pyscf_molecule)
  pyscf_scf.verbose = 0
  pyscf_scf.run()
  molecule.hf_energy = float(pyscf_scf.e_tot)

  # Hold pyscf data in molecule. They are required to compute density
  # matrices and other quantities.
  molecule._pyscf_data = pyscf_data = {}
  pyscf_data['mol'] = pyscf_molecule
  pyscf_data['scf'] = pyscf_scf

  # Populate fields.
  molecule.canonical_orbitals = pyscf_scf.mo_coeff.astype(float)
  molecule.orbital_energies = pyscf_scf.mo_energy.astype(float)

  # Get integrals.
  one_body_integrals, two_body_integrals = compute_integrals(pyscf_molecule, pyscf_scf)
  molecule.one_body_integrals = one_body_integrals
  molecule.two_body_integrals = two_body_integrals
  molecule.overlap_integrals = pyscf_scf.get_ovlp()

  # Run FCI.
  if run_fci:
    pyscf_fci = fci.FCI(pyscf_molecule, pyscf_scf.mo_coeff)
    pyscf_fci.verbose = 0
    molecule.fci_energy = pyscf_fci.kernel()[0]
    pyscf_data['fci'] = pyscf_fci

  # Return updated molecule instance.
  pyscf_molecular_data = PyscfMolecularData.__new__(PyscfMolecularData)
  pyscf_molecular_data.__dict__.update(molecule.__dict__)
  return pyscf_molecular_data

# ↑↑↑ openfermionpyscf/_run_pyscf.py ↑↑↑


# ↓↓↓ qupack.vqe ↓↓↓

def get_mol(name:Name, geo:Geo, run_fci:bool=True) -> MolecularData:
  geometry = [[name[i], list(e)] for i, e in enumerate(geo.reshape(len(name), -1))]
  mol = MolecularData(geometry, 'sto3g', multiplicity=1)
  # NOTE: make integral calculation info for `mol.get_molecular_hamiltonian()`
  return run_pyscf_hijack(mol, run_fci=run_fci)

def get_ham(mol:MolecularData) -> ESConserveHam:
  ham_of = mol.get_molecular_hamiltonian()
  inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
  ham_hiq = FermionOperator(inter_ops)  # len(ham_hiq) == 1861 for LiH; == 37 for H2
  ham_fo = normal_ordered(ham_hiq)
  ham = ESConserveHam(ham_fo.real)      # len(ham.terms) == 631 for LiH; == 15 for H2
  #mat = ham.ham.matrix().todense()      # [4096, 4096] for LiH, 12-qubits; [16, 16] for H2, 4-qubits
  #diag = np.diag(mat)
  return ham

def gen_uccsd(mol:MolecularData) -> Circuit:
  ucc_fermion_ops = uccsd_singlet_generator(mol.n_qubits, mol.n_electrons, anti_hermitian=False)
  circ = Circuit()
  for term in ucc_fermion_ops: circ += ExpmPQRSFermionGate(term)
  if args.track: circ.summary()
  return circ

def run_uccsd(ham:ESConserveHam, circ:Circuit, sim:ESConservation) -> float:
  grad_ops = sim.get_expectation_with_grad(ham, circ)

  def fun(tht, grad_ops):
    f, g = grad_ops(tht)
    return f.real, g.real

  tht = np.random.uniform(size=len(circ.params_name)) * 0.01
  res = minimize(fun, tht, args=(grad_ops,), jac=True, method=args.optim)
  return res.fun

# ↑↑↑ qupack.vqe ↑↑↑


def optim_fn(geo:Geo, name:Name, args) -> float:
  global circ, sim, steps

  if args.objective == 'pyscf':
    mol = get_mol(name, geo, run_fci=True)
    res: float = mol.fci_energy
  elif args.objective == 'uccsd':
    mol = get_mol(name, geo)    # only need *.h5 file
    ham = get_ham(mol)
    if circ is None:
      circ = gen_uccsd(mol)
      sim = ESConservation(mol.n_qubits, mol.n_electrons)
    res = run_uccsd(ham, circ, sim)
  else:
    raise ValueError(f'unknown objective: {args.objective}')

  time_elapse = time() - global_T

  if args.track:
    steps += 1
    if steps % 10 == 0:
      print(f'>> [Step {steps}] energy: {res}, total time elapse: {time_elapse:.3f} s')
    track_ene.append(res)
    track_geo.append(geo.reshape(len(name), -1))
  
  if not args.track and time_elapse > TIMEOUT_LIMIT:
    best_x = geo.reshape(len(name), -1)
    write_csv(args.output_mol, name, best_x)
    exit(0)

  return res

def get_init_linear(npoints:int) -> ndarray:
  geo = np.zeros([npoints, 3])
  geo[:, -1] = np.linspace(0.0, 1.0, npoints)   # 等分点
  return geo.flatten()

def get_init_eq_kd(npoints:int, dim:int=2) -> ndarray:
  assert 3 <= npoints <= 4 
  assert dim in [2, 3]
  geo = None

  if dim == 2:
    if npoints == 3:
      geo = np.asarray([    # 等边三角形
        [0, 0, 0],
        [0, 0, 1],
        [0, np.sqrt(3)/2, 0],
      ])
    elif npoints == 4:
      geo = np.asarray([    # 正方形
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
      ])
  elif dim == 3:
    if npoints == 4:
      geo = np.asarray([    # 正四面体
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
      ])

  return geo.flatten() if geo is not None else None


@timer
def run(args, name:Name, init_geo:Geo) -> Tuple[Name, Geo]:
  geo = None
  if   args.init == 'randu':  geo = np.random.uniform(low=-1.0, high =1.0, size=len(init_geo))
  elif args.init == 'randn':  geo = np.random.normal (loc= 0.0, scale=1.0, size=len(init_geo))
  elif args.init == 'linear': geo = get_init_linear(len(name))
  elif args.init == 'eq-2d':  geo = get_init_eq_kd(len(name), dim=2)
  elif args.init == 'eq-3d':  geo = get_init_eq_kd(len(name), dim=3)
  elif args.init == 'orig':   geo = init_geo
  if geo is None: return  # ignore if geo not applicable
  init_x = geo.tolist()

  s = time()
  res = minimize(
    optim_fn, 
    x0=geo, 
    args=(name, args), 
    method=args.optim, 
    tol=args.eps, 
    options={
      'maxiter': args.maxiter, 
      'disp': args.track,
    },
  )
  t = time()
  best_x = res.x    # flattened

  if args.track:
    import json
    import matplotlib.pyplot as plt
  
    fci = get_mol(name, best_x, run_fci=True).fci_energy
    print('final fci:', fci)

    track_geo_np: ndarray = np.stack(track_geo, axis=0)
    with open(os.path.join(args.log_dp, 'stats.json'), 'w', encoding='utf-8') as fh:
      data = {
        'args': vars(args),
        'final_fci': fci,
        'ts': t - s,
        'init_x': init_x,
        'energy': track_ene,
        'name': name,
        'geometry': track_geo_np.tolist(),
      }
      json.dump(data, fh, indent=2, ensure_ascii=False)

    energy, geometry = track_ene, track_geo_np

    if 'plot energy':
      plt.clf()
      plt.subplot(211) ; plt.plot(energy)                 ; plt.title('Energy')
      plt.subplot(212) ; plt.plot(np.log(np.abs(energy))) ; plt.title('log(|Energy|)')
      plt.tight_layout()
      plt.savefig(os.path.join(args.log_dp, 'energy.png'))

    if 'plot geometry':
      T, N, D = geometry.shape    # (T=480, N=4, D=3)
      plt.clf()
      nrow = int(N**0.5)
      ncol = int(np.ceil(N / nrow))
      for i in range(nrow):
        for j in range(ncol):
          idx = i * ncol + j
          plt.subplot(nrow, ncol, idx + 1)
          plt.plot(geometry[:, idx, 0], 'r', label='x')
          plt.plot(geometry[:, idx, 1], 'g', label='y')
          plt.plot(geometry[:, idx, 2], 'b', label='z')
          plt.title(name[idx])
      plt.suptitle('Geometry')
      plt.tight_layout()
      plt.savefig(os.path.join(args.log_dp, 'geometry.png'))

  return name, best_x

@timer
def run_all(args):
  name, geo = read_csv(args.input_mol)

  def setup_exp(args) -> bool:
    # reset globals
    global steps, circ, sim, track_ene, track_geo

    steps = 0
    circ = None
    sim = None
    track_ene = []
    track_geo = []

    # setup log_dp
    expname = f'O={args.optim}_I={args.init}'
    args.log_dp = os.path.abspath(os.path.join(args.log_path, expname))
    if os.path.exists(args.log_dp): return True
    os.makedirs(args.log_dp, exist_ok=True)
    return False

  from traceback import print_exc
  
  for optim in OPTIM_METH:
    args.optim = optim
    for init in INIT_METH:
      args.init = init

      done = setup_exp(args)
      if done: continue

      print(f'>> run optim={optim}, init={init}')
      try: run(args, name, geo)
      except: print_exc()

@timer
def run_compound(args):
  name, best_x = read_csv(args.input_mol)

  def go():
    nonlocal name, best_x
    name, best_x = run(args, name, best_x)
    best_geo = best_x.reshape(len(name), -1)
    write_csv(args.output_mol, name, best_geo)

  if args.no_comp: go()
  else:
    configs = [
      'COBYLA',         # this is fast
      'BFGS',           # this is fine
      #'trust-constr',   # this is precise (but overfit?)
    ]

    maxiter = args.maxiter
    for i, optim in enumerate(configs):
      print(f'>> round {i}: optim use {optim}')
      args.optim = optim
      args.init = args.init if i == 0 else 'orig'
      args.maxiter = maxiter // 2 if 1 == 0 else maxiter

      go()


def get_args():
  parser = ArgumentParser()
  # run
  parser.add_argument('-i', '--input-mol',  help='input molecular *.csv',  default='h4.csv')
  parser.add_argument('-x', '--output-mol', help='output molecular *.csv', default='h4_best.csv')
  parser.add_argument('-Z', '--objective',  help='optim target objective', default='uccsd', choices=['pyscf', 'uccsd'])
  parser.add_argument('-O', '--optim',      help='optim method',           default='BFGS', choices=OPTIM_METH)
  parser.add_argument('--init',     help='init method',    default='linear', choices=INIT_METH)
  parser.add_argument('--eps',      help='tol eps',        default=1e-8, type=float)
  parser.add_argument('--maxiter',  help='max optim iter', default=1000, type=int)
  # dev
  parser.add_argument('--check',    help='get predicted FCI from *.csv',   action='store_true')
  parser.add_argument('--no_comp',  help='do not use compound optim',      action='store_true')
  parser.add_argument('--run_all',  help='run all optim-init grid search', action='store_true')
  parser.add_argument('--log_path', help='track log out base path', default='log', type=str)
  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()

  if args.check:
    get_fci_from_csv(args)
    exit(0)

  if args.run_all:
    print(f'[dev mode] objective: {args.objective}')
    args.track = True
    run_all(args)

  else:
    print(f'[submit mode] objective: {args.objective}')
    args.track = False
    run_compound(args)
