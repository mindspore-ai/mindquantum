#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/02 

# implementation of original VQE and FSM in "A variational eigenvalue solver on a quantum processor"

from .common import *


def run(mol:MolecularData, ham:Ham, config:Config) -> Tuple[QVM, float, Params]:
  # Declare the simulator
  sim = get_sim(mol, ham)
  # Construct ansatz circuit: ψ(λ)
  circ, init_amp = get_ansatz(mol, config['ansatz'], config)

  # Load cached params if exists
  if config.get('dump', False):
    fp = CACHE_PATH / f'vqe-{config["ansatz"]}-{Path(mol.filename).name}.npy'
    if fp.exists():
      print('>> try using cached E0 states')
      try:
        params = np.load(fp)
        ene = run_expectaion(sim, ham, circ, params)
        print('E0 energy:', ene)
        return sim, ene, params
      except:
        print('>> cache file error')
        fp.unlink()
  
  # Get the expectation and gradient calculating function: <ψ(λ)|H|ψ(λ)>
  grad_ops = sim.get_expectation_with_grad(ham, circ)

  # Define the objective function to be minimized
  def func(x:ndarray, grad_ops:Callable, hparam:Config) -> Tuple[float, ndarray]:
    f, g = grad_ops(x)
    f, g = [np.squeeze(x) for x in [f, g]]
    if DEBUG: print('gs:', f.real)
    return np.real(f), np.real(g)
  
  # Get optimized results
  params = optim_scipy(func, init_amp, (grad_ops,), config)

  # Make params cache
  if config.get('dump', False):
    np.save(fp, params)
 
  # Get energy, evolve `sim` to `circ`
  ene = run_expectaion(sim, ham, circ, params)

  print('E0 energy:', ene)
  return sim, ene, params


def vqe_solver(mol:MolecularData, config:Config) -> float:
  ham = get_ham(mol, config)

  # Ground state E0: |ψ(λ0)>
  _, gs_ene, _ = run(mol, ham, config)

  return gs_ene


def fsm_solver(mol:MolecularData, config1:Config, config2:Config) -> float:
  ''' NOTE: 已知 特征值λ(能量) 求 特征向量(态) '''

  ham = get_ham(mol, config1)

  # Ground state E0: |ψ(λ0)>
  gs_ene = vqe_solver(mol, config1)

  # Guess a E1 larger than E0
  lmbd1 = gs_ene + 0.1

  # Make folded spectrum: |H-λ|^2
  ham_hat = type(ham)((ham.hamiltonian - QubitOperator('') * lmbd1) ** 2)
  es_sim, _, _ = run(mol, ham_hat, config2)

  # Excited state Ek: |ψ(λk)>
  es_ene = run_expectaion(es_sim, ham, Circuit(), [])

  return gs_ene - es_ene
