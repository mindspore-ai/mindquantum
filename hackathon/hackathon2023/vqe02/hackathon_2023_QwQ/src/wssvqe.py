#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/20

# implementation of weighted SS-VQE in "Subspace-search variational quantum eigensolver for excited states"

from .common import *


def run(mol:MolecularData, ham:Ham, config:Config) -> Tuple[float, float]:
  # Declare the simulator
  sim = get_sim(mol, ham)
  # Construct encoding circ for preparing orthogonal init state |ψj>
  q0_enc, q1_enc = get_encoder_ortho()
  # Construct U ansatz circuit: U|ψj>
  ansatz, init_amp = get_ansatz(mol, config['ansatz'], config, no_hfw=True)
  # Full circuit
  q0_circ = q0_enc + ansatz
  q1_circ = q1_enc + ansatz
  # Get the expectation and gradient calculating function: <φj|U'HU|φj>
  q0_grad_ops = sim.get_expectation_with_grad(ham, q0_circ)
  q1_grad_ops = sim.get_expectation_with_grad(ham, q1_circ)

  # Define the objective function to be minimized
  def func(x:ndarray, q0_grad_ops:Callable, q1_grad_ops:Callable, hparam:Config) -> Tuple[float, ndarray]:
    w: float = hparam['w']
    f0, g0 = q0_grad_ops(x)
    f1, g1 = q1_grad_ops(x)
    f0, f1, g0, g1 = [np.squeeze(x) for x in [f0, f1, g0, g1]]
    if DEBUG: print('gs:', f0.real, 'es:', f1.real)
    return np.real(f0 + w * f1), np.real(g0 + w * g1)
  
  # Get optimized result
  params = optim_scipy(func, init_amp, (q0_grad_ops, q1_grad_ops,), config)

  # Get the energies
  sim.reset() ; f0 = run_expectaion(sim, ham, q0_circ, params)
  sim.reset() ; f1 = run_expectaion(sim, ham, q1_circ, params)

  if PEEK:
    print('E0 energy:', f0)
    print('E1 energy:', f1)
  return f0, f1


def wssvqe_solver(mol:MolecularData, config:Config) -> float:
  ham = get_ham(mol, config)

  # Find the lowest two energies by min. U|φ0> + w*U|φ1>, sat. <φ0|φ1> = 0
  assert 0.0 < config['w'] < 1.0
  _, es_ene = run(mol, ham, config)

  return es_ene
