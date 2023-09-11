#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/19 

# implementation of SS-VQE in "Subspace-search variational quantum eigensolver for excited states"
# NOTE: this does now work properly, still need investigation...

from .common import *


def run_U(mol:MolecularData, ham:Ham, config:Config) -> QVM:
  # Declare the U simulator
  sim = get_sim(mol, ham)
  # Construct encoding circ for preparing orthogonal init state
  q0_enc, q1_enc = get_encoder_ortho()
  # Construct U ansatz circuit: |ψ(λ0)>
  U_circ, init_amp = get_ansatz(mol, config['ansatz'], config, no_hfw=True)
  # Full circuit
  q0_circ = q0_enc + U_circ
  q1_circ = q1_enc + U_circ
  # Get the expectation and gradient calculating function: <φ|U'HU|φ>
  q0_grad_ops = sim.get_expectation_with_grad(ham, q0_circ)
  q1_grad_ops = sim.get_expectation_with_grad(ham, q1_circ)

  # Define the objective function to be minimized
  def func(x:ndarray, q0_grad_ops:Callable, q1_grad_ops:Callable, hparam:Config) -> Tuple[float, ndarray]:
    f0, g0 = q0_grad_ops(x)
    f1, g1 = q1_grad_ops(x)
    f0, f1, g0, g1 = [np.squeeze(x) for x in [f0, f1, g0, g1]]
    if DEBUG: print('f0:', f0.real, 'f1:', f1.real)
    return np.real(f0 + f1), np.real(g0 + g1)
  
  # Get optimized result
  params = optim_scipy(func, init_amp, (q0_grad_ops, q1_grad_ops), config)

  # Get the energyies
  sim.reset() ; f0 = run_expectaion(sim, ham, q0_circ, params)
  sim.reset() ; f1 = run_expectaion(sim, ham, q1_circ, params)
  if PEEK: print('lowest energies:', [f0, f1])

  # arbitarily choose a circ to evolve into
  sim.reset()
  sim.apply_circuit(q0_circ, params)

  return sim


def run_V(mol:MolecularData, ham:Ham, config:Config, sim:QVM) -> float:
  # Construct V ansatz circuit: VU|φj>
  V_circ, init_amp = get_ansatz(mol, config['ansatz'], config)
  # Get the expectation and gradient calculating function: <φj|U'V'HVU|φj>
  grad_ops = sim.get_expectation_with_grad(ham, V_circ)

  # Define the objective function to be maximized
  def func(x:ndarray, grad_ops:Callable, hparam:Config) -> Tuple[float, ndarray]:
    f, g = grad_ops(x)
    f, g = [np.squeeze(x) for x in [f, g]]
    if DEBUG: print('f:', f.real)
    return -np.real(f), -np.real(g)
  
  # Get Optimized result: min. E0 = <ψ(λ0)|H|ψ(λ0)>
  params = optim_scipy(func, init_amp, (grad_ops,), config)

  # NOTE: do not use `run_expectaion()` because it will reset QVM
  es_ene, _ = grad_ops(params)
  es_ene = es_ene.item().real

  if PEEK: print('E1 energy:', es_ene)
  return es_ene


def ssvqe_solver(mol:MolecularData, config1:Config, config2:Config) -> float:
  ham = get_ham(mol, config1)

  # Shrink the subspace, expand ansatz state: U|φ>
  sim = run_U(mol, ham, config1)
  # Find the highest energy Ek: VU|φ>
  es_ene = run_V(mol, ham, config2, sim)

  return es_ene
