#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/19 

# implementation of OC-VQE in "Variational Quantum Computation of Excited States"

from .common import *
from .vqe import run as run_gs

# save current reg term for check
punish_f: float = 1e5


def run_es(mol:MolecularData, ham:Ham, gs_sim:QVM, config:Config, init_params:ndarray=None) -> Tuple[float, Params]:
  # Declare the excited state simulator
  es_sim, HAM_NULL = get_sim(mol, ham, ret_null_ham=True)
  # Construct excited state ansatz circuit: |ψ(λ1)>
  es_circ, init_amp = get_ansatz(mol, config['ansatz'], config)
  # Initialize amplitudes
  if config.get('cont_evolve', False) and init_params is not None and len(init_params) == len(es_circ.all_paras):
    init_amp = init_params + init_randu(init_params.shape, mul=1e-3)

  # Get the expectation and gradient calculating function: <ψ(λ1)|H|ψ(λ1)>
  es_grad_ops = es_sim.get_expectation_with_grad(ham, es_circ)
  # Get the expectation and gradient calculating function of inner product: <ψ(λ0)|ψ(λ1)> where H is I
  ip_grad_ops = es_sim.get_expectation_with_grad(HAM_NULL, es_circ, circ_left=Circuit(), simulator_left=gs_sim)

  # Define the objective function to be minimized
  def func(x:ndarray, es_grad_ops:Callable, ip_grad_ops:Callable, hparam:Config) -> Tuple[float, ndarray]:
    global punish_f
    beta: float = hparam['beta']
    f0, g0 = es_grad_ops(x)
    f1, g1 = ip_grad_ops(x)
    # Remove extra dimension of the array
    f0, f1, g0, g1 = [np.squeeze(x) for x in [f0, f1, g0, g1]]
    # reg term: `+ beta * |f1| ** 2``, where f1 = β * <ψ(λ0)|ψ(λ1)>
    punish_f = beta * np.abs(f1) ** 2
    # grad of reg term: `+ beta * (g1' * f1 + g1 * f1')`
    punish_g = beta * (np.conj(g1) * f1 + g1 * np.conj(f1))
    if DEBUG: print('es:', f0.real, 'ip:', f1.real)
    return np.real(f0 + punish_f), np.real(g0 + punish_g)
  
  # Get Optimized result: min. E1 = <ψ(λ1)|H|ψ(λ1)> + |<ψ(λ1)|ψ(λ0)>|^2
  params = optim_scipy(func, init_amp, (es_grad_ops, ip_grad_ops), config)

  # Get the energy
  es_ene = run_expectaion(es_sim, ham, es_circ, params)

  if PEEK:
    print('E1 energy:', es_ene)
    print('reg_term:', punish_f)
  return es_ene, params


def ocvqe_solver(mol:MolecularData, config1:Config, config2:Config) -> float:
  ham = get_ham(mol, config1)

  # Ground state E0: |ψ(λ0)>
  gs_sim, gs_ene, params = run_gs(mol, ham, config1)

  # Retry on case failed:
  #  - excited state E1 should be lower than E0
  #  - reg term should be small enough
  es_ene = gs_ene + 1
  while gs_ene >= es_ene or punish_f > config2['eps']:
    # Excited state E1: |ψ(λ1)>
    es_ene, params = run_es(mol, ham, gs_sim, config2, params)
    # double the reg_term coeff
    config2['beta'] *= 2

  return es_ene
