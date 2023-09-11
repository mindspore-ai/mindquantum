#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/19 

import os
import random
import tempfile
from pathlib import Path
from typing import Callable, Any, Union, List, Tuple, Dict

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
import mindspore.nn as nn
import mindquantum as mq
from mindquantum.framework import MQAnsatzOnlyLayer
from openfermion.chem import MolecularData
from mindquantum.core.gates import H, X, Y, Z, RX, RY, RZ
from mindquantum.core.circuit import Circuit, UN
from mindquantum.simulator import Simulator
from mindquantum.core.operators import InteractionOperator, FermionOperator, normal_ordered
from mindquantum.core.operators import Hamiltonian, QubitOperator, TimeEvolution
from mindquantum.algorithm.nisq.chem import Transform
from mindquantum.algorithm.nisq.chem import (
  UCCAnsatz,
  QubitUCCAnsatz,
  HardwareEfficientAnsatz,
  generate_uccsd,
  uccsd_singlet_generator,
  uccsd_singlet_get_packed_amplitudes,
)
from qupack.vqe import ESConservation, ESConserveHam, ExpmPQRSFermionGate
ESConservation._check_qubits_max = lambda *args, **kwargs: None   # monkey-patching avoid FileNotFoundError

from src.hijack.QubitUCCAnsatz_hijack import QubitUCCAnsatz_hijack
from src.hijack.uccsd_singlet_generator_hijack import uccsd_singlet_generator_hijack, uccsd_singlet_get_packed_amplitudes_hijack

CACHE_PATH = Path(tempfile.gettempdir())

PEEK         = os.environ.get('PEEK',         False)
DEBUG        = os.environ.get('DEBUG',        False)
DEBUG_HAM    = os.environ.get('DEBUG_HAM',    False)
DEBUG_ANSATZ = os.environ.get('DEBUG_ANSATZ', False)

Ham = Union[Hamiltonian, ESConserveHam]
QVM = Union[Simulator, ESConservation]
Config = Dict[str, Any]
Params = ndarray


def seed_everything(seed:Union[int, str]=None):
  if seed in [None, '']:
    seed = random.randint(0, 2**31-1)
  elif isinstance(seed, str):
    seed = int(seed)
  print(f'>> seed: {seed}')
  
  random.seed(seed)
  np.random.seed(seed)
  ms.set_seed(seed)


def approx_terms(tensor:ndarray, n_prec:int=-1, eps:float=0.0) -> ndarray:
  if n_prec > 0:
    tensor = np.round(tensor, decimals=n_prec)
  if eps > 0.0:
    tensor *= (np.abs(tensor) >= eps)
  return tensor


def get_ham(mol:MolecularData, config:Config) -> Ham:
  ham_of = mol.get_molecular_hamiltonian()
  # shapes: [], [nq, nq], [nq, nq, nq, nq]
  constant, one_body_tensor, two_body_tensor = ham_of.n_body_tensors.values()
  # NOTE: too much terms
  n_terms = 1 + (one_body_tensor != 0.0).sum() + (two_body_tensor != 0.0).sum()
  one_body_tensor = approx_terms(one_body_tensor, config.get('round_one', -1), config.get('trunc_one', 0.0))
  two_body_tensor = approx_terms(two_body_tensor, config.get('round_two', -1), config.get('trunc_two', 0.0))
  inter_ops = InteractionOperator(constant, one_body_tensor, two_body_tensor)
  ham_hiq = FermionOperator(inter_ops)
  if 'compress' in config:
    ham_hiq.compress(config.get('compress', 1e-8))
  n_terms_approx = len(ham_hiq)
  print(f'n_terms: {n_terms} => {n_terms_approx}')

  if 'QP' in config['ansatz']:
    ham_fo = normal_ordered(ham_hiq)
    ham_op = ham_fo.real
    ham = ESConserveHam(ham_op)   # ham of a FermionOperator
  else:
    ham_qo: QubitOperator = Transform(ham_hiq).jordan_wigner()   # fermi => pauli
    ham_op = ham_qo.real          # FIXME: why discard imag part?
    ham = Hamiltonian(ham_op)     # ham of a QubitOperator

  if DEBUG_HAM:
    mat = ham.hamiltonian.matrix().todense()
    diag = np.diag(mat).astype(np.float32)
    np.save('ham.npy', diag)

  return ham


def get_encoder_ortho() -> Tuple[Circuit, Circuit]:
  q0 = Circuit()
  q0 += X.on(0)
  q1 = Circuit()
  q1 += X.on(1)
  return q0, q1


def get_ansatz(mol:MolecularData, ansatz:str, config:Config, no_hfw:bool=False) -> Tuple[Circuit, Params]:
  # Construct hartreefock wave function circuit: |0...> -> |1...>
  # NOTE: this flip is important though do not know why; H does not work
  if not no_hfw:
    hartreefock_wfn_circuit = UN(X, mol.n_electrons)

  init_amp = None
  if ansatz == 'UCC':
    # H2O: 1000/25080 gates, 65 parameters
    ansatz_circuit = UCCAnsatz(mol.n_qubits, mol.n_electrons, trotter_step=config['trotter']).circuit
  elif ansatz == 'QUCC':
    # H2O: 310/3090 gates, 310 parameters
    ansatz_circuit = QubitUCCAnsatz(mol.n_qubits, mol.n_electrons, trotter_step=config['trotter']).circuit
  elif ansatz == 'QUCC-hijack':
    ansatz_circuit = QubitUCCAnsatz_hijack(mol.n_qubits, mol.n_electrons, trotter_step=config['trotter']).circuit
  elif ansatz == 'UCCSD':
    # H2O: 1000/25080 gates, 65 parameters
    if not 'sugar':
      ansatz_circuit, init_amp, _, _, _, _ = generate_uccsd(mol, threshold=config['thresh'])
    else:
      # H2O: 170 terms
      ucc_fermion_ops = uccsd_singlet_generator(mol.n_qubits, mol.n_electrons, anti_hermitian=True)
      # H2O: 2000 terms
      ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()
      ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag).circuit    # ucc_qubit_ops 中已经包含了复数因子 i 
      # 65 values
      init_amp_ccsd: Dict[str, float] = uccsd_singlet_get_packed_amplitudes(mol.ccsd_single_amps, mol.ccsd_double_amps, mol.n_qubits, mol.n_electrons)
      init_amp = np.asarray([init_amp_ccsd[i] for i in ansatz_circuit.params_name])
  elif ansatz == 'UCCSD-QP':
    # H2O: 170 terms
    ucc_fermion_ops = uccsd_singlet_generator(mol.n_qubits, mol.n_electrons, anti_hermitian=False)
    circ = Circuit()
    # H2O: 170 gates, 65 parameters
    for term in ucc_fermion_ops: circ += ExpmPQRSFermionGate(term)
  elif ansatz == 'UCCSD-QP-hijack':
    # H2O: 170*k gates, 65*k parameters
    circ = uccsd_singlet_generator_hijack(mol.n_qubits, mol.n_electrons, anti_hermitian=False, n_trotter=config['trotter'])
    init_amp_ccsd = uccsd_singlet_get_packed_amplitudes_hijack(mol.ccsd_single_amps, mol.ccsd_double_amps, mol.n_qubits, mol.n_electrons, n_trotter=config['trotter'])
    init_amp = np.asarray([init_amp_ccsd[i] for i in circ.params_name])
    # NOTE: this is important to eliminate param symmetricity
    init_amp += init_randu(init_amp.shape, mul=1e-3)
  elif ansatz == 'HEA':
    rot_gates = [globals()[g] for g in config['rot_gates']]
    entgl_gate = globals()[config['entgl_gate']]
    ansatz_circuit = HardwareEfficientAnsatz(mol.n_qubits, rot_gates, entgl_gate, depth=config['depth']).circuit
  else:
    raise ValueError(f'unknown ansatz: {ansatz}')

  if ansatz.startswith('UCCSD-QP'):
    vqc = circ
  elif no_hfw:
    vqc = ansatz_circuit
  else:
    vqc = hartreefock_wfn_circuit + ansatz_circuit

  if DEBUG_ANSATZ:
    vqc.summary()
    with open('circ.txt', 'w', encoding='utf-8') as fh:
      fh.write(str(vqc))

  if init_amp is None:
    init_amp = init_randu(len(vqc.all_paras))

  return vqc, init_amp


def get_sim(mol:MolecularData, ham:Ham, ret_null_ham:bool=False) -> Union[QVM, Tuple[QVM, Ham]]:
  if isinstance(ham, ESConserveHam):
    sim = ESConservation(mol.n_qubits, mol.n_electrons)
    if ret_null_ham:
      HAM_NULL = ESConserveHam(FermionOperator(''))
  else:
    sim = Simulator('mqvector', mol.n_qubits)
    if ret_null_ham:
      HAM_NULL = Hamiltonian(QubitOperator(''))
  return (sim, HAM_NULL) if ret_null_ham else sim


def init_randu(len:int, sub:float=0.5, mul:float=0.5):
  ''' default uniform on [-0.25, 0.25] '''
  return (np.random.random(len) - sub) * mul


def run_expectaion(sim:QVM, ham:Ham, circ:Circuit, params:ndarray) -> float:
  # Construct parameter resolver of the taregt state circuit
  pr = dict(zip(circ.params_name, params))
  # Calculate energy of ground state
  if isinstance(sim, ESConservation):
    E = sim.get_expectation(ham, circ, pr)
  else:
    # Evolve into tagert state
    sim.apply_circuit(circ, pr)
    if 'sugar':
      E = sim.get_expectation(ham)
    else:
      qs1 = sim.get_qs()
      H = ham.hamiltonian.matrix().todense()
      E = (qs1 @ H @ qs1)

  return E.real


scipy_callback = None

def optim_scipy(func:Callable, init_x:ndarray, grad_ops:Tuple[Callable, ...], config:Config) -> ndarray:
  global scipy_callback
  optim_plans: List[Config] = config.get('optims', [config])
  for i, cfg in enumerate(optim_plans):
    optim = cfg.get('optim', 'BFGS')
    print(f'[{i+1}/{len(optim_plans)}] optim: {optim}')
    res = minimize(
      func,
      init_x,
      args=tuple([*grad_ops, cfg]),
      method=optim,
      jac=True,
      tol=cfg.get('tol', 1e-6),
      options={
        'maxiter': cfg.get('maxiter', 1000), 
        'disp': cfg.get('debug', False),
      },
      callback=scipy_callback,
    )
    init_x = res.x
  return np.asarray(res.x, dtype=np.float32)


def optim_mindspore(pqc:MQAnsatzOnlyLayer, config:Config) -> ndarray:
  optim_cls = getattr(nn.optim, config.get('optim', 'Adagrad'))
  optimizer = optim_cls(pqc.trainable_params(), learning_rate=config.get('lr', 4e-2))
  train_pqc = nn.TrainOneStepCell(pqc, optimizer)
  for _ in range(config.get('maxiter', 1000)): train_pqc()
  return np.asarray(pqc.weight, dtype=np.float32)
