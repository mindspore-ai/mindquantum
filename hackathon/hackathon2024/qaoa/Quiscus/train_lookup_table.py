#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/07 

# 微调预制表 (Euler step 方法)

import random
from copy import deepcopy
from re import compile as Regex
from argparse import ArgumentParser

import mindspore as ms
import mindspore.ops.functional as F
from mindspore.nn.optim import Adam
from mindspore.nn import TrainOneStepCell
from mindquantum.simulator import Simulator
from mindquantum.core.operators import Hamiltonian
from mindquantum.framework import MQAnsatzOnlyLayer
import numpy as np
from tqdm import tqdm

from utils.path import LOG_PATH
from utils.lookup_table import load_lookup_table_original, load_lookup_table, dump_lookup_table, load_lookup_table_ex, dump_lookup_table_ex, SIM_EQ, NON_EQ
from utils.qcirc import qaoa_hubo, build_ham_high
from score import load_data
from main import ave_D, order, trans_gamma, rescale_factor

ms.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE, pynative_synchronize=True)

R_ITER = Regex('iter=(\d+)')
THRESHOLD = 1.0


def train(args):
  ''' Data '''
  dataset = []
  for propotion in [0.3, 0.6, 0.9]:
    for k in range(2, 6):
      for coef in ['std', 'uni', 'bimodal']:
        for r in range(10):
          Jc_dict = load_data(f"data/k{k}/{coef}_p{propotion}_{r}.json")
          dataset.append(Jc_dict)

  ''' Ckpt '''
  if args.ex:
    if not args.load and not args.load_base:
      raise ValueError('>> you must start from some pretrained weights, fresh training is not supported yet.')
    if args.load:
      lookup_table_ex = load_lookup_table_ex(args.load)
      lookup_table_ex_moment = {}
      try:
        init_iter = int(R_ITER.findall(args.load)[0])
      except:
        init_iter = 0
    elif args.load_base:
      lookup_table = load_lookup_table(args.load_base)
      lookup_table_ex = {
        SIM_EQ: deepcopy(lookup_table),    # std(weights) <= THRESHOLD, for near-equal edge weights
        NON_EQ: deepcopy(lookup_table),    # std(weights) > THRESHOLD, for non-equal edge weights
      }
      lookup_table_ex_moment = {}
      del lookup_table
      try:
        init_iter = int(R_ITER.findall(args.load_base)[0])
      except:
        init_iter = 0
  else:
    if args.load:
      lookup_table = load_lookup_table(args.load)
      lookup_table_moment = {}
      try:
        init_iter = int(R_ITER.findall(args.load)[0])
      except:
        init_iter = 0
    else:
      lookup_table = load_lookup_table_original()
      lookup_table_moment = {}
      init_iter = 0
      save_fp = LOG_PATH / 'lookup_table-original.json'
      if not save_fp.exists():
        dump_lookup_table(lookup_table, save_fp)
  
  ''' Save '''
  save_dp = (LOG_PATH / args.name) if args.name else LOG_PATH
  save_dp.mkdir(exist_ok=True)

  ''' Simulator '''
  Nq = 12
  sim = Simulator('mqvector', n_qubits=Nq)

  ''' Train '''
  for iter in tqdm(range(init_iter, args.iters), initial=init_iter, total=args.iters):
    # random pick a sample and circuit depth
    if iter % len(dataset) == 0: random.shuffle(dataset)
    Jc_dict = dataset[iter % len(dataset)]
    p = random.choice([4, 8])
    ham = Hamiltonian(build_ham_high(Jc_dict))
    D = ave_D(Jc_dict, Nq)
    k = min(order(Jc_dict), 6)

    if args.ex:
      vals_std = np.asarray(list(Jc_dict.values())).std()
      w = SIM_EQ if vals_std < THRESHOLD else NON_EQ

    # vqc
    gamma_params = [f'g{i}' for i in range(p)]
    beta_params  = [f'b{i}' for i in range(p)]
    circ = qaoa_hubo(Jc_dict, Nq, gamma_params, beta_params, p=p)

    # init_p
    if args.ex:
      params = lookup_table_ex[w][p][k]
    else:
      params = lookup_table[p][k]
    gammas, betas = np.split(params, 2)
    factor = rescale_factor(Jc_dict) * args.rescaler    # rescale gamma
    gammas = trans_gamma(gammas, D) * factor

    # align order with qcir (param => weights)
    init_weights = []
    for pname in circ.params_name:
      which = pname[0]
      idx = int(pname[1:])
      if which == 'g':
        init_weights.append(gammas[idx])
      else:
        init_weights.append(betas[idx])
    init_weights = ms.tensor(init_weights)

    # train
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    net = MQAnsatzOnlyLayer(grad_ops, weight=init_weights)
    opt = Adam(net.trainable_params(), learning_rate=args.lr)
    train_step = TrainOneStepCell(net, opt)
    E_before = net().item()
    for step in range(args.steps):
      loss = train_step()
      if step % 10 == 0:
        print(f'>> [step {step}] loss: {loss.item()}')
    E_after = net().item()

    L1 = F.l1_loss(net.weight, init_weights)
    print(f'>> [dist] L1: {L1.mean()}, Linf: {L1.max()}')

    # de-align order to lookup_table (weights => param)
    tuned_weights = net.weight.asnumpy()
    tuned_gammas = [None] * p
    tuned_betas  = [None] * p
    for pname, pvalue in zip(circ.params_name, tuned_weights):
      which = pname[0]
      idx = int(pname[1:])
      if which == 'g':
        tuned_gammas[idx] = pvalue
      else:
        tuned_betas[idx] = pvalue

    # update lookup table
    tuned_gammas = np.asarray(tuned_gammas)
    tuned_gammas /= factor * np.arctan(1 / np.sqrt(D - 1))  # inv rescale gamma
    tuned_betas = np.asarray(tuned_betas)
    tuned_params = np.concatenate([tuned_gammas, tuned_betas])

    # adaptive step size: lr ∝ ΔE
    dx_decay = args.dx_decay ** (init_iter // args.dx_decay_every)
    ΔE = E_before - E_after
    lr = args.dx * dx_decay * np.log1p(ΔE)
    if args.ex:
      if args.momentum:
        if w not in lookup_table_ex_moment: lookup_table_ex_moment[w] = {}
        if p not in lookup_table_ex_moment[w]: lookup_table_ex_moment[w][p] = {}
        if k not in lookup_table_ex_moment[w][p]: lookup_table_ex_moment[w][p][k] = tuned_params.copy()
        lookup_table_ex_moment[w][p][k] = (1 - args.momentum) * lookup_table_ex_moment[w][p][k] + args.momentum * tuned_params
        lookup_table_ex[w][p][k] = (1 - lr) * params + lr * lookup_table_ex_moment[w][p][k]
      else:
        lookup_table_ex[w][p][k] = (1 - lr) * params + lr * tuned_params
    else:
      if args.momentum:
        if p not in lookup_table_moment: lookup_table_moment[p] = {}
        if k not in lookup_table_moment[p]: lookup_table_moment[p][k] = tuned_params.copy()
        lookup_table_moment[p][k] = (1 - args.momentum) * lookup_table_moment[p][k] + args.momentum * tuned_params
        lookup_table[p][k] = (1 - lr) * params + lr * lookup_table_moment[p][k]
      else:
        lookup_table[p][k] = (1 - lr) * params + lr * tuned_params

    # tmp ckpt
    if (iter + 1) % 100 == 0:
      if args.ex:
        dump_lookup_table_ex(lookup_table_ex, save_dp / f'lookup_table-iter={iter+1}.json')
      else:
        dump_lookup_table(lookup_table, save_dp / f'lookup_table-iter={iter+1}.json')

  ''' Ckpt '''
  if args.ex:
    dump_lookup_table_ex(lookup_table_ex, save_dp / 'lookup_table.json')
  else:
    dump_lookup_table(lookup_table, save_dp / 'lookup_table.json')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--iters', default=10000, type=int, help='optim iter case count')
  parser.add_argument('--steps', default=100, type=int, help='optim steps per sample case')
  parser.add_argument('--lr', default=1e-5, type=float)
  parser.add_argument('--dx', default=0.1, type=float)
  parser.add_argument('--dx_decay', default=0.98, type=float)
  parser.add_argument('--dx_decay_every', default=100, type=int)
  parser.add_argument('--momentum', default=0.0, type=float, help='impact of of the new param, >= 0.6 recommended')
  parser.add_argument('--rescaler', default=1.275, type=float, help='gamma rescaler')
  parser.add_argument('--load', help='load from lookup table of same type')
  parser.add_argument('--load_base', help='load from lookup table of standard')
  parser.add_argument('--ex', action='store_true', help='enable ex mode, divide and conquer simeq & noneq cases')
  parser.add_argument('--name')
  args = parser.parse_args()

  train(args)
