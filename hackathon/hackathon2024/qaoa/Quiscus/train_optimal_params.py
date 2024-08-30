#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/07 

# 为每个样例寻找其最优参数 (i.e. 刷分上限)

import json
from pathlib import Path
from argparse import ArgumentParser

import mindspore as ms
import mindspore.ops.functional as F
from mindspore.nn.optim import Adam
from mindspore.nn import TrainOneStepCell
from mindspore import context
from mindquantum.simulator import Simulator
from mindquantum.core.operators import Hamiltonian
from mindquantum.framework import MQAnsatzOnlyLayer
import numpy as np
from tqdm import tqdm

from utils.path import DATA_OPT_PATH ; DATA_OPT_PATH.mkdir(exist_ok=True)
from utils.lookup_table import load_lookup_table_original
from utils.qcirc import qaoa_hubo, build_ham_high
from score import load_data
from main import ave_D, order, trans_gamma, rescale_factor

context.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE, pynative_synchronize=True)


def train(args):
  ''' Data (check) '''
  score = 0
  fps_for_score = []
  for propotion in [0.3, 0.9]:
    for k in range(2, 5):
        for coef in ['std', 'uni', 'bimodal']:
            for r in range(5):
                fps_for_score.append(f"data/k{k}/{coef}_p{propotion}_{r}.json")

  ''' Data '''
  dataset = []
  for propotion in [0.3, 0.6, 0.9]:
    for k in range(2, 6):
      for coef in ['std', 'uni', 'bimodal']:
        for r in range(10):
          fp = f"data/k{k}/{coef}_p{propotion}_{r}.json"
          Jc_dict = load_data(fp)
          dataset.append((fp, Jc_dict))

  ''' Ckpt '''
  lookup_table = load_lookup_table_original()

  ''' Simulator '''
  Nq = 12
  sim = Simulator('mqvector', n_qubits=Nq)

  ''' Optimize '''
  for fp, Jc_dict in tqdm(dataset):
    if args.peek and fp not in fps_for_score: continue
    opt_params = {}
    for p in [4, 8]:
      # ham
      ham = Hamiltonian(build_ham_high(Jc_dict))
      D = ave_D(Jc_dict, Nq)
      k = min(order(Jc_dict), 6)

      # vqc
      gamma_params = [f'g{i}' for i in range(p)]
      beta_params  = [f'b{i}' for i in range(p)]
      circ = qaoa_hubo(Jc_dict, Nq, gamma_params, beta_params, p=p)

      # init_p
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
      last_loss = None
      for step in range(args.steps):
        loss = train_step()
        if last_loss is not None:
          loss_diff = loss - last_loss
          if loss_diff.abs() < 1e-6: break
        last_loss = loss
        if step % 100 == 0:
          print(f'>> [step {step}] loss: {loss.item()}')
      
      if fp in fps_for_score:
        score -= last_loss.item()

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

      # save optimal params
      tuned_gammas = np.asarray(tuned_gammas)
      tuned_gammas /= factor * np.arctan(1 / np.sqrt(D - 1))  # inv rescale gamma
      tuned_betas = np.asarray(tuned_betas)
      tuned_params = np.concatenate([tuned_gammas, tuned_betas])

      opt_params[p] = tuned_params.tolist()

    # save opt params
    save_fp: Path = DATA_OPT_PATH / fp.replace('data/', '')
    save_fp.parent.mkdir(exist_ok=True, parents=True)
    with open(save_fp, 'w', encoding='utf-8') as fh:
      json.dump(opt_params, fh, indent=2, ensure_ascii=False)

  print(f'>> optimal possible score: {score:.5f}')  # 23654.99239


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--steps', default=1000, type=int, help='optim steps per sample case')
  parser.add_argument('--lr', default=1e-4, type=float)
  parser.add_argument('--rescaler', default=1.275, type=float, help='gamma rescaler')
  parser.add_argument('--peek', action='store_true', help='only peek for optimal score')
  args = parser.parse_args()

  train(args)
