#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/05 

import math
import json
import random
import pickle as pkl
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import Tensor
from torch.nn import Parameter
import torch.storage
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from run_baseline import modulate_and_transmit

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'    # CPU is even faster :(

BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)

# Eq. 15 ~ 16 from arXiv:2306.16264
σ = F.sigmoid
sw = lambda x: x * σ(x)
φ_s = lambda x, Λ=10: (1 / Λ) * (Λ * sw(x + 1) - Λ * sw(x - 1)) - 1
ψ_s = lambda x, A=100, B=1.01: σ(A * (torch.abs(x) - B))


class DU_LM_SB(nn.Module):

  ''' arXiv:2306.16264 Deep Unfolded Simulated Bifurcation for Massive MIMO Signal Detection '''

  def __init__(self, T:int, batch_size:int=100):
    super().__init__()

    self.T = T
    self.batch_size = batch_size
    # Eq. 4
    self.a = torch.linspace(0, 1, T)

    # the T+2 trainable parameters :)
    self.Δ = Parameter(torch.ones  ([T],    dtype=torch.float32), requires_grad=True)
    self.η = Parameter(torch.tensor([1.0],  dtype=torch.float32), requires_grad=True)
    self.λ = Parameter(torch.tensor([25.0], dtype=torch.float32), requires_grad=True)

  def get_J_h(self, H:Tensor, y:Tensor, nbps:int) -> Tuple[Tensor, Tensor]:
    return to_ising_ext(H, y, nbps, self.λ)

  def forward(self, H:Tensor, y:Tensor, nbps:int, **kwargs) -> Tensor:
    ''' LM part '''
    J, h = self.get_J_h(H, y, nbps, **kwargs)

    ''' DU-SB part '''
    # Eq. 6 and 12
    B = self.batch_size
    N = J.shape[0]
    # from essay, this will NOT work
    #c_0: float = 2 * math.sqrt((N - 1) / (J**2).sum())
    # from qaia lib
    c_0: Tensor = 0.5 * math.sqrt(N - 1) / torch.linalg.norm(J, ord='fro')

    # rand init x and y
    x = 0.02 * (torch.rand(N, B, device=H.device) - 0.5)
    y = 0.02 * (torch.rand(N, B, device=H.device) - 0.5)

    # Eq. 11 ~ 14
    for k, Δ_k in enumerate(self.Δ):
      y = y + Δ_k * (-(1 - self.a[k]) * x + self.η * c_0 * (J @ x + h))
      x = x + Δ_k * y
      x = φ_s(x)
      y = y * (1 - ψ_s(x))

    # [B=100, rb*c*Nt=256]
    spins = x.T

    return spins


class pReg_LM_SB(DU_LM_SB):

  ''' parametraized-regularizing DU-LM-SB '''

  def __init__(self, T:int, batch_size:int=100):
    super().__init__(T, batch_size)

    # the trainable lmmse-like part :)
    self.U_λ_res_64  = Parameter(torch.diag(self.λ.sqrt() * torch.ones([2*64],  dtype=torch.float32)), requires_grad=True)
    self.U_λ_res_128 = Parameter(torch.diag(self.λ.sqrt() * torch.ones([2*128], dtype=torch.float32)), requires_grad=True)

  def get_J_h(self, H:Tensor, y:Tensor, nbps:int) -> Tuple[Tensor, Tensor]:
    if H.shape[0] == 64:
      U_λ_res = self.U_λ_res_64
    elif H.shape[0] == 128:
      U_λ_res = self.U_λ_res_128
    else: raise ValueError(f'not support H.shape: {H.shape}')
    return to_ising_ext(H, y, nbps, self.λ, U_λ_res)

  def gather_weights_pickle(self) -> Dict[str, Any]:
      return {
        'deltas': self.Δ.detach().cpu().numpy().tolist(),
        'eta':    self.η.detach().cpu().item(),
        'lmbd':   self.λ.detach().cpu().item(),
        'lmbd_res': {
          128: self.U_λ_res_128.detach().cpu().numpy(),
          64:  self.U_λ_res_64 .detach().cpu().numpy(),
        }
      }


class ppReg_LM_SB(pReg_LM_SB):

  ''' projective parametraized-regularizing DU-LM-SB '''

  def __init__(self, T:int, batch_size:int=100):
    super().__init__(T, batch_size)

  def get_J_h(self, H:Tensor, y:Tensor, nbps:int) -> Tuple[Tensor, Tensor]:
    if H.shape[0] == 64:
      U_λ_res = self.U_λ_res_64
    elif H.shape[0] == 128:
      U_λ_res = self.U_λ_res_128
    else: raise ValueError(f'not support H.shape: {H.shape}')
    return to_ising_ext(H, y, nbps, self.λ, U_λ_res, lmbd_res_mode='proj')


class pppReg_LM_SB(DU_LM_SB):

  ''' per-SNR projective parametraized-regularizing DU-LM-SB '''

  def __init__(self, T:int, batch_size:int=100):
    super().__init__(T, batch_size)

    # the trainable lmmse-like part :)
    self.snr_list = [10, 15, 20]
    self.N_list = [64, 128]
    for snr in self.snr_list:
      for N in self.N_list:
        name = f'U_λ_{snr}_{N}'
        p = Parameter(torch.diag(self.λ.sqrt() * torch.ones([2*N],  dtype=torch.float32)), requires_grad=True)
        self.register_parameter(name, p)

  def get_J_h(self, H:Tensor, y:Tensor, nbps:int, **kwargs) -> Tuple[Tensor, Tensor]:
    snr = kwargs['snr']
    U_λ_res = self.get_parameter(f'U_λ_{snr}_{H.shape[0]}')
    assert U_λ_res is not None, f'not support snr: {snr}, H.shape: {H.shape}'
    return to_ising_ext(H, y, nbps, self.λ, U_λ_res, lmbd_res_mode='proj')

  def gather_weights_pickle(self) -> Dict[str, Any]:
      return {
        'deltas': self.Δ.detach().cpu().numpy().tolist(),
        'eta':    self.η.detach().cpu().item(),
        'lmbd':   self.λ.detach().cpu().item(),
        'lmbd_res': {
          snr: {
            N: self.get_parameter(f'U_λ_{snr}_{N}').detach().cpu().numpy() 
              for N in self.N_list
          } for snr in self.snr_list
        }
      }


def to_ising_ext(H:Tensor, y:Tensor, nbps:int, lmbd:Tensor, lmbd_res:Tensor=None, lmbd_res_mode:str='res') -> Tuple[Tensor, Tensor]:
  ''' pytorch version of to_ising_ext() '''

  # the size of constellation, the M-QAM where M in {16, 64, 256}
  M = 2**nbps
  # n_elem at TX side (c=2 for real/imag, 1 symbol = 2 elem)
  Nr, Nt = H.shape
  N = 2 * Nt
  # n_bits/n_spins that one elem decodes to
  rb = nbps // 2
  # QAM variance for normalization
  qam_var = 2 * (M - 1) / 3

  # Eq. 7 the transform matrix T from arXiv:2105.10535
  I = torch.eye(N, device=H.device)
  # [rb, N, N]
  T: Tensor = (2**(rb - 1 - torch.arange(rb, device=H.device)))[:, None, None] * I[None, ...]
  # [rb*N, N] => [N, rb*N]
  T = T.reshape(-1, N).T

  # Eq. 1
  H_tilde = torch.vstack([
    torch.hstack([H.real, -H.imag]), 
    torch.hstack([H.imag,  H.real]),
  ])
  y_tilde = torch.cat([y.real, y.imag])

  # Eq. 10
  # LMMSE-like part, with our fix
  if lmbd_res is None:
    U_λ = torch.linalg.inv(H_tilde @ H_tilde.T + lmbd * I) / lmbd
  else:
    if lmbd_res_mode == 'res':
      U_λ = torch.linalg.inv(H_tilde @ H_tilde.T + lmbd_res @ lmbd_res.T) / lmbd
    elif lmbd_res_mode == 'proj':
      U_λ = lmbd_res
  H_tilde_T = H_tilde @ T
  J = -H_tilde_T.T @ U_λ @ H_tilde_T * (2 / qam_var)
  J = J * (1 - torch.eye(J.shape[0], device=H.device))    # mask diagonal to zeros
  z = (y_tilde - H_tilde_T @ torch.ones([N * rb, 1], device=H.device) + (math.sqrt(M) - 1) * H_tilde @ torch.ones([N, 1], device=H.device)) / math.sqrt(qam_var)
  h = 2 * H_tilde_T.T @ (U_λ @ z)

  # [rb*N, rb*N], [rb*N, 1]
  return J, h


def ber_loss(spins:Tensor, bits:Tensor, loss_fn:str='mse') -> Tensor:
  ''' differentiable version of compute_ber() '''
  if False:
    from judger import compute_ber
    assert compute_ber

  # convert the bits from sionna style to constellation style
  # Sionna QAM16 map: https://nvlabs.github.io/sionna/examples/Hello_World.html
  bits_constellation = 1 - torch.cat([bits[..., 0::2], bits[..., 1::2]], dim=-1)

  # Fig. 2 from arXiv:2001.04014, the QuAMax paper converting QuAMax to gray coded
  nbps = bits_constellation.shape[1]
  rb = nbps // 2
  spins = torch.reshape(spins, (rb, 2, -1))  # [rb, c=2, Nt]
  spins = torch.permute(spins, (2, 1, 0))    # [Nt, c=2, rb]
  spins = torch.reshape(spins, (-1, 2*rb))   # [Nt, 2*rb]
  bits_hat = (spins + 1) / 2                 # Ising {-1, +1} to QUBO {0, 1}

  # QuAMax => intermediate code
  bits_final = bits_hat.clone()                           # copy b[0]
  index = torch.nonzero(bits_hat[:, rb-1] > 0.5)[:, -1]   # select even columns
  bits_hat[index, rb:] = 1 - bits_hat[index, rb:]         # invert bits of high part (flip upside-down)
  # Differential bit encoding, intermediate code => gray code (constellation-style)
  for i in range(1, nbps):                                # b[i] = b[i] ^ b[i-1]
    x = bits_hat[:, i] + bits_hat[:, i - 1]
    x_dual = 2 - x
    bits_final[:, i] = torch.where(x <= x_dual, x, x_dual)
  # calc BER
  if loss_fn in ['l2', 'mse']:
    return F.mse_loss(bits_final, bits_constellation)
  elif loss_fn in ['l1', 'mae']:
    return F.l1_loss(bits_final, bits_constellation)
  elif loss_fn == 'bce':
    pseudo_logits = bits_final * 2 - 1
    return F.binary_cross_entropy_with_logits(pseudo_logits, bits_constellation)


def make_random_transmit(bits_shape:torch.Size, H:Tensor, nbps:int, SNR:int) -> Tuple[Tensor, Tensor]:
  ''' transmit random bits through given channel mix H '''
  bits = np.random.uniform(size=bits_shape) < 0.5
  x, y = modulate_and_transmit(bits.astype(np.float32), H.cpu().numpy(), nbps, SNR=10)   # SNR
  bits = torch.from_numpy(bits).to(device, torch.float32)
  y    = torch.from_numpy(y)   .to(device, torch.complex64)
  return bits, y


def load_data(limit:int) -> List[Tuple]:
  dataset = []
  for idx in tqdm(range(150)):
    if idx > limit > 0: break
    with open(f'MLD_data/{idx}.pickle', 'rb') as fh:
      data = pkl.load(fh)
      H = data['H']
      y = data['y']
      bits = data['bits']
      nbps: int = data['num_bits_per_symbol']
      SNR: int = data['SNR']

      H    = torch.from_numpy(H)   .to(device, torch.complex64)
      y    = torch.from_numpy(y)   .to(device, torch.complex64)
      bits = torch.from_numpy(bits).to(device, torch.float32)
      dataset.append([H, y, bits, nbps, SNR])
  return dataset


class ValueWindow:

  def __init__(self, nlen=10):
    self.values: List[float] = []
    self.nlen = nlen

  def add(self, v:float):
    self.values.append(v)
    self.values = self.values[-self.nlen:]

  @property
  def mean(self):
    return sum(self.values) / len(self.values) if self.values else 0.0


def train(args):
  print('device:', device)
  print('hparam:', vars(args))
  exp_name = f'{args.M.replace("_", "-")}_T={args.n_iter}_lr={args.lr}{"_overfit" if args.overfit else ""}'

  ''' Data '''
  dataset = load_data(args.limit)

  ''' Model '''
  model: DU_LM_SB = globals()[args.M](args.n_iter, args.batch_size).to(device)
  optim = Adam(model.parameters(), args.lr)

  ''' Ckpt '''
  init_step = 0
  losses = []
  if args.load:
    print(f'>> resume from {args.load}')
    ckpt = torch.load(args.load, map_location='cpu')
    init_step = ckpt['steps']
    losses.extend(ckpt['losses'])
    model.load_state_dict(ckpt['model'], strict=False)
    try:
      optim.load_state_dict(ckpt['optim'])
    except:
      optim_state_ckpt = ckpt['optim']
      optim_state_cur = optim.state_dict()
      optim_state_ckpt['param_groups'][0]['params'] = optim_state_cur['param_groups'][0]['params']
      optim_state_ckpt['state'] = optim_state_cur['state']
      optim.load_state_dict(optim_state_ckpt)

  ''' Bookkeep '''
  loss_wv = ValueWindow(100)
  steps_minor = 0
  steps = init_step

  ''' Train '''
  model.train()
  try:
    pbar = tqdm(total=args.steps-init_step)
    while steps < init_step + args.steps:
      if not args.no_shuffle and steps_minor % len(dataset) == 0:
        random.shuffle(dataset)
      sample = dataset[steps_minor % len(dataset)]

      H, y, bits, nbps, SNR = sample
      if not args.overfit:
        bits, y = make_random_transmit(bits.shape, H, nbps, SNR)

      if isinstance(model, pppReg_LM_SB):
        spins = model(H, y, nbps, snr=SNR)
      else:
        spins = model(H, y, nbps)
      loss_each = torch.stack([ber_loss(sp, bits, args.loss_fn) for sp in spins])
      loss = getattr(loss_each, args.agg_fn)()
      loss_for_backward: Tensor = loss / args.grad_acc
      loss_for_backward.backward()

      loss_wv.add(loss.item())

      steps_minor += 1

      if args.grad_acc == 1 or steps_minor % args.grad_acc:
        optim.step()
        optim.zero_grad()
        steps += 1
        pbar.update()

      if not 'debug best pred':
        with torch.no_grad():
          from judger import compute_ber
          soluts = torch.sign(spins).detach().cpu().numpy()
          bits_np = bits.cpu().numpy()
          ber = [compute_ber(solut, bits_np) for solut in soluts]
          print('ber:', ber)
          breakpoint()

      if steps % 50 == 0:
        losses.append(loss_wv.mean)
        print(f'>> [step {steps}] loss: {losses[-1]}')
  except KeyboardInterrupt:
    pass

  ''' Ckpt '''
  ckpt = {
    'steps': steps,
    'losses': losses,
    'model': model.state_dict(),
    'optim': optim.state_dict(),
  }
  torch.save(ckpt, LOG_PATH / f'{exp_name}.pth')

  with torch.no_grad():
    params = {
      'deltas': model.Δ.detach().cpu().numpy().tolist(),
      'eta':    model.η.detach().cpu().item(),
      'lmbd':   model.λ.detach().cpu().item(),
    }
    print('params:', params)

    with open(LOG_PATH / f'{exp_name}.json', 'w', encoding='utf-8') as fh:
      json.dump(params, fh, indent=2, ensure_ascii=False)

  if args.M in ['pReg_LM_SB', 'ppReg_LM_SB', 'pppReg_LM_SB']:
    with torch.no_grad():
      model: pReg_LM_SB
      weights = model.gather_weights_pickle()
    with open(LOG_PATH / f'{exp_name}.pkl', 'wb') as fh:
      pkl.dump(weights, fh)

  plt.plot(losses)
  plt.tight_layout()
  plt.savefig(LOG_PATH / f'{exp_name}.png', dpi=600)


if __name__ == '__main__':
  METHODS = [name for name, value in globals().items() if type(value) == type(DU_LM_SB) and issubclass(value, DU_LM_SB)]

  parser = ArgumentParser()
  parser.add_argument('-M', default='DU_LM_SB', choices=METHODS)
  parser.add_argument('-T', '--n_iter', default=10, type=int)
  parser.add_argument('-B', '--batch_size', default=32, type=int, help='SB candidate batch size')
  parser.add_argument('--steps', default=3000, type=int)
  parser.add_argument('--loss_fn', default='mse', choices=['mse', 'l1', 'bce'])
  parser.add_argument('--agg_fn', default='mean', choices=['mean', 'max'])
  parser.add_argument('--grad_acc', default=1, type=int, help='training batch size')
  parser.add_argument('--lr', default=1e-2, type=float)
  parser.add_argument('--load', help='ckpt to resume')
  parser.add_argument('-L', '--limit', default=-1, type=int, help='limit dataset n_sample')
  parser.add_argument('--overfit', action='store_true', help='overfit to given dataset')
  parser.add_argument('--no_shuffle', action='store_true', help='no shuffle dataset')
  parser.add_argument('--log_every', default=50, type=int)
  args = parser.parse_args()

  if args.overfit:
    print('[WARN] you are trying to overfit to the given dataset!')

  train(args)
