#!/usr/bin/env python3
# Author: Armit
# Create Time: 周六 2025/05/17

'''
bSB优化相位的时候，需要一个把 spin 组转为 phase 的映射函数，希望这个转换是连续稳定的(格雷码)
[nq=1]
  + -> +1 (0°)
  - -> -1 (180°)
[nq=2] 碰巧是是格雷码
  ++ -> +1 (0°)
  +- -> +i (90°)
  -- -> -1 (180°)
  -+ -> -i (270°)
[nq=3]
  arXiv:2409.19938 提出的一种线性映射法，但不是格雷码
[nq=4]
  线性映射的自由度已经不够了...
  我们需要一点非线性性 :)
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import Tensor
import numpy as np

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('>> device:', device)

# 格雷码: https://blog.csdn.net/NJUzzf98/article/details/124408584
GREY_CODE_3 = [
  '000',
  '001',
  '011',
  '010',
  '110',
  '111',
  '101',
  '100',
]
GREY_CODE_4 = [
  '0000',
  '0001',
  '0011',
  '0010',
  '0110',
  '0111',
  '0101',
  '0100',
  '1100',
  '1101',
  '1111',
  '1110',
  '1010',
  '1011',
  '1001',
  '1000',
]


def get_data(nq:int):
  vq = 2 * np.pi / 2**nq
  GREY_CODE = GREY_CODE_3 if nq == 3 else GREY_CODE_4
  x = np.asarray([[1-int(b)*2 for b in list(bits)] for bits in GREY_CODE], dtype=np.float32).round(4)
  y = np.asarray([np.exp(1j*i*vq) for i in range(2**nq)], dtype=np.complex64).round(4)
  X = torch.from_numpy(x.real).float()
  Y = torch.stack([
    torch.from_numpy(y.real),
    torch.from_numpy(y.imag),
  ], dim=-1)
  return X, Y


class MLP(nn.Module):
  def __init__(self, d_in:int, d_hid:int):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(d_in, d_hid),
      nn.ReLU(inplace=True),
      nn.Linear(d_hid, 2, bias=False),
    )
  def forward(self, x:Tensor) -> Tensor:
    return self.mlp(x)


def train(nq:int, model:nn.Module, X:Tensor, Y:Tensor):
  save_fp = f'mlp-{nq}.pth'
  if os.path.exists(save_fp):
    print('>> ignore file exist:', save_fp)
    return

  X = X.to(device)
  Y = Y.to(device)
  model = model.to(device)
  optim = Adam(model.parameters(), lr=0.001)
  print('>> param_cnt:', sum(p.numel() for p in model.parameters()))

  iter = 0
  model.train()
  while True:
    optim.zero_grad()
    out = model(X)
    loss = F.mse_loss(out, Y)
    loss.backward()
    optim.step()
    iter += 1

    with torch.inference_mode():
      if iter % 100 == 0:
        print('loss:', loss.item())
      if loss < 1e-10:
        print(f'>> Converged at iter: {iter}')
        break

  model = model.eval().cpu()
  print(f'>> save model to {save_fp}')
  torch.save(model.state_dict(), save_fp)


def train_nq3():
  nq = 3
  X, Y = get_data(nq)
  model = MLP(nq, 4)  # n_param=24
  train(nq, model, X, Y)

def train_nq4():
  nq = 4
  X, Y = get_data(nq)
  model = MLP(nq, 6)  # n_param=42
  train(nq, model, X, Y)


if __name__ == '__main__':
  # 有不收敛的可能性，需要反复重试
  # 不就为了省点儿参数量嘛....
  train_nq3()
  train_nq4()
