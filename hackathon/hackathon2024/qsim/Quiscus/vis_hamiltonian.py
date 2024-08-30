#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/04 

# 查看系统哈密顿量的真实基态和能量
# - 真实基态 psi
#  - HF处主振幅 0.9677，次主振幅1项 0.1877，一般振幅18项 0.01~0.001，剩下分量绝大多数为0
#  - geometry 的微小改变只会改变各振幅强度，不太会改变项的有无
# - 线路制备态 psi_hat
#  - HEA 能保底达到 0.99，运气好能冲 0.9999
#  - HEAc 比 HEA 差一点，保底 0.98
#  - CC 能达到 0.999 能制备最好的态
#  - pCHC 完全不行 

from pprint import pprint as pp
import pickle as pkl
from solution import *
from benchmark_ansatz import save_fp

def state_vec_to_sparse_amp(psi:ndarray, eps:float=1e-4) -> Dict[str, float]:
  return {'|' + bin(idx)[2:].rjust(nq, '0') + '>': val.item() for idx, val in enumerate(psi.real) if np.abs(val).item() > eps}


''' Data '''
molecule = [
  ['H', [0, 0, 0.0]],
  ['H', [0, 0, 1.0]],
  ['H', [0, 0, 2.0]],
  ['H', [0, 0, 3.0]],
]
mol = generate_molecule(molecule)
ham = get_molecular_hamiltonian(mol)

''' Matrix '''
H = ham.matrix().todense()
E_hf = min(np.diag(H))
evs, vecs = np.linalg.eigh(H)
E_fci, psi = evs[0], vecs[:, 0]
print('E_hf:', E_hf)
print('E_fci (λmin):', E_fci)
print('top-k small λ:', evs[:10])
print('top-k large λ:', evs[-10:][::-1])
#print('|ψ>:', np.round(psi.real, 4).T)
nq = int(np.log2(H.shape[0]))
print('|ψ> sparse amplitudes:')
pp(state_vec_to_sparse_amp(psi))

'''
[Sorted by absolute Amplitude]
|00001111>:  0.9677106263678632
|00110011>: -0.18768784145590353
|01101001>:  0.07892321707925219
|10010110>:  0.07892321707925219
|00111100>: -0.06500319468952202
|11001100>: -0.052037940497991805
|01100110>: -0.048611561597010695
|10011001>: -0.04861156159701068
|11000011>: -0.0416976983518129
|01011010>: -0.03031165548224152
|10100101>: -0.030311655482241504
|11110000>:  0.021663372022213884  (↑ 共12个主要项)
|00011110>:  0.005759131529488659
|00101101>: -0.005759131529488631
|01001011>:  0.004410107871199615
|10000111>: -0.004410107871199616
|11010010>:  0.0037101578490584645
|11100001>: -0.003710157849058464
|01111000>:  0.0026251185155691174
|10110100>: -0.0026251185155691152
'''

# 对比各线路制备的态
with open(save_fp, 'rb') as fh:
  stats = pkl.load(fh)

E_min = 99999
psi_hat = None
best_name = None
print('[state fidelity]')
for name, stat in stats.items():
  qs = np.expand_dims(stat['qs'], -1)
  fid = np.abs(psi.T @ qs).item()
  E = np.real(qs.T @ H @ qs).item()
  print(f'  {name}: fid {fid}, E {E}')
  if E < E_min:
    E_min = E
    psi_hat = qs
    best_name = name

# 最好的态仍然有很大的 L1 error, 说明线路先验还是不行
psi_diff = np.abs(psi_hat - psi)
print('psi L1:', psi_diff.mean())
print('psi Linf:', psi_diff.max())

#psi_hat_fix = np.where(np.abs(psi_hat) <= 1e-4, 0, psi_hat)
#print(f'best |ψ_U> from {best_name}:', np.round(psi_hat_fix.real, 4).T)
print(f'best |ψ_U> ({best_name}) sparse amplitudes:')
pp(state_vec_to_sparse_amp(psi_hat))
