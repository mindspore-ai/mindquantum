import json
from pathlib import Path
from typing import List, Tuple, Dict

import pickle as pkl
import numpy as np
from numpy import ndarray

from qaia import QAIA, NMFA, SimCIM, CAC, CFC, SFC, ASB, BSB, DSB, LQA
from qaia import DUSB

BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log'
DU_LM_SB_weights = LOG_PATH / 'DU-LM-SB_T=10_lr=0.0001.json'
pReg_LM_SB_weights = LOG_PATH / 'pReg-LM-SB_T=6_lr=0.0001.pkl'
ppReg_LM_SB_weights = LOG_PATH / 'ppReg-LM-SB_T=10_lr=0.01_overfit.pkl'
pppReg_LM_SB_weights = LOG_PATH / 'pppReg-LM-SB_T=10_lr=0.01_overfit.pkl'

#run_cfg = 'DU_LM_SB'
run_cfg = 'DU_LM_SB-approx'

if run_cfg.startswith('DU_LM_SB'):
    with open(DU_LM_SB_weights, 'r', encoding='utf-8') as fh:
        params = json.load(fh)
        deltas: List[float] = params['deltas']
        eta: float = params['eta']
        lmbd: float = params['lmbd']
elif run_cfg == 'pReg_LM_SB':
    with open(pReg_LM_SB_weights, 'rb') as fh:
        params = pkl.load(fh)
        deltas: ndarray = params['deltas']
        eta: float = params['eta']
        lmbd: float = params['lmbd']
        lmbd_res: Dict[int, ndarray] = params['lmbd_res']
    lmbd_res = {k: v @ v.T for k, v in lmbd_res.items()}    # precompute
elif run_cfg == 'ppReg_LM_SB':
    with open(ppReg_LM_SB_weights, 'rb') as fh:
        params = pkl.load(fh)
        deltas: ndarray = params['deltas']
        eta: float = params['eta']
        lmbd: float = params['lmbd']
        lmbd_res: Dict[int, ndarray] = params['lmbd_res']
elif run_cfg == 'pppReg_LM_SB':
    with open(pppReg_LM_SB_weights, 'rb') as fh:
        params = pkl.load(fh)
        deltas: ndarray = params['deltas']
        eta: float = params['eta']
        lmbd: float = params['lmbd']
        lmbd_res: Dict[int, Dict[int, ndarray]] = params['lmbd_res']


J_h = Tuple[ndarray, ndarray]

I_cache: Dict[int, ndarray] = {}
def get_I(N:int) -> ndarray:
    key = N
    if key not in I_cache:
        I_cache[key] = np.eye(N)
    return I_cache[key]

ones_cache: Dict[int, ndarray] = {}
def get_ones(N:int) -> ndarray:
    key = N
    if key not in ones_cache:
        ones_cache[key] = np.ones((N, 1))
    return ones_cache[key]

T_cache: Dict[Tuple[int, int], ndarray] = {}
def get_T(N:int, rb:int) -> ndarray:
    key = (N, rb)
    if key not in T_cache:
        # Eq. 7 the transform matrix T
        I = get_I(N)
        # [rb, N, N]
        T = (2**(rb - 1 - np.arange(rb)))[:, np.newaxis, np.newaxis] * I[np.newaxis, ...]
        # [rb*N, N] => [N, rb*N]
        T = T.reshape(-1, N).T
        T_cache[key] = T
    return T_cache[key]


def np_linagl_inv_hijack(a):
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)

    signature = 'D->D' if isComplexType(t) else 'd->d'
    extobj = get_linalg_error_extobj(_raise_linalgerror_singular)
    ainv = _umath_linalg.inv(a, signature=signature, extobj=extobj)
    return wrap(ainv.astype(result_t, copy=False))


def to_ising(H:ndarray, y:ndarray, nbps:int) -> J_h:
    '''
    Reduce MIMO detection problem into Ising problem.

    Reference
    ---------
    [1] Singh A K, Jamieson K, McMahon P L, et al. Ising machines’ dynamics and regularization for near-optimal mimo detection. 
        IEEE Transactions on Wireless Communications, 2022, 21(12): 11080-11094.
    [2] Ising Machines’ Dynamics and Regularization for Near-Optimal Large and Massive MIMO Detection. arXiv: 2105.10535v3

    Input
    -----
    H: [Nr, Nt], np.complex
        Channel matrix
    y: [Nr, 1], np.complex
        Received signal
    num_bits_per_symbol: int
        The number of bits per constellation symbol, e.g., 4 for QAM16.

    Output
    ------
    J: [rb*2*Nt, rb*2*Nt], np.float
        The coupling matrix of Ising problem
    h: [rb*2*Nt, 1], np.float
        The external field
    '''

    # the size of constellation, the M-QAM where M in {16, 64, 256}
    M = 2**nbps
    # n_elem at TX side (c=2 for real/imag, 1 symbol = 2 elem)
    Nr, Nt = H.shape
    N = 2 * Nt
    # n_bits/n_spins that one elem decodes to
    rb = nbps // 2

    # QAM variance for normalization
    # ref: https://dsplog.com/2007/09/23/scaling-factor-in-qam/
    #qam_var: float = 1 / (2**(rb - 2)) * np.sum(np.linspace(1, 2**rb - 1, 2**(rb - 1))**2)
    qam_var = 2 * (M - 1) / 3

    # Eq. 7 the transform matrix T
    I = np.eye(N)
    # [rb, N, N]
    T = (2**(rb - 1 - np.arange(rb)))[:, np.newaxis, np.newaxis] * I[np.newaxis, ...]
    # [rb*N, N] => [N, rb*N]
    T = T.reshape(-1, N).T

    # Eq. 4 and 5
    H_tilde = np.vstack([
        np.hstack([H.real, -H.imag]), 
        np.hstack([H.imag,  H.real]),
    ])
    y_tilde = np.concatenate([y.real, y.imag])

    # Eq. 8, J is symmetric with diag=0, J[i,j] signifies spin interaction of σi and σj in the Ising model
    # This is different from the original paper because we use normalized transmitted symbol
    # J = -ZeroDiag(T.T * H.T * H * T))
    J = T.T @ H_tilde.T @ H_tilde @ T * (-2 / qam_var)
    J[np.diag_indices_from(J)] = 0
    # h = 2 * H * T.T * H.T * (y - H * T * 1 + (sqrt(M) - 1) * H * 1)
    z = y_tilde / np.sqrt(qam_var) - H_tilde @ T @ np.ones((N * rb, 1)) / qam_var + (np.sqrt(M) - 1) * H_tilde @ np.ones((N, 1)) / qam_var
    h = 2 * z.T @ H_tilde @ T

    # [rb*N, rb*N], [rb*N, 1]
    return J, h.T

def to_ising_ext(H:ndarray, y:ndarray, nbps:int, lmbd:float=25, lmbd_res:ndarray=None, lmbd_mode:str='inv', lmbd_res_mode:str='res') -> J_h:
    # the size of constellation, the M-QAM where M in {16, 64, 256}
    M = 2**nbps
    # n_elem at TX side (c=2 for real/imag, 1 symbol = 2 elem)
    Nr, Nt = H.shape
    N = 2 * Nt
    # n_bits/n_spins that one elem decodes to
    rb = nbps // 2
    # QAM variance for normalization
    qam_var = 2 * (M - 1) / 3

    # Eq. 7 the transform matrix T
    I = get_I(N)
    T = get_T(N, rb)

    # Eq. 1
    H_tilde = np.empty([N, N], dtype=np.float32)
    H_tilde[:Nt, :Nt] = H.real
    H_tilde[:Nt, Nt:] = -H.imag
    H_tilde[Nt:, :Nt] = H.imag
    H_tilde[Nt:, Nt:] = H.real
    y_tilde = np.concatenate([y.real, y.imag])

    # Eq. 10
    if lmbd_res is None:                # DU
        # LM-SB from arXiv:2306.16264, the LMMSE-like part with our divisor fix :)
        if lmbd_mode == 'inv':
            U_λ = np.linalg.inv(H_tilde @ H_tilde.T + lmbd * I)
        elif lmbd_mode == 'approx':
            # https://en.wikipedia.org/wiki/Neumann_series
            #   T^-1 = Σk (I - T)^k
            # hence we have
            #   inv(λI + A) = inv(λ(I + A/λ))
            #               ~ inv(I + A/λ) / λ
            #               ~ inv(I + A|A|)    # λ make no difference experimentally, wtf?
            #               ~ Σk (A/|A|)^k
            A = H_tilde @ H_tilde.T #/ lmbd
            A /= np.linalg.norm(A)
            if not 'the theoretical way':
                it = I - A
                U_λ = it
                for _ in range(24):
                    U_λ = it @ (I + U_λ)    # 华罗庚公式
            else:                           # wtf, it just works?!
                U_λ = I - A
                for _ in range(5):          # NOTE: cherry-picked magic number!!
                    U_λ = np.matmul(U_λ, I + U_λ, out=U_λ)  # inplace!
    else:       # method on this branch is not that good :(
        if lmbd_res_mode == 'res':      # pReg
            # learnable identity-residual LMMSE-like part
            # NOTE: here `lmbd_res` should be the precomputed symmetric, this is not the same as in training!!
            U_λ = np.linalg.inv(H_tilde @ H_tilde.T + lmbd_res) / lmbd
        elif lmbd_res_mode == 'proj':   # ppReg
            # fully learnable projection space
            U_λ = lmbd_res

    # J = -ZeroDiag(T.T * H.T * H * T))
    H_tilde_T = H_tilde @ T
    J = H_tilde_T.T @ (U_λ * (-2 / qam_var)) @ H_tilde_T
    for j in range(J.shape[0]): J[j, j] = 0
    # h = 2 * H * T.T * H.T * (y - H * T * 1 + (sqrt(M) - 1) * H * 1)
    # NOTE: y_tilde should devide by `qam_var`, but `sqrt(qam_var)` gives the same outputs (wtf?)
    z = y_tilde - H_tilde @ T.sum(axis=-1, keepdims=True) + (np.sqrt(M) - 1) * H_tilde.sum(axis=-1, keepdims=True)
    h = H_tilde_T.T @ (U_λ @ (2 / np.sqrt(qam_var) * z))

    # [rb*N, rb*N], [rb*N, 1]
    return J, h

def to_ising_MDI_MIMO(H:ndarray, y:ndarray, nbps:int) -> J_h:
    ''' MDI-MIMO from [2304.12830] Uplink MIMO Detection using Ising Machines: A Multi-Stage Ising Approach '''


def solver_qaia_lib(qaia_cls, J:ndarray, h:ndarray) -> ndarray:
    bs = 1
    solver: QAIA = qaia_cls(J, h, batch_size=bs, n_iter=10)
    solver.update()                     # [rb*N, B]
    if bs > 1:
        energy = solver.calc_energy()   # [1, B]
        opt_index = np.argmin(energy)
    else:
        opt_index = 0
    solution = np.sign(solver.x[:, opt_index])  # [rb*N], vset {-1, 1}
    return solution

def solver_DU_LM_SB(J:ndarray, h:ndarray) -> ndarray:
    global deltas, eta
    bs = 1
    solver = DUSB(J, h, deltas, eta, batch_size=bs)
    solver.update()                     # [rb*N, B]
    if bs > 1:
        energy = solver.calc_energy()   # [1, B]
        opt_index = np.argmin(energy)
    else:
        opt_index = 0
    solution = np.sign(solver.x[:, opt_index])  # [rb*N], vset {-1, 1}
    return solution


# 选手提供的Ising模型生成函数，可以用我们提供的to_ising
def ising_generator(H:ndarray, y:ndarray, nbps:int, snr:float) -> J_h:
    if run_cfg == 'baseline':
        return to_ising(H, y, nbps)
    elif run_cfg == 'LM_SB':
        return to_ising_ext(H, y, nbps, lmbd=25)
    elif run_cfg == 'DU_LM_SB':
        return to_ising_ext(H, y, nbps, lmbd=lmbd)
    elif run_cfg == 'DU_LM_SB-approx':
        return to_ising_ext(H, y, nbps, lmbd=lmbd, lmbd_mode='approx')
    elif run_cfg == 'pReg_LM_SB':
        return to_ising_ext(H, y, nbps, lmbd=lmbd, lmbd_res=lmbd_res[H.shape[0]])
    elif run_cfg == 'ppReg_LM_SB':
        return to_ising_ext(H, y, nbps, lmbd=lmbd, lmbd_res=lmbd_res[H.shape[0]], lmbd_res_mode='proj')
    elif run_cfg == 'pppReg_LM_SB':
        return to_ising_ext(H, y, nbps, lmbd=lmbd, lmbd_res=lmbd_res[snr][H.shape[0]], lmbd_res_mode='proj')

# 选手提供的qaia MLD求解器，用mindquantum.algorithms.qaia
def qaia_mld_solver(J:ndarray, h:ndarray) -> ndarray:
    if run_cfg == 'baseline':
        return solver_qaia_lib(BSB, J, h)
    elif run_cfg == 'LM_SB':
        return solver_qaia_lib(BSB, J, h)
    elif run_cfg in ['DU_LM_SB', 'DU_LM_SB-approx', 'pReg_LM_SB', 'ppReg_LM_SB', 'pppReg_LM_SB']:
        return solver_DU_LM_SB(J, h)
