"""
Schnorr's algorithm.
"""

import os
import time
import math
from typing import Tuple, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum, unique

import numpy as np
import sympy
from numba import njit
from fpylll import IntegerMatrix, LLL

from xor_linear_system_solver import XORLinearSystemSolver
from qaoa_search import QAOASearch, QAOAConfig
from brute_force_search import BruteForceSearch


@unique
class SearchType(Enum):
    """The search type to optimize Schnorr's algorithm."""
    QAOA = "QAOA"
    BF = "BF"
    NONE = "NONE"


@dataclass
class SchnorrConfig:
    cap_n: int               # The integer N that will be factorized.
    smooth_b1: int = 5       # The number of primes, e.g it's 4, means the maximum prime is 7
    # since the prime list is [2, 3, 5, 7, 11, ...].
    # The second number of primes, in paper it's `2 * (smooth_b1^2)`, make sure `smooth_b2 >= smooth_b1`.
    smooth_b2: int = 5
    max_iter: int = 1e6      # The maximum number of iterations
    # The number of smooth-pair that will sample, if None, it will be set as 2*smooth_b2
    n_pair: int = None
    base: int = 10           # The base used in the last line of CVP matrix.
    pwr: float = None        # The power used in the last line of CVP matrix.
    # Random sample `pwr` from `pwr_range` when `pwr` is None.
    pwr_range: Tuple = (0.5, 6.0)
    # Search type that optimizes the root of babai's algorithm.
    search_type: SearchType = SearchType.NONE
    # Only valid when `search_type` is `SearchType.QAOA`.
    qaoa_config: QAOAConfig = None


def get_primes(smooth_b: int) -> List:
    """Get the first `smooth_b` primes."""
    return np.array([sympy.prime(i) for i in range(1, smooth_b + 1)])


def construct_lattice(cap_n: int, n_basis: int, pwr: float, base=Optional[None], primes_b1=None):
    """Construct lattice initial basis and target vector.

    Args:
        cap_n (int): The big integer number that will be factorized.
        n_basis (int): Number of basis.
        pwr (float | int): The power of base.
        base (None | int): The base number.

    Return:
        lat_basis (np.ndarray): shape=(n+1, n). The lattice basis. Each column is a base vector.
        t (np.ndarray): shape=(n+1,1). The target vector.
    """
    if not base:
        base = cap_n
    eps = 1e-2
    modify_nums = np.array([i for i in range(1, n_basis+1)]) + eps
    def f(x): return np.round(x / 2)
    diag_values = f(np.random.permutation(modify_nums))
    lat_basis = np.vstack([np.diag(diag_values),
                           (base**pwr * np.log(primes_b1)).reshape((1, -1))])
    lat_basis = np.round(lat_basis)
    t = np.zeros((n_basis+1, 1))
    t[n_basis, 0] = base**pwr * np.log(cap_n)
    t = np.round(t)
    return lat_basis, t


@njit
def gram_schmidt(lat_basis):
    """Gram-Schmidt Orthogonalization for input basis.

    Args:
        lat_basis (np.ndarray): Input basis, each column is a base.

    Return:
        mat_c (np.ndarray): The orthogonalized basis after gram-schmidt orthogonalization.
    """
    mat_c = lat_basis.copy()
    n = mat_c.shape[1]
    for i in range(1, n):
        ci = mat_c[:, i:i+1]
        for j in range(i):
            cj = mat_c[:, j:j+1]
            mu = np.dot(ci.ravel(), cj.ravel()) / \
                (np.dot(cj.ravel(), cj.ravel()) + 1e-8)
            mat_c[:, i:i+1] -= mu * cj
    return mat_c


@njit
def lll_reduction(lat_basis, delta=0.75):
    """LLL-Reduction for lat_basis.

    Args:
        lat_basis: The lattice basis.

    Return:
        lll_basis: The LLL-reduction basis.
    """
    lll_basis = lat_basis
    n = lat_basis.shape[1]
    while True:
        mat_c = gram_schmidt(lat_basis)
        for i in range(1, n):
            bi = lat_basis[:, i]
            for j in range(i-1, -1, -1):
                cj = mat_c[:, j]
                mu_ij = np.dot(bi.flatten(), cj.flatten()) / \
                    np.dot(cj.flatten(), cj.flatten())
                lat_basis[:, i] -= np.round(mu_ij) * lat_basis[:, j]
        stop = True
        for i in range(n-1):
            mat_ci = mat_c[:, i].flatten()
            mu = np.dot(lat_basis[:, i+1].flatten(),
                        mat_ci) / np.dot(mat_ci, mat_ci)
            tmp = (mu * mat_c[:, i] + mat_c[:, i+1]).flatten()
            if delta * np.dot(mat_ci, mat_ci) > np.dot(tmp, tmp):
                col = lat_basis[:, i].copy()
                lat_basis[:, i] = lat_basis[:, i+1]
                lat_basis[:, i+1] = col
                stop = False
                break
        if stop:
            break
    return lll_basis


def lll_reduction_fpylll(lat_basis, delta=0.75):
    """LLL-reduction realized by `fpylll` library [https://github.com/fplll/fpylll], which is about 10x faster.
    The LLL-reduction is a time-consuming part in Schnorr's algorithm.

    Args:
        lat_basis: The lattice basis.

    Return:
        lll_basis: The LLL-reduction basis.
    """

    mat_b = IntegerMatrix.from_matrix(
        np.round(lat_basis.T).astype(int).tolist())
    mat_b = LLL.reduction(mat_b, delta=delta)
    m, n = mat_b.nrows, mat_b.ncols
    lll_basis = np.zeros((m, n), dtype=int)
    mat_b.to_matrix(lll_basis)
    lll_basis = lll_basis.astype(np.float64).T
    return lll_basis


def babai_algorithm_extension(lat_basis, t, delta=0.75, search_type=SearchType.NONE, qaoa_config=None):
    """The Babai-algorithm.

    Args:
        lat_basis (np.ndarray): shape=(m, n), the initial basis.
        t (np.ndarray): shape=(m, 1), the target vector.
        delta (float): The const coefficient in algorithm.

    Return:
        bop (np.ndarray): shape=(m, 1), the closest vector.
        symbol (np.ndarray): shape=(n), each element is in {-1, 1}, is 1 when
            `\mu_j > round(\mu_j)` else -1.
    """
    lat_basis = lat_basis.copy()
    b = t.copy()

    n = lat_basis.shape[1]
    # lll_basis = lll_reduction(lat_basis, delta)
    lll_basis = lll_reduction_fpylll(lat_basis, delta)

    ort_basis = gram_schmidt(lll_basis)
    symbol = np.zeros(n)
    for j in range(n-1, -1, -1):
        d = lll_basis[:, j:j+1]
        g = ort_basis[:, j:j+1]
        muj = np.dot(b.ravel(), g.ravel()) / np.dot(g.ravel(), g.ravel())
        cj = np.round(muj)
        symbol[j] = 1 if muj > cj else -1
        b -= cj * d
    bop = t - b
    # Using QAOA or Brute-Force to optimize the closest vector is not a part
    # of Babai's algorithm, we extend it here to make code clean.
    diff = -b
    if search_type == SearchType.QAOA:
        assert qaoa_config is not None, "The QAOA method need QAOA config, but get None."
        qaoa = QAOASearch(qaoa_config)
        bop = qaoa(lll_basis, diff, symbol, bop)
        pass
    elif search_type == SearchType.BF:
        bf = BruteForceSearch()
        bop = bf(lll_basis.copy(), diff, bop)
    return bop


def get_smooth_factor(d: int, primes_b2: List[int]) -> Tuple[bool, List[int]]:
    """Factorize a digit and check if it's smooth.

    Args:
        d: The digit that will be factorized.

    Return:
        is_smooth (bool): If the `d` is smooth.
        result (list): If `is_smooth == False`, the `result` is invalid. If `is_smooth == True`,
            the result gives the power of each prime (include -1 at first) for `d`.
    """
    n_smooth = len(primes_b2)
    if d == 0 or n_smooth < 1:
        return False, []
    # new_primes = [-1, primes[0], primes[2], ...], use result[0] to show if it's less than 0.
    new_primes = [-1] + list(primes_b2)
    n_smooth += 1
    result = [0] * n_smooth

    if d < 0:
        d = -d
        result[0] = 1
    else:
        result[0] = 0    # (-1)^0 = 1

    idx = 1
    is_smooth = True
    while True:
        if d % new_primes[idx] == 0:
            result[idx] += 1
            d //= new_primes[idx]
        else:
            idx += 1
        if d == 1:
            break
        elif idx >= n_smooth:
            is_smooth = False
            break
    return is_smooth, result


def get_smooth_pair(cap_n: int, lat_basis, bop, primes_b1: List[int], primes_b2: List[int], int_dtype=np.int64):
    """Get smooth pair by given lattice basis and closest vector.

    Args:
        cap_n: The integer N.
        lat_basis (np.ndarray): shape=(n+1, n). Lattice basis.
        bop (np.ndarray): shape=(n, 1). The closest vector.

    Return:
        is_smooth (bool): If the result is a smooth-pair.
        x_e (list): Only valid when `is_smooth` is True. The power of first factor.
        y_e (list): Only valid when `is_smooth` is True. The power of second factor.
    """
    # `lat_basis @ x == bop`, here we need to get the vector `x` which is the coordinate
    # of `bop` under `lat_basis` basis.
    n = lat_basis.shape[1]
    x = np.linalg.inv(lat_basis.T @ lat_basis) @ lat_basis.T @ bop
    x = np.round(x).ravel().astype(int_dtype)

    # validate
    lat_basis = np.round(lat_basis).astype(int_dtype)
    prod = (lat_basis @ x).ravel()
    bop = np.round(bop).ravel().astype(int_dtype)
    succeed = (prod == bop).all()
    if not succeed:
        print("Warn: construct smooth pair failed.")

    u_e = [max(e, 0) for e in x]
    v_e = [-min(e, 0) for e in x]

    u = v = 1
    for i in range(n):
        u *= (int(primes_b1[i])**int(u_e[i]))
        v *= (int(primes_b2[i])**int(v_e[i]))
    if u == 0:
        print("Good, cap_n is factorized.")
    x = u
    y = u - v * cap_n
    x_e = [0] + u_e + [0] * (len(primes_b2) - len(primes_b1))
    # check if `u - v*N` is smooth.
    is_smooth, y_e = get_smooth_factor(y, primes_b2=primes_b2)
    # print(x_e, "*", y_e, "&", len(x_e), len(y_e))
    return is_smooth, (x_e, y_e)


def get_u_v(u_e: List[int], v_e: List[int], primes_b2: List[int]):
    """We express a factor result with two ways, the first one is using the power of each prime,
    the second one is give the factor, which is a integer. `get_u_v` will transform the first way
    to the second way.
    e.g. if u_e = [1,2,0,3], primes=[2,3,5,7], then u=2**1 * 3**2 * 5**0 * 7**3, where ** means pow.

    Args:
        u_e: The power of `u`.
        v_e: The power of `v`.

    Return:
        u: The integer corresponding to `u_e`.
        v: The integer corresponding to `v_e`.
    """
    assert len(u_e) == len(v_e), "`u_e` and `v_e` should have the same length."
    # Sometime the input parameters may be np.ndarray, convert to python list.
    u_e = list(u_e)
    v_e = list(v_e)
    primes = list(primes_b2)

    u = v = 1
    for i in range(len(u_e)):
        u *= primes[i]**u_e[i]
        v *= primes[i]**v_e[i]
    return u, v


def smooth_pair_sampling(config: SchnorrConfig, verbose=1, log_step=500):
    """Smooth pair sampling. It's the most time consuming part.

    Args:
        config: Sample config.
        verbose: If it > 0, there are some log output.
    """
    global primes_b1, primes_b2
    n_pair = config.n_pair
    if config.n_pair is None:
        n_pair = 2 * config.smooth_b2 + 2

    primes_b1 = get_primes(config.smooth_b1)
    primes_b2 = get_primes(config.smooth_b2)
    e_pos = []
    e_neg = []

    cnt_pair = 0
    cnt_iter = 0
    max_iter = config.max_iter

    # counter for repeated result.
    cnt_repeat = 0
    t1 = time.time()
    while (cnt_pair < n_pair and cnt_iter < max_iter):
        pwr = config.pwr
        if pwr is None:
            pwr = np.random.uniform(*config.pwr_range)
        lat_basis, t = construct_lattice(config.cap_n,
                                         config.smooth_b1,
                                         pwr,
                                         config.base,
                                         primes_b1=primes_b1)

        bop = babai_algorithm_extension(lat_basis, t, search_type=config.search_type,
                                        qaoa_config=config.qaoa_config)
        is_smooth, pair = get_smooth_pair(
            config.cap_n, lat_basis, bop, primes_b1=primes_b1, primes_b2=primes_b2)

        cnt_iter += 1
        if is_smooth and (pair[0] in e_pos) and (pair[1] in e_neg):
            cnt_repeat += 1
        if is_smooth and (
                (pair[0] not in e_pos) or (pair[1] not in e_neg)):
            cnt_pair += 1
            e_pos.append(pair[0])
            e_neg.append(pair[1])
        if cnt_iter >= max_iter:
            print(
                f"Warn: only find {cnt_pair}/{n_pair} pairs with {cnt_iter} iterations.")
            break
        if verbose:
            if cnt_iter % log_step == 0:
                t2 = time.time()
                print(
                    f"Iter [{cnt_iter}]: cnt_pair/n_pair = {cnt_pair}/{n_pair}. Number of repeated pairs = {cnt_repeat}. Spend: {t2 - t1:.3f} seconds.")
                t1 = t2
        if cnt_pair >= n_pair:
            if verbose:
                print(
                    f"Successfully find {cnt_pair} smooth pair with {cnt_iter} iterations. Number of repeated pairs = {cnt_repeat}.")
            break

    e_mat = np.vstack([np.array(e_pos).T, np.array(e_neg).T])
    return e_mat


def reconstruct_factors(e_mat, root, cap_n, new_primes):
    """Find the factor of `cap_n(N)` by the root of modular linear system. The `new_primes` should be contains -1,
    which means `new_primes` is [-1, 2, 3, 5, ...].
    """
    x = y = 1
    row = e_mat.shape[0]
    out = e_mat @ root
    # The first half rows is the power of first
    outp = out[:row//2]
    outm = out[row//2:]
    for i in range(len(new_primes)):
        # calculate the power of each prime.
        for _ in range(outp[i] // 2):
            x *= int(new_primes[i])
            x %= cap_n
        for _ in range(outm[i] // 2):
            y *= int(new_primes[i])
            y %= cap_n

    p = math.gcd(x + y, cap_n)
    q = cap_n // p
    return p, q


def run_factorize(config: SchnorrConfig, verbose=0, outdir="output/", log_step=500):
    """Run factorization.

    Args:
        config: Config for Schnorr's algorithm.
        verbose: If `verbose > 0` means debug mode and output log.
    """
    print(
        f"Factorize N = {config.cap_n} begin. It has classically {len(bin(config.cap_n)) - 2} bits.")
    t1 = time.time()
    primes_b2 = get_primes(config.smooth_b2)

    mat_e = smooth_pair_sampling(config, verbose, log_step=log_step)
    mat_01 = np.mod(mat_e, 2)
    # The all equations equal to zero.
    mat_01 = np.c_[mat_01, np.zeros((mat_01.shape[0], 1), dtype=mat_01.dtype)]
    solver = XORLinearSystemSolver()
    roots = solver(mat_01)
    succeed = False
    p = 1
    q = config.cap_n
    new_primes = [-1]+list(primes_b2)
    for root in roots:
        p, q = reconstruct_factors(mat_e, root, config.cap_n, new_primes)
        if p not in [1, config.cap_n] and p * q == config.cap_n:
            print(f"Successfully factorize cap_n({config.cap_n}) = {p} * {q}")
            succeed = True
            break
    if not succeed:
        print(f"Warn: Factorize failed!")
    t2 = time.time()
    dt = t2 - t1
    if dt > 60:
        print(f"Total spend {dt/60:.3f} minutes.")
    else:
        print(f"Total spend {dt:.3f} seconds.")
    if outdir:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        filename = f'{outdir}/fac_{config.cap_n}_B1_{config.smooth_b1}_B2_{config.smooth_b2} '\
                   f'_{config.search_type.name}.txt'
        with open(filename, 'w') as f:
            print(f"config:\n{asdict(config)}", file=f)
            print("XOR linear system is:", file=f)
            print(mat_e.tolist(), file=f)
            print(f"the roots are:", file=f)
            print(roots, file=f)
            print(f"succeed={succeed}  p={p}  q={q}", file=f)
            print(f"Total spend {dt/60:.3f} minutes.", file=f)
            if succeed:
                exponent = list(mat_e @ np.array(root))
                nn = len(exponent)
                print(f"primes:                 {new_primes}", file=f)
                print(f"succeed root for p:     {root[:nn//2]}", file=f)
                print(f"succeed exponent for p: {exponent[:nn//2]}", file=f)
                print(f"succeed root for q:     {root[nn//2:]}", file=f)
                print(f"succeed exponent for q: {exponent[nn//2:]}", file=f)
    return succeed, (p, q)


if __name__ == "__main__":
    np.random.seed(0)
    p = sympy.prime(80)
    q = sympy.prime(60)
    # cap_n = p * q
    cap_n = 48567227
    # cap_n = 1961
    # cap_n = 78742675849
    # cap_n = 261980999226229
    # cap_n = 1109434480987307

    smooth_b1 = 13
    smooth_b2 = smooth_b1
    # smooth_b2 = 2 * smooth_b1**2
    primes_b2 = get_primes(smooth_b2)
    print(primes_b2)
    # config = SampleConfig(cap_n, n_prime=n_prime, n_pair=None, max_iter=int(1e8), pwr_range=(0.5, 6))
    qaoa_config = QAOAConfig(n_layer=5, max_iter=3001, verbose=0)
    sample_config = SchnorrConfig(cap_n,
                                  smooth_b1=smooth_b1,
                                  smooth_b2=smooth_b2,
                                  n_pair=2 * smooth_b2+2,
                                  max_iter=int(1e8),
                                  pwr_range=(0.5, 6),
                                  search_type=SearchType.NONE,
                                  #  qaoa_config=qaoa_config,
                                  )

    succeed, (p, q) = run_factorize(sample_config, verbose=1)
    print(f"succeed = {succeed}, p = {p}, q = {q}.")
