# pylint: disable=unused-argument, unused-variable, invalid-name, consider-using-enumerate, unnecessary-list-index-lookup, wrong-import-position, broad-exception-caught, use-dict-literal, no-else-return, superfluous-parens
"""

DQI 经典期望值计算模块。

Classical expectation computation for DQI.

本模块实现了 DQI 算法的经典期望值计算，无需运行量子电路。
通过分析哪些错误模式在经典 BP 解码器下可成功解码，计算 A 矩阵形式主义。

核心公式 (Core formulas):
  1. 最优权重 w_k: (ℓ+1)×(ell+1) 三对角矩阵的主特征向量
  2. A 矩阵: A_{k,k'} = (1/√(T_k·T_k')) · Σ_{y∈D_k, y'∈D_k'} Σ_i [δ(B^T y = B^T(y'+e_i)) · (-1)^{v_i+a}]
  3. 期望值: <f> = <w|A|w> / norm², <S> = (<f> + m) / 2

其中：
  - D_k: 权重为 k 且可解码为零综合征的错误模式集合
  - T_k = C(m, k): 权重为 k 的总模式数
  - ε_k: 解码失败率 = 1 - |D_k| / T_k

这是复现 Fig 11 的核心计算。
"""

import math
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.special import comb


def compute_optimal_w(m, ell, p=2, r=1):
    """
    Compute the optimal weight vector w_k for the DQI polynomial.

    This implementation matches the original get_optimal_w from
    dqi/pipelines/dicke_states.py.

    w_k are the components of the principal eigenvector of a sparse
    (ell+1) x (ell+1) tridiagonal matrix A where:
        A_{k,k} = k * d              (d = (p-2r)/sqrt(r(p-r)))
        A_{k,k+1} = A_{k+1,k} = sqrt(k * (m - k + 1))

    For max-XORSAT (p=2, r=1): d = 0, diagonal is all zeros.

    Args:
        m: number of constraints.
        ell: polynomial degree (max error weight).
        p: field size (default 2 for GF(2)).
        r: constraint size parameter (default 1 for XORSAT).

    Returns:
        w: ndarray, optimal weights padded to next power of 2.
    """
    d = (p - 2 * r) / np.sqrt(r * (p - r))
    diag = np.arange(ell + 1) * d
    off = np.sqrt(np.arange(1, ell + 1) * (m - np.arange(1, ell + 1) + 1))

    # Build sparse tridiagonal matrix
    weight_mat = diags([off, diag, off], offsets=(-1, 0, 1), format="csr")

    # Get principal eigenvector
    _, vecs = eigsh(weight_mat, k=1, which="LA")
    w = vecs.flatten()
    w /= np.linalg.norm(w)

    # Pad up to next power of two
    orig = w.size
    target = 1 << math.ceil(math.log2(orig))
    return np.pad(w, (0, target - orig))



def _precompute_syndromes(constraint_mat_t, v, y_in_decodable, ell):
    """Precompute syndromes and phases for all error patterns."""
    y_syn = []
    y_phase = []
    for k in range(ell + 1):
        syn_list = []
        phase_list = []
        for y in y_in_decodable[k]:
            y_arr = np.array(y, dtype=int)
            syn = np.mod(constraint_mat_t @ y_arr, 2)
            syn_int = int("".join(str(int(b)) for b in syn), 2)
            syn_list.append(syn_int)
            a = int(np.mod(np.sum(y_arr * v), 2))
            phase_list.append((-1) ** a)
        y_syn.append(syn_list)
        y_phase.append(phase_list)
    return y_syn, y_phase


def compute_weight_matrix(ell, constraint_mat, v, y_in_decodable):
    """
    Build the (ell+1) x (ell+1) A matrix using vectorized operations.

    A_{k,k'} = (1/sqrt(T_k * T_kp)) * sum_{y in D_k, y' in D_k'}
               sum_i [delta(B^T y = B^T(y'+e_i)) * (-1)^{v_i + a}]

    Uses integer syndrome encoding and precomputed column lookup
    for O(|D_k| * |D_kp|) complexity instead of O(|D_k| * |D_kp| * m).
    """
    m, _ = constraint_mat.shape
    constraint_mat_t = constraint_mat.T

    # Normalisation factors
    if m <= 20:
        norm = np.array([comb(m, k, exact=True) for k in range(ell + 1)], dtype=float)
    else:
        norm = np.array([comb(m, k) for k in range(ell + 1)], dtype=float)
        norm = np.where(np.isinf(norm), 1e10, norm)

    # Precompute column syndrome signs
    col_syn_sign = {}
    for i in range(m):
        col = constraint_mat_t[:, i]
        syn_int = int("".join(str(int(b)) for b in col), 2)
        col_syn_sign[syn_int] = col_syn_sign.get(syn_int, 0) + ((-1) ** int(v[i]))

    y_syn, y_phase = _precompute_syndromes(constraint_mat_t, v, y_in_decodable, ell)

    weight_mat = np.zeros((ell + 1, ell + 1))
    for k in range(ell + 1):
        for kp in range(ell + 1):
            if not y_in_decodable[k] or not y_in_decodable[kp]:
                continue
            acc = 0.0
            for i, sy in enumerate(y_syn[k]):
                for j, syp in enumerate(y_syn[kp]):
                    acc += y_phase[k][i] * y_phase[kp][j] * col_syn_sign.get(sy ^ syp, 0)
            denom = np.sqrt(norm[k] * norm[kp])
            weight_mat[k, kp] = acc / denom if denom > 0 else 0.0

    return weight_mat


def compute_expected_values(_weight_mat, w, m, epsilon_k):
    """
    Compute the expected number of satisfied constraints <S>.

    Uses the formula:
        f_expected = <w|A|w> / norm^2
        s_expected = (f_expected + m) / 2

    where norm^2 = sum_k w_k^2 * (1 - epsilon_k) accounts for
    decoding failures.

    This implementation handles padded weight vectors (power of 2)
    by truncating to match epsilon_k length for norm computation,
    then padding back for matrix multiplication.

    Args:
        A: ndarray of shape (ell+1, ell+1), A matrix.
        w: ndarray (possibly padded to power of 2), optimal weights.
        m: number of constraints.
        epsilon_k: ndarray of shape (ell+1,), decoding failure rates.

    Returns:
        f_expected: expected objective function value <f>.
        s_expected: expected number of satisfied constraints <S>.
    """
    # Remove padding from w to match epsilon_k length
    if len(w) != len(epsilon_k):
        w = w[: len(epsilon_k)]

    norm2 = np.sum(w**2 * (1 - epsilon_k))
    if norm2 <= 0:
        return -m, 0.0

    # Pad w back to match A dimensions for matrix multiplication
    if len(w) != weight_mat.shape[0]:
        w = np.pad(w, (0, weight_mat.shape[0] - len(w)), mode="constant", constant_values=0)

    f_expected = w.dot(weight_mat.dot(w)) / norm2
    s_expected = (f_expected + m) / 2.0
    return f_expected, s_expected


def expected_constraints_dqi(
    constraint_mat,
    v,
    ell,
    bp_func,
    config=None,
):
    """
    Full DQI expected constraint satisfaction computation.

    Orchestrates: compute optimal w -> find valid error patterns D_k ->
    build A matrix -> compute expected values.

    Args:
        constraint_mat: ndarray of shape (m, n), constraint matrix.
        v: ndarray of shape (m,), RHS vector.
        ell: polynomial degree.
        bp_func: callable, BP decoder (parity_check, syndrome) -> (e_hat, success).
        max_iter: max iterations for BP decoder.
        n_jobs: number of parallel workers for compute_y_in_decodable_set.
        max_patterns: optional subsampling limit for error patterns.
        threshold_mode: 'unanimous', 'majority', or 'minority' for bp1_decode.

    Returns:
        result: dict with keys 'f_expected', 's_expected', 's_normalized',
                'w', 'epsilon_k', 'y_in_Dk_sizes'.
    """
    from .bp_decoders import compute_y_in_decodable_set

    if config is None:
        config = {}
    max_iter = config.get("max_iter", 50)
    n_jobs = config.get("n_jobs", 1)
    max_patterns = config.get("max_patterns", None)
    threshold_mode = config.get("threshold_mode", "majority")

    m = constraint_mat.shape[0]
    w = compute_optimal_w(m, ell)
    y_in_decodable, epsilon_k = compute_y_in_decodable_set(
        constraint_mat,
        ell,
        bp_func,
        max_iter,
        config={"n_jobs": n_jobs, "max_patterns": max_patterns, "threshold_mode": threshold_mode},
    )
    weight_mat = compute_weight_matrix(ell, constraint_mat, v, y_in_decodable)
    f_exp, s_exp = compute_expected_values(weight_mat, w, m, epsilon_k)

    return {
        "f_expected": f_exp,
        "s_expected": s_exp,
        "s_normalized": s_exp / m,
        "w": w,
        "epsilon_k": epsilon_k,
        "y_in_Dk_sizes": [len(dk) for dk in y_in_decodable],
        "weight_mat": weight_mat,
    }
