# pylint: disable=unused-argument, unused-variable, invalid-name, consider-using-enumerate, unnecessary-list-index-lookup, wrong-import-position, broad-exception-caught, use-dict-literal, no-else-return, superfluous-parens
"""

GF(2) 上的信念传播解码器模块。

Belief Propagation decoders over GF(2) for DQI.

本模块提供三种解码器，用于求解 H · e = syndrome (mod 2)：
  1. bp1_decode: 硬判决 Gallager-A/B 风格 BP
     - 每次迭代翻转所有连接的校验节点都不满足的变量节点
     - 适用于 LDPC 码的快速解码
  2. bp2_decode: 软判决 Sum-Product BP (LDPC 风格)
     - 使用 tanh-based 校验节点更新
     - 通常比 BP1 有更好的纠错能力
  3. gauss_jordan_solve: 精确代数解码器 (高斯消元)
     - 提供最大似然解码基线
     - 用于验证 BP 解码器的正确性

辅助函数：
  - bp2_greedy_flip: BP2 后处理的贪心比特翻转
  - compute_y_in_decodable_set: 找出所有可解码的错误模式 D_k（DQI 期望计算的核心）
"""

import numpy as np


def bp1_decode(parity_check, syndrome, max_iter=50, threshold_mode="unanimous"):
    """
    BP1 (Gallager-A/B style) hard-decision decoder over GF(2).

    Solves H * e = syndrome (mod 2) where H is an n x m parity-check matrix.
    Starts from e=0 and iteratively flips variable nodes to match the target syndrome.

    Args:
        parity_check: ndarray of shape (n, m), parity-check matrix.
        syndrome: ndarray of shape (n,), target syndrome vector.
        max_iter: maximum number of BP iterations.
        threshold_mode: 'unanimous' or 'majority'.

    Returns:
        e_hat: ndarray of shape (m,), decoded error vector.
        success: bool, whether the syndrome is satisfied.
    """
    _, m_vars = parity_check.shape
    e_hat = np.zeros(m_vars, dtype=int)
    connected = [np.where(parity_check[:, j] == 1)[0] for j in range(m_vars)]

    for _ in range(max_iter):
        residual = (parity_check.dot(e_hat) + syndrome) % 2
        flip_candidates = np.zeros(m_vars, dtype=int)
        for j in range(m_vars):
            cnodes = connected[j]
            if len(cnodes) == 0:
                continue
            unsatisfied = np.sum(residual[cnodes])
            if threshold_mode == "unanimous":
                threshold = len(cnodes)
            else:
                threshold = (len(cnodes) + 1) // 2
            if unsatisfied >= threshold:
                flip_candidates[j] = 1
        if np.sum(flip_candidates) == 0:
            break
        e_hat = (e_hat + flip_candidates) % 2
        if np.array_equal(parity_check.dot(e_hat) % 2, syndrome):
            return e_hat, True

    return e_hat, np.array_equal(parity_check.dot(e_hat) % 2, syndrome)


def bp1_decode_to_zero(parity_check, y, max_iter=5):
    """
    BP1 decoder: start from error pattern y, iteratively try to decode to zero.

    与参考实现 dqi/pipelines/belief_propagations.py 中的
    belief_propagation_gallager 行为完全一致。

    这是 DQI 量子电路中 BP1 解码器的经典模拟：
      1. 从 bits = y（错误模式）开始
      2. 每次迭代计算 syndrome = (parity_check @ bits) % 2
      3. 当某变量节点的所有连接校验节点都不满足时，翻转该变量
      4. 成功条件：最终 bits 变为全零

    Args:
        parity_check: ndarray of shape (n, m), parity-check matrix (B^T).
        y: ndarray of shape (m,), error pattern to decode.
        max_iter: maximum number of BP iterations.

    Returns:
        bits: ndarray of shape (m,), final decoded vector.
        success: bool, whether decoded to all zeros.
    """
    bits = y.copy()
    num_vnodes = parity_check.shape[1]
    connected = [np.where(parity_check[:, j] == 1)[0] for j in range(num_vnodes)]

    for _ in range(max_iter):
        syndrome = (parity_check @ bits) % 2
        flip_candidates = np.zeros_like(bits)
        for j in range(num_vnodes):
            cnodes = connected[j]
            if len(cnodes) == 0:
                continue
            unsatisfied = np.sum(syndrome[cnodes])
            if unsatisfied >= len(cnodes):
                flip_candidates[j] = 1
        bits = (bits + flip_candidates) % 2

    success = np.all(bits == 0)
    return bits, success


def bp2_decode(parity_check, syndrome, max_iter=50, p=0.001):
    """
    BP2 (Sum-Product) decoder over GF(2).

    Uses tanh-based check node updates typical of LDPC decoding.

    Args:
        parity_check: ndarray of shape (n, m), parity-check matrix.
        syndrome: ndarray of shape (n,), target syndrome vector.
        max_iter: maximum number of BP iterations.
        p: crossover probability for the BSC channel model.

    Returns:
        e_hat: ndarray of shape (m,), decoded error vector.
        success: bool, whether the syndrome is satisfied.
    """
    n_checks, m_vars = parity_check.shape
    # Initialize LLRs from channel observation: log((1-p)/p)
    channel_llr = np.log((1 - p) / p)
    # Initial messages from variable nodes to check nodes
    log_q = np.full((n_checks, m_vars), channel_llr) * parity_check

    for _ in range(max_iter):
        # Check node update (tanh sum-product)
        log_r = np.zeros((n_checks, m_vars))
        for i in range(n_checks):
            var_nodes = np.where(parity_check[i, :] == 1)[0]
            if len(var_nodes) < 2:
                continue
            for j in var_nodes:
                other = [v for v in var_nodes if v != j]
                # Compute product of tanh(log_q/2) from other nodes, with syndrome sign
                prod = (-1) ** syndrome[i]
                for v in other:
                    prod *= np.tanh(log_q[i, v] / 2.0)
                # Clip to avoid numerical issues
                prod = np.clip(prod, -1 + 1e-15, 1 - 1e-15)
                log_r[i, j] = 2.0 * np.arctanh(prod)

        # Variable node update
        log_q_new = np.zeros((n_checks, m_vars))
        log_total = np.zeros(m_vars)
        for j in range(m_vars):
            check_nodes = np.where(parity_check[:, j] == 1)[0]
            total_llr = channel_llr
            for i in check_nodes:
                total_llr += log_r[i, j]
            log_total[j] = total_llr
            for i in check_nodes:
                log_q_new[i, j] = total_llr - log_r[i, j]
        log_q = log_q_new

        e_hat = (log_total < 0).astype(int)
        if np.array_equal(parity_check.dot(e_hat) % 2, syndrome):
            return e_hat, True

    return e_hat, False


def bp2_greedy_flip(parity_check, syndrome, e_init, max_flips=3):
    """
    BP2 post-processing: greedy bit-flipping decoder.

    Attempts to correct residual errors by flipping bits that resolve
    the most unsatisfied parity checks.

    Args:
        parity_check: ndarray of shape (n, m), parity-check matrix.
        syndrome: ndarray of shape (n,), target syndrome vector.
        e_init: ndarray of shape (m,), initial error estimate.
        max_flips: maximum number of bit flips.

    Returns:
        e_hat: ndarray of shape (m,), corrected error vector.
        success: bool, whether the syndrome is satisfied.
    """
    e_hat = e_init.copy()
    current_syndrome = parity_check.dot(e_hat) % 2
    unmatched = current_syndrome != syndrome

    for _ in range(max_flips):
        if np.sum(unmatched) == 0:
            return e_hat, True

        best_flip = -1
        max_resolved = 0
        n_satisfied = np.sum(current_syndrome == syndrome)

        for j in range(len(e_hat)):
            test_e = e_hat.copy()
            test_e[j] ^= 1
            test_synd = parity_check.dot(test_e) % 2
            resolved = np.sum(test_synd == syndrome)
            if resolved > max_resolved:
                max_resolved = resolved
                best_flip = j

        if best_flip != -1 and max_resolved > n_satisfied:
            e_hat[best_flip] ^= 1
            current_syndrome = parity_check.dot(e_hat) % 2
            unmatched = current_syndrome != syndrome
        else:
            break

    success = np.array_equal(parity_check.dot(e_hat) % 2, syndrome)
    return e_hat, success


def gauss_jordan_solve(parity_check, syndrome):
    """
    Solve H * e = syndrome (mod 2) using Gauss-Jordan elimination.

    Provides an exact (maximum-likelihood) decoder baseline.

    Args:
        parity_check: ndarray of shape (n, m), parity-check matrix.
        syndrome: ndarray of shape (n,), target syndrome vector.

    Returns:
        e_hat: ndarray of shape (m,), decoded error vector (minimum weight
               particular solution if underdetermined).
        success: bool, whether the system is consistent and solved.
    """
    n, m = parity_check.shape
    # Augmented matrix
    aug = np.concatenate([parity_check, syndrome.reshape(-1, 1)], axis=1).astype(int)

    row = 0
    col = 0
    pivot_cols = []
    while row < n and col < m:
        # Find pivot
        pivot = None
        for r in range(row, n):
            if aug[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            col += 1
            continue
        # Swap
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        # Eliminate other rows
        for r in range(n):
            if r != row and aug[r, col] == 1:
                aug[r] ^= aug[row]
        pivot_cols.append(col)
        row += 1
        col += 1

    # Check consistency
    for r in range(row, n):
        if np.array_equal(aug[r, :-1], 0) and aug[r, -1] == 1:
            return np.zeros(m, dtype=int), False

    # Back-substitution: free variables set to 0
    e_hat = np.zeros(m, dtype=int)
    for r, c in enumerate(pivot_cols):
        e_hat[c] = aug[r, -1]

    success = np.array_equal(parity_check.dot(e_hat) % 2, syndrome)
    return e_hat, success


def _y_in_decodable_worker(args):
    """
    Worker function for parallel compute_y_in_decodable_set.

    Processes a chunk of (m, k) combination tuples, decoding each with
    the specified BP decoder. Must be at module level for pickling.

    Args:
        args: tuple of (bit_patterns, pc_data, pc_shape, bp_func_name, threshold_mode, max_iter)

    Returns:
        list of tuples (bit positions) that decoded successfully.
    """
    bit_patterns, pc_data, pc_shape, bp_func_name, threshold_mode, max_iter = args
    parity_check = np.array(pc_data).reshape(pc_shape)

    successful = []
    for bits in bit_patterns:
        y = np.zeros(parity_check.shape[1], dtype=int)
        for pos in bits:
            y[pos] = 1

        if bp_func_name == "bp1_decode":
            _, success = bp1_decode_to_zero(parity_check, y, max_iter=max_iter)
        elif bp_func_name == "bp2_decode":
            syndrome = parity_check.dot(y) % 2
            _, success = bp2_decode(parity_check, syndrome, max_iter=max_iter)
        elif bp_func_name == "gauss_jordan_solve":
            syndrome = parity_check.dot(y) % 2
            _, success = gauss_jordan_solve(parity_check, syndrome)
        else:
            syndrome = parity_check.dot(y) % 2
            _, success = bp1_decode(
                parity_check, syndrome, max_iter=max_iter, threshold_mode=threshold_mode
            )

        if success:
            successful.append(bits)
    return successful



def _decode_single_pattern(parity_check, y, bp_func_name, bp_func, config):
    """Decode a single error pattern y using the specified BP decoder."""
    max_iter = config.get("max_iter", 50)
    threshold_mode = config.get("threshold_mode", "unanimous")
    if bp_func_name == "bp1_decode":
        _, success = bp1_decode_to_zero(parity_check, y, max_iter=max_iter)
    elif bp_func_name == "bp2_decode":
        syndrome = parity_check.dot(y) % 2
        _, success = bp_func(parity_check, syndrome, max_iter=max_iter)
    elif bp_func_name == "gauss_jordan_solve":
        syndrome = parity_check.dot(y) % 2
        _, success = bp_func(parity_check, syndrome)
    else:
        syndrome = parity_check.dot(y) % 2
        _, success = bp_func(
            parity_check, syndrome, max_iter=max_iter, threshold_mode=threshold_mode
        )
    return success


def _process_patterns_sequential(patterns_to_check, m, parity_check, bp_func_name, config):
    """Process patterns sequentially and return valid patterns."""
    bp_func = config.get("bp_func")
    valid_patterns = []
    for bits in patterns_to_check:
        y = np.zeros(m, dtype=int)
        for pos in bits:
            y[pos] = 1
        if _decode_single_pattern(parity_check, y, bp_func_name, bp_func, config):
            valid_patterns.append(y)
    return valid_patterns


def _process_patterns_parallel(patterns_to_check, m, parity_check, bp_func_name, config):
    """Process patterns in parallel and return valid patterns."""
    from multiprocessing import Pool
    n_jobs = config.get("n_jobs", 1)
    threshold_mode = config.get("threshold_mode", "unanimous")
    max_iter = config.get("max_iter", 50)
    chunks = np.array_split(patterns_to_check, n_jobs)
    pc_data = parity_check.tolist()
    pc_shape = parity_check.shape
    args_list = [
        (chunk.tolist() if hasattr(chunk, "tolist") else chunk,
         pc_data, pc_shape, bp_func_name, threshold_mode, max_iter)
        for chunk in chunks
    ]
    with Pool(n_jobs) as pool:
        results = pool.map(_y_in_decodable_worker, args_list)
    valid_patterns = []
    for r in results:
        for bits in r:
            y = np.zeros(m, dtype=int)
            for pos in bits:
                y[pos] = 1
            valid_patterns.append(y)
    return valid_patterns


def compute_y_in_decodable_set(
    constraint_mat,
    ell,
    bp_func,
    max_iter=50,
    config=None,
):
    """
    For each k = 0..ell, find all error patterns y of weight k
    that the BP decoder can successfully decode (reduce to zero).

    与参考实现 dqi/pipelines/DQI_classical.py 中的 compute_y_in_decodable_set 行为一致：
      - 对于 BP1：传递错误向量 y 给 bp1_decode_to_zero，检查是否能清零
      - 对于 BP2/GJ：计算综合征，传递给解码器

    These sets D_k define the error subspaces reachable by DQI.

    Args:
        constraint_mat: ndarray of shape (m, n), max-XORSAT constraint matrix.
        ell: maximum error weight (polynomial degree).
        bp_func: callable, BP decoder function.
        max_iter: max iterations for the BP decoder.
        config: dict with optional keys 'n_jobs', 'max_patterns', 'threshold_mode'.

    Returns:
        y_in_decodable: list of lists, where y_in_decodable[k] contains error patterns
                 (as ndarrays) of weight k that decode successfully.
        epsilon_k: ndarray of shape (ell+1,), decoding failure rate per k.
    """
    if config is None:
        config = {}
    n_jobs = config.get("n_jobs", 1)
    max_patterns = config.get("max_patterns", None)
    threshold_mode = config.get("threshold_mode", "unanimous")
    from scipy.special import comb
    from itertools import combinations

    m = constraint_mat.shape[0]
    parity_check = constraint_mat.T  # (n, m) parity-check matrix
    y_in_decodable = [[] for _ in range(ell + 1)]
    eps_k = np.ones(ell + 1)

    # k=0: always trivially decodes (zero syndrome)
    y_in_decodable[0].append(np.zeros(m, dtype=int))
    eps_k[0] = 0.0

    bp_func_name = bp_func.__name__
    rng = np.random.default_rng(42)

    for k in range(1, ell + 1):
        total = int(comb(m, k, exact=True))

        # Subsample if total exceeds max_patterns — use lazy sampling to avoid OOM
        if max_patterns is not None and total > max_patterns:
            idx = set(rng.choice(total, size=max_patterns, replace=False))
            patterns_to_check = []
            for i, combo in enumerate(combinations(range(m), k)):
                if i in idx:
                    patterns_to_check.append(combo)
                if len(patterns_to_check) == max_patterns:
                    break
            total_checked = max_patterns
        else:
            patterns_to_check = list(combinations(range(m), k))
            total_checked = total

        valid_count = 0

        if n_jobs > 1 and len(patterns_to_check) > 1000:
            valid_patterns = _process_patterns_parallel(
                patterns_to_check, m, parity_check, bp_func_name,
                {"n_jobs": n_jobs, "max_iter": max_iter, "threshold_mode": threshold_mode}
            )
        else:
            valid_patterns = _process_patterns_sequential(
                patterns_to_check, m, parity_check, bp_func_name,
                {"bp_func": bp_func, "max_iter": max_iter, "threshold_mode": threshold_mode}
            )

        y_in_decodable[k].extend(valid_patterns)
        valid_count = len(valid_patterns)

        eps_k[k] = 1.0 - valid_count / total_checked if total_checked > 0 else 1.0

    return y_in_decodable, eps_k
