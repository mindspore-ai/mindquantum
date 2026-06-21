# pylint: disable=unused-argument, unused-variable, invalid-name, consider-using-enumerate, unnecessary-list-index-lookup, wrong-import-position, broad-exception-caught, use-dict-literal, no-else-return, superfluous-parens
"""

矩阵生成与降采样模块。

Matrix generation and downscaling for DQI experiments.

本模块提供从参考 B 矩阵生成降采样实例的功能：
  1. extract_degree_distribution: 提取行/列度分布
  2. mcmc_sample: MCMC 2×2 边交换随机化（保持边缘和不变）
  3. generate_downsized_matrices: 生成降采样矩阵集

降采样流程 (Downscaling workflow):
  1. 从原始 ILP 转换的 B 矩阵中提取度分布 δ (列度) 和 κ (行度)
  2. 对每个目标尺寸 (m, n)，从度分布中采样行和/列和
  3. 贪心构造初始矩阵，然后用 MCMC 随机化
  4. 生成随机右端向量 v

预定义尺寸 (Pre-defined sizes):
  DOWNSIZED_SIZES: 28 个标准尺寸 (23×12 ~ 563×282)
  PERFORMANCE_SIZES: 前 7 个尺寸 (23×12 ~ 143×72)，用于 Fig 11 复现
"""

import numpy as np


def extract_degree_distribution(constraint_mat):
    """
    Extract normalized row (constraint) and column (variable) degree distributions.

    Args:
        constraint_mat: ndarray of shape (m, n), binary matrix.

    Returns:
        delta: dict, column degree distribution (fraction of columns with degree d).
        kappa: dict, row degree distribution (fraction of rows with degree d).
    """
    m, n = constraint_mat.shape
    col_degrees = np.sum(constraint_mat, axis=0)
    row_degrees = np.sum(constraint_mat, axis=1)

    delta = {}
    for d in range(int(col_degrees.max()) + 1):
        delta[d] = np.sum(col_degrees == d) / n

    kappa = {}
    for d in range(int(row_degrees.max()) + 1):
        kappa[d] = np.sum(row_degrees == d) / m

    return delta, kappa


def _sample_degrees(distribution, size, rng):
    """Sample integer degrees from a normalized distribution."""
    degs = list(distribution.keys())
    probs = [distribution[d] for d in degs]
    return rng.choice(degs, size=size, p=probs).astype(int)


def _generate_initial_matrix(row_sums, col_sums, rng, max_attempts=10000):
    """
    Greedy construction of a binary matrix with specified row and column sums.

    Uses a sequential filling algorithm with backtracking.

    Args:
        row_sums: ndarray of length m, target row sums.
        col_sums: ndarray of length n, target column sums.
        rng: numpy random number generator.
        max_attempts: maximum construction attempts.

    Returns:
        constraint_mat: ndarray of shape (m, n), binary matrix, or None if construction fails.
    """
    m = len(row_sums)
    n = len(col_sums)

    for _ in range(max_attempts):
        constraint_mat = np.zeros((m, n), dtype=int)
        current_row = row_sums.copy()
        current_col = col_sums.copy()

        # Process columns from largest sum to smallest
        col_order = np.argsort(-current_col)
        success = True

        for j in col_order:
            needed = current_col[j]
            if needed == 0:
                continue
            # Find rows that can still accept a 1
            candidates = [i for i in range(m) if current_row[i] > 0]
            if len(candidates) < needed:
                success = False
                break
            # Choose randomly among candidates
            chosen = rng.choice(candidates, size=int(needed), replace=False)
            for i in chosen:
                constraint_mat[i, j] = 1
                current_row[i] -= 1
            current_col[j] -= needed

        if success and np.all(current_row == 0) and np.all(current_col == 0):
            return constraint_mat

    return None


def _is_valid_swap(constraint_mat_arg, i1, i2, j1, j2):
    """Check if a 2x2 submatrix allows a swap (preserving marginals)."""
    return (constraint_mat_arg[i1, j1] == 1 and constraint_mat_arg[i1, j2] == 0
            and constraint_mat_arg[i2, j1] == 0 and constraint_mat_arg[i2, j2] == 1)


def _perform_swap(constraint_mat_arg, i1, i2, j1, j2):
    """Perform a 2x2 edge swap."""
    constraint_mat_arg[i1, j1] = 0
    constraint_mat_arg[i1, j2] = 1
    constraint_mat_arg[i2, j1] = 1
    constraint_mat_arg[i2, j2] = 0


def mcmc_sample(constraint_mat, iterations=10000, seed=None):
    """
    Randomize a binary matrix while preserving row and column sums
    using MCMC (random 2x2 edge swaps).

    Args:
        constraint_mat: ndarray, input binary matrix.
        iterations: number of MCMC steps.
        seed: random seed for reproducibility.

    Returns:
        rand_mat: ndarray, randomized binary matrix.
    """
    output_mat = constraint_mat.copy()
    m, n = output_mat.shape
    rng = np.random.default_rng(seed)

    for _ in range(iterations):
        i1, i2 = rng.choice(m, size=2, replace=False)
        j1, j2 = rng.choice(n, size=2, replace=False)

        if _is_valid_swap(output_mat, i1, i2, j1, j2):
            _perform_swap(output_mat, i1, i2, j1, j2)
        elif _is_valid_swap(output_mat, i2, i1, j1, j2):
            _perform_swap(output_mat, i2, i1, j1, j2)

    return output_mat


def generate_downsized_matrices(
    delta,
    kappa,
    target_sizes,
    n_samples=5,
    config=None,
):
    """
    Generate downscaled B matrices preserving degree distributions.

    For each target (m, n) size, generates n_samples matrices with
    the specified degree distributions using MCMC randomization.

    Args:
        delta: dict, column degree distribution from reference matrix.
        kappa: dict, row degree distribution from reference matrix.
        target_sizes: list of (m, n) tuples.
        n_samples: number of random instances per size.
        config: dict with optional keys 'mcmc_steps', 'max_attempts_factor', 'seed'.

    Returns:
        matrices: dict mapping (m, n) -> list of (constraint_mat, v) tuples.
    """
    if config is None:
        config = {}
    mcmc_steps = config.get("mcmc_steps", 10000)
    max_attempts_factor = config.get("max_attempts_factor", 50)
    seed = config.get("seed", None)

    rng = np.random.default_rng(seed)
    results = {}

    for m, n in target_sizes:
        samples = []
        attempts = 0
        max_attempts = max_attempts_factor * n_samples

        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            # Sample degree sequences from distributions
            row_sums = _sample_degrees(kappa, m, rng)
            col_sums = _sample_degrees(delta, n, rng)

            # Check basic feasibility
            if sum(row_sums) != sum(col_sums):
                continue

            init_mat = _generate_initial_matrix(row_sums, col_sums, rng)
            if init_mat is None:
                continue

            rand_mat = mcmc_sample(init_mat, iterations=mcmc_steps,
                                   seed=int(rng.integers(0, 2**31)))
            # Generate random RHS vector with 50% sparsity
            v = rng.integers(0, 2, size=m)
            samples.append((rand_mat, v))

        results[(m, n)] = samples
        print(
            f"  ({m}, {n}): generated {len(samples)}/{n_samples} samples "
            f"in {attempts} attempts"
        )

    return results


# Standard downscaled sizes from the reference experiment
DOWNSIZED_SIZES = [
    (23, 12),
    (43, 22),
    (63, 32),
    (83, 42),
    (103, 52),
    (123, 62),
    (143, 72),
    (163, 82),
    (183, 92),
    (203, 102),
    (223, 112),
    (243, 122),
    (263, 132),
    (283, 142),
    (303, 152),
    (323, 162),
    (343, 172),
    (363, 182),
    (383, 192),
    (403, 202),
    (423, 212),
    (443, 222),
    (463, 232),
    (483, 242),
    (503, 252),
    (523, 262),
    (543, 272),
    (563, 282),
]

# Performance evaluation subset (first 7 sizes, m*n from 276 to 10296)
PERFORMANCE_SIZES = DOWNSIZED_SIZES[:7]
