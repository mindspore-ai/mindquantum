# pylint: disable=unused-argument, unused-variable, invalid-name, consider-using-enumerate, unnecessary-list-index-lookup, wrong-import-position, broad-exception-caught, use-dict-literal, no-else-return, superfluous-parens
"""

随机采样基线模块。

Random sampling baseline for DQI performance comparison.

提供均匀随机采样的性能评估，作为 DQI 算法的对比基线。

核心函数：
  - objective_function_f: 计算目标函数 f(x) = Σ_i (-1)^{v_i + B_i·x mod 2}
  - satisfied_constraints: 计算满足的约束数 s = Σ_i [B·x mod 2 == v]
  - random_sampling_performance: 运行随机采样并返回统计结果

与 DQI 的关系：
  随机采样对应 DQI 中 ℓ=0 的情况（不使用量子多项式增强）。
  DQI 的优势体现在 ℓ≥1 时，<S>/m 显著高于随机基线的 0.5。
"""

import numpy as np


def objective_function_f(x, constraint_mat, v):
    """
    Compute f(x) = sum_i (-1)^{v_i + B_i · x (mod 2)}.

    Returns +1 per satisfied constraint, -1 per violated constraint.

    Args:
        x: ndarray of shape (n,), candidate solution.
        constraint_mat: ndarray of shape (m, n), constraint matrix.
        v: ndarray of shape (m,), RHS vector.

    Returns:
        f: integer, objective value in [-m, m].
    """
    return np.sum((-1) ** ((constraint_mat.dot(x) + v) % 2))


def satisfied_constraints(x, constraint_mat, v):
    """
    Count the number of satisfied XOR constraints.

    Args:
        x: ndarray of shape (n,), candidate solution.
        constraint_mat: ndarray of shape (m, n), constraint matrix.
        v: ndarray of shape (m,), RHS vector.

    Returns:
        s: integer, number of satisfied constraints in [0, m].
    """
    return np.sum((constraint_mat.dot(x) % 2) == v)


def random_sampling_performance(constraint_mat, v, shots=10000, seed=None):
    """
    Evaluate random uniform sampling baseline.

    Args:
        constraint_mat: ndarray of shape (m, n), constraint matrix.
        v: ndarray of shape (m,), RHS vector.
        shots: number of random samples.
        seed: random seed.

    Returns:
        result: dict with keys:
            'avg_s': average satisfied constraints.
            'std_s': standard deviation of satisfied constraints.
            'avg_f': average objective value.
            's_hist': array of length m+1, histogram of satisfied constraints.
            's_normalized': avg_s / m.
    """
    m, n = constraint_mat.shape
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, 2, size=(shots, n))

    s_vals = np.array([satisfied_constraints(x, constraint_mat, v) for x in samples])
    f_vals = 2 * s_vals - m

    s_hist = np.zeros(m + 1, dtype=float)
    for s in s_vals:
        s_hist[s] += 1
    s_hist /= shots

    return {
        "avg_s": float(np.mean(s_vals)),
        "std_s": float(np.std(s_vals)),
        "avg_f": float(np.mean(f_vals)),
        "s_hist": s_hist,
        "s_normalized": float(np.mean(s_vals) / m),
    }
