"""
经典算法模块。

Classical algorithms for DQI.

提供纯 numpy/scipy 实现的核心算法：
  - BP 解码器 (bp_decoders): BP1 硬判决、BP2 软判决、Gauss-Jordan 精确解码
  - DQI 期望计算 (dqi_expectation): 最优权重、A 矩阵、<S> 期望值
  - 矩阵生成 (matrix_generation): 度分布提取、MCMC 降采样
  - 随机采样 (random_sampling): 均匀随机基线
"""

from .bp_decoders import bp1_decode, bp2_decode, gauss_jordan_solve, compute_y_in_Dk
from .dqi_expectation import (
    compute_optimal_w,
    compute_A_matrix,
    compute_expected_values,
    expected_constraints_dqi,
)
from .matrix_generation import (
    extract_degree_distribution,
    mcmc_sample,
    generate_downsized_matrices,
)
from .random_sampling import random_sampling_performance

__all__ = [
    "bp1_decode",
    "bp2_decode",
    "gauss_jordan_solve",
    "compute_y_in_Dk",
    "compute_optimal_w",
    "compute_A_matrix",
    "compute_expected_values",
    "expected_constraints_dqi",
    "extract_degree_distribution",
    "mcmc_sample",
    "generate_downsized_matrices",
    "random_sampling_performance",
]
