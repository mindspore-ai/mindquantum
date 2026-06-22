"""
复现文献2 (arXiv:2509.08328) Fig 15：DQI 量子电路采样的约束满足直方图。

Reproduce Fig 15 from 文献2: histogram of satisfied constraints from DQI sampling.

论文要求 (Paper requirements):
  - 使用固定的 8×6 max-XORSAT 实例（论文公式 (72)）
  - 运行完整量子电路模拟，ℓ=1, 2, 3
  - 使用 10,000 次采样 (shots)
  - 与均匀随机采样进行对比
  - BP 解码器迭代次数 T=1

量子电路架构 (Circuit architecture):
  1. 权重编码 (WeightedUnaryEncoding)：在 error_register 上制备 w_k 叠加态
  2. Dicke 态制备 (UnkStatePreparation)：生成 |D_k^m> 态
  3. 相位 Z 编码：应用 (-1)^{v_i} 相位
  4. 综合征计算：CNOT 网络计算 s = B^T y mod 2
  5. BP1 解码器：将 error register 反计算为 |0>
  6. Hadamard 变换：在 syndrome register 上应用 H 门

后选择 (Post-selection):
  测量后仅保留 error_register = |0⟩ 的结果，提取 syndrome 计算满足的约束数。
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quantum.full_circuit import run_dqi_histogram  # pylint: disable=wrong-import-position
from classical.random_sampling import random_sampling_performance  # pylint: disable=wrong-import-position

# ========================= 实例定义 =========================
# 固定的 8×6 max-XORSAT 实例，来自论文公式 (72)
# B: 8 个约束 × 6 个变量的二进制矩阵
# v: 右端向量 (RHS vector)

B = np.array(
    [
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
    ],
    dtype=int,
)

v = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)

M, N = B.shape  # m=8, n=6，总量子比特数 = 3m + n = 28
SHOTS = 10000  # 论文要求 10^4 次采样
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    """
    运行 Fig 15 复现实验。

    流程：
      1. 计算随机采样基线 (10^4 次采样)
      2. 对 ℓ=1, 2, 3 分别运行 DQI 量子电路模拟
      3. 绘制分组柱状图对比各方法的约束满足分布
      4. 输出汇总表格（平均满足约束数、归一化值、后选择比例）
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("Reproducing Fig 15: DQI Quantum Histogram")
    print(f"Instance: {M}x{N}, Shots: {SHOTS}")
    print("=" * 60)

    # Random sampling baseline
    print("\n[1/4] Random sampling baseline...")
    random_result = random_sampling_performance(B, v, shots=SHOTS, seed=42)
    print(f"  Average satisfied constraints: {random_result['avg_s']:.4f}")

    # DQI for ell=1, 2, 3
    dqi_results = {}
    for ell in [1, 2, 3]:
        print(f"\n[{ell + 1}/4] DQI with ell={ell}...")
        result = run_dqi_histogram(B, v, ell=ell, shots=SHOTS, seed=42)
        dqi_results[ell] = result
        print(f"  Average satisfied constraints: {result['avg_s']:.4f}")
        print(f"  Post-selection fraction: {result['post_fraction']:.4f}")

    # Plot Fig 15: grouped bar chart
    print("\n[4/4] Generating plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(M + 1)  # 0..8 satisfied constraints
    width = 0.18
    colors = ["#6b6b6b", "#4c72b0", "#55a868", "#c44e52"]
    labels = ["Random", r"DQI $\ell=1$", r"DQI $\ell=2$", r"DQI $\ell=3$"]

    # Random histogram
    ax.bar(
        x - 1.5 * width,
        random_result["s_hist"],
        width,
        label=labels[0],
        color=colors[0],
        edgecolor="white",
    )

    # DQI histograms
    for idx, ell in enumerate([1, 2, 3]):
        offset = (idx - 0.5) * width
        ax.bar(
            x + offset,
            dqi_results[ell]["s_hist"],
            width,
            label=labels[ell],
            color=colors[ell],
            edgecolor="white",
        )

    ax.set_xlabel("Number of Satisfied Constraints $s$", fontsize=14)
    ax.set_ylabel("Probability", fontsize=14)
    ax.set_title(
        f"Fig 15: DQI Sampling Histogram ($8\\times 6$ instance, {SHOTS} shots)",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig15_dqi_histogram.png")
    fig.savefig(fig_path, dpi=150)
    print(f"\nFigure saved to: {fig_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"{'Method':<20} {'<S>':<10} {'<S>/m':<10}")
    print("-" * 40)
    print(
        f"{'Random':<20} {random_result['avg_s']:<10.4f} "
        f"{random_result['s_normalized']:<10.4f}"
    )
    for ell in [1, 2, 3]:
        r = dqi_results[ell]
        print(
            f"{'DQI l=' + str(ell):<20} {r['avg_s']:<10.4f} {r['s_normalized']:<10.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
