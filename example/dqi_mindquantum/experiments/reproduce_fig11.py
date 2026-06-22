# pylint: disable=unused-argument, unused-variable, invalid-name, consider-using-enumerate, unnecessary-list-index-lookup, wrong-import-position, broad-exception-caught, use-dict-literal, no-else-return, superfluous-parens, redefined-outer-name
"""

复现文献2 (arXiv:2509.08328) Fig 11：DQI 性能随问题规模的缩放曲线。

Reproduce Fig 11 from 文献2: DQI performance scaling for downscaled max-XORSAT
instances derived from the automotive bundling ILP.

论文要求 (Paper requirements):
  - 对 7 个降采样尺寸 (23×12 ~ 143×72) 的 max-XORSAT 实例进行测试
  - 每个尺寸独立采样 5 个实例，取平均值
  - 使用 BP1 和 BP2 两种解码器，迭代次数 T=5
  - 测试 ℓ=1, 2, 3 的 DQI 性能
  - 与随机采样和 Gurobi 最优解进行对比

算法流程 (Algorithm flow):
  1. 运行时生成降采样矩阵（或从预计算 .npz 文件加载）
  2. 对每个 (尺寸, 实例, ℓ, 解码器) 组合，计算经典 DQI 期望值 <S>/m
     - 核心计算：compute_y_in_decodable_set (找出可解码的错误模式 D_k)
     - 构建 A 矩阵并计算 <w|A|w> 期望
  3. 使用 Gurobi ILP 求解器计算最优解作为上界
  4. 绘制 broken y-axis 图展示结果

性能优化 (Performance optimization):
  - compute_y_in_decodable_set 使用 multiprocessing.Pool 并行处理 C(m,k) 个错误模式
  - 对大规模 C(m,k) 支持随机子采样 (max_patterns 参数)
  - 使用 ProcessPoolExecutor 并行运行独立的 DQI 任务
"""

import os
import sys
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import gurobipy as gp
    from gurobipy import GRB

    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classical.bp_decoders import bp1_decode, bp2_decode, gauss_jordan_solve
from classical.dqi_expectation import expected_constraints_dqi
from classical.random_sampling import random_sampling_performance
from classical.matrix_generation import (
    extract_degree_distribution,
    generate_downsized_matrices,
    PERFORMANCE_SIZES,
)

# ========================= 实验参数配置 =========================
# 与论文对齐的参数设置 (aligned with paper Section 5.2)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")  # 输出目录
CACHE_FILE = os.path.join(OUTPUT_DIR, "fig11_results.pkl")  # 结果缓存文件（可选）

N_TRIALS = 5  # 论文要求：每个尺寸 5 个独立实例取平均 (paper: 5 instances)
ELL_VALUES = [1, 2, 3]  # 论文测试的 ℓ 值 (polynomial degrees)
BP_MAX_ITER = 5  # 论文要求：BP 迭代次数 T=5 (paper: T=5 iterations)
N_JOBS = 8  # 并行 worker 数 (parallel workers for compute_y_in_decodable_set)
MAX_PATTERNS = 500  # 子采样上限：当 C(m,k) 超过此值时随机采样


def solve_max_xorsat_gurobi(constraint_mat, v, time_limit=60):
    """
    使用 Gurobi ILP 求解器精确求解 max-XORSAT 问题。

    Solve max-XORSAT optimally using Gurobi ILP solver.

    Args:
        constraint_mat: ndarray of shape (m, n), 约束矩阵 (constraint matrix).
        v: ndarray of shape (m,), 右端向量 (RHS vector).
        time_limit: Gurobi 求解时间上限（秒）。

    Returns:
        s_opt: int, 最大满足约束数，未安装 Gurobi 或超时返回 None。
    """
    if not HAS_GUROBI:
        return None

    m, n = constraint_mat.shape
    model = gp.Model("MaxXORSAT")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

    x_vars = model.addVars(n, vtype=GRB.BINARY, name="x")
    z_vars = model.addVars(m, vtype=GRB.BINARY, name="z")
    k_vars = model.addVars(m, vtype=GRB.INTEGER, lb=-float("inf"), name="k")

    for i in range(m):
        vars_in_sum = [x_vars[j] for j in range(n) if constraint_mat[i, j] == 1]
        if not vars_in_sum:
            if v[i] == 0:
                model.addConstr(z_vars[i] == 1)
            else:
                model.addConstr(z_vars[i] == 0)
        else:
            lhs = gp.quicksum(vars_in_sum)
            rhs = int(v[i]) + 2 * k_vars[i] + (1 - z_vars[i])
            model.addConstr(lhs == rhs, f"xor_{i}")

    model.setObjective(z_vars.sum(), GRB.MAXIMIZE)
    model.optimize()

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
        return int(model.objVal)
    return None


def generate_reference_matrix():
    """
    Generate a reference B matrix with LDPC-like structure.

    Rows (constraints) have degree 3-6 (from CARRY gadget structure).
    Columns (variables) have degree 2-4.
    """
    rng = np.random.default_rng(42)
    m_ref, n_ref = 200, 100
    constraint_mat = np.zeros((m_ref, n_ref), dtype=int)
    for i in range(m_ref):
        k = rng.integers(3, 7)
        cols = rng.choice(n_ref, size=k, replace=False)
        constraint_mat[i, cols] = 1
    col_sums = constraint_mat.sum(axis=0)
    zero_cols = np.where(col_sums == 0)[0]
    for c in zero_cols:
        rows = rng.choice(m_ref, size=3, replace=False)
        constraint_mat[rows, c] = 1
    return constraint_mat


def run_single_trial(args):
    """
    运行单个 (constraint_mat, v, ℓ, 解码器) 组合的 DQI 期望值计算。

    Run a single (constraint_mat, v, ell, decoder) combination.

    此函数设计为模块级函数，支持 ProcessPoolExecutor 序列化 (pickling)。

    计算流程：
      1. 调用 expected_constraints_dqi 计算经典 DQI 期望值
      2. 内部会执行 compute_y_in_decodable_set 找出可解码的错误模式
      3. 构建 A 矩阵并计算 <S>/m

    Args:
        args: tuple of (constraint_mat, v_vec, ell, decoder_name, n_jobs, max_patterns)

    Returns:
        s_normalized: float, <S>/m 归一化期望值，失败返回 None。
    """
    constraint_mat, v_vec, ell, decoder_name, n_jobs, max_patterns = args

    decoder_map = {
        "bp1": bp1_decode,
        "bp2": bp2_decode,
        "gauss": gauss_jordan_solve,
    }
    bp_func = decoder_map[decoder_name]

    try:
        # 使用论文指定的 T=5 次 BP 迭代
        config = {
            "max_iter": BP_MAX_ITER,
            "n_jobs": n_jobs,
            "max_patterns": max_patterns,
        }
        result = expected_constraints_dqi(
            constraint_mat,
            v_vec,
            ell,
            bp_func,
            config=config,
        )
        return result["s_normalized"]
    except (ValueError, RuntimeError) as e:
        print(f"  ERROR: (ell={ell}, {decoder_name}): {e}")
        return None



def _collect_baselines(matrices, results):
    """Collect random sampling and Gurobi baselines for all instances."""
    if not HAS_GUROBI:
        print("  NOTE: gurobipy not installed, skipping Gurobi optimal baseline")
    for (m, n), samples in matrices.items():
        key = f"{m}x{n}"
        for trial_idx, (constraint_mat, v_vec) in enumerate(samples):
            rand_res = random_sampling_performance(constraint_mat, v_vec, shots=10000)
            results[key]["random"].append(rand_res["s_normalized"])
            time_lim = 120 if m * n <= 5000 else 30
            s_opt = solve_max_xorsat_gurobi(constraint_mat, v_vec, time_limit=time_lim)
            results[key]["gurobi"].append(s_opt / m if s_opt is not None else np.nan)


def _run_dqi_tasks(matrices, results):
    """Run DQI expectation tasks in parallel."""
    tasks = []
    for (m, n), samples in matrices.items():
        for trial_idx, (constraint_mat, v_vec) in enumerate(samples):
            for ell in ELL_VALUES:
                for dec_name in ["bp1", "bp2"]:
                    tasks.append((f"{m}x{n}", trial_idx, ell, dec_name, constraint_mat, v_vec))

    print(f"  Total DQI tasks: {len(tasks)} ({len(matrices)} sizes x "
          f"{N_TRIALS} trials x {len(ELL_VALUES)} ell x 2 decoders)")

    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {
            executor.submit(
                run_single_trial,
                (constraint_mat, v_vec, ell, dec_name, N_JOBS, MAX_PATTERNS)
            ): (key, trial_idx, ell, dec_name)
            for (key, trial_idx, ell, dec_name, constraint_mat, v_vec) in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Progress"):
            key, trial_idx, ell, dec_name = futures[future]
            try:
                s_norm = future.result()
                if s_norm is not None:
                    results[key][dec_name][ell].append(s_norm)
            except (ValueError, RuntimeError) as e:
                print(f"  Failed: {key} ell={ell} {dec_name}: {e}")


def run_experiment():
    """
    运行完整的 Fig 11 实验，使用多进程加速。

    Run the full Fig 11 experiment with multiprocessing.

    流程：
      1. 运行时生成降采样矩阵（或从缓存加载）
      2. 对每个实例计算随机采样基线和 Gurobi 最优解
      3. 并行运行 DQI 任务 (ℓ=1,2,3 × BP1/BP2 × 5 trials × 7 sizes)

    Returns:
        results: dict, 键为 "m×n" 字符串，值为各方法的性能数据。
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(CACHE_FILE):
        print(f"Loading cached results from {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("=" * 60)
    print("Reproducing Fig 11: DQI Performance Scaling (multiprocessing)")
    print("=" * 60)

    # Step 1: 运行时生成降采样矩阵
    print("\n[1/3] Generating downsized matrices at runtime...")
    ref_mat = generate_reference_matrix()
    delta, kappa = extract_degree_distribution(ref_mat)
    matrices = generate_downsized_matrices(
        delta, kappa, PERFORMANCE_SIZES, n_samples=N_TRIALS, config={"mcmc_steps": 10000, "seed": 42}
    )

    # Step 2: Collect baselines and run DQI tasks
    print(f"\n[2/3] Collecting baselines and running DQI tasks (T={BP_MAX_ITER})...")

    results = {}
    for (m, n), _ in matrices.items():
        key = f"{m}x{n}"
        results[key] = {"m": m, "n": n, "random": [], "gurobi": [],
                        "bp1": {ell: [] for ell in ELL_VALUES},
                        "bp2": {ell: [] for ell in ELL_VALUES}}

    _collect_baselines(matrices, results)
    _run_dqi_tasks(matrices, results)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(results, f)

    return results



def _prepare_plot_data(results, sizes):
    """Prepare data arrays for plotting."""
    random_means = np.array([np.mean(results[k]["random"]) for k in sizes])
    random_stds = np.array([np.std(results[k]["random"]) for k in sizes])
    gurobi_means = np.array([np.mean(results[k]["gurobi"]) for k in sizes])
    gurobi_stds = np.array([np.std(results[k]["gurobi"]) for k in sizes])

    bp_data = {}
    for ell in ELL_VALUES:
        for dec in ["bp1", "bp2"]:
            means = np.array([np.mean(results[k][dec][ell]) if results[k][dec][ell] else 0 for k in sizes])
            stds = np.array([np.std(results[k][dec][ell]) if results[k][dec][ell] else 0 for k in sizes])
            bp_data[(dec, ell)] = (means, stds)

    return random_means, random_stds, gurobi_means, gurobi_stds, bp_data


def plot_fig11(results):
    """
    生成 Fig 11 的 broken y-axis 图。
    Args:
        results: dict, run_experiment() 的输出。
    """
    print("\n[3/3] Generating Fig 11 plot...")
    sizes = sorted(results.keys(), key=lambda k: int(k.split("x")[0]))
    x_vals = [results[k]["m"] * results[k]["n"] for k in sizes]
    x = np.arange(len(sizes))
    random_means, random_stds, gurobi_means, gurobi_stds, bp_data = _prepare_plot_data(results, sizes)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.5]})
    fig.subplots_adjust(hspace=0.08)
    bp1_colors = {1: "#4c72b0", 2: "#55a868", 3: "#c44e52"}
    bp2_colors = {1: "#8172b2", 2: "#ccb974", 3: "#64b5cd"}
    markers = {1: "o", 2: "s", 3: "D"}
    ax1.set_ylim(0.80, 1.02)
    ax2.set_ylim(0.495, 0.67)

    for ax in [ax1, ax2]:
        ax.errorbar(x, random_means, yerr=random_stds, fmt="--",
                    color="#999999", label="Random", capsize=3, alpha=0.7)
        ax.errorbar(x, gurobi_means, yerr=gurobi_stds, fmt="-",
                    color="#222222", label="Gurobi (optimal)", capsize=3, linewidth=2)
        for ell in ELL_VALUES:
            m1, s1 = bp_data[("bp1", ell)]
            ax.errorbar(x, m1, yerr=s1, marker=markers[ell],
                        color=bp1_colors[ell], label=f"BP1 $\\ell={ell}$", capsize=3)
            m2, s2 = bp_data[("bp2", ell)]
            ax.errorbar(x, m2, yerr=s2, marker=markers[ell],
                        color=bp2_colors[ell], linestyle="--", label=f"BP2 $\\ell={ell}$", capsize=3)

    ax1.set_title("Fig 11: DQI Performance Scaling", fontsize=14)
    ax1.set_ylabel("$\\langle S \\rangle / m$", fontsize=13)
    ax2.set_ylabel("$\\langle S \\rangle / m$", fontsize=13)
    ax2.set_xlabel("Problem Size $m \\times n$", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(v) for v in x_vals], rotation=45, fontsize=9)
    ax1.legend(fontsize=8, ncol=2, loc="upper right")
    for ax in [ax1, ax2]:
        ax.grid(alpha=0.3)

    d = 0.015
    kwargs = {"transform": ax1.transAxes, "color": "k", "clip_on": False}
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig11_dqi_performance_scaling.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Figure saved to: {fig_path}")
    plt.close()


def print_summary(results):
    """Print summary table."""
    print("\nResults Summary:")
    print("=" * 100)
    header = f"{'Size':<10} {'m*n':<8} {'Random':<10} {'Gurobi':<10}"
    for ell in ELL_VALUES:
        header += f" {'BP1 l=' + str(ell):<12} {'BP2 l=' + str(ell):<12}"
    print(header)
    print("-" * 100)
    for key in sorted(results.keys(), key=lambda k: int(k.split("x")[0])):
        r = results[key]
        line = f"{key:<10} {r['m'] * r['n']:<8} {np.mean(r['random']):<10.4f} {np.mean(r['gurobi']):<10.4f}"
        for ell in ELL_VALUES:
            bp1_m = np.mean(r["bp1"][ell]) if r["bp1"][ell] else 0
            bp2_m = np.mean(r["bp2"][ell]) if r["bp2"][ell] else 0
            line += f" {bp1_m:<12.4f} {bp2_m:<12.4f}"
        print(line)
    print("=" * 100)


if __name__ == "__main__":
    # 运行 Fig 11 复现实验
    # 首次运行会执行完整的 DQI 期望值计算（约 7-10 分钟）
    # 后续运行会加载缓存结果直接绘图
    t_start = time.time()
    results = run_experiment()
    plot_fig11(results)
    print_summary(results)
    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")
