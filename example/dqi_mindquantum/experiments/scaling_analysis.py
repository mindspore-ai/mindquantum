# pylint: disable=unused-argument, unused-variable, invalid-name, consider-using-enumerate, unnecessary-list-index-lookup, wrong-import-position, broad-exception-caught, use-dict-literal, no-else-return, superfluous-parens
"""

DQI 量子资源扩展性分析。

DQI resource scaling analysis.

分析 DQI 算法的量子比特数和门数如何随问题规模 (m, n) 和算法参数 (ℓ, T) 变化。

理论复杂度 (Theoretical complexity):
  - 量子比特数：O(n + m)，线性扩展
  - LUT 解码门数：O(m · 2^n)，指数级（n>20 不可行）
  - BP1 解码门数：O(T · m · d)，多项式级（d 为平均列度）
  - Dicke 态制备：O(ℓ · m)，线性
  - 权重反计算：O(C(m, ℓ))，组合级（限制 ℓ 取小值）

输出图表 (Output figures):
  1. scaling_qubits.png — 量子比特数随问题规模的扩展曲线
  2. scaling_gate_breakdown.png — 单个实例的门数分解（各组件占比）
  3. scaling_gates_vs_size.png — 总门数随问题规模的扩展曲线
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quantum.full_circuit import DQICircuit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "output")


def theoretical_qubit_count(m, n, ell, bp_iters=1, use_bp=False):
    """
    Compute theoretical qubit count.

    DQI 电路的量子比特数不依赖 ℓ（ℓ 只影响权重寄存器的角度，不影响比特数）。

    LUT decoding: N = n + m (error + syndrome registers)
    BP1 decoding: N = 3m + n (error + syndrome + 2m BP1 aux)

    Args:
        m: number of constraints.
        n: number of variables.
        ell: polynomial degree (不影响量子比特数).
        T: BP iterations.
        use_bp: if True, use quantum BP1 formula.

    Returns:
        n_qubits: total qubit count.
    """
    if use_bp:
        # BP1: error(m) + syndrome(n) + bp1_aux(2m)
        return 3 * m + n
    else:
        # LUT: error(m) + syndrome(n)
        return m + n


def theoretical_gate_count(m, n, ell, use_lut=True):
    """
    Estimate gate count for different components.

    门数公式（基于实际 MindQuantum 电路结构）：
      - Weight embedding: O(2^n_w) = O(ell)，RY/CRY 级联
      - Dicke preparation: O(ell * m)，受控 SCS 门
      - Weight uncomputation: O(C(m, ell) * n_w)，枚举所有重量-k 模式
        注：这是理论最坏情况，实际电路可能更高效
      - Phase Z: O(m)
      - Syndrome computation: O(nnz(constraint_mat)) ~ O(m * d)
      - BP1 decoding: O(m * d)，AND 门 + 翻转 + 反计算
      - Hadamard: O(n)

    Args:
        m: number of constraints.
        n: number of variables.
        ell: polynomial degree.
        use_lut: if True, include LUT gate count (exponential, 仅小规模可用).

    Returns:
        dict with gate counts per component.
    """
    from scipy.special import comb

    n_w = int(np.ceil(np.log2(ell + 1)))

    # Weight embedding: O(2^n_w) = O(ell)
    gates_embed = 2 ** (n_w + 1)

    # Dicke preparation: O(ell * m) gates
    gates_dicke = ell * m * 5  # ~5 gates per SCS

    # Weight uncomputation: O(C(m, ell) * n_w)
    # 这是理论估计，实际电路使用多控 X 门而非枚举
    gates_uncompute = int(comb(m, min(ell, 3), exact=True)) * n_w * 2

    # Phase Z: O(m)
    gates_phase = m

    # Syndrome computation: O(nnz(constraint_mat)) ~ O(m * d) where d is avg degree
    gates_syndrome = m * 4  # assume avg degree ~4

    # BP1 decoding: O(m * d) -- AND gates + flip + uncompute
    gates_bp1 = m * 4  # ~4 gates per variable node

    # LUT decoding: O(m * 2^n) -- exponential, only for small n
    if use_lut and n <= 20:
        gates_lut = m * (2**n) // 10  # approximate
    else:
        gates_lut = 0

    # Hadamard: O(n)
    gates_hadamard = n

    total = (
        gates_embed
        + gates_dicke
        + gates_uncompute
        + gates_phase
        + gates_syndrome
        + gates_bp1
        + gates_lut
        + gates_hadamard
    )

    return {
        "embed": gates_embed,
        "dicke": gates_dicke,
        "uncompute": gates_uncompute,
        "phase": gates_phase,
        "syndrome": gates_syndrome,
        "bp1": gates_bp1,
        "lut": gates_lut,
        "hadamard": gates_hadamard,
        "total": total,
    }


def analyze_actual_circuit(m, n, ell):
    """
    Build an actual DQI circuit and count gates using MindQuantum.

    Args:
        m: number of constraints.
        n: number of variables.
        ell: polynomial degree.

    Returns:
        summary dict from the circuit.
    """
    # Generate a random B matrix
    rng = np.random.default_rng(42)
    constraint_mat = rng.integers(0, 2, size=(m, n))
    # Ensure no zero rows/columns
    for i in range(m):
        if B[i].sum() == 0:
            B[i, rng.integers(0, n)] = 1
    for j in range(n):
        if B[:, j].sum() == 0:
            B[rng.integers(0, m), j] = 1

    v = rng.integers(0, 2, size=m)

    try:
        dqi = DQICircuit(constraint_mat, v, ell=ell)
        circ = dqi.build_circuit()
        summary = dqi.get_circuit_summary(circ)
        return summary
    except (ValueError, RuntimeError) as e:
        return {
            "n_qubits": dqi.total_qubits if "dqi" in dir() else 0,
            "n_gates": 0,
            "error": str(e),
        }



def _plot_qubit_scaling(fig_dir):
    """Plot qubit scaling analysis."""
    print("\n[1/3] Qubit scaling analysis...")
    sizes = [(m, int(m * 0.5)) for m in range(10, 310, 20)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    m_vals = [m * n for m, n in sizes]
    qubits_lut = [theoretical_qubit_count(m, n, ell=1, use_bp=False) for m, n in sizes]
    qubits_bp = [theoretical_qubit_count(m, n, ell=1, bp_iters=3, use_bp=True) for m, n in sizes]

    ax1.plot(m_vals, qubits_lut, "o-", color="#4c72b0", label="N = m + n")
    ax2.plot(m_vals, qubits_bp, "s-", color="#c44e52", label="N = 3m + n (T=3)")
    for ax, title in [(ax1, "LUT Decoding: $N = m + n$"), (ax2, "BP1 Decoding: $N = 3m + n$")]:
        ax.set_xlabel("Problem Size $m \times n$", fontsize=12)
        ax.set_ylabel("Qubit Count", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
    fig.suptitle(r"Qubit Scaling: $\mathcal{O}(m + n)$", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "scaling_qubits.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: scaling_qubits.png")


def _plot_gate_breakdown(fig_dir):
    """Plot gate count breakdown."""
    print("\n[2/3] Gate count analysis...")
    m_test, n_test, ell_test = 50, 25, 2
    gates = theoretical_gate_count(m_test, n_test, ell_test, use_lut=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    components = [k for k, v in gates.items() if k != "total" and v > 0]
    counts = [gates[k] for k in components]
    colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
    bars = ax.bar(components, counts, color=colors, edgecolor="gray")
    ax.set_yscale("log")
    ax.set_ylabel("Gate Count (log scale)", fontsize=12)
    ax.set_title(f"Gate Count Breakdown (BP1, $m={m_test}$, $n={n_test}$, $\\ell={ell_test}$)", fontsize=13)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{count}", ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "scaling_gate_breakdown.png"), dpi=150)
    plt.close()
    print("  Saved: scaling_gate_breakdown.png")


def _plot_gate_scaling(fig_dir):
    """Plot gate scaling with problem size."""
    print("\n[3/3] Gate count vs problem size...")
    sizes2 = [(m, int(m * 0.5)) for m in range(10, 101, 10)]
    fig, ax = plt.subplots(figsize=(10, 6))
    for ell in [1, 2, 3]:
        m_vals = [m * n for m, n in sizes2]
        total_gates = [theoretical_gate_count(m, n, ell, use_lut=False)["total"] for m, n in sizes2]
        ax.plot(m_vals, total_gates, marker="o", label=f"$\\ell={ell}$")
    ax.set_xlabel(r"Problem Size $m \times n$", fontsize=12)
    ax.set_ylabel("Total Gate Count", fontsize=12)
    ax.set_title("Gate Count Scaling (BP1 decoding)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "scaling_gates_vs_size.png"), dpi=150)
    plt.close()
    print("  Saved: scaling_gates_vs_size.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=" * 60)
    print("DQI Resource Scaling Analysis")
    print("=" * 60)
    _plot_qubit_scaling(FIG_DIR)
    _plot_gate_breakdown(FIG_DIR)
    _plot_gate_scaling(FIG_DIR)
    print("All scaling figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
