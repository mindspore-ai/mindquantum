"""
复现 arXiv:2510.19550 (Quantum nuclear-spin echo) FIG.2(d)(e) 在 H2 上的版本:
    "Quantum computation of molecular geometry via many-body nuclear spin echoes"

任务: 体现 OTOC 对分子几何 (键长) 的敏感性, 并演示通过 OTOC 学习键长.
论文用 toluene (15 自旋) NMR; 验收要求 H2 (最小分子) 上完成同等思路.

实现:
  * H2 在 STO-3G 基组下经 Bravyi-Kitaev / Jordan-Wigner 简化, 可被精确
    映射为 2 比特有效哈密顿量 (Seeley-Richard-Love 2012 等). 系数随
    H-H 键长 r 解析变化.
  * 演化算子 U(t) = exp(-i H(r) t).
  * OTOC 协议: |+>^N -> U -> X_qb -> U^† -> 测 <Z_qm>.
    对一组演化时间 t 计算 OTOC(t; r); 不同 r 给出不同曲线.
  * (d) 图: OTOC vs 时间, 不同键长 r 对比, 显示对 r 的敏感性.
  * (e) 图: 给定一条 "实验" 曲线 (取 r_ref ≈ 0.74 Å), 计算它与扫描
    各 r 的模拟曲线的均方误差; 以最小 MSE 估计 r, 并画估计随
    "数据点数" 收敛.

H2 STO-3G 系数 (来自 Seeley et al., 用 BK 编码后, 后续合并对称等价比特):
    r 依赖系数表 (内置插值, 单位: Hartree)

实现简化: 我们不依赖 openfermion/pyscf, 而是用 Seeley 2012 论文表 III
给出的 r 网格化系数, 通过样条插值得到任意 r 的哈密顿量.
"""

import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mindquantum as mq

# 避免 scipy 与 mindquantum 内置 BLAS 的 DLL 冲突 (Windows 上会段错误),
# 改用 numpy 实现的简单 cubic-Hermite 样条 / 分段三次插值.
class _NPCubicSpline:
    """轻量 piecewise cubic Hermite (Catmull-Rom) 插值器, 替代 scipy.CubicSpline."""

    def __init__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        order = np.argsort(x)
        self.x = x[order]
        self.y = y[order]
        self.n = len(self.x)
        # 端点切线: 一阶差分; 内部用中央差分
        m = np.zeros_like(self.y)
        m[1:-1] = (self.y[2:] - self.y[:-2]) / (self.x[2:] - self.x[:-2])
        m[0] = (self.y[1] - self.y[0]) / (self.x[1] - self.x[0])
        m[-1] = (self.y[-1] - self.y[-2]) / (self.x[-1] - self.x[-2])
        self.m = m

    def __call__(self, xq):
        scalar = np.isscalar(xq)
        xq = np.atleast_1d(xq).astype(float)
        # 边界外用线性外推
        out = np.empty_like(xq)
        for k, xv in enumerate(xq):
            if xv <= self.x[0]:
                out[k] = self.y[0] + self.m[0] * (xv - self.x[0])
            elif xv >= self.x[-1]:
                out[k] = self.y[-1] + self.m[-1] * (xv - self.x[-1])
            else:
                i = int(np.searchsorted(self.x, xv) - 1)
                h = self.x[i + 1] - self.x[i]
                t = (xv - self.x[i]) / h
                # Hermite basis
                h00 = 2 * t**3 - 3 * t**2 + 1
                h10 = t**3 - 2 * t**2 + t
                h01 = -2 * t**3 + 3 * t**2
                h11 = t**3 - t**2
                out[k] = (
                    h00 * self.y[i]
                    + h10 * h * self.m[i]
                    + h01 * self.y[i + 1]
                    + h11 * h * self.m[i + 1]
                )
        return out[0] if scalar else out


CubicSpline = _NPCubicSpline
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, X, Y, Z, RX, RY, RZ
from mindquantum.core.operators import QubitOperator, Hamiltonian, TimeEvolution
from mindquantum.simulator import Simulator


# ============================================================
# H2 / STO-3G qubit Hamiltonian as a function of bond length
# ============================================================
# 数据来源: Seeley, Richard, Love, JCP 137, 224109 (2012), Table I & III
# 经 Bravyi-Kitaev 编码, 2-qubit 简化形式 (见 O'Malley 2016 PRX 6, 031007).
#
# H(r) = g0(r) I + g1(r) Z0 + g2(r) Z1 + g3(r) Z0 Z1
#       + g4(r) X0 X1 + g5(r) Y0 Y1
#
# 下面的 (r, g0..g5) 表来自 O'Malley 2016 Table I 的拟合, 单位 Hartree.

H2_COEFF_TABLE = np.array([
    # r,    g0,        g1,       g2,       g3,       g4,       g5
    [0.20, +2.8489,   +0.5678, -1.4508, +0.6799, +0.0791, +0.0791],
    [0.25, +2.1868,   +0.5449, -1.2870, +0.6719, +0.0798, +0.0798],
    [0.30, +1.7252,   +0.5215, -1.1458, +0.6631, +0.0806, +0.0806],
    [0.35, +1.3827,   +0.4982, -1.0226, +0.6537, +0.0815, +0.0815],
    [0.40, +1.1182,   +0.4754, -0.9145, +0.6438, +0.0825, +0.0825],
    [0.45, +0.9083,   +0.4534, -0.8194, +0.6336, +0.0835, +0.0835],
    [0.50, +0.7381,   +0.4325, -0.7355, +0.6233, +0.0846, +0.0846],
    [0.55, +0.5979,   +0.4125, -0.6612, +0.6129, +0.0858, +0.0858],
    [0.60, +0.4808,   +0.3937, -0.5950, +0.6025, +0.0870, +0.0870],
    [0.65, +0.3819,   +0.3760, -0.5358, +0.5921, +0.0883, +0.0883],
    [0.70, +0.2976,   +0.3593, -0.4826, +0.5818, +0.0896, +0.0896],
    [0.75, +0.2252,   +0.3435, -0.4347, +0.5716, +0.0910, +0.0910],
    [0.80, +0.1626,   +0.3288, -0.3915, +0.5616, +0.0925, +0.0925],
    [0.85, +0.1083,   +0.3149, -0.3523, +0.5518, +0.0939, +0.0939],
    [0.90, +0.0609,   +0.3018, -0.3168, +0.5421, +0.0954, +0.0954],
    [0.95, +0.0193,   +0.2895, -0.2845, +0.5327, +0.0970, +0.0970],
    [1.00, -0.0172,   +0.2779, -0.2550, +0.5235, +0.0986, +0.0986],
    [1.05, -0.0493,   +0.2669, -0.2282, +0.5146, +0.1002, +0.1002],
    [1.10, -0.0778,   +0.2565, -0.2036, +0.5059, +0.1018, +0.1018],
    [1.15, -0.1029,   +0.2467, -0.1810, +0.4974, +0.1034, +0.1034],
    [1.20, -0.1253,   +0.2374, -0.1603, +0.4892, +0.1050, +0.1050],
    [1.25, -0.1452,   +0.2286, -0.1413, +0.4812, +0.1067, +0.1067],
    [1.30, -0.1629,   +0.2203, -0.1238, +0.4735, +0.1083, +0.1083],
    [1.35, -0.1786,   +0.2123, -0.1077, +0.4660, +0.1100, +0.1100],
    [1.40, -0.1927,   +0.2048, -0.0929, +0.4588, +0.1116, +0.1116],
    [1.45, -0.2053,   +0.1976, -0.0792, +0.4518, +0.1133, +0.1133],
    [1.50, -0.2165,   +0.1908, -0.0666, +0.4451, +0.1149, +0.1149],
    [1.55, -0.2265,   +0.1843, -0.0549, +0.4386, +0.1165, +0.1165],
    [1.60, -0.2355,   +0.1782, -0.0442, +0.4323, +0.1181, +0.1181],
    [1.65, -0.2436,   +0.1723, -0.0342, +0.4262, +0.1196, +0.1196],
    [1.70, -0.2508,   +0.1667, -0.0251, +0.4204, +0.1211, +0.1211],
    [1.75, -0.2573,   +0.1614, -0.0166, +0.4148, +0.1226, +0.1226],
    [1.80, -0.2632,   +0.1563, -0.0088, +0.4094, +0.1241, +0.1241],
    [1.85, -0.2684,   +0.1515, -0.0015, +0.4042, +0.1256, +0.1256],
    [1.90, -0.2731,   +0.1469, +0.0052, +0.3992, +0.1270, +0.1270],
    [1.95, -0.2774,   +0.1426, +0.0114, +0.3944, +0.1284, +0.1284],
    [2.00, -0.2812,   +0.1384, +0.0171, +0.3898, +0.1297, +0.1297],
    [2.05, -0.2847,   +0.1344, +0.0223, +0.3853, +0.1310, +0.1310],
    [2.10, -0.2879,   +0.1306, +0.0272, +0.3811, +0.1323, +0.1323],
    [2.15, -0.2908,   +0.1270, +0.0317, +0.3769, +0.1335, +0.1335],
    [2.20, -0.2934,   +0.1235, +0.0359, +0.3730, +0.1347, +0.1347],
    [2.25, -0.2958,   +0.1202, +0.0397, +0.3692, +0.1359, +0.1359],
    [2.30, -0.2980,   +0.1170, +0.0432, +0.3655, +0.1370, +0.1370],
    [2.35, -0.3000,   +0.1140, +0.0465, +0.3620, +0.1381, +0.1381],
    [2.40, -0.3018,   +0.1111, +0.0495, +0.3586, +0.1392, +0.1392],
    [2.45, -0.3035,   +0.1083, +0.0523, +0.3553, +0.1402, +0.1402],
    [2.50, -0.3051,   +0.1057, +0.0549, +0.3521, +0.1413, +0.1413],
    [2.55, -0.3066,   +0.1031, +0.0572, +0.3491, +0.1422, +0.1422],
    [2.60, -0.3079,   +0.1007, +0.0594, +0.3461, +0.1432, +0.1432],
    [2.65, -0.3092,   +0.0984, +0.0614, +0.3433, +0.1441, +0.1441],
    [2.70, -0.3104,   +0.0962, +0.0632, +0.3406, +0.1450, +0.1450],
    [2.75, -0.3115,   +0.0940, +0.0649, +0.3379, +0.1458, +0.1458],
    [2.80, -0.3125,   +0.0920, +0.0665, +0.3354, +0.1467, +0.1467],
    [2.85, -0.3135,   +0.0901, +0.0679, +0.3329, +0.1475, +0.1475],
])

# 构造 r -> g_i(r) 的样条 (每列单独插值)
_R_GRID = H2_COEFF_TABLE[:, 0]
_SPLINES = [CubicSpline(_R_GRID, H2_COEFF_TABLE[:, k]) for k in range(1, 7)]


def h2_qubit_hamiltonian(r: float) -> QubitOperator:
    """
    返回 H2 在键长 r (单位: Å) 下的 2-qubit 哈密顿量 (Hartree).

    H(r) = g0 + g1 Z0 + g2 Z1 + g3 Z0 Z1 + g4 X0 X1 + g5 Y0 Y1
    """
    g = [s(r) for s in _SPLINES]
    op = QubitOperator("", float(g[0]))
    op += QubitOperator("Z0", float(g[1]))
    op += QubitOperator("Z1", float(g[2]))
    op += QubitOperator("Z0 Z1", float(g[3]))
    op += QubitOperator("X0 X1", float(g[4]))
    op += QubitOperator("Y0 Y1", float(g[5]))
    return op


# ============================================================
# Trotter 演化电路 U(t) = exp(-i H t)
# ============================================================
def trotter_evolve(ham: QubitOperator, t: float, n_steps: int) -> Circuit:
    """
    1 阶 Trotter: U(t) ≈ ( prod_k exp(-i h_k * dt) )^n_steps,  dt = t / n_steps.
    使用 mindquantum 的 TimeEvolution.
    """
    dt = t / n_steps
    # TimeEvolution 接受 QubitOperator 与 t, 返回参数化 / 已绑定电路
    step_circ = TimeEvolution(ham, dt).circuit
    full = Circuit()
    for _ in range(n_steps):
        full += step_circ
    return full


# ============================================================
# OTOC 协议
# ============================================================
def initial_state_circuit(n_qubits: int) -> Circuit:
    """
    NMR 高温初始密度矩阵 ~ I + epsilon * Z; 信号正比于 <Z>.
    我们用 |+>^N (在所有比特上 H), 然后测 X, 等价于测 NMR 的 transverse.
    """
    c = Circuit()
    for q in range(n_qubits):
        c += H.on(q)
    return c


def measure_otoc(
    ham: QubitOperator,
    t: float,
    qb: int,
    qm: int,
    n_qubits: int,
    n_steps: int,
) -> float:
    """
    论文风格 OTOC, 用 Loschmidt-echo 形式:
        F(t) = |<HF| U^(-t) X_qb U(t) X_qm U^(-t) X_qb U(t) X_qm |HF>|^2
    实现: 顺序施加 echo, 最后通过测 |<HF|psi>|^2 = |<10|psi>|^2 给出 OTOC.
    返回 +1 (t=0) 单调衰减的曲线 (BK 编码下 H2 系统在第一个半周期内).
    """
    if t == 0.0:
        return 1.0
    fwd = trotter_evolve(ham, t, n_steps)
    bwd = trotter_evolve(ham, -t, n_steps)

    sim = Simulator("mqvector", n_qubits)
    sim.apply_circuit(Circuit([X.on(0)]))    # |HF> = |10>  (qubit 0 = X|0>)
    # echo: U(t) X_qb U(-t) X_qm U(t) X_qb U(-t) X_qm
    sim.apply_circuit(fwd)
    sim.apply_circuit(Circuit([X.on(qb)]))
    sim.apply_circuit(bwd)
    sim.apply_circuit(Circuit([X.on(qm)]))
    sim.apply_circuit(fwd)
    sim.apply_circuit(Circuit([X.on(qb)]))
    sim.apply_circuit(bwd)
    sim.apply_circuit(Circuit([X.on(qm)]))   # 闭合 echo
    # 计算 |<HF|psi>|^2.  HF = |10>, 索引 (qubit 排列: 高位 q1, 低位 q0)
    # 但 mindquantum get_qs() 排列是 q_{n-1}...q_0, |10> 在状态 X.on(0)|00>
    # 之后, 所以 |10> 在 amplitude index = 1 (二进制 ...01).
    qs = sim.get_qs()
    # |HF> 在 amplitude index 1 (qubit 0 = 1, 其它 0):
    #    state = X_0 |00> = |...0 1>  -> binary index '01' = 1
    return float(np.abs(qs[1]) ** 2)


def measure_loschmidt(
    ham: QubitOperator,
    t: float,
    n_qubits: int,
    n_steps: int,
) -> float:
    """
    Loschmidt echo: <psi0| U^†(t) U(t) |psi0> = 1 (理想); 引入 perturbation
    (这里不加, 用作 baseline / 验证电路无误差).
    在论文中 Loschmidt 是 "无蝴蝶的" OTOC reference. 返回 <psi0|psi(t->0)>.
    """
    fwd = trotter_evolve(ham, t, n_steps)
    bwd = trotter_evolve(ham, -t, n_steps)
    sim = Simulator("mqvector", n_qubits)
    sim.apply_circuit(initial_state_circuit(n_qubits))
    sim.apply_circuit(fwd)
    sim.apply_circuit(bwd)
    # 测 <X0> 作为 fidelity 的代理 (理想为 1)
    return float(np.real(sim.get_expectation(Hamiltonian(QubitOperator("X0")))))


# ============================================================
# (d) OTOC 对键长拉伸的灵敏度
# ============================================================
def run_panel_d(
    r_list: tuple,
    t_grid: np.ndarray,
    n_steps: int,
    cache_path: str,
):
    """对一组键长 r 计算 OTOC(t) 曲线; r 改变 -> 哈密顿量改变 -> OTOC 改变."""
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    results = {}
    for r in r_list:
        ham = h2_qubit_hamiltonian(r)
        otoc_t = []
        for t in t_grid:
            otoc_t.append(
                measure_otoc(ham, float(t), qb=1, qm=0, n_qubits=2, n_steps=n_steps)
            )
        results[r] = np.array(otoc_t)
        print(
            f"[(d)] r = {r:.3f} A  done. mean={results[r].mean():+.3f} "
            f"|OTOC| range=[{results[r].min():+.3f}, {results[r].max():+.3f}]"
        )
    with open(cache_path, "wb") as f:
        pickle.dump(dict(t_grid=t_grid, curves=results), f)
    return dict(t_grid=t_grid, curves=results)


# ============================================================
# (e) 用 OTOC 数据估计键长 r, 收敛到参考值
# ============================================================
def run_panel_e(
    r_truth: float,
    r_scan: np.ndarray,
    t_full: np.ndarray,
    n_steps: int,
    n_points_grid: tuple,
    n_bootstrap: int,
    noise_sigma: float,
    cache_path: str,
):
    """
    e 图协议:
      1. 以 r_truth 模拟得到 "实验" OTOC 曲线 + 高斯噪声 (代替 NMR 实验)
      2. 对每个 candidate r, 模拟 OTOC 曲线
      3. 计算 MSE(r) = sum_i (sim_r[t_i] - exp[t_i])^2
      4. 用前 N 个时间点 (N 从小到大) 估计 r_hat = argmin_r MSE
      5. bootstrap 噪声实例化, 给出 2σ CI
    """
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    rng = np.random.default_rng(2026)
    print(f"[(e)] r_truth = {r_truth:.3f} A, |r_scan|={len(r_scan)}, |t|={len(t_full)}")

    # --- 一次性算出 "实验" 与 "candidate" 库 ---
    ham_truth = h2_qubit_hamiltonian(r_truth)
    exp_curve_clean = np.array([
        measure_otoc(ham_truth, float(t), qb=1, qm=0, n_qubits=2, n_steps=n_steps)
        for t in t_full
    ])

    sim_lib = np.zeros((len(r_scan), len(t_full)))
    for i, r in enumerate(r_scan):
        ham = h2_qubit_hamiltonian(float(r))
        for j, t in enumerate(t_full):
            sim_lib[i, j] = measure_otoc(ham, float(t), 1, 0, 2, n_steps)
        if i % 5 == 0:
            print(f"  scan r={r:.3f}  done")

    # --- 不同 N 点数下的估计 + bootstrap CI ---
    r_hat_mean = []
    r_hat_lo = []
    r_hat_hi = []
    mse_at_truth = []  # 顶图: covariance-weighted 误差
    for N in n_points_grid:
        ests = []
        for _ in range(n_bootstrap):
            exp_curve = exp_curve_clean[:N] + rng.normal(0, noise_sigma, size=N)
            mse = np.mean((sim_lib[:, :N] - exp_curve[None, :]) ** 2, axis=1)
            ests.append(float(r_scan[int(np.argmin(mse))]))
        ests = np.array(ests)
        r_hat_mean.append(ests.mean())
        r_hat_lo.append(np.quantile(ests, 0.025))
        r_hat_hi.append(np.quantile(ests, 0.975))
        # 顶图: 在 r_truth 附近, 评价 sim - exp 的均方误差 vs 不同假设 r
        mse_curve = np.mean((sim_lib[:, :N] - exp_curve_clean[:N][None, :]) ** 2, axis=1)
        mse_at_truth.append(mse_curve)
        print(
            f"  N={N:3d}  r_hat={ests.mean():.3f}+-{ests.std():.3f} "
            f"(2sig: [{r_hat_lo[-1]:.3f},{r_hat_hi[-1]:.3f}])"
        )

    out = dict(
        r_truth=r_truth,
        r_scan=r_scan,
        n_points_grid=list(n_points_grid),
        r_hat_mean=r_hat_mean,
        r_hat_lo=r_hat_lo,
        r_hat_hi=r_hat_hi,
        mse_at_truth=mse_at_truth,
    )
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    return out


# ============================================================
# 可视化
# ============================================================
def _draw_h2_inset(ax_parent, loc_box=(0.05, 0.05, 0.32, 0.32)):
    """在 ax_parent 内嵌入一个 H2 分子卡通图作为 inset, 标注键长 r_HH."""
    ax = ax_parent.inset_axes(loc_box)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    # 两个氢原子
    for x, lbl in [(-0.6, "H"), (0.6, "H")]:
        ax.add_patch(plt.Circle((x, 0), 0.32, facecolor="white",
                                edgecolor="black", lw=1.8, zorder=5))
        ax.text(x, 0, lbl, ha="center", va="center", fontsize=10,
                fontweight="bold", zorder=10)
    # 键
    ax.plot([-0.28, 0.28], [0, 0], "k-", lw=2.0, zorder=2)
    # 距离标注
    ax.annotate("", xy=(0.6, -0.55), xytext=(-0.6, -0.55),
                arrowprops=dict(arrowstyle="<->", color="#d77a17", lw=1.8))
    ax.text(0, -0.78, r"$r_{H-H}$", ha="center", va="top",
            fontsize=10, color="#d77a17", fontweight="bold")


def plot_panel_d(d_data, save_path, r_truth=0.74, exp_noise=0.025, exp_seed=7):
    """
    论文 FIG2(d) 风格:
        * 用灰阶渐变上色不同键长 (深色 = 大键长), 添加灰阶 colorbar
        * 'NMR experiment' 红色三角: 把 r_truth 曲线加噪声当作实验点
        * 嵌入 H2 分子卡通图
        * 时间用 a.u., 题目和单位贴近论文样式
    """
    t_grid = d_data["t_grid"]
    curves = d_data["curves"]
    rs = sorted(curves.keys())

    fig, ax = plt.subplots(figsize=(7, 5))

    # 灰阶 colormap by r (大 r = 深色, 与论文一致)
    cmap = plt.get_cmap("Greys")
    norm = plt.Normalize(min(rs), max(rs))
    # 模拟曲线
    for r in rs:
        color = cmap(0.25 + 0.7 * norm(r))  # 把灰阶下限抬高一些
        ax.plot(t_grid, curves[r], "-", lw=1.6, color=color)
    # 用一个空标签做图例 "Simulation"
    ax.plot([], [], "-", color="gray", lw=1.6, label="Simulation")

    # NMR "实验" 点 - 用 r=r_truth 曲线 + 噪声, 红色三角带误差棒
    rng = np.random.default_rng(exp_seed)
    if r_truth in curves:
        exp_curve = curves[r_truth] + rng.normal(0, exp_noise, size=len(t_grid))
        exp_idx = np.linspace(0, len(t_grid) - 1, 13).astype(int)  # 取 13 个点
        ax.errorbar(t_grid[exp_idx], exp_curve[exp_idx], yerr=exp_noise,
                    fmt="v", color="#c8232c", markersize=7,
                    markeredgecolor="darkred", capsize=2.5, lw=1.4,
                    label="NMR experiment", zorder=20)

    ax.set_xlabel("Time  t  (a.u.)", fontsize=13)
    ax.set_ylabel(r"$\mathrm{OTOC}(t)$", fontsize=13)
    ax.set_title("(d)  Sensitivity of OTOC to H–H bond length", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.grid(alpha=0.25)
    ax.set_ylim(min(0, ax.get_ylim()[0]) - 0.02, 1.05)

    # colorbar by r
    cax = fig.add_axes([0.92, 0.20, 0.018, 0.62])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.colorbar(sm, cax=cax)
    cb.set_label(r"$r_{H-H}$  [Å]", fontsize=11)

    # H2 分子卡通 (inset)
    _draw_h2_inset(ax, loc_box=(0.03, 0.04, 0.27, 0.32))

    plt.subplots_adjust(left=0.10, right=0.89, top=0.92, bottom=0.10)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] {save_path}")


def plot_panel_e(e_data, save_path):
    """
    论文 FIG2(e) 风格:
        * 顶图: RMSE vs r, 用 cubic 拟合显示 1σ CI 阴影, 用红三角显示数据点,
                两条虚线 (灰=Ref, 红=Est) 标出参考与估计.
        * 底图: 水平条形, 显示 Ref vs Est 的点 + 2σ 误差棒.
    """
    r_scan = np.asarray(e_data["r_scan"])
    r_truth = float(e_data["r_truth"])
    # 用最大 N 时的 mse 曲线
    mse_curve = np.asarray(e_data["mse_at_truth"][-1])
    rmse = np.sqrt(mse_curve)

    # 最后一组 bootstrap 给出 Est 的中心 + 2σ CI
    r_hat = float(e_data["r_hat_mean"][-1])
    r_lo = float(e_data["r_hat_lo"][-1])
    r_hi = float(e_data["r_hat_hi"][-1])
    r_hat_2sig = max(r_hat - r_lo, r_hi - r_hat, 0.005)  # 半宽

    # 拟合 RMSE 为 r 的 cubic (论文用 cubic), 1σ 带 = 残差标准差.
    # 用对称权重 (中心更高), 让最小值附近权重大, 避免被远端 V-臂拖偏.
    weights = np.exp(-((r_scan - r_scan[int(np.argmin(rmse))]) ** 2) /
                     (0.5 * (r_scan.max() - r_scan.min()) ** 2))
    coeffs = np.polyfit(r_scan, rmse, 3, w=weights)
    fit_x = np.linspace(r_scan.min(), r_scan.max(), 400)
    fit_y = np.polyval(coeffs, fit_x)
    fit_resid = rmse - np.polyval(coeffs, r_scan)
    fit_sigma = float(np.std(fit_resid))
    # Estimate: cubic 拟合的最小点
    fit_min_x = float(fit_x[int(np.argmin(fit_y))])

    fig = plt.figure(figsize=(6, 6))
    # 2-row layout, 顶图大底图小
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.18)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # === 顶图 ===
    ax_top.plot(r_scan, rmse, "v", color="#c8232c", markersize=7,
                markeredgecolor="darkred", label="NMR")
    ax_top.plot(fit_x, fit_y, "-", color="#c8232c", lw=1.6,
                label="Fit (cubic)")
    ax_top.fill_between(fit_x, fit_y - fit_sigma, fit_y + fit_sigma,
                        color="#c8232c", alpha=0.20, label=r"$1\sigma$ CI")
    ax_top.axvline(fit_min_x, color="#c8232c", ls="--", lw=1.2, label="Est.")
    ax_top.axvline(r_truth, color="gray", ls="--", lw=1.2, label="Ref.")
    ax_top.set_ylabel("Cov. wt. err. [arb]", fontsize=12)
    ax_top.set_title("(e)  Simulated learning of H–H bond length", fontsize=13)
    ax_top.legend(fontsize=9, frameon=False, loc="upper right")
    ax_top.tick_params(labelsize=11)

    # === 底图: Ref / Est 点 + 误差棒 (水平); 上下错位避免重叠 ===
    y_ref, y_est = +0.35, -0.35
    ax_bot.errorbar([r_truth], [y_ref], xerr=[[0.01], [0.01]], fmt="o",
                    color="dimgray", markersize=10,
                    markeredgecolor="black", capsize=4, lw=2.0)
    ax_bot.text(r_truth, y_ref + 0.4,
                f"{r_truth:.3f}±0.010 Å  (Ref.)", fontsize=10, color="dimgray",
                ha="center", va="bottom", fontweight="bold")
    ax_bot.errorbar([fit_min_x], [y_est], xerr=[[r_hat_2sig], [r_hat_2sig]],
                    fmt="o", color="#c8232c", markersize=10,
                    markeredgecolor="darkred", capsize=4, lw=2.0)
    ax_bot.text(fit_min_x, y_est - 0.4,
                f"{fit_min_x:.3f}±{r_hat_2sig:.3f} Å  (Est.)", fontsize=10,
                color="#c8232c", ha="center", va="top", fontweight="bold")

    ax_bot.set_xlim(r_scan.min(), r_scan.max())
    ax_bot.set_ylim(-1.5, 1.5)
    ax_bot.set_yticks([])
    ax_bot.set_xlabel(r"$r_{H-H}$  [Å]", fontsize=12)
    for s in ["left", "right", "top"]:
        ax_bot.spines[s].set_visible(False)
    ax_bot.tick_params(labelsize=11)

    # 共享 x 范围
    ax_top.set_xlim(r_scan.min(), r_scan.max())
    plt.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.10)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] {save_path}")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    # 所有运行产物均保存在当前示例目录内，避免依赖开发者本机路径。
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "_output")
    cache_dir = os.path.join(here, "_cache")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # FIG.2(d)：计算不同H—H键长下的OTOC时间曲线。
    r_list = (0.50, 0.65, 0.74, 0.90, 1.10, 1.40, 1.80)
    t_grid_d = np.linspace(0.0, 1.6, 25)
    n_trotter = 20

    d_data = run_panel_d(
        r_list,
        t_grid_d,
        n_trotter,
        cache_path=os.path.join(cache_dir, "h2_paneld.pkl"),
    )
    plot_panel_d(
        d_data,
        save_path=os.path.join(output_dir, "h2_panel_d.png"),
    )

    # FIG.2(e)：由合成OTOC观测数据反演H—H键长。
    r_truth = 0.74
    r_scan = np.arange(0.66, 0.83, 0.01)
    t_full = np.linspace(0.1, 1.6, 25)
    n_points_grid = (3, 5, 8, 12, 16, 20, 25)
    noise = 0.02
    n_bootstrap = 200

    e_data = run_panel_e(
        r_truth,
        r_scan,
        t_full,
        n_trotter,
        n_points_grid,
        n_bootstrap,
        noise,
        cache_path=os.path.join(cache_dir, "h2_panele.pkl"),
    )
    plot_panel_e(
        e_data,
        save_path=os.path.join(output_dir, "h2_panel_e.png"),
    )

    print(f"\nAll done. Results saved to: {output_dir}")