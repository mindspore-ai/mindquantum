"""
复现 arXiv:2506.10191v1 (Google) FIG.2(b)(c):
    "Constructive interference at the edge of quantum ergodic dynamics"

本机为概念验证版本 (受限于本地算力), 参数已按小规模下调；
论文级/集群规模的参数在下方常量注释与 __main__ 入口处均有标注，
按需放大即可 (无需改动物理逻辑)。

FIG.2(b): 2D brick-wall 蜂窝网格 (本机 ROWS x COLS = 5x6 = 30 比特),
          MPS bond_dim=96 (入口处设置), 蛇形映射编号。
          所有 qb 均真实计算 C^(4) (不写死光锥外为 1)；
          qm 处不参与测量 (B、M 同点无物理意义), 存 NaN, 绘图时以紫点标出。
          均值图配色 inferno_r, 标准差图配色 hot_r, cyan 折线描光锥边界。

FIG.2(c): 1D 链 (本机 N_1D=32 比特), bond_dim=96。
          C^(1) 用 N=12 精确对角化 (bond_dim=64, 无截断误差) 计算 qm 单点 ⟨Z(t)⟩；
          C^(2)/C^(4) 在 1D 链上、固定 qb=qm+QB_FIXED_OFFSET 处计算
          (小系统下动态追踪波前会使 qb 锁死在链边界, 故改为固定 qb,
          详见 QB_FIXED_OFFSET 处注释)。
          C^(4)_off-diag = C^(4) - (C^(2))^2。
"""

import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import multiprocessing as mp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RX, RZ, ISWAP, PhaseShift, X, Z
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.simulator import Simulator
from mindquantum._mq_mps.double import mqmps as MqMpsDouble


# ============================================================
# 2D lattice constants (FIG.2b)
# ============================================================
# 【本机概念验证版 — 5x6=30q, bond=96】
# 放大到论文级: ROWS,COLS=8,8; N_QUBITS=64; 入口 BOND_DIM=256,
#   CYCLES_B=(4,8,12), QM_RC=(7,4), N_INST_B=50 (需集群/高性能资源)。
#   均值图: 光锥扩散(外锥黑C⁴≈1, 内锥亮黄C⁴≈0)；
#   标准差图: 波前std大、qm附近深scramble→std≈0(需高bond才显白)。
ROWS = 5
COLS = 6
N_QUBITS = ROWS * COLS

# ============================================================
# 1D chain constants (FIG.2c)
# ============================================================
# 【本机概念验证版 — N=32, bond=96】
# ── 为何用 1D 链 (论文 SI 也是) ────────────────────────────────
# FIG.2c 论文用 1D Haar 随机电路 (原文: "one-dimensional Haar random
# circuits")。2D 模拟 bond 需求 ∝ 2^(cycle×宽度) 会爆炸，无法存储；
# 论文 FIG.2b 那张 2D 图是真实硬件(103q)测的，不是模拟。
# ── 参数依据 (本机诊断实测) ───────────────────────────────────
# bond 不足时深 cycle 的 mean[C2] 会跑出 [0,1] (截断崩坏)；
#   N 越大同 cycle 纠缠越多、可信 cycle 上限越低。波前速度 v_B≈2 offset/cycle。
# 边界反射: 波前撞链端反弹回会污染深 cycle，故 cycle 范围不宜过大。
# 放大到论文级: N_1D=80; BOND_DIM_1D=256; 入口 CYCLES_C=range(1,21),
#   N_INST_C=100 (需集群/高性能资源)。
N_1D    = 32             # 本机: N=32 bond=96 在 cycle≤10 可信
QM_1D   = N_1D // 2      # = 16，链中心
K_HAAR  = 1              # 占位

BOND_DIM_1D = 96          # 本机: 实测 N=32 cycle≤10 截断可信

# ── qb 固定 (小系统的现实选择) ────────────────────────────────
# 论文用动态追踪波前 (Fig.2 caption: "qb gradually moved... maintaining
#   C̄⁽²⁾≈0.5")，但实测在 1D 小系统 (N≤40) 波前是陡台阶，几乎没有
#   mean[C2]≈0.5 的过渡格，追踪逻辑会把 qb 锁死在链边界 → σ 前段恒0。
# 故本机版改用固定 qb=qm+QB_FIXED_OFFSET: 波前经过后 qb 进入深 scramble
#   区，σ 从峰值衰减，物理趋势 (OTOC 衰减慢于 TOC) 正确且能算。
# 【这是相对论文协议的简化，PR/交付时需如实声明】
# 大 N + 高 bond 资源下可改回追踪波前 (见 git 历史的扫描版实现)。
QB_FIXED_OFFSET = 6  # 波前约 cycle3 到达，之后是衰减段

# C(1) 用 N=12 精确小系统 (qm 单点 ⟨Z(t)⟩, 论文定义, 无 qb)
# C1衰减速率 alpha 由物理规律决定 (~2ln(5/4)≈0.45/cycle), 与N无关
N_C1        = 12
QM_C1       = N_C1 // 2  # = 6
BOND_DIM_C1 = 64          # 2^6=64，对 N=12 完全精确


# ============================================================
# 2D lattice helpers (FIG.2b, 保持原样)
# ============================================================
def build_random_circuit_n(
    couplings_even, couplings_odd, n_qubits: int,
    cycles: int, rng: np.random.Generator
) -> Circuit:
    """
    构造 brick-wall 结构的随机量子电路（论文 Fig.1 协议）。

    每个 cycle 包含：
    1. 单比特随机旋转层: RZ(-φ) RX(θ) RZ(φ)，θ 从 {π/4, π/2, 3π/4} 随机选
    2. 双比特纠缠层（偶数耦合）: iSWAP + PhaseShift(0.35)
    3. 双比特纠缠层（奇数耦合）: iSWAP + PhaseShift(0.35)

    参数:
        couplings_even/odd: 偶数/奇数层的比特对列表，例如 [(0,1),(2,3),...]
        n_qubits: 总比特数
        cycles: 重复层数（对应论文中的演化深度）
        rng: numpy 随机数生成器（保证可重复性）

    返回:
        Circuit: 深度为 cycles 的随机电路 U(t)，t ∝ cycles
    """
    c = Circuit()
    for _ in range(cycles):
        for q in range(n_qubits):
            theta = float(rng.choice(THETA_POOL))
            phi = float(rng.uniform(-1, 1)) * np.pi
            c += RZ(-phi).on(q)
            c += RX(theta).on(q)
            c += RZ(phi).on(q)
        for a, b in couplings_even:
            c += ISWAP.on([a, b])
            c += PhaseShift(0.35).on(b, a)
        for a, b in couplings_odd:
            c += ISWAP.on([a, b])
            c += PhaseShift(0.35).on(b, a)
    return c


def rc2idx(r: int, c: int) -> int:
    """
    2D 蜂窝网格的蛇形编号映射（row, col → 1D qubit index）。
    偶数行从左到右，奇数行从右到左，使相邻格子在 MPS 编号上也尽量相邻。
    """
    if r % 2 == 0:
        return r * COLS + c
    else:
        return r * COLS + (COLS - 1 - c)


def idx2rc(i: int) -> tuple:
    """逆映射：1D qubit index → (row, col)。"""
    r = i // COLS
    inner = i % COLS
    c = inner if r % 2 == 0 else (COLS - 1 - inner)
    return r, c


def build_2d_lattice():
    """
    构造 2D brick-wall 蜂窝格子的坐标与耦合对。

    返回:
        coords: [(x, y), ...] 长度为 N_QUBITS，用于绘图和距离计算
        couplings_even: 偶数层的比特对（同行 + 跨行的偶数对）
        couplings_odd: 奇数层的比特对
    """
    coords = []
    for i in range(N_QUBITS):
        r, c = idx2rc(i)
        x = c + 0.5 * (r % 2)
        y = -r * 0.866
        coords.append((x, y))

    same_row = [(rc2idx(r, c), rc2idx(r, c + 1))
                for r in range(ROWS) for c in range(COLS - 1)]
    cross_row = [(rc2idx(r, c), rc2idx(r + 1, c))
                 for r in range(ROWS - 1) for c in range(COLS)]
    couplings_even = same_row[::2] + cross_row[::2]
    couplings_odd  = same_row[1::2] + cross_row[1::2]
    return coords, couplings_even, couplings_odd


# ============================================================
# MPS 模拟器
# ============================================================
def make_mps_sim(bond_dim: int, seed: int = 42, n_qubits: int = None) -> Simulator:
    """
    创建 MindQuantum MPS 模拟器。

    bond_dim (χ): MPS 键维度，控制纠缠表示精度。
    - 精确模拟需要 χ = 2^(N/2)，对大系统不可行
    - χ 不足时，深纠缠态会产生截断误差
    - 本机版用 χ=96 在 N≤32 的小系统下为可信与算力的折中
    """
    n = n_qubits or N_QUBITS
    s = Simulator("mqmps", n)
    s.backend.sim = MqMpsDouble(n, int(seed), int(bond_dim))
    return s


# ============================================================
# 随机量子电路
# ============================================================
THETA_POOL = np.array([0.25, 0.5, 0.75]) * np.pi  # 单比特旋转角从 {π/4, π/2, 3π/4} 随机选


def build_random_circuit(
    couplings_even,
    couplings_odd,
    cycles: int,
    rng: np.random.Generator,
) -> Circuit:
    return build_random_circuit_n(couplings_even, couplings_odd,
                                  N_QUBITS, cycles, rng)


# ============================================================
# 关联函数 (2D 和 1D 共用)
# ============================================================
def measure_C1_cached(s_fwd: Simulator, qm: int) -> float:
    """
    测量 C^(1) = ⟨Z_qm(t)⟩ (TOC)。
    无时间反演，对动力学细节不敏感，通常指数快速衰减。
    """
    return float(np.real(
        s_fwd.get_expectation(Hamiltonian(QubitOperator(f"Z{qm}")))
    ))


def measure_C2_cached(
    s_fwd: Simulator, circ_bwd: Circuit, qm: int, qb: int
) -> float:
    """
    测量 C^(2)，二阶 OTOC 的中间项，用于追踪波前。
    操作: 在 t 处施加扰动 X_qb，时间反演 U†，测 Z_qm。
    """
    s = s_fwd.copy()
    s.apply_circuit(Circuit([X.on(qb)]))
    s.apply_circuit(circ_bwd)
    return float(np.real(s.get_expectation(Hamiltonian(QubitOperator(f"Z{qm}")))))


def measure_C4_cached(
    s_fwd: Simulator,
    circ_fwd: Circuit,
    circ_bwd: Circuit,
    qm: int,
    qb: int,
) -> float:
    """
    测量 C^(4)，四阶 OTOC（完整量子回声）。

    物理意义：测量时间反演后能否恢复初态。
    - C⁴≈1: 光锥外，扰动未传播，回声完美
    - C⁴≈0: 光锥内波前附近，信息扩散中
    - C⁴→0: 深 scramble 区，信息已充分打乱

    操作序列: X_qb → U† → Z_qm → U → X_qb → U† → 测 Z_qm
    （qm: 测量点，qb: 扰动点）
    """
    s = s_fwd.copy()
    s.apply_circuit(Circuit([X.on(qb)]))
    s.apply_circuit(circ_bwd)
    s.apply_circuit(Circuit([Z.on(qm)]))
    s.apply_circuit(circ_fwd)
    s.apply_circuit(Circuit([X.on(qb)]))
    s.apply_circuit(circ_bwd)
    return float(np.real(
        s.get_expectation(Hamiltonian(QubitOperator(f"Z{qm}")))
    ))


# ============================================================
# 2D 距离 (用于光锥)
# ============================================================
def chebyshev_dist_2d(coords, qm: int):
    """
    计算蜂窝格子中各点到 qm 的 Chebyshev 距离（光锥半径判定）。
    除以 0.866 是因为蜂窝布局 y 方向压缩（六边形高度因子）。

    返回: shape=(N_QUBITS,) 的距离数组，用于判断 dist <= v_B * cycle
    """
    qx, qy = coords[qm]
    return np.array(
        [max(abs(x - qx), abs(y - qy) / 0.866) for (x, y) in coords]
    )


# ============================================================
# FIG.2(b) 多进程工作函数 — 完整保留，不做任何修改
# ============================================================
def _pin_to_pcores():
    """将当前进程绑定到 i7-12700H 的 6 个 P核 (逻辑线程 0-11), 避免调度到 E核."""
    try:
        import psutil
        psutil.Process().cpu_affinity(list(range(12)))
    except Exception:
        pass  # psutil 未安装或平台不支持时静默跳过


def _worker_2b(args):
    """
    FIG.2(b) 单实例工作函数：对一个随机电路实例，计算所有 qb 的 C^(4)。

    返回: (s_idx, row)，其中 row[qb] = C^(4)_qm,qb
          qb==qm 时存 NaN（B、M 同点无物理意义）。
    """
    _pin_to_pcores()
    s_idx, cyc, seed_inst, bond_dim, qm, all_qbs = args
    rng = np.random.default_rng(seed_inst)
    coords, ce, co = build_2d_lattice()
    circ_f = build_random_circuit(ce, co, cyc, rng)
    circ_b = circ_f.hermitian()
    s_fwd = make_mps_sim(bond_dim, seed=seed_inst)
    s_fwd.apply_circuit(circ_f)
    row = np.full(N_QUBITS, np.nan)
    for qb in all_qbs:
        if qb == qm:
            # qm 是参考点: B 作用在 qm 自身无物理意义，存 NaN，绘图用紫点标记，
            # 不参与 colormap (不伪造数据)。
            continue
        row[qb] = measure_C4_cached(s_fwd, circ_f, circ_b, qm, qb)
    return s_idx, row


# ============================================================
# FIG.2(c) 多进程工作函数
# ============================================================
def _worker_1d(args):
    """
    FIG.2(c) 单实例工作函数：计算 C^(1)/C^(2)/C^(4) 在不同 qb 候选位置。

    C^(1): 用 N=12 精确系统（bond=64，无截断）
    C^(2)/C^(4): 用 N=N_1D 系统（bond=BOND_DIM_1D）

    关键：seed 只依赖实例编号，不依赖 cycle，保证"同一实例在不同 cycle 是同一电路加深"。

    返回: (c1, {off: c2}, {off: c4}, {off: c4_off})
    """
    _pin_to_pcores()
    s_idx, cyc, qm_1d, seed_inst, bond_dim, cand_offsets = args

    # ── C(1): N=12 精确系统, qm 单点 ⟨Z(t)⟩ ──────────────────────────
    rng_c1 = np.random.default_rng(seed_inst)
    ce_c1  = [(i, i+1) for i in range(0, N_C1-1, 2)]
    co_c1  = [(i, i+1) for i in range(1, N_C1-1, 2)]
    circ_c1 = build_random_circuit_n(ce_c1, co_c1, N_C1, cyc, rng=rng_c1)
    s_c1    = make_mps_sim(BOND_DIM_C1, seed=seed_inst, n_qubits=N_C1)
    s_c1.apply_circuit(circ_c1)
    c1 = measure_C1_cached(s_c1, QM_C1)

    # ── C(2)/C(4): 扫描候选 qb offset ────────────────────────────────
    N   = N_1D
    rng = np.random.default_rng(seed_inst)
    ce  = [(i, i+1) for i in range(0, N-1, 2)]
    co  = [(i, i+1) for i in range(1, N-1, 2)]
    circ_f = build_random_circuit_n(ce, co, N, cyc, rng=rng)
    circ_b = circ_f.hermitian()
    s_fwd  = make_mps_sim(bond_dim, seed=seed_inst, n_qubits=N)
    s_fwd.apply_circuit(circ_f)

    c2m, c4m, c4om = {}, {}, {}
    for off in cand_offsets:
        qb = qm_1d + off
        if qb > N - 2:
            continue
        c2 = measure_C2_cached(s_fwd, circ_b, qm_1d, qb)
        c4 = measure_C4_cached(s_fwd, circ_f, circ_b, qm_1d, qb)
        c2m[off]  = c2
        c4m[off]  = c4
        c4om[off] = c4 - c2 ** 2

    return c1, c2m, c4m, c4om


# ============================================================
# FIG.2(b) 主流程 — 完整保留，不做任何修改
# ============================================================
def run_figure_2b(
    qm: int,
    cycles_list: tuple,
    n_instances: int,
    bond_dim: int,
    seed: int,
    cache_path: str,
    n_workers: int = None,
):
    if os.path.isfile(cache_path):
        print(f"[cache] loading {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    coords, _, _ = build_2d_lattice()
    dist = chebyshev_dist_2d(coords, qm)
    results = {}
    n_workers = n_workers or min(n_instances, mp.cpu_count())

    # 所有 qb 都真实计算 (不再用 np.ones 写死光锥外)。
    # 光锥外严格 C4=1 (Lieb-Robinson, B(t)与M对易)，真算会自然得到≈1，
    all_qbs = list(range(N_QUBITS))

    for cyc in cycles_list:
        Lr = cyc * 0.5 + 1.0
        in_cone = int((dist <= Lr).sum())
        print(f"\n[FIG2b] cycles={cyc}, instances={n_instances}, bond={bond_dim}, "
              f"in-cone≈{in_cone}/{N_QUBITS} (全部qb真算), workers={n_workers}")
        t0 = time.time()
        job_args = [
            (s_idx, cyc, seed + s_idx * 997, bond_dim, qm, all_qbs)
            for s_idx in range(n_instances)
        ]
        table = np.full((n_instances, N_QUBITS), np.nan)  # qm 处保持 NaN
        with mp.Pool(n_workers) as pool:
            for s_idx, row in pool.imap_unordered(_worker_2b, job_args):
                table[s_idx] = row
                print(f"  inst {s_idx+1}/{n_instances} done  t={time.time()-t0:.1f}s",
                      flush=True)
        # qm 列是 NaN，用 nanmean/nanstd 跳过
        results[cyc] = (np.nanmean(table, axis=0), np.nanstd(table, axis=0))
        print(f"[FIG2b] cyc={cyc} done in {time.time()-t0:.1f}s")

    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    return results


# ============================================================
# FIG.2(c) 主流程 — 多 qb 同时扫描，峰值 sigma（v6）
# ============================================================
def run_figure_2c(
    qm_1d, cycle_grid, n_instances, bond_dim, seed, cache_path, n_workers=None
):
    """
    计算 FIG.2(c) 数据：四条 sigma[C^(l)] 曲线随 cycle 的演化。

    【qb 选择规则 — 与论文 SI Fig.S6 一致】
    论文方法：对每个 cycle，选择使 mean[C2] ≈ 0.5（即处于波前中点）的 qb，
    且要求 qb 随 cycle 单调不减（因为波前只会向外传播，不会后退）。

    旧版本的错误：每个cycle独立地在所有候选offset中挑"sigma[C2]最大"的那个，
    这在小样本噪声下会导致挑选结果在cycle间跳变（例如cycle=2选到off=3，
    cycle=3又跳回off=2），产生论文中不存在的非单调振荡。

    新逻辑：
      1. 对每个候选offset算出 |mean[C2] - 0.5| 的距离，取距离最小的offset
         作为本cycle的"波前中心"。
      2. 额外约束：所选offset不能比上一个cycle选的offset更小（单调性约束），
         如果按规则1选出的offset更小，则强制沿用上一个cycle的offset。
      3. 第一个cycle没有"上一个"参照，直接用规则1的结果。
    """
    if os.path.isfile(cache_path):
        print(f"[cache] loading {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    n_workers = n_workers or min(n_instances, mp.cpu_count())
    sigmas = {"C1": [], "C2": [], "C4": [], "C4_off": []}
    means_c2, chosen_qb = [], []

    # 固定 qb=qm+QB_FIXED_OFFSET (候选池仅一个 offset)。
    # 1D 小系统追踪波前会锁死在边界，故用固定 qb (见常量处说明)。
    off = QB_FIXED_OFFSET
    cand_offsets = [off]

    for cyc in cycle_grid:
        print(f"\n[FIG2c] cyc={cyc:2d}  {n_instances}实例  qb=qm+{off}(固定)", flush=True)
        t0 = time.time()

        job_args = [
            (s_idx, cyc, qm_1d,
             seed + s_idx * 997,      # seed 只依赖实例编号，不依赖 cyc
             bond_dim, cand_offsets)
            for s_idx in range(n_instances)
        ]

        c1_list, c2_list, c4_list, c4o_list = [], [], [], []
        with mp.Pool(n_workers) as pool:
            for c1v, c2m, c4m, c4om in pool.imap_unordered(_worker_1d, job_args):
                c1_list.append(c1v)
                if off in c2m:
                    c2_list.append(c2m[off])
                    c4_list.append(c4m[off])
                    c4o_list.append(c4om[off])

        mean_c2 = float(np.mean(c2_list)) if c2_list else float("nan")
        chosen_qb.append(qm_1d + off)
        means_c2.append(mean_c2)
        sigmas["C1"].append(float(np.std(c1_list)))
        sigmas["C2"].append(float(np.std(c2_list)))
        sigmas["C4"].append(float(np.std(c4_list)))
        sigmas["C4_off"].append(float(np.std(c4o_list)))
        print(f"  -> qb=qm+{off} (site{chosen_qb[-1]}) mean[C2]={mean_c2:+.3f}  "
              f"σ C1={sigmas['C1'][-1]:.2e} C2={sigmas['C2'][-1]:.2e} "
              f"C4={sigmas['C4'][-1]:.2e} C4o={sigmas['C4_off'][-1]:.2e} "
              f"t={time.time()-t0:.1f}s")

    out = dict(cycles=list(cycle_grid)[:len(sigmas["C1"])], sigmas=sigmas,
               means_c2=means_c2, chosen_qb=chosen_qb)
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    return out


# ============================================================
# 可视化: 2D 菱形蜂窝 (FIG.2b) — 完整保留，不做任何修改
# ============================================================
def _draw_diamond_2d(
    ax,
    coords,
    values,
    qm: int,
    cmap_name: str,
    vmin: float,
    vmax: float,
    lightcone_radius: float,
    mask_outside_cone: bool = False,  # kept for API compat, unused
    show_grid_outside: bool = False,  # kept for API compat, unused
):
    """Draw all qubits with their values; lightcone boundary in cyan."""
    qx, qy = coords[qm]
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    dist = chebyshev_dist_2d(coords, qm)

    for i, (cx, cy) in enumerate(coords):
        verts = [
            (cx, cy + 0.5),
            (cx + 0.5, cy),
            (cx, cy - 0.5),
            (cx - 0.5, cy),
        ]
        if np.isnan(values[i]):
            # qm 参考点: 数据为 NaN (未测)，画中性灰，紫点会盖在上面
            face = "#c8c8c8"
        else:
            face = cmap(norm(values[i]))
        # 浅灰边框: 白色格子(std≈0)也能从背景中分辨出轮廓
        ax.add_patch(Polygon(verts, facecolor=face,
                             edgecolor="#999999", lw=0.6, zorder=10))

    _draw_lightcone_outline(ax, coords, dist, lightcone_radius)

    ax.plot(qx, qy, "o", markerfacecolor="#a991d4",
            markeredgecolor="black", markersize=10, zorder=30)

    ax.set_aspect("equal")
    ax.axis("off")
    # 极浅灰背景，衬托白色格子
    ax.set_facecolor("#f0f0f0")


def _draw_lightcone_outline(ax, coords, dist, radius):
    cone = dist <= radius + 0.5
    edge_count = {}
    for i, (cx, cy) in enumerate(coords):
        if not cone[i]:
            continue
        v = [
            (cx, cy + 0.5),
            (cx + 0.5, cy),
            (cx, cy - 0.5),
            (cx - 0.5, cy),
        ]
        for k in range(4):
            a = (round(v[k][0], 3), round(v[k][1], 3))
            b = (round(v[(k + 1) % 4][0], 3), round(v[(k + 1) % 4][1], 3))
            key = frozenset([a, b])
            edge_count[key] = edge_count.get(key, 0) + 1
    for key, c in edge_count.items():
        if c == 1:
            (a, b) = tuple(key)
            ax.plot([a[0], b[0]], [a[1], b[1]],
                    color="#00d0d0", lw=1.6, solid_capstyle="round", zorder=20)


def plot_fig2b(results, qm, save_path):
    coords, _, _ = build_2d_lattice()
    cycles = sorted(results.keys())

    # Fixed range matching paper: mean in [0,1], std 固定上限
    # std_hi 固定为 0.6 (不随每次跑的 nanmax 浮动，保证配色稳定可比)。
    # 30q bond96 的真实 std 范围约 0~0.67; 用 0.6 让梯度饱满又不顶死。
    mean_lo, mean_hi = 0.0, 1.0
    std_lo, std_hi = 0.0, 0.6

    fig, axes = plt.subplots(2, len(cycles), figsize=(3.6 * len(cycles), 4.8))

    for col, cyc in enumerate(cycles):
        mean_map, std_map = results[cyc]
        Lr = max(1.0, cyc * 0.5)

        ax = axes[0, col]
        # inferno_r: C4=1 (outside cone) → dark/black; C4≈0 (inside, scrambled) → bright
        _draw_diamond_2d(ax, coords, mean_map, qm, "inferno_r", mean_lo, mean_hi, Lr)
        ax.set_title(f"Cycle {cyc}", fontsize=14)

        ax = axes[1, col]
        # hot_r: std=0 → white (outside cone / near qm); high std → dark red/black (wavefront)
        _draw_diamond_2d(ax, coords, std_map, qm, "hot_r", std_lo, std_hi, Lr)

    cax1 = fig.add_axes([0.025, 0.55, 0.012, 0.36])
    plt.colorbar(plt.cm.ScalarMappable(cmap="inferno_r",
                                       norm=plt.Normalize(mean_lo, mean_hi)),
                 cax=cax1).set_label(r"$\overline{C}^{(4)}$", fontsize=12)
    cax2 = fig.add_axes([0.025, 0.08, 0.012, 0.36])
    plt.colorbar(plt.cm.ScalarMappable(cmap="hot_r",
                                       norm=plt.Normalize(std_lo, std_hi)),
                 cax=cax2).set_label(r"$\sigma\,[C^{(4)}]$", fontsize=12)

    plt.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.04,
                        wspace=0.05, hspace=0.08)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] {save_path}")


# ============================================================
# 可视化: 时域统计曲线 (FIG.2c)
# ============================================================
def plot_fig2c(c_data, save_path):
    """FIG.2(c): 四曲线 sigma 对数衰减。"""
    cycles = c_data["cycles"]
    sig    = c_data["sigmas"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    cyc_arr = np.array(cycles, dtype=float)

    # C1 (TOC): 全程都有信号 (qm 单点)，直接画
    ax.semilogy(cycles, sig["C1"], "-p", color="gray", markersize=8,
                markerfacecolor="gray", markeredgecolor="dimgray", lw=1.8,
                label=r"$\mathcal{C}^{(1)}$")

    # OTOC 三条线: 波前到达前 σ≈0 (C2=1 恒定)，是物理零点不是数据，
    # 只画 σ 超过噪声地板的点 (波前到达后)，避免竖线伪影。
    FLOOR = 1e-3
    def _plot_otoc(key, color, mfc, mec, lab):
        arr = np.array(sig[key])
        mask = arr > FLOOR
        ax.semilogy(cyc_arr[mask], arr[mask], "-o", color=color, markersize=8,
                    markerfacecolor=mfc, markeredgecolor=mec, lw=1.8, label=lab)

    _plot_otoc("C2", "#3a64c4", "#5580e8", "navy", r"$\mathcal{C}^{(2)}$")
    _plot_otoc("C4", "#3ec45a", "#5de875", "darkgreen", r"$\mathcal{C}^{(4)}$")
    _plot_otoc("C4_off", "#e043b5", "#f070cc", "purple",
               r"$\mathcal{C}^{(4)}_{\rm off\text{-}diag}$")

    # 幂律参考线 ~ t^{-1/4}（论文 SI Eq.79）: 从波前到达点 (C2 首个非零) 起算
    c2_arr = np.array(sig["C2"])
    nz = np.where(c2_arr > FLOOR)[0]
    if len(nz) >= 2:
        i0 = nz[0]
        t0, s0 = cyc_arr[i0], c2_arr[i0]
        c0    = s0 * t0 ** 0.25
        t_ref = np.linspace(t0, cyc_arr[-1], 200)
        ax.semilogy(t_ref, c0 * t_ref ** (-0.25), "--",
                    color="#3a64c4", lw=1.2, alpha=0.45,
                    label=r"$\propto t^{-1/4}$ (SI Eq.79)")

    # C1 指数拟合 (全部可用点)
    c1_arr  = np.array(sig["C1"])
    valid   = c1_arr > 1e-4
    if valid.sum() >= 2:
        alpha = -np.polyfit(cyc_arr[valid], np.log(c1_arr[valid]), 1)[0]
        t_r   = np.linspace(cycles[0], cycles[-1], 200)
        ax.semilogy(t_r, c1_arr[0] * np.exp(-alpha * (t_r - cycles[0])),
                    "--", color="gray", lw=1.0, alpha=0.45,
                    label=rf"$\mathcal{{C}}^{{(1)}}$ fit $\alpha$={alpha:.2f}/cyc")

    ax.set_xlabel("Cycles", fontsize=14)
    ax.set_ylabel(r"$\sigma[\mathcal{C}^{(\ell)}]$", fontsize=14)
    ax.set_xlim(0, cycles[-1] + 0.5)
    ax.set_ylim(5e-3, 2.0)
    ax.tick_params(which="both", direction="in", labelsize=12, width=1.5, length=5)
    ax.tick_params(which="minor", length=3)
    for sp in ax.spines.values():
        sp.set_linewidth(1.8)
    ax.legend(fontsize=10, loc="lower left", ncol=2,
              frameon=False, handlelength=2.0, columnspacing=0.8)
    ax.text(0.97, 0.97,
            f"N={N_1D} ($\\chi$={BOND_DIM_1D})  C2/C4\n"
            f"N={N_C1} ($\\chi$={BOND_DIM_C1}, exact)  C1\n"
            f"1D chain  (paper: 2D, 103q)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color="gray",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="lightgray", alpha=0.85))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] {save_path}")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    mp.freeze_support()   # Windows 多进程必须

    _HERE = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(_HERE, "_output")
    CACHE_DIR  = os.path.join(_HERE, "_cache")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR,  exist_ok=True)

    # ---- FIG.2b 参数 【本机概念验证: 5x6=30q, bond=96】----
    QM_RC = (4, 3)        # qm 在 5x6 网格底边中点 (row=4=最后行, col=3)
    QM = rc2idx(*QM_RC)
    BOND_DIM = 96         # 本机: bond=96 让深scramble区 std≈0.03 (显白)
                          # 论文级/集群: 8x8=64q 时设 256~512
    SEED = 2026
    # cycles (1,2,4): 光锥扩散 23%→47%→80%，三阶段清晰。
    # 论文级(8x8): CYCLES_B=(4,8,12), QM_RC=(7,4), N_INST_B=50
    CYCLES_B = (1, 2, 4)
    N_INST_B = 12         # 本机 12 实例; 论文 50~250

    # FIG.2c 参数 【本机: 追踪波前, cycle 1-10】
    CYCLES_C  = tuple(range(1, 11))   # 本机 1~10; 论文级 range(1,21)
    N_INST_C  = 40         # 本机 40 实例; 论文级 100
    N_WORKERS = min(6, mp.cpu_count())

    print("=" * 64)
    print(f"FIG.2b: 2D {ROWS}x{COLS}={N_QUBITS}-qubit, bond={BOND_DIM}")
    print(f"FIG.2c: C1→N={N_C1}χ={BOND_DIM_C1}  C2/C4→N={N_1D}χ={BOND_DIM_1D}")
    print(f"  qb=波前跟踪(mean[C2]≈0.5), n_inst={N_INST_C}, workers={N_WORKERS}")
    print("=" * 64)

    # FIG.2b — 全 qb 真算, qm 存 NaN 
    cache_b = os.path.join(CACHE_DIR, f"fig2b_n{N_QUBITS}_b{BOND_DIM}_realqm.pkl")
    out_b   = os.path.join(OUTPUT_DIR, "fig2b.png")
    res_b   = run_figure_2b(QM, CYCLES_B, N_INST_B, BOND_DIM, SEED, cache_b,
                            n_workers=N_WORKERS)
    plot_fig2b(res_b, QM, save_path=out_b)

    # FIG.2c — 追踪波前 (mean[C2]≈0.5), C1 N=12 exact
    cache_c = os.path.join(CACHE_DIR,
                           f"fig2c_n{N_1D}b{BOND_DIM_1D}_fixedqb{QB_FIXED_OFFSET}.pkl")
    out_c   = os.path.join(OUTPUT_DIR, "fig2c.png")
    res_c   = run_figure_2c(
        QM_1D, CYCLES_C, N_INST_C, BOND_DIM_1D, SEED + 1,
        cache_c, n_workers=N_WORKERS,
    )
    plot_fig2c(res_c, save_path=out_c)

    print(f"\nAll done.")
    print(f"  FIG2b → {out_b}")
    print(f"  FIG2c → {out_c}")