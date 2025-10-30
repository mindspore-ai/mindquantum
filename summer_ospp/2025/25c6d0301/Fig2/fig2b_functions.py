"""
Fig2b 核心函数模块
包含 VQE vs SH-VQE 表达能力指标 Δt 比较的所有核心逻辑
"""

from typing import List, Tuple, Optional, Any, Callable, Dict
import numpy as np
import matplotlib.pyplot as plt

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RX, RY, RZ, Rxx, Ryy, Rzz, CNOT, H, U3
from mindquantum.core.parameterresolver import PRGenerator
from mindquantum.simulator import Simulator

# 常量
EPS_LOG = 1e-15  # 防止log(0)的小常数


# ===================== 电路构建函数 =====================

def build_param_luc_ansatz(n_qubits: int,
                           depth: int,
                           prefix: str = 'theta',
                           single_rot: str = 'rx-ry-rz',
                           entangler: str = 'rxx') -> Tuple[Circuit, List[str]]:
    """
    构造参数化 LUC 电路（作为 ansatz）：
      - 单比特层: 每层对每个比特加参数化旋转 (默认 'rx-ry-rz')
      - 两比特层: 砖墙耦合，按 entangler 选择 'rxx'(参数) 或 'cnot'(非参数)
    返回:
      (ansatz_circuit, 参数名列表)
    """
    assert n_qubits >= 1 and depth >= 0
    gen = PRGenerator(prefix)
    circ = Circuit()

    single_rot = single_rot.lower()
    entangler = entangler.lower()

    for l in range(depth):
        # --- 单比特参数层 ---
        for q in range(n_qubits):
            if single_rot == 'rx-ry-rz':
                circ += RX(gen.new(suffix=f'l{l}q{q}rx')).on(q)
                circ += RY(gen.new(suffix=f'l{l}q{q}ry')).on(q)
                circ += RZ(gen.new(suffix=f'l{l}q{q}rz')).on(q)
            elif single_rot == 'rx-ry':
                circ += RX(gen.new(suffix=f'l{l}q{q}rx')).on(q)
                circ += RY(gen.new(suffix=f'l{l}q{q}ry')).on(q)
            elif single_rot == 'ry-rz':
                circ += RY(gen.new(suffix=f'l{l}q{q}ry')).on(q)
                circ += RZ(gen.new(suffix=f'l{l}q{q}rz')).on(q)
            else:
                # 默认回退到 rx-ry-rz
                circ += RX(gen.new(suffix=f'l{l}q{q}rx')).on(q)
                circ += RY(gen.new(suffix=f'l{l}q{q}ry')).on(q)
                circ += RZ(gen.new(suffix=f'l{l}q{q}rz')).on(q)

        # --- 砖墙两比特层 ---
        start = l % 2
        for i in range(start, n_qubits - 1, 2):
            if entangler == 'rxx':
                circ += Rxx(gen.new(suffix=f'l{l}p{i}_{i+1}')).on([i, i+1])  # 参数化 Rxx
            else:
                # MindQuantum: .on(目标, 控制)
                circ += CNOT.on(i + 1, i)  # 非参数 CNOT
    # 标记为 ansatz
    circ.as_ansatz()  # 方便下游提取 ansatz 参数名等信息
    return circ, list(circ.ansatz_params_name)


def _random_disjoint_pairs_all2all(n_qubits: int,
                                   rng: np.random.Generator,
                                   force_crosscut: bool = True) -> List[Tuple[int, int]]:
    """
    相比 1D 砖墙，全连接随机配对能在同一深度内并行耦合更"分散"的比特对；
    强制包含一对跨越 A|B 的配对（A={0..n/2-1}, B={n/2..n-1}），
    能显著提升跨割纠缠的注入速率，从而更快逼近 t-design（Δ_t 更快→0）。
    """
    used = set()
    pairs = []

    # 强制跨割（若可行）
    if force_crosscut and n_qubits >= 2:
        left  = list(range(n_qubits // 2))
        right = list(range(n_qubits // 2, n_qubits))
        if left and right:
            a = int(rng.choice(left))
            b = int(rng.choice(right))
            pairs.append((a, b))
            used.update({a, b})

    # 其余随机不相交配对
    rem = [i for i in range(n_qubits) if i not in used]
    rng.shuffle(rem)
    for i in range(0, len(rem) - 1, 2):
        pairs.append((rem[i], rem[i + 1]))
    return pairs


def _add_single_rot(circ: Circuit, gen: PRGenerator, q: int, tag: str, single_rot: str):
    """
    单比特去对称层：在每层每个比特上放 RX–RY 或 RX–RY–RZ。
    目的：打破可交换/对称结构，增强与两比特核的非对易性 → 更快"搅乱"与混合。
    """
    sr = single_rot.lower()
    if sr == 'rx-ry':
        circ += RX(gen.new(suffix=f'{tag}_rx')).on(q)
        circ += RY(gen.new(suffix=f'{tag}_ry')).on(q)
    elif sr == 'ry-rz':
        circ += RY(gen.new(suffix=f'{tag}_ry')).on(q)
        circ += RZ(gen.new(suffix=f'{tag}_rz')).on(q)
    else:  # 默认 'rx-ry-rz'
        circ += RX(gen.new(suffix=f'{tag}_rx')).on(q)
        circ += RY(gen.new(suffix=f'{tag}_ry')).on(q)
        circ += RZ(gen.new(suffix=f'{tag}_rz')).on(q)


def build_param_luc_ansatz_strong(n_qubits: int,
                                         depth: int,
                                         prefix: str = 'theta',
                                         single_rot: str = 'rx-ry-rz',
                                         entangler: str = 'xyz-2p') -> Tuple[Circuit, List[str]]:
    """
    构造"更强但不复杂"的参数化 LUC（与原函数相同的输入/输出模式）。
    增强点：
      1) 随机全连接不相交配对 + 每层强制至少一对跨割（更快跨越 A|B 注水）；
      2) 两比特核用少参数的非对易组合：'xx-yy'（2参数）或 'xyz-2p'（2参数：XX=YY 与 ZZ 分别一参）或 'xyz-1p'（1参数）。
         它们都比单一 Rxx 更"强"，但远比完整 KAK 简单。
    """
    assert n_qubits >= 1 and depth >= 0
    gen = PRGenerator(prefix)
    circ = Circuit()

    ent = (entangler or 'xyz-2p').lower()
    sr  = (single_rot or 'rx-ry-rz').lower()
    rng = np.random.default_rng()  # 如需复现，可在外部设置全局种子

    for l in range(depth):
        # 1) 单比特去对称层
        for q in range(n_qubits):
            _add_single_rot(circ, gen, q, tag=f'l{l}q{q}', single_rot=sr)

        # 2) 全连接随机不相交配对（强制跨割）
        pairs = _random_disjoint_pairs_all2all(n_qubits, rng, force_crosscut=True)

        # 3) 两比特核（少参数但非对易 ⇒ 比单一 Rxx 更强）
        for (i, j) in pairs:
            if ent == 'xx-yy':
                # 2 参数：Rxx(θ1) + Ryy(θ2)（XX 与 YY 非对易 → 更强）
                circ += Rxx(gen.new(suffix=f'l{l}p{i}_{j}_xx')).on([i, j])
                circ += Ryy(gen.new(suffix=f'l{l}p{i}_{j}_yy')).on([i, j])
            elif ent == 'xyz-1p':
                # 1 参数（最简 Heisenberg 近似）：同一角度串联 Rxx, Ryy, Rzz
                name = gen.new(suffix=f'l{l}p{i}_{j}')
                circ += Rxx(name).on([i, j])
                circ += Ryy(name).on([i, j])
                circ += Rzz(name).on([i, j])
            elif ent == 'xyz-2p':
                # 2 参数（各向异性 Heisenberg）：θ_xy 给 XX/YY，共用；θ_z 给 ZZ
                name_xy = gen.new(suffix=f'l{l}p{i}_{j}_xy')
                name_z  = gen.new(suffix=f'l{l}p{i}_{j}_z')
                circ += Rxx(name_xy).on([i, j])
                circ += Ryy(name_xy).on([i, j])
                circ += Rzz(name_z).on([i, j])
            elif ent == 'rxx':
                # 兼容旧版：单一 Rxx（表达力较弱）
                circ += Rxx(gen.new(suffix=f'l{l}p{i}_{j}')).on([i, j])
            else:
                # 兼容旧版：CNOT（非参数）
                # MindQuantum: .on(目标, 控制)
                circ += CNOT.on(j, i)

    circ.as_ansatz()
    return circ, list(circ.ansatz_params_name)


def _cz_on(circ: Circuit, ctrl: int, tgt: int):
    """
    在 (ctrl, tgt) 上施加 CZ。
    为兼容性稳妥，这里用分解: CZ = (I ⊗ H) · CNOT(ctrl→tgt) · (I ⊗ H)。
    MindQuantum 的 CNOT.on(目标, 控制)，所以 CNOT.on(tgt, ctrl)。
    """
    circ += H.on(tgt)
    circ += CNOT.on(tgt, ctrl)
    circ += H.on(tgt)


def build_hardware_efficient_luc(n_qubits: int,
                                 depth: int,
                                 prefix: str = 'theta',
                                 single_rot: str = 'u3',     # 'u3' 或 'rx-ry-rz'
                                 entangler: str = 'cz'       # 预留接口，当前固定用 'cz'
                                 ) -> Tuple[Circuit, List[str]]:
    """
    硬件高效（Hardware‑Efficient）LUC：
      每层 = [单比特任意旋转层] + [CZ 纠缠层(砖墙式)]
    - 单比特层：默认 U3(θ, φ, λ)；也可选 'rx-ry-rz' 欧拉分解（通用到全局相位）。
    - CZ 层：偶层耦合 (0,1),(2,3)...；奇层耦合 (1,2),(3,4)...
    返回:
      (ansatz_circuit, 参数名列表)
    """
    assert n_qubits >= 1 and depth >= 0
    circ = Circuit()
    gen = PRGenerator(prefix)
    sr  = (single_rot or 'u3').lower()

    for l in range(depth):
        # ---- ① 单比特任意旋转层（参数向量 θ⃗） ----
        for q in range(n_qubits):
            if sr == 'rx-ry-rz':
                circ += RX(gen.new(suffix=f'l{l}q{q}_rx')).on(q)
                circ += RY(gen.new(suffix=f'l{l}q{q}_ry')).on(q)
                circ += RZ(gen.new(suffix=f'l{l}q{q}_rz')).on(q)
            else:
                # 默认 U3(θ, φ, λ) —— 任意单比特酉（差一个全局相位）
                th = gen.new(suffix=f'l{l}q{q}_th')
                ph = gen.new(suffix=f'l{l}q{q}_ph')
                la = gen.new(suffix=f'l{l}q{q}_la')
                circ += U3(th, ph, la).on(q)

        # ---- ② CZ 纠缠层（砖墙偶/奇交替）----
        start = l % 2
        for i in range(start, n_qubits - 1, 2):
            _cz_on(circ, ctrl=i, tgt=i+1)

    # 标记为 ansatz，方便下游取参数名/赋值
    circ.as_ansatz()
    return circ, list(circ.ansatz_params_name)


def build_random_clifford_T(n_qubits: int,
                            n_gates: int = 500,
                            seed: Optional[int] = None) -> Circuit:
    """
    构造随机 Clifford 电路 T（随机 H/S/CNOT）
    """
    from mindquantum.core.gates import H, S
    rng = np.random.default_rng(seed)
    circ = Circuit()
    for _ in range(n_gates):
        t = int(rng.integers(0, 3))  # 0:H, 1:S, 2:CNOT
        if t == 0:
            q = int(rng.integers(0, n_qubits))
            circ += H.on(q)
        elif t == 1:
            q = int(rng.integers(0, n_qubits))
            circ += S.on(q)
        else:
            if n_qubits < 2:
                continue
            tgt = int(rng.integers(0, n_qubits))
            ctrl_choices = [c for c in range(n_qubits) if c != tgt]
            ctrl = int(rng.choice(ctrl_choices))
            circ += CNOT.on(tgt, ctrl)  # 目标在前，控制在后
    return circ


# ===================== 态生成和偏迹函数 =====================

def _simulate_state_vector(circ: Circuit,
                           n_qubits: int,
                           params: Optional[dict] = None,
                           seed: Optional[int] = None) -> np.ndarray:
    """在 'mqvector' 后端运行电路并返回终态向量。"""
    sim = Simulator('mqvector', n_qubits, seed=seed)
    from mindquantum.core.parameterresolver import ParameterResolver as PR
    pr = PR(params) if (params is not None) else None
    sim.apply_circuit(circ, pr=pr)
    return np.asarray(sim.get_qs(), dtype=np.complex128)


def _pure_state_density(psi: np.ndarray) -> np.ndarray:
    psi = psi.reshape(-1, 1)
    return (psi @ psi.conj().T).astype(np.complex128)


def rdm_first_half_from_state_vec(psi: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    更高效的 对纯态向量 psi 直接 reshape 计算（避免构造完整 ρ）。
    """
    dA = 2 ** (n_qubits // 2)
    dB = 2 ** (n_qubits - n_qubits // 2)
    psi2 = psi.reshape(dA, dB)
    rhoA = psi2 @ psi2.conj().T
    return 0.5 * (rhoA + rhoA.conj().T)


def trace_power(rho: np.ndarray, t: int) -> float:
    """
    计算密度矩阵 rho 的 t 次幂的迹: Tr(ρ^t)
    """
    if t == 1:
        return np.trace(rho).real
    else:
        rho_power = np.linalg.matrix_power(rho, t)
        return np.trace(rho_power).real


def sample_params_uniform(param_names: List[str], rng: np.random.Generator,
                         low: float = 0.0, high: float = 2*np.pi) -> Dict[str, float]:
    """从均匀分布采样参数值"""
    return {name: float(rng.uniform(low, high)) for name in param_names}


def _build_ansatz(n_qubits: int,
                  depth: int,
                  prefix: str,
                  single_rot: str,
                  entangler: str,
                  ansatz_builder: Optional[Callable[..., Tuple[Circuit, List[str]]]] = None,
                  ansatz_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Circuit, List[str]]:
    builder = build_param_luc_ansatz if ansatz_builder is None else ansatz_builder
    extra_kwargs = {} if ansatz_kwargs is None else dict(ansatz_kwargs)
    return builder(n_qubits, depth, prefix=prefix, single_rot=single_rot, entangler=entangler, **extra_kwargs)


def sample_E_Haar_trace_power(n_qubits: int, t: int, num_samples: int, seed: Optional[int] = None) -> float:
    """
    采样 Haar 随机纯态 |ψ>，计算 Tr(ρ_{n/2}^t) 的样本平均。
    """
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    acc = 0.0
    for _ in range(num_samples):
        g = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        psi = g / np.linalg.norm(g)
        rhoA = rdm_first_half_from_state_vec(psi, n_qubits)
        acc += trace_power(rhoA, t)
    return acc / num_samples


def sample_E_ens_trace_power_random(n_qubits: int,
                                    t: int,
                                    depth: int,
                                    num_samples: int,
                                    which: str,              # 'vqe' 或 'shvqe'
                                    n_clifford: int = 500,
                                    seed: Optional[int] = None,
                                    ansatz_builder: Optional[Callable[..., Tuple[Circuit, List[str]]]] = None,
                                    ansatz_kwargs: Optional[Dict[str, Any]] = None) -> float:
    """
    用随机 LUC（与论文 Fig.2(b) 一致）生成 ensemble。可通过 ansatz_builder / ansatz_kwargs 替换默认的 LUC ansatz：
      - which='vqe'  : U|0>
      - which='shvqe': T U |0>
    返回 E_ens[Tr(ρ_{n/2}^t)] 的样本平均。
    """
    which = which.lower()
    if which not in {'vqe', 'shvqe'}:
        raise ValueError("which 应为 'vqe' 或 'shvqe'。")

    rng = np.random.default_rng(seed)
    acc = 0.0
    circuit_U, param_names = _build_ansatz(
        n_qubits,
        depth,
        prefix='theta',
        single_rot='rx-ry-rz',
        entangler='rxx',
        ansatz_builder=ansatz_builder,
        ansatz_kwargs=ansatz_kwargs,
    )

    for _ in range(num_samples):
        sample_seed = int(rng.integers(0, 2**31 - 1))
        local_rng = np.random.default_rng(sample_seed)
        params = sample_params_uniform(param_names, local_rng)
        if which == 'vqe':
            psi = _simulate_state_vector(circuit_U, n_qubits, params=params, seed=None)
        else:
            clifford_seed = int(local_rng.integers(0, 2**31 - 1))
            circuit_T = build_random_clifford_T(n_qubits, n_gates=n_clifford, seed=clifford_seed)
            psi = _simulate_state_vector(circuit_U + circuit_T, n_qubits, params=params, seed=None)
        rhoA = rdm_first_half_from_state_vec(psi, n_qubits)
        acc += trace_power(rhoA, t)
    return acc / num_samples


def delta_t_random(n_qubits: int,
                   t: int,
                   depth: int,
                   num_samples_haar: int,
                   num_samples_ens: int,
                   which: str,                 # 'vqe' 或 'shvqe'
                   n_clifford: int = 500,
                   seed: Optional[int] = None,
                   log_base: str = 'e',
                   ansatz_builder: Optional[Callable[..., Tuple[Circuit, List[str]]]] = None,
                   ansatz_kwargs: Optional[Dict[str, Any]] = None) -> float:
    """
    计算 Δ_t = log(E_Haar) - log(E_ens)。可通过 ansatz_builder / ansatz_kwargs 替换 ensemble 中的 LUC ansatz。
    """
    E_H = sample_E_Haar_trace_power(n_qubits, t, num_samples=num_samples_haar, seed=seed)
    E_S = sample_E_ens_trace_power_random(n_qubits, t, depth, num_samples=num_samples_ens,
                                          which=which, n_clifford=n_clifford, seed=seed,
                                          ansatz_builder=ansatz_builder, ansatz_kwargs=ansatz_kwargs)
    if log_base == '2':
        return float(np.log2(E_H) - np.log2(E_S))
    elif log_base == '10':
        return float(np.log10(E_H) - np.log10(E_S))
    else:
        return float(np.log(E_H) - np.log(E_S))


# ===================== 可视化函数 =====================

def compute_and_plot_delta_t(results_by_t, depths_vqe, depths_shvqe, figsize=(12, 10)):
    """
    计算并绘制 Δ_t 随深度的变化
    
    Args:
        results_by_t: 按 t 值组织的实验结果字典
        depths_vqe: VQE 深度列表
        depths_shvqe: SH-VQE 深度列表
        figsize: 图形大小
        
    Returns:
        fig: matplotlib figure 对象
    """
    t_orders = sorted(results_by_t.keys())
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for ax, t_order in zip(axes, t_orders):
        vqe_depths = list(results_by_t[t_order]['vqe'].keys())
        vqe_values = list(results_by_t[t_order]['vqe'].values())
        shvqe_depths = list(results_by_t[t_order]['shvqe'].keys())
        shvqe_values = list(results_by_t[t_order]['shvqe'].values())

        ax.plot(vqe_depths, vqe_values, 'o-', color='red', linewidth=2,
                markersize=6, label='VQE', markerfacecolor='red')
        ax.plot(shvqe_depths, shvqe_values, 'o-', color='blue', linewidth=2,
                markersize=6, label='SH-VQE', markerfacecolor='blue')
        ax.set_title(f't = {t_order}')
        ax.set_xlabel('Schrödinger circuit depth')
        ax.set_ylabel('Δ$_t$')
        ax.set_xticks(depths_vqe)
        ax.set_xlim(min(depths_vqe) - 2, max(depths_vqe) + 2)
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[0].legend(frameon=False)
    plt.tight_layout()
    return fig


def run_fig2b_experiment(n_qubits=12, t_orders=[3, 6, 9, 12], 
                         depths_vqe=[0, 2, 4, 8, 12, 18, 24, 36, 48, 60, 72],
                         depths_shvqe=[0, 2, 4, 8, 12, 16],
                         nHaar=200, nEns=200, seed=42,
                         ansatz_builder=None, ansatz_kwargs=None):
    """
    运行完整的 Fig2b 实验
    
    Args:
        n_qubits: 量子比特数
        t_orders: t 的取值列表
        depths_vqe: VQE 深度列表
        depths_shvqe: SH-VQE 深度列表
        nHaar: Haar 采样数
        nEns: Ensemble 采样数
        seed: 随机种子
        ansatz_builder: ansatz 构建函数
        ansatz_kwargs: ansatz 构建的额外参数
        
    Returns:
        results_by_t: 实验结果字典
    """
    results_by_t = {}

    for t_order in t_orders:
        vqe_results = {
            depth: delta_t_random(
                n_qubits, t_order, depth, nHaar, nEns,
                which='vqe', n_clifford=500, seed=seed,
                ansatz_builder=ansatz_builder,
                ansatz_kwargs=ansatz_kwargs,
            )
            for depth in depths_vqe
        }
        shvqe_results = {
            depth: delta_t_random(
                n_qubits, t_order, depth, nHaar, nEns,
                which='shvqe', n_clifford=500, seed=seed,
                ansatz_builder=ansatz_builder,
                ansatz_kwargs=ansatz_kwargs,
            )
            for depth in depths_shvqe
        }
        results_by_t[t_order] = {
            'vqe': vqe_results,
            'shvqe': shvqe_results,
        }
        print(f"t={t_order}")
        print("VQE结果:", vqe_results)
        print("SH-VQE结果:", shvqe_results)
    
    return results_by_t

