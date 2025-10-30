"""
Fig3(b)(c) 核心函数模块
包含结构搜索训练和fidelity比较的所有核心逻辑
"""

from typing import List, Tuple, Optional, Dict, Union, Any, Callable
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from functools import lru_cache
from dataclasses import dataclass

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RX, RY, RZ, Z, X
from mindquantum.core.parameterresolver import PRGenerator
from mindquantum.simulator import Simulator
from mindquantum.core.operators import Hamiltonian, QubitOperator

# 全局随机数生成器
_rng = np.random.default_rng(7)


# ===================== Hamiltonian 构建 =====================

def hamiltonian_xxz(n: int, delta: float = 1.0, pbc: bool = True) -> Hamiltonian:
    """构建 XXZ 模型的 Hamiltonian"""
    if n <= 1:
        raise ValueError("n must be >= 2.")
    op = QubitOperator("", 0.0)
    def add_pair(pauli: str, i: int, j: int, coeff: float = 1.0) -> QubitOperator:
        return coeff * QubitOperator(f"{pauli}{i} {pauli}{j}")
    for i in range(n - 1):
        j = i + 1
        op += add_pair("X", i, j)
        op += add_pair("Y", i, j)
        op += add_pair("Z", i, j, delta)
    if pbc:
        op += add_pair("X", n - 1, 0)
        op += add_pair("Y", n - 1, 0)
        op += add_pair("Z", n - 1, 0, delta)
    return Hamiltonian(op)


# ===================== Clifford 电路构建 =====================

def ti_elementary_edges(n_qubits: int, distance: int) -> List[Tuple[int, int]]:
    """返回距离 d 的平移不变边集"""
    assert n_qubits >= 2 and 1 <= distance <= n_qubits // 2
    edges = []
    for i in range(n_qubits):
        j = (i + distance) % n_qubits
        # 偶数 n 且 2d==n 的"跨半圈"需要去重（避免同一 CZ 重复 -> CZ^2=I）
        if 2 * distance == n_qubits and j < i:
            continue
        edges.append((i, j))
    return edges


def code_from_distances(n_qubits: int, distances: List[int]) -> str:
    """将距离列表转换为代码串"""
    k = n_qubits // 2
    bits = ['0'] * k
    for d in distances:
        if 1 <= d <= k:
            bits[d - 1] = '1'
        else:
            raise ValueError(f"distance d={d} 越界（应在 1..{k}）")
    return ''.join(bits)


def distances_from_code(code: str) -> List[int]:
    """从代码串提取距离列表"""
    return [i + 1 for i, b in enumerate(code) if b == '1']


def clifford_ti_from_code(n_qubits: int, code: str) -> Circuit:
    """根据代码串构建平移不变的 Clifford 电路"""
    k = n_qubits // 2
    if len(code) != k or set(code) - {'0', '1'}:
        raise ValueError(f"code 长度必须为 {k}，且仅包含 0/1。")
    circ = Circuit()
    for idx, bit in enumerate(code):  # idx=0 -> d=1
        if bit == '1':
            d = idx + 1
            for c, t in ti_elementary_edges(n_qubits, d):
                circ += Z.on(t, c)     # MindQuantum: CZ = Z.on(target, control)
    return circ


def _softmax2_rows(alpha: np.ndarray) -> np.ndarray:
    """计算2维softmax"""
    if alpha.ndim != 2 or alpha.shape[1] != 2:
        raise ValueError("alpha 形状应为 (k, 2)。")
    A = alpha - alpha.max(axis=1, keepdims=True)
    e = np.exp(A)
    return e / e.sum(axis=1, keepdims=True)


def alpha_to_p1(alpha: np.ndarray) -> np.ndarray:
    """从alpha计算每个距离的开启概率"""
    return _softmax2_rows(alpha)[:, 1]


def sample_code_from_alpha(alpha: np.ndarray, rng: Optional[np.random.Generator] = None) -> str:
    """从alpha采样代码串"""
    rng = rng or np.random.default_rng()
    p1 = alpha_to_p1(alpha)
    bits = (rng.random(size=len(p1)) < p1).astype(int)
    return ''.join(map(str, bits.tolist()))


def threshold_code_from_alpha(alpha: np.ndarray, tau: float = 0.5) -> str:
    """从alpha阈值化得到代码串"""
    p1 = alpha_to_p1(alpha)
    bits = (p1 >= tau).astype(int)
    return ''.join(map(str, bits.tolist()))


CliffordConfig = Union[
    str,                       # '0000' / '1111' / '1010' 等代码串
    List[int],                 # 距离列表，如 [1,3]
    Dict[str, Union[str, int, float, np.ndarray]]  # {'alpha':..., 'mode':..., 'tau':..., 'seed':...}
]


def build_clifford_layer(n_qubits: int, config: CliffordConfig, rng: Optional[np.random.Generator] = None) -> Tuple[Circuit, str, Dict]:
    """
    返回: (T_cz 电路, code 字符串, meta 信息)
    支持三种用法：
      - 直接给 code（str）
      - 距离集合（list[int]）
      - {'alpha': ndarray(k,2), 'mode': 'sample'|'threshold', 'tau':0.5, 'seed':...}
    """
    if isinstance(config, str):
        code = config
    elif isinstance(config, list):
        code = code_from_distances(n_qubits, config)
    elif isinstance(config, dict) and 'alpha' in config:
        alpha = np.asarray(config['alpha'], dtype=float)
        mode  = str(config.get('mode', 'sample')).lower()
        if rng is None and 'seed' in config:
            rng = np.random.default_rng(int(config['seed']))
        if mode in ('sample', 'sampling'):
            code = sample_code_from_alpha(alpha, rng)
        elif mode in ('threshold', 'thres', 'det'):
            tau = float(config.get('tau', 0.5))
            code = threshold_code_from_alpha(alpha, tau)
        else:
            raise ValueError("alpha 模式仅支持 'sample' 或 'threshold'.")
    else:
        raise ValueError("config 必须是 code(str) / distances(list[int]) / {'alpha':..., ...} 三种之一。")

    T_cz = clifford_ti_from_code(n_qubits, code)
    meta = {'distances_on': distances_from_code(code)}
    return T_cz, code, meta


# ===================== VQE and SH-VQE 电路构建 =====================

def _cz_brick_layer(n_qubits: int, start: int, pbc: bool) -> Circuit:
    """
    砖墙 CZ：
      start=0 -> (0,1),(2,3),...,(n-2,n-1)
      start=1 -> (1,2),(3,4),...,(n-3,n-2) [+ (n-1,0) 若 pbc=True]
    """
    c = Circuit()
    for i in range(start, n_qubits - 1, 2):
        c += Z.on(i + 1, i)
    if pbc and (start == 1) and (n_qubits % 2 == 0):
        c += Z.on(0, n_qubits - 1)  # 仅在奇起点时做环回
    return c


def _single_qubit_layer(n_qubits: int,
                        prefix: str,
                        layer: int,
                        single_rot: str,
                        gen: PRGenerator) -> Circuit:
    """单比特旋转层"""
    c = Circuit()
    srot = single_rot.lower()
    for q in range(n_qubits):
        if srot == 'rx-ry':
            c += RX(gen.new(suffix=f'l{layer}q{q}rx')).on(q)
            c += RY(gen.new(suffix=f'l{layer}q{q}ry')).on(q)
        elif srot == 'ry-rz':
            c += RY(gen.new(suffix=f'l{layer}q{q}ry')).on(q)
            c += RZ(gen.new(suffix=f'l{layer}q{q}rz')).on(q)
        elif srot == 'rz-rx-rz':
            c += RZ(gen.new(suffix=f'l{layer}q{q}rz1')).on(q)
            c += RX(gen.new(suffix=f'l{layer}q{q}rx')).on(q)
            c += RZ(gen.new(suffix=f'l{layer}q{q}rz2')).on(q)
        elif srot == 'rz-ry-rz':
            c += RZ(gen.new(suffix=f'l{layer}q{q}rz1')).on(q)
            c += RY(gen.new(suffix=f'l{layer}q{q}ry')).on(q)
            c += RZ(gen.new(suffix=f'l{layer}q{q}rz2')).on(q)
        else:  # 'rx-ry-rz' (default)
            c += RX(gen.new(suffix=f'l{layer}q{q}rx')).on(q)
            c += RY(gen.new(suffix=f'l{layer}q{q}ry')).on(q)
            c += RZ(gen.new(suffix=f'l{layer}q{q}rz')).on(q)
    return c


def vqe_circuit(n_qubits: int,
                depth: int,
                prefix: str = 'theta',
                single_rot: str = 'rx-ry-rz',
                pbc: bool = True) -> Tuple[Circuit, List[str]]:
    """
    VQE（Schrödinger 电路，HEA）
    线路开头先叠加一层单量子门，然后每一层依次执行
      单量子门 -> CZ 砖墙（偶/奇层交替配对）。
    仅考虑偶数个比特。
    返回：(电路, θ 参数名列表)
    """
    assert n_qubits % 2 == 0 and depth >= 0
    gen = PRGenerator(prefix)
    circ = Circuit()

    # 额外的首层单比特旋转
    circ += _single_qubit_layer(
        n_qubits, prefix=prefix, layer='init',
        single_rot=single_rot, gen=gen
    )

    for l in range(depth):
        circ += _single_qubit_layer(
            n_qubits, prefix=prefix, layer=l,
            single_rot=single_rot, gen=gen
        )
        circ += _cz_brick_layer(n_qubits, start=(l % 2), pbc=pbc)

    circ.as_ansatz()
    return circ, list(circ.ansatz_params_name)


def shvqe_circuit_with_config(n_qubits: int,
                              depth: int,
                              clifford_config: CliffordConfig,
                              prefix_theta: str = 'theta',
                              prefix_phi: str = 'phi',
                              single_rot_u: str = 'rx-ry-rz',
                              single_rot_t: str = 'rz-rx-rz',
                              pbc: bool = True):
    """构建 SH-VQE 电路"""
    # Schrödinger 层（复用 VQE）
    U_circ, theta_names = vqe_circuit(
        n_qubits, depth, prefix=prefix_theta, single_rot=single_rot_u, pbc=pbc
    )

    # Clifford 层（通过"构型"生成）
    T_cz, code, meta = build_clifford_layer(n_qubits, clifford_config)

    # Heisenberg 的 1q 参数层
    gen_phi = PRGenerator(prefix_phi)
    T_1q = _single_qubit_layer(
        n_qubits, prefix=prefix_phi, layer='T', single_rot=single_rot_t, gen=gen_phi
    )
    T_1q.as_ansatz()
    phi_names = list(T_1q.ansatz_params_name)

    total = U_circ + T_cz + T_1q
    total.as_ansatz()
    return total, theta_names, phi_names, {'code': code, **meta}


# ===================== 基态和Fidelity计算 =====================

def _xxz_dense_matrix(n: int, delta: float = 1.0, pbc: bool = True) -> np.ndarray:
    """构建XXZ模型的稠密矩阵"""
    I = np.array([[1, 0], [0, 1]], complex)
    Xp = np.array([[0, 1], [1, 0]], complex)
    Y  = np.array([[0, -1j], [1j, 0]], complex)
    Zm = np.array([[1, 0], [0, -1]], complex)

    def embed_two(A, i, B, j):
        mats = [I] * n
        mats[i] = A; mats[j] = B
        M = mats[0]
        for t in range(1, n):
            M = np.kron(M, mats[t])
        return M

    H = np.zeros((2**n, 2**n), complex)
    for i in range(n - 1):
        H += embed_two(Xp, i, Xp, i + 1)
        H += embed_two(Y,  i, Y,  i + 1)
        H += delta * embed_two(Zm, i, Zm, i + 1)
    if pbc:
        H += embed_two(Xp, n - 1, Xp, 0)
        H += embed_two(Y,  n - 1, Y,  0)
        H += delta * embed_two(Zm, n - 1, Zm, 0)
    return H


_GS_CACHE: Dict[Tuple[int, float, bool], Tuple[float, np.ndarray]] = {}


def ground_state_of_xxz_cached(n: int, delta: float = 1.0, pbc: bool = True) -> Tuple[float, np.ndarray]:
    """缓存XXZ基态"""
    key = (n, float(delta), bool(pbc))
    if key in _GS_CACHE:
        return _GS_CACHE[key]
    H = _xxz_dense_matrix(n, delta, pbc)
    e, v = np.linalg.eigh(H)
    idx = int(np.argmin(e.real))
    E0, psi0 = float(e[idx].real), v[:, idx]
    psi0 = psi0 / np.linalg.norm(psi0)
    _GS_CACHE[key] = (E0, psi0)
    return E0, psi0


@lru_cache(None)
def mq_perm_be_to_mq(n: int) -> np.ndarray:
    """
    返回 perm（len=2^n），使得 psi_mq = psi_be[perm]。
    通过在 MQ 中分别制备 |1_j>，探测每个物理比特 j 在振幅向量中的位权。
    """
    pos_for_q = np.zeros(n, dtype=int)
    sim = Simulator('mqvector', n)
    for j in range(n):
        sim.reset()
        c = Circuit() + X.on(j)
        sim.apply_circuit(c)
        amp = np.asarray(sim.get_qs())
        idx = int(np.argmax(np.abs(amp)))
        pos_for_q[j] = int(round(np.log2(idx)))
    perm = np.zeros(2**n, dtype=int)
    for i in range(2**n):
        idx_mq = 0
        # 大端：q=0 在最高位
        for q in range(n):
            bit = (i >> (n - 1 - q)) & 1
            if bit:
                idx_mq |= (1 << pos_for_q[q])
        perm[i] = idx_mq
    return perm


@lru_cache(None)
def ground_state_mq_order(n: int, delta: float = 1.0, pbc: bool = True) -> Tuple[float, np.ndarray]:
    """获取MQ位序的基态"""
    E0, psi_be = ground_state_of_xxz_cached(n, delta=delta, pbc=pbc)
    perm = mq_perm_be_to_mq(n)
    psi_mq = psi_be[perm]
    psi_mq = psi_mq / np.linalg.norm(psi_mq)
    return E0, psi_mq


def pack_param_dict(theta_names: List[str], phi_names: List[str],
                    theta_vals: np.ndarray, phi_vals: np.ndarray) -> Dict[str, float]:
    """打包参数字典"""
    if len(theta_vals) != len(theta_names) or len(phi_vals) != len(phi_names):
        raise ValueError("参数长度与名称列表不匹配。")
    pr = {n: float(v) for n, v in zip(theta_names, theta_vals)}
    pr.update({n: float(v) for n, v in zip(phi_names,   phi_vals)})
    return pr


class SHVQEEvaluator:
    """
    评估能量/末态/Fidelity；对采样的多结构做"计数去重"加权平均；CPU-only。
    """
    def __init__(self,
                 n: int,
                 H: Hamiltonian,
                 U_circ: Circuit, theta_names: List[str],
                 T1q_circ: Circuit, phi_names: List[str],
                 code2circ_fn=None,
                 backend: str = 'mqvector'):
        self.n = int(n)
        self.H = H
        self.U = U_circ
        self.theta_names = list(theta_names)
        self.T1q = T1q_circ
        self.phi_names = list(phi_names)
        self.backend = backend
        # code -> Clifford 层缓存
        self._code_cache: Dict[str, Circuit] = {}
        if code2circ_fn is None:
            try:
                self._code2circ = globals()['clifford_ti_from_code']
            except KeyError:
                raise RuntimeError("需要提供 code2circ_fn 或在全局定义 clifford_ti_from_code。")
        else:
            self._code2circ = code2circ_fn

    def _get_Tcz(self, code: str) -> Circuit:
        if code not in self._code_cache:
            self._code_cache[code] = self._code2circ(self.n, code)
        return self._code_cache[code]

    def energy_for_code(self, code: str, theta_vals: np.ndarray, phi_vals: np.ndarray,
                        seed: Optional[int] = None) -> float:
        """单结构能量"""
        pr = pack_param_dict(self.theta_names, self.phi_names, theta_vals, phi_vals)
        total = self.U + self._get_Tcz(code) + self.T1q
        sim = Simulator(self.backend, self.n, seed=seed)
        sim.apply_circuit(total, pr=pr)
        val = sim.get_expectation(self.H)
        return float(np.real(val))

    def mean_energy_over_codes(self, codes: List[str], theta_vals: np.ndarray, phi_vals: np.ndarray,
                               seed: Optional[int] = None) -> float:
        """多结构样本平均能量（计数去重）"""
        pr = pack_param_dict(self.theta_names, self.phi_names, theta_vals, phi_vals)
        cnt = Counter(codes)
        tot_e, tot_m = 0.0, 0
        for code, m in cnt.items():
            total = self.U + self._get_Tcz(code) + self.T1q
            sim = Simulator(self.backend, self.n, seed=seed)
            sim.apply_circuit(total, pr=pr)
            e = sim.get_expectation(self.H)
            tot_e += float(np.real(e)) * int(m)
            tot_m += int(m)
        return tot_e / max(1, tot_m)

    def final_state_for_code(self, code: str, theta_vals: np.ndarray, phi_vals: np.ndarray,
                             seed: Optional[int] = None) -> np.ndarray:
        """最终态"""
        pr = pack_param_dict(self.theta_names, self.phi_names, theta_vals, phi_vals)
        total = self.U + self._get_Tcz(code) + self.T1q
        sim = Simulator(self.backend, self.n, seed=seed)
        sim.apply_circuit(total, pr=pr)
        psi = np.asarray(sim.get_qs(), dtype=complex)
        norm = np.linalg.norm(psi)
        return psi if norm == 0 else (psi / norm)

    def fidelity_with_ground_state(self, code: str, theta_vals: np.ndarray, phi_vals: np.ndarray,
                                   delta: float = 1.0, pbc: bool = True,
                                   seed: Optional[int] = None) -> float:
        """Fidelity：与"重排到 MQ 位序"的 XXZ 基态"""
        _, psi0_mq = ground_state_mq_order(self.n, delta=delta, pbc=pbc)
        code_physical = ''.join('0' if ch == '1' else '1' if ch == '0' else ch for ch in code)
        psi = self.final_state_for_code(code_physical, theta_vals, phi_vals, seed=seed)
        return float(np.abs(np.vdot(psi0_mq, psi)))


# ===================== 策略和优化器 =====================

def _softmax2(rows):
    """2维softmax"""
    z = rows - rows.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class GraphPolicy:
    """L 位 bitstring，每位二元 softmax；带温度 τ 便于退火。"""
    def __init__(self, L, tau=2.0, alpha_init_scale=1e-6):
        self.L = int(L)
        self.tau = float(tau)
        self.alpha = _rng.normal(0.0, alpha_init_scale, size=(self.L, 2))

    def set_tau(self, tau: float):
        self.tau = float(tau)

    def probs_bits(self):
        """返回每位的 [p(0), p(1)]，shape=(L,2)。"""
        return _softmax2(self.alpha / self.tau)

    def sample(self, N: int):
        """采样 N 个代码串；返回 bits[N,L]（0/1），logp[N]，codes(list[str])。"""
        p = self.probs_bits()
        u = _rng.random(size=(N, self.L))
        bits = (u < p[:, 1].reshape(1, -1)).astype(int)
        logp = (bits * np.log(p[:, 1]).reshape(1, -1)
                + (1 - bits) * np.log(p[:, 0]).reshape(1, -1)).sum(axis=1)
        codes = [''.join(map(str, r.tolist())) for r in bits]
        return bits, logp, codes

    def grad_logp(self, bits):
        """∇_α log p(k|α) = (1/τ)(one_hot - p)。返回 [N,L,2]。"""
        p = self.probs_bits()
        N = bits.shape[0]
        one_hot = np.zeros((N, self.L, 2), dtype=float)
        I0 = np.repeat(np.arange(N), self.L)
        I1 = np.tile(np.arange(self.L), N)
        I2 = bits.reshape(-1)
        one_hot[I0, I1, I2] = 1.0
        return (one_hot - p.reshape(1, self.L, 2)) / self.tau

    def step(self, grad_alpha, lr=0.06, clip=5.0, weight_decay=0.0):
        if weight_decay != 0.0:
            grad_alpha = grad_alpha + weight_decay * self.alpha
        gnorm = np.linalg.norm(grad_alpha)
        if clip and gnorm > clip and gnorm > 0:
            grad_alpha = grad_alpha * (clip / gnorm)
        self.alpha -= lr * grad_alpha
        self.alpha = np.clip(self.alpha, -20.0, 20.0)


@dataclass
class Adam:
    lr: float = 0.03
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    t: int = 0
    def step(self, w, g, m=None, v=None):
        if m is None: m = np.zeros_like(w)
        if v is None: v = np.zeros_like(w)
        self.t += 1
        m = self.beta1 * m + (1 - self.beta1) * g
        v = self.beta2 * v + (1 - self.beta2) * (g * g)
        mhat = m / (1 - self.beta1 ** self.t)
        vhat = v / (1 - self.beta2 ** self.t)
        w = w - self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return w, m, v


# ===================== 梯度操作缓存 =====================

class GradOpsCache:
    """
    为每个 code 构建并缓存：
      - total 电路
      - grad_ops = sim.get_expectation_with_grad(H, total)
      - 参数名顺序（映射到 θ / ϕ）
    """
    def __init__(self, n_qubits, depth, H,
                 single_rot_u='rx-ry-rz', single_rot_t='rz-rx-rz', pbc=True):
        self.n = int(n_qubits)
        self.depth = int(depth)
        self.H = H
        self.single_rot_u = single_rot_u
        self.single_rot_t = single_rot_t
        self.pbc = bool(pbc)
        self._cache = {}

    def _build_one(self, code: str):
        total, theta_names, phi_names, _ = shvqe_circuit_with_config(
            self.n, self.depth, code,
            single_rot_u=self.single_rot_u, single_rot_t=self.single_rot_t,
            pbc=self.pbc
        )
        sim = Simulator('mqvector', self.n)
        grad_ops = sim.get_expectation_with_grad(self.H, total)
        order = list(total.ansatz_params_name)
        idx_theta = np.array([order.index(nm) for nm in theta_names], dtype=int)
        idx_phi   = np.array([order.index(nm) for nm in phi_names], dtype=int)
        return {
            'grad_ops': grad_ops,
            'order': order,
            'idx_theta': idx_theta,
            'idx_phi': idx_phi,
            'theta_names': list(theta_names),
            'phi_names': list(phi_names),
        }

    def get(self, code: str):
        if code not in self._cache:
            self._cache[code] = self._build_one(code)
        return self._cache[code]


class MQEnergyOracle:
    """
    统一接口：
      energy_and_grad(code, theta, phi) -> (E, g_theta, g_phi)
      energy_only(code, theta, phi)     -> E
    其中 theta/phi 为连续参数的 numpy 向量。
    """
    def __init__(self, n_qubits, depth, H,
                 single_rot_u='rx-ry-rz', single_rot_t='rz-rx-rz', pbc=True):
        self.cache = GradOpsCache(
            n_qubits, depth, H,
            single_rot_u=single_rot_u, single_rot_t=single_rot_t, pbc=pbc
        )

    @staticmethod
    def _call_grad_ops(grad_ops, ansatz_vec: np.ndarray):
        try:
            out = grad_ops(ansatz_vec.astype(np.float32))
        except TypeError:
            out = grad_ops(None, ansatz_vec.astype(np.float32))

        if isinstance(out, tuple):
            if len(out) == 2:
                measure, ans_grad = out
            elif len(out) == 3:
                measure, _, ans_grad = out
            else:
                raise RuntimeError("Unexpected return from grad_ops.")
        else:
            measure, ans_grad = out, None

        e_arr = np.array(measure).reshape(-1)
        e = float(np.real_if_close(e_arr[0], tol=1e-8))

        g = None
        if ans_grad is not None:
            g_arr = np.array(ans_grad).reshape(-1)
            g = np.real_if_close(g_arr, tol=1e-8).astype(np.float64, copy=False)

        return e, g

    def energy_and_grad(self, code: str, theta: np.ndarray, phi: np.ndarray):
        item = self.cache.get(code)
        P = len(item['order'])
        x = np.zeros(P, dtype=np.float32)
        x[item['idx_theta']] = theta.astype(np.float32, copy=False)
        x[item['idx_phi']]   = phi.astype(np.float32, copy=False)
        e, g_full = self._call_grad_ops(item['grad_ops'], x)
        g_theta = g_full[item['idx_theta']].astype(np.float64, copy=False)
        g_phi   = g_full[item['idx_phi']].astype(np.float64, copy=False)
        return e, g_theta, g_phi

    def energy_only(self, code: str, theta: np.ndarray, phi: np.ndarray):
        item = self.cache.get(code)
        P = len(item['order'])
        x = np.zeros(P, dtype=np.float32)
        x[item['idx_theta']] = theta.astype(np.float32, copy=False)
        x[item['idx_phi']]   = phi.astype(np.float32, copy=False)
        e, _ = self._call_grad_ops(item['grad_ops'], x)
        return float(e)


def all_code_probs_from_bits_prob(p_bits: np.ndarray):
    """
    把每位 [p0, p1] 转为所有 2^L 代码串的概率向量和标签（大端，'0000'..'1111'）。
    """
    L = p_bits.shape[0]
    total = 1 << L
    probs = np.zeros(total)
    labels = []
    for m in range(total):
        bits = np.array([(m >> (L-1-k)) & 1 for k in range(L)], dtype=int)
        labels.append(''.join(map(str, bits.tolist())))
        p = 1.0
        for j, b in enumerate(bits):
            p *= p_bits[j, b]
        probs[m] = p
    return probs, labels


def alternating_train_mq(
    n_qubits: int,
    depth: int,
    delta: float = 1.0,
    pbc: bool = True,
    T: int = 300,
    N: int = 800,
    tau0: float = 2.0, tau_min: float = 0.25, tau_decay: float = 0.993,
    lr_alpha: float = 0.06, entropy_coef: float = 0.01, baseline_ema: float = 0.06,
    lr_theta: float = 0.03, lr_phi: float = 0.03,
    single_rot_u: str = 'rx-ry-rz', single_rot_t: str = 'rz-rx-rz',
    seed: int = 1234
):
    """
    交替训练：固定 α 更新 (θ,ϕ)，固定 (θ,ϕ) 更新 α
    """
    rng = np.random.default_rng(seed)
    assert n_qubits % 2 == 0
    L = n_qubits // 2

    # Hamiltonian 与能量/梯度 oracle
    H = hamiltonian_xxz(n_qubits, delta=delta, pbc=pbc)
    oracle = MQEnergyOracle(n_qubits, depth, H,
                            single_rot_u=single_rot_u, single_rot_t=single_rot_t, pbc=pbc)

    # 获取 θ/ϕ 维度
    tmp = oracle.cache.get('0' * L)
    d_theta = len(tmp['idx_theta'])
    d_phi   = len(tmp['idx_phi'])

    # 连续参数初始化 + Adam
    theta = rng.normal(0.0, 0.5, size=d_theta)
    phi   = rng.normal(0.0, 0.5, size=d_phi)
    opt_theta = Adam(lr=lr_theta); mth = np.zeros_like(theta); vth = np.zeros_like(theta)
    opt_phi   = Adam(lr=lr_phi);   mph = np.zeros_like(phi);   vph = np.zeros_like(phi)

    # 结构策略 α
    policy = GraphPolicy(L=L, tau=tau0, alpha_init_scale=1e-6)

    # 记录
    cost_hist = []
    prob_hist = []
    total = 1 << L
    labels_orig = [''.join(map(str, [(i >> (L-1-k)) & 1 for k in range(L)])) for i in range(total)]
    labels = [''.join('1' if ch == '0' else '0' for ch in label) for label in labels_orig]
    idx_label_1111 = labels.index('1111')
    baseline = None

    for t in range(1, T + 1):
        # 温度退火
        tau_t = max(tau_min, tau0 * (tau_decay ** (t - 1)))
        policy.set_tau(tau_t)

        # Step 1. 采样 S_t（共用样本）
        bits, logp, codes = policy.sample(N)

        # 分组统计
        uniq, counts = np.unique(np.array(codes), return_counts=True)
        weights = counts.astype(float) / float(N)

        # Step 2. 固定 α：用 MQ 梯度 + Adam 更新 (θ,ϕ)
        gth = np.zeros_like(theta)
        gph = np.zeros_like(phi)
        for code_u, w_u in zip(uniq, weights):
            e_u, gth_u, gph_u = oracle.energy_and_grad(code_u, theta, phi)
            gth += w_u * gth_u
            gph += w_u * gph_u
        theta, mth, vth = opt_theta.step(theta, gth, mth, vth)
        phi,   mph, vph = opt_phi.step(phi,   gph, mph, vph)

        # Step 3. 固定 (θ,ϕ)：在同一批样本上重新评估能量，更新 α
        e_lookup = {code_u: oracle.energy_only(code_u, theta, phi) for code_u in uniq}
        energies = np.array([e_lookup[c] for c in codes])
        mean_e = float(energies.mean())
        if baseline is None: baseline = mean_e
        else: baseline = (1 - baseline_ema) * baseline + baseline_ema * mean_e

        grad_logp = policy.grad_logp(bits)
        advantage = energies - baseline - entropy_coef * logp
        grad_alpha = (advantage.reshape(-1, 1, 1) * grad_logp).mean(axis=0)
        policy.step(grad_alpha, lr=lr_alpha, clip=5.0, weight_decay=0.0)

        # 记录（经验概率）
        freq_map = {
            ''.join('1' if ch == '0' else '0' for ch in cu): float(cnt) / float(N)
            for cu, cnt in zip(uniq.tolist(), counts.tolist())
        }
        probs_display = np.array([freq_map.get(label, 0.0) for label in labels], dtype=float)
        prob_hist.append(probs_display)

        cost_hist.append(mean_e)
        if t == 1 or t % max(1, T // 10) == 0:
            p1111 = float(probs_display[idx_label_1111])
            print(f"[arch] step {t:4d}/{T}, E = {mean_e:.6f}, p('1111') = {p1111:.6f}")

    return {
        'cost': np.asarray(cost_hist),
        'prob': np.asarray(prob_hist),
        'labels': labels,
        'theta': theta, 'phi': phi, 'policy': policy,
    }


# ===================== 可视化函数 =====================

def plot_fig3b(hist):
    """绘制Fig3b的两个子图"""
    # 上图：Cost（线性坐标）
    plt.figure()
    y = hist['cost']
    plt.plot(np.arange(1, len(y)+1), y)
    plt.xlabel("Iteration")
    plt.ylabel("Cost value")
    plt.title("Cost (mean energy) vs Iteration")
    plt.show()

    # 下图：所有 2^L 个图的概率随迭代变化
    plt.figure()
    P = hist['prob']; labels = hist['labels']
    for k in range(P.shape[1]):
        plt.plot(np.arange(1, P.shape[0]+1), P[:, k], label=labels[k])
    plt.xlabel("Iteration")
    plt.ylabel("Probability")
    plt.title("Probabilities of all {} graphs".format(P.shape[1]))
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_fig3c(evaluator, codes_array, theta, phi):
    """绘制Fig3c的fidelity柱状图"""
    fid_array = np.array([evaluator.fidelity_with_ground_state(code, theta, phi) for code in codes_array])
    
    x = np.arange(len(codes_array))
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(x, fid_array, color="#f5c08f", edgecolor="#8a5d29", linewidth=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(codes_array, fontsize=10)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0, 1.0)
    
    ax.set_xlabel("Graph Connection type")
    ax.set_ylabel("Fidelity")
    
    plt.tight_layout()
    return fig

