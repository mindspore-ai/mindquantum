"""
Fig3(d) 核心函数模块
包含不同qubit和depth下VQE vs SH-VQE的fidelity比较的所有核心逻辑
"""

from typing import Optional, List, Tuple, Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RX, RY, RZ, Z
from mindquantum.core.parameterresolver import PRGenerator
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.simulator import Simulator
from scipy.sparse.linalg import eigsh

import mindspore as ms
from mindspore import Tensor
from mindspore.nn import Adam, TrainOneStepCell
from mindquantum.framework import MQAnsatzOnlyLayer

# ===================== 优化配置 =====================

DEFAULT_THREADS = max(1, os.cpu_count() // 2)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, str(DEFAULT_THREADS))

EIGSH_TOL = 1e-8
EIGSH_NCV = 40
USE_REAL_H = True


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


@lru_cache(maxsize=None)
def cached_hamiltonian(n: int, delta: float = 1.0, pbc: bool = True):
    """缓存的Hamiltonian"""
    return hamiltonian_xxz(n, delta=delta, pbc=pbc)


# ===================== 电路构建 =====================

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
        else:  # 'rx-ry-rz' (default)
            c += RX(gen.new(suffix=f'l{layer}q{q}rx')).on(q)
            c += RY(gen.new(suffix=f'l{layer}q{q}ry')).on(q)
            c += RZ(gen.new(suffix=f'l{layer}q{q}rz')).on(q)
    return c


def _cz_brick_layer(n_qubits: int, start: int, pbc: bool) -> Circuit:
    """砖墙 CZ 层"""
    c = Circuit()
    for i in range(start, n_qubits - 1, 2):
        c += Z.on(i + 1, i)
    if pbc and (start == 1) and (n_qubits % 2 == 0):
        c += Z.on(0, n_qubits - 1)
    return c


def _full_graph_cz(n_qubits: int) -> Circuit:
    """全连接 CZ Clifford（无参数）"""
    c = Circuit()
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            c += Z.on(j, i)
    return c


def vqe_circuit(n_qubits: int,
                depth: int,
                prefix: str = 'theta',
                single_rot: str = 'rx-ry-rz',
                pbc: bool = False) -> Tuple[Circuit, List[str]]:
    """VQE电路"""
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


def shvqe_circuit(n_qubits: int,
                  depth: int,
                  prefix: str = 'theta',
                  single_rot: str = 'rx-ry-rz',
                  pbc: bool = False) -> Tuple[Circuit, List[str]]:
    """SH-VQE 电路"""
    S, _ = vqe_circuit(n_qubits, depth, prefix, single_rot, pbc)
    C = _full_graph_cz(n_qubits)
    total = S + C

    # 末端追加单比特旋转
    final_gen = PRGenerator(f'{prefix}_final')
    final_layer = _single_qubit_layer(
        n_qubits, prefix=prefix, layer='final',
        single_rot=single_rot, gen=final_gen
    )
    total += final_layer

    total.as_ansatz()
    theta_names = list(total.ansatz_params_name)
    return total, theta_names


# ===================== 基态和Fidelity计算 =====================

def exact_ground_state(H, k: int = 2) -> Tuple[float, np.ndarray]:
    """求基态能量与向量"""
    A = H.sparse_matrix.tocsr()
    if USE_REAL_H:
        A = A.astype(np.float64)
        A = (A + A.T) * 0.5
    else:
        A = A.astype(np.complex128)
        A = (A + A.getH()) * 0.5
    k = min(int(k), A.shape[0]-1)
    ncv = min(A.shape[0]-1, EIGSH_NCV)
    vals, vecs = eigsh(A, k=k, which='SA', tol=EIGSH_TOL, ncv=ncv)
    i = int(np.argmin(vals.real))
    gs = vecs[:, i]
    gs = gs / np.linalg.norm(gs)
    return float(vals[i].real), gs


@lru_cache(maxsize=None)
def cached_ground_state(n: int, delta: float = 1.0, pbc: bool = True):
    """缓存的基态"""
    H = cached_hamiltonian(n, delta=delta, pbc=pbc)
    return exact_ground_state(H)


def _reorder_bits(v: np.ndarray, n: int) -> np.ndarray:
    """在不同比特序之间转换"""
    return v.reshape([2]*n).transpose(tuple(range(n-1, -1, -1))).reshape(-1)


def fidelity(psi: np.ndarray, gs: np.ndarray) -> float:
    """计算fidelity |<gs|psi>|^2"""
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    gs  = np.asarray(gs , dtype=np.complex128).reshape(-1)
    n = int(round(np.log2(psi.size)))
    psi = psi / np.linalg.norm(psi)
    gs  = gs  / np.linalg.norm(gs)
    F1 = abs(np.vdot(gs, psi))
    F2 = abs(np.vdot(gs, _reorder_bits(psi, n)))
    return float(max(F1, F2))


# ===================== 训练函数 =====================

def adam_train_energy_ansatz(grad_ops,
                             steps: int,
                             lr: float,
                             seed: int,
                             verbose: bool = False,
                             initial_weights: Optional[np.ndarray] = None,
                             init_noise_scale: float = 1e-2) -> Tuple[float, Dict[str, float], List[float]]:
    """
    用 Adam 最小化 <H>；仅 ansatz 参数可训练。
    """
    ms.set_seed(seed)
    rng = np.random.default_rng(seed)

    param_names = list(grad_ops.ansatz_params_name)

    if initial_weights is None:
        weights = rng.normal(loc=0.0, scale=init_noise_scale, size=len(param_names)).astype(np.float32)
    else:
        weights = np.asarray(initial_weights, dtype=np.float32).reshape(-1)
        if weights.shape[0] != len(param_names):
            raise ValueError(f"initial_weights length {weights.shape[0]} mismatch with ansatz params {len(param_names)}.")

    net = MQAnsatzOnlyLayer(grad_ops, weight=Tensor(weights))
    opt = Adam(net.trainable_params(), learning_rate=lr)
    train_net = TrainOneStepCell(net, opt)

    trace: List[float] = []
    for i in range(steps):
        out = train_net()
        arr = out.asnumpy()
        E = float(np.real(np.asarray(arr).reshape(-1)[0]))
        trace.append(E)
        if verbose and (i % max(1, steps // 10) == 0):
            print(f"  step {i:4d}: E = {E:.6f}")

    weights_final = net.trainable_params()[0].asnumpy().reshape(-1)
    pr = dict(zip(param_names, weights_final.tolist()))
    return trace[-1], pr, trace


def run_vqe_experiment(
    n: int,
    depths: Tuple[int, ...],
    delta: float = 1.0,
    pbc_brick: bool = False,
    RESTARTS: int = 5,
    BASE_SEED: int = 21,
    STEPS: int = 100,
    LR: float = 0.05,
    WARM_START: bool = False,
    INIT_NOISE_SCALE: float = 0.01,
    VERBOSE_TRAIN: bool = False
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """运行VQE实验"""
    print(
        f"=== n = {n} | restarts = {RESTARTS} | steps = {STEPS} | "
        f"lr = {LR} | warm_start = {WARM_START} | noise = {INIT_NOISE_SCALE} ==="
    )

    H = cached_hamiltonian(n, delta=delta, pbc=True)
    _, gs = cached_ground_state(n, delta=delta, pbc=True)

    sim = Simulator('mqvector', n)

    last_best_pr: Optional[Dict[str, float]] = None
    results: List[Dict[str, Any]] = []
    fid_by_depth: Dict[int, float] = {}

    for depth_idx, depth in enumerate(depths):
        circuit, _ = vqe_circuit(
            n,
            depth,
            prefix='theta',
            single_rot='rx-ry-rz',
            pbc=pbc_brick,
        )
        grad_ops = sim.get_expectation_with_grad(H, circuit)
        param_names = list(grad_ops.ansatz_params_name)

        if WARM_START and last_best_pr:
            base_vector = np.array(
                [last_best_pr.get(name, 0.0) for name in param_names],
                dtype=np.float32,
            )
        else:
            base_vector = np.zeros(len(param_names), dtype=np.float32)

        best_result = None

        for restart in range(RESTARTS):
            seed_base = BASE_SEED + depth_idx * RESTARTS + restart
            rng = np.random.default_rng(seed_base)

            init_weights = base_vector + rng.normal(
                0.0,
                INIT_NOISE_SCALE,
                size=len(param_names),
            ).astype(np.float32)

            energy, pr, trace = adam_train_energy_ansatz(
                grad_ops=grad_ops,
                steps=STEPS,
                lr=LR,
                seed=seed_base,
                verbose=VERBOSE_TRAIN,
                initial_weights=init_weights,
                init_noise_scale=INIT_NOISE_SCALE,
            )

            psi = circuit.get_qs(pr=pr)
            fid = fidelity(psi, gs)

            if (best_result is None) or (fid > best_result['fidelity']):
                best_result = {
                    'depth': depth,
                    'fidelity': fid,
                    'energy': energy,
                    'pr': pr,
                    'trace': trace,
                    'restart': restart,
                }

        if WARM_START and best_result is not None:
            last_best_pr = best_result['pr']

        if best_result is not None:
            fid_by_depth[depth] = float(best_result['fidelity'])
            print(
                f"  depth = {depth:2d} | VQE best restart #{best_result['restart']:02d} "
                f"| fid = {best_result['fidelity']:.3f} | E = {best_result['energy']:.6f}"
            )
            results.append(best_result)
        else:
            fid_by_depth[depth] = float('nan')

    result_fid = [fid_by_depth.get(d, float('nan')) for d in depths]
    return results, result_fid


def run_sh_vqe_experiment(
    n: int,
    depths: Tuple[int, ...],
    delta: float = 1.0,
    pbc_brick: bool = False,
    RESTARTS: int = 5,
    BASE_SEED: int = 21,
    STEPS: int = 100,
    LR: float = 0.05,
    WARM_START: bool = False,
    INIT_NOISE_SCALE: float = 0.01,
    VERBOSE_TRAIN: bool = False
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """运行SH-VQE实验"""
    print(
        f"=== [SH-VQE] n = {n} | restarts = {RESTARTS} | steps = {STEPS} | "
        f"lr = {LR} | warm_start = {WARM_START} | noise = {INIT_NOISE_SCALE} ==="
    )

    H = cached_hamiltonian(n, delta=delta, pbc=True)
    _, gs = cached_ground_state(n, delta=delta, pbc=True)

    sim = Simulator('mqvector', n)

    last_best_pr: Optional[Dict[str, float]] = None
    results: List[Dict[str, Any]] = []
    fid_by_depth: Dict[int, float] = {}

    for depth_idx, depth in enumerate(depths):
        circuit, _ = shvqe_circuit(
            n,
            depth,
            prefix='theta',
            single_rot='rx-ry-rz',
            pbc=pbc_brick,
        )
        grad_ops = sim.get_expectation_with_grad(H, circuit)
        param_names = list(grad_ops.ansatz_params_name)

        if WARM_START and last_best_pr:
            base_vector = np.array(
                [last_best_pr.get(name, 0.0) for name in param_names],
                dtype=np.float32,
            )
        else:
            base_vector = np.zeros(len(param_names), dtype=np.float32)

        best_result = None

        for restart in range(RESTARTS):
            seed_base = BASE_SEED + depth_idx * RESTARTS + restart
            rng = np.random.default_rng(seed_base)

            init_weights = base_vector + rng.normal(
                0.0,
                INIT_NOISE_SCALE,
                size=len(param_names),
            ).astype(np.float32)

            energy, pr, trace = adam_train_energy_ansatz(
                grad_ops=grad_ops,
                steps=STEPS,
                lr=LR,
                seed=seed_base,
                verbose=VERBOSE_TRAIN,
                initial_weights=init_weights,
                init_noise_scale=INIT_NOISE_SCALE,
            )

            psi = circuit.get_qs(pr=pr)
            fid = fidelity(psi, gs)

            if (best_result is None) or (fid > best_result['fidelity']):
                best_result = {
                    'depth': depth,
                    'fidelity': fid,
                    'energy': energy,
                    'pr': pr,
                    'trace': trace,
                    'restart': restart,
                }

        if WARM_START and best_result is not None:
            last_best_pr = best_result['pr']

        if best_result is not None:
            fid_by_depth[depth] = float(best_result['fidelity'])
            print(
                f"  depth = {depth:2d} | SH-VQE best restart #{best_result['restart']:02d} "
                f"| fid = {best_result['fidelity']:.3f} | E = {best_result['energy']:.6f}"
            )
            results.append(best_result)
        else:
            fid_by_depth[depth] = float('nan')

    result_fid = [fid_by_depth.get(d, float('nan')) for d in depths]
    return results, result_fid


# ===================== 可视化函数 =====================

def plot_fig3d(data_dict, depths_dict, figsize=(6, 5), dpi=150):
    """
    绘制Fig3d的结果
    data_dict: {qubit: {'vqe': [...], 'sh': [...]}}
    depths_dict: {qubit: [depth_list]}
    """
    colors = {8:"tab:blue", 10:"tab:green", 12:"tab:red", 14:"tab:orange", 16:"tab:purple"}
    markers = {"vqe": "o", "sh": "*"}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    handles_vqe, labels_vqe = [], []
    handles_sh,  labels_sh  = [], []

    for n in sorted(data_dict):
        for method in ("vqe", "sh"):
            if method not in data_dict[n]:
                continue
            y = data_dict[n][method]
            x = depths_dict[n][:len(y)]

            (line,) = ax.plot(
                x, y,
                marker=markers[method],
                color=colors[n],
                markersize=6 if method == "vqe" else 8,
                markerfacecolor="none" if method == "vqe" else None,
                linewidth=1.6,
            )

            if method == "vqe":
                handles_vqe.append(line)
                labels_vqe.append(f"n={n}")
            else:
                handles_sh.append(line)
                labels_sh.append(f"n={n}")

    ax.set_xlabel("Schrödinger circuit depth")
    ax.set_ylabel("Fidelity")
    ax.set_title("Comparison of solved ground-state fidelities")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(0.0, 1.0)

    leg_sh = ax.legend(
        handles_sh, labels_sh, title="SH-VQE",
        loc="lower right", bbox_to_anchor=(1.0, 0.00), frameon=True
    )
    ax.add_artist(leg_sh)

    leg_vqe = ax.legend(
        handles_vqe, labels_vqe, title="VQE",
        loc="lower right", bbox_to_anchor=(1.0, 0.36), frameon=True
    )

    plt.tight_layout()
    return fig

