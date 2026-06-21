# pylint: disable=unused-argument, unused-variable, invalid-name, consider-using-enumerate, unnecessary-list-index-lookup, wrong-import-position, broad-exception-caught, use-dict-literal, no-else-return, superfluous-parens
"""

DQI 量子电路的权重编码与反计算模块。

Weight encoding and uncomputation for DQI quantum circuits.

本模块提供两个核心功能：
  1. compute_optimal_w: 计算 DQI 多项式的最优权重向量 w_k
     - w_k 是 (ℓ+1)×(ell+1) 三对角矩阵的主特征向量
     - 用于最大化 DQI 的期望约束满足数
  2. build_embed_weights: 构建权重编码电路
     - 使用 RY/CRY 级联门制备 Σ_k w_k |k⟩ 叠加态
     - 将经典权重向量编码到量子态中

辅助函数：
  - _add_multi_ctrl_x: 多控 X 门（支持指定控制值）
  - _generate_bit_strings: 生成指定汉明重量的所有二进制串
  - build_uncompute_weight: 权重寄存器反计算（枚举 C(m,k) 模式，仅适用于小规模）
"""

import math
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X, RY


def compute_optimal_w(m, ell, p=2, r=1):
    """
    Compute the optimal weight vector w_k for the DQI polynomial.

    This implementation matches the original get_optimal_w from
    dqi/pipelines/dicke_states.py.

    w_k are the components of the principal eigenvector of a sparse
    (ell+1) x (ell+1) tridiagonal matrix.

    Args:
        m: number of constraints.
        ell: polynomial degree (max error weight).
        p: field size (default 2 for GF(2)).
        r: constraint size (default 1 for XORSAT).

    Returns:
        w: ndarray, optimal weights padded to next power of 2.
    """
    d = (p - 2 * r) / np.sqrt(r * (p - r))
    diag = np.arange(ell + 1) * d
    off = np.sqrt(np.arange(1, ell + 1) * (m - np.arange(1, ell + 1) + 1))

    # Build sparse tridiagonal matrix
    weight_mat = diags([off, diag, off], offsets=(-1, 0, 1), format="csr")

    # Get principal eigenvector
    _, vecs = eigsh(A, k=1, which="LA")
    w = vecs.flatten()
    w /= np.linalg.norm(w)

    # Pad up to next power of two
    orig = w.size
    target = 1 << math.ceil(math.log2(orig))
    return np.pad(w, (0, target - orig))


def build_embed_weights(w_k, weight_register):
    """
    Build circuit to prepare sum_k w_k |k> on the weight register.

    Uses a cascade of RY gates. For ell <= 3 (2 weight qubits):
        RY(theta0) on q0 splits amplitude between even/odd parity groups
        Controlled-RY(theta1) on q1 (ctrl=q0=0) mixes |00> and |01>
        Controlled-RY(theta2) on q1 (ctrl=q0=1) mixes |10> and |11>

    Args:
        w_k: ndarray of optimal weights.
        weight_register: list of weight qubit indices.

    Returns:
        circ: Circuit for weight embedding.
    """
    circ = Circuit()
    w = np.asarray(w_k, dtype=float)
    n_w = len(weight_register)

    if n_w == 1:
        # Single weight qubit: |0> -> w0|0> + w1|1>
        theta = 2.0 * np.arccos(np.clip(w[0], -1.0, 1.0))
        circ += RY(theta).on(weight_register[0])
        return circ

    if n_w >= 2:
        # Qubit 0: create superposition of even and odd parity groups
        # |0> -> sqrt(w0^2+w2^2)|0> + sqrt(w1^2+w3^2)|1>
        alpha = np.sqrt(w[0] ** 2 + w[2] ** 2)
        theta0 = 2.0 * np.arccos(np.clip(alpha, -1.0, 1.0))
        circ += RY(theta0).on(weight_register[0])

        # Qubit 1, branch q0=|0>: mix |00> and |01>
        # After RY(theta1): cos(theta1/2)|0> + sin(theta1/2)|1>
        # We need cos^2(theta1/2) = w0/alpha, sin^2(theta1/2) = w1/alpha
        theta1 = 2.0 * np.arccos(np.clip(w[0] / alpha, -1.0, 1.0))
        circ += X.on(weight_register[0])
        circ += RY(theta1).on(weight_register[1], weight_register[0])
        circ += X.on(weight_register[0])

        # Qubit 1, branch q0=|1>: mix |10> and |11>
        # We need cos^2(theta2/2) = w2/beta, sin^2(theta2/2) = w3/beta
        beta = np.sqrt(w[2] ** 2 + w[3] ** 2)
        if beta > 1e-10:
            theta2 = 2.0 * np.arccos(np.clip(w[2] / beta, -1.0, 1.0))
            circ += RY(theta2).on(weight_register[1], weight_register[0])

    return circ


def _add_multi_ctrl_x(circ, target, controls, control_values):
    """
    Add a multi-controlled X gate with specified control values.

    For controls with value 0, wraps with X gates (flip-apply-unflip).

    Args:
        circ: Circuit to append to.
        target: target qubit index.
        controls: list of control qubit indices.
        control_values: list of 0/1 values for each control.

    Returns:
        circ: Modified circuit.
    """
    ctrls = list(controls)
    for cw, cv in zip(ctrls, control_values):
        if cv == 0:
            circ += X.on(cw)
    if len(ctrls) > 0:
        circ += X.on(target, ctrls)
    else:
        circ += X.on(target)
    for cw, cv in zip(ctrls, control_values):
        if cv == 0:
            circ += X.on(cw)
    return circ


def _generate_bit_strings(length, hamming_weight):
    """Generate all binary strings of given length and Hamming weight."""
    from itertools import combinations

    results = []
    for positions in combinations(range(length), hamming_weight):
        bs = [0] * length
        for pos in positions:
            bs[pos] = 1
        results.append(bs)
    return results


def build_uncompute_weight(m_val, k, weight_register, error_register):
    """
    Build circuit to uncompute (erase) the weight register.

    Uses multi-controlled X gates to detect each Dicke state pattern
    and reset the weight register.

    WARNING: Enumerates C(m, k) patterns. Only feasible for small m, k.

    Args:
        m_val: size of error register.
        k: Hamming weight to match.
        weight_register: list of weight qubit indices.
        error_register: list of error qubit indices.

    Returns:
        circ: Uncomputation circuit.
    """
    circ = Circuit()
    num_weight_qubits = len(weight_register)
    bit_strings = _generate_bit_strings(m_val, k)

    for bs in bit_strings:
        for j in range(num_weight_qubits):
            qubit_state = (k >> j) & 1
            if qubit_state == 1:
                circ = _add_multi_ctrl_x(circ, weight_register[j], error_register, bs)
    return circ
