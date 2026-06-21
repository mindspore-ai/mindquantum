"""
完整 DQI 量子电路组装与模拟模块。

Full DQI quantum circuit assembly and simulation using MindQuantum.

本模块提供 DQICircuit 类，负责组装所有子组件并运行量子模拟。
对应论文中的完整 DQI 算法流程（Algorithm 1）。

电路架构 (Circuit architecture):
  与 Qiskit 参考实现一致：WeightedUnaryEncoding 和 UnkStatePreparation
  顺序应用于同一组量子比特 (error_register)，产生正确的加权 Dicke 态叠加：
    |ψ⟩ = Σ_k w_k |u_{m,k}⟩

寄存器布局 (Register layout):
  - error_register: m 个量子比特（错误模式寄存器）
  - syndrome_register: n 个量子比特（综合征寄存器，存储 B^T y）
  - bp1_aux: 2m 个量子比特（BP1 解码器辅助比特）
  - 总计: 3m + n 个量子比特

算法步骤 (Algorithm steps):
  1. 权重编码 (Weight encoding): 在 error_register 上制备 Σ w_k |k⟩ 叠加态
  2. Dicke 态制备 (Dicke state prep): 将 |k⟩ 展开为 |D_k^m⟩ 均匀叠加
  3. 相位 Z 编码 (Phase Z): 应用 (-1)^{v·y} 相位
  4. 综合征计算 (Syndrome): 计算 s = B^T y mod 2
  5. BP1 解码 (BP1 decode): 将 error register 反计算为 |0⟩
  6. Hadamard 变换: 在 syndrome register 上应用 H 门
"""

import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import Measure
from mindquantum.simulator import Simulator

from .weight_encoding import compute_optimal_w
from .unk_state import build_weighted_unary_encoding, build_unk_state_preparation
from .circuit_components import build_phase_z, build_syndrome_compute, build_hadamard
from .decoding import build_bp1_decode


class DQICircuit:
    """
    DQI 量子电路构建器与模拟器。

    DQI quantum circuit builder and simulator.

    与 Qiskit 参考实现一致：WeightedUnaryEncoding 和 UnkStatePreparation
    顺序应用于 error_register，产生加权 Dicke 态叠加。

    寄存器布局 (Register layout):
        - error_register: m 个量子比特 (错误模式)
        - syndrome_register: n 个量子比特 (B^T y 综合征)
        - bp1_aux: 2m 个量子比特 (BP1 解码器辅助)

    Args:
        constraint_mat: ndarray of shape (m, n), 约束矩阵。
        v: ndarray of shape (m,), 右端向量。
        ell: int, 多项式度数 (最大错误权重)。
    """

    def __init__(self, constraint_mat, v, ell=2):
        self.constraint_mat = np.array(constraint_mat)
        self.v = np.array(v)
        self.m, self.n = self.constraint_mat.shape
        self.ell = ell

        # Register layout: error + syndrome + bp1_aux
        offset = 0
        self.error_register = list(range(offset, offset + self.m))
        offset += self.m
        self.syndrome_register = list(range(offset, offset + self.n))
        offset += self.n

        # BP1 auxiliary qubits
        self.bp1_aux_start = offset
        self.bp1_aux_count = 2 * self.m
        offset += self.bp1_aux_count

        self.total_qubits = offset

        # Compute optimal weights
        self.w_k = compute_optimal_w(self.m, self.ell)

    def build_circuit(self):
        """
        构建完整的 DQI 量子电路。

        Build the complete DQI quantum circuit.

        六步流程 (对应论文 Algorithm 1):
          Step 1: 权重编码 — 在 error_register 上制备 w_k 叠加态
                  使用 RY/CRY 级联门实现
          Step 2: Dicke 态制备 — 将 |k⟩ 展开为 |D_k^m⟩
                  使用受控 SCS 门的递推分解
          Step 3: 相位 Z 编码 — 应用 (-1)^{v_i} 相位到每个 error qubit
          Step 4: 综合征计算 — CNOT 网络计算 s = B^T y mod 2
          Step 5: BP1 解码 — 反计算 error register 为 |0⟩
                  使用 AND 门 + 翻转 + 反计算 + 综合征重算
          Step 6: Hadamard 变换 — 在 syndrome register 上应用 H 门

        Returns:
            circ: 完整的 DQI Circuit。
        """
        circ = Circuit()

        # Step 1: Weight encoding on error register
        circ += build_weighted_unary_encoding(self.w_k, self.error_register)

        # Step 2: Unk state preparation on error register (sequential, same qubits)
        max_errors = int(np.nonzero(self.w_k)[0][-1]) if np.any(self.w_k) else 0
        circ += build_unk_state_preparation(self.m, max_errors, self.error_register)

        # Step 3: Phase Z encoding
        circ += build_phase_z(self.v, self.error_register)

        # Step 4: Syndrome computation
        circ += build_syndrome_compute(
            self.constraint_mat, self.error_register, self.syndrome_register
        )

        # Step 5: BP1 decoder
        bp1_circ, _ = build_bp1_decode(
            self.constraint_mat,
            self.syndrome_register,
            self.error_register,
            aux_start=self.bp1_aux_start,
        )
        circ += bp1_circ

        # Step 6: Hadamard transform
        circ += build_hadamard(self.syndrome_register)

        return circ

    def add_measurements(self, circ):
        """Add measurement gates on error and syndrome registers."""
        for wire in self.error_register:
            circ += Measure(f"e{wire}").on(wire)
        for wire in self.syndrome_register:
            circ += Measure(f"s{wire}").on(wire)
        return circ

    def get_circuit_summary(self, circ):
        """Return a summary dict for the circuit."""
        return {
            "n_qubits": self.total_qubits,
            "n_gates": len(circ),
            "depth": circ.depth(),
            "n_measurements": sum(
                1
                for _ in circ
                if hasattr(_, "gate_type") and str(_.gate_type) == "Measure"
            ),
        }


def run_dqi_histogram(constraint_mat, v, ell, shots=10000, seed=None):
    """
    运行 DQI 量子模拟并返回约束满足直方图。

    Run DQI quantum simulation and return the histogram of satisfied constraints.

    后选择流程 (Post-selection):
      1. 测量所有 error_register 和 syndrome_register 量子比特
      2. 仅保留 error_register = |0⟩ 的测量结果（成功解码）
      3. 从 syndrome bits 计算满足的约束数：s = Σ_i [B·syndrome mod 2 == v]
      4. 统计 s 的概率分布

    Args:
        constraint_mat: ndarray of shape (m, n), 约束矩阵。
        v: ndarray of shape (m,), 右端向量。
        ell: int, 多项式度数。
        shots: int, 测量采样次数。
        seed: int, 随机种子（可复现）。

    Returns:
        result: dict, 包含以下键：
            's_hist': ndarray of length m+1, 每个 s 值的概率。
            'avg_s': float, 平均满足约束数 <S>。
            's_normalized': float, <S>/m 归一化值。
            'post_fraction': float, 后选择通过的比例。
    """
    m, n = constraint_mat.shape

    dqi = DQICircuit(constraint_mat, v, ell=ell)
    circ = dqi.build_circuit()
    dqi.add_measurements(circ)

    sim = Simulator("mqvector", dqi.total_qubits)
    result = sim.sampling(circ, shots=shots, seed=seed)

    s_hist = np.zeros(m + 1, dtype=float)
    total_post = 0

    for bitstring, count in result.data.items():
        bits = [int(b) for b in bitstring[::-1]]
        error_bits = bits[:m]

        if all(b == 0 for b in error_bits):
            syndrome_bits = np.array(bits[m : m + n])
            s_val = int(np.sum((constraint_mat.dot(syndrome_bits) % 2) == v))
            s_hist[s_val] += count
            total_post += count

    if total_post > 0:
        s_hist /= total_post

    avg_s = np.sum(np.arange(m + 1) * s_hist)

    return {
        "s_hist": s_hist,
        "avg_s": float(avg_s),
        "s_normalized": float(avg_s / m),
        "post_fraction": total_post / shots,
    }
