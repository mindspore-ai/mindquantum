# pylint: disable=unused-argument, unused-variable, invalid-name, consider-using-enumerate, unnecessary-list-index-lookup, wrong-import-position, broad-exception-caught, use-dict-literal, no-else-return, superfluous-parens
"""

DQI 量子电路的可复用组件模块。

Reusable quantum circuit components for DQI.

本模块提供 DQI 电路的基础构建块：
  1. build_phase_z: 相位 Z 编码 — 对 v_i=1 的 error qubit 应用 Z 门
     实现 (-1)^{v·y} 相位编码
  2. build_syndrome_compute: 综合征计算 — CNOT 网络计算 s = B^T y mod 2
  3. build_syndrome_uncompute: 综合征反计算 — 清除 syndrome register
  4. build_hadamard: Hadamard 变换 — 将计算基转换为傅里叶基
"""

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, Z, X


def build_phase_z(v, error_register):
    """
    Apply Z gates to encode the constraint RHS vector v.

    For each i where v[i] == 1, applies Z on the corresponding error qubit.
    This imparts phase (-1)^{v·y} to the error state |y>.

    Args:
        v: ndarray of shape (m,), binary RHS vector.
        error_register: list of error qubit indices (length m).

    Returns:
        circ: Phase encoding circuit.
    """
    circ = Circuit()
    for i, _ in enumerate(v):
        if v[i] == 1:
            circ += Z.on(error_register[i])
    return circ


def build_syndrome_compute(_constraint_mat, error_register, syndrome_register):
    """
    Build circuit to compute syndrome s = B^T y (mod 2).

    Uses a CNOT network: for each 1 at B[i,j], apply CX(error[j], syndrome[i]).

    Args:
        B: ndarray of shape (m, n), constraint matrix.
        error_register: list of error qubit indices (length m).
        syndrome_register: list of syndrome qubit indices (length n).

    Returns:
        circ: Syndrome computation circuit.
    """
    circ = Circuit()
    m, n = B.shape
    for row_idx in range(n):
        for col_idx in range(m):
            if B[col_idx, row_idx] == 1:
                circ += X.on(syndrome_register[row_idx], error_register[col_idx])
    return circ


def build_syndrome_uncompute(_constraint_mat, error_register, syndrome_register):
    """
    Uncompute syndrome register: set it back to |0⟩ from B^T y.

    This is the same CNOT network as syndrome computation (CNOT is self-inverse).
    Used after the decoder to clear the syndrome register before Hadamard.

    Args:
        B: ndarray of shape (m, n), constraint matrix.
        error_register: list of error qubit indices (length m).
        syndrome_register: list of syndrome qubit indices (length n).

    Returns:
        circ: Syndrome uncomputation circuit.
    """
    circ = Circuit()
    m, n = B.shape
    for row_idx in range(n):
        for col_idx in range(m):
            if B[col_idx, row_idx] == 1:
                circ += X.on(syndrome_register[row_idx], error_register[col_idx])
    return circ


def build_hadamard(syndrome_register):
    """
    Apply Hadamard gates to all syndrome qubits.

    This transforms the computational basis into the Fourier basis,
    converting the encoded polynomial into a sampling distribution.

    Args:
        syndrome_register: list of syndrome qubit indices.

    Returns:
        circ: Hadamard transform circuit.
    """
    circ = Circuit()
    for wire in syndrome_register:
        circ += H.on(wire)
    return circ
