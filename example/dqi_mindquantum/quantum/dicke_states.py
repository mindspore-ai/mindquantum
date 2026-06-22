"""
Dicke 态制备电路模块。

Dicke state preparation circuits for DQI using MindQuantum.

Dicke 态 |D_k^m⟩ 是所有 m 比特、汉明重量为 k 的字符串的均匀叠加：
  |D_k^m⟩ = (1/√C(m,k)) Σ_{|y|=k} |y⟩

本模块使用 SCS (Split & Cyclic Shift) 门的递推分解实现 Dicke 态制备：
  递推关系：|D_k^m⟩ = √(k/m) |D_{k-1}^{m-1}⟩|1⟩ + √((m-k)/m) |D_k^{m-1}⟩|0⟩

实现方式：
  - build_controlled_scs: 受控 SCS 门，实现递推的一步
  - build_controlled_dicke_prep: 受控 Dicke 态制备
    Stage 1: SCS(m,k) → SCS(m-1,k) → ... → SCS(k+1,k) [降序]
    Stage 2: SCS(k,k-1) → SCS(k-1,k-2) → ... → SCS(2,1) [细化]
"""

import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X, RY


def build_controlled_scs(m_val, k, ctrl_wires):
    """
    Build a controlled SCS (Split & Cyclic Shift) gate.

    SCS_{m,k} implements one step of the inductive Dicke state preparation.
    The recurrence is:
        |D_k^m> = sqrt(k/m) |D_{k-1}^{m-1}>|1> + sqrt((m-k)/m) |D_k^{m-1}>|0>

    Args:
        m_val: total number of qubits in this SCS stage.
        k: target Hamming weight.
        ctrl_wires: list of control qubit indices.

    Returns:
        circ: Circuit implementing the controlled SCS gate.
    """
    circ = Circuit()
    m_idx = m_val + len(ctrl_wires) - 1  # top qubit in the SCS group
    cw = list(ctrl_wires)

    # Level 1: CRY(2*arccos(sqrt(1/m))) on the first qubit
    angle = 2.0 * np.arccos(np.sqrt(1.0 / m_val))
    circ += X.on(m_idx, [m_idx - 1] + cw)
    circ += RY(angle).on(m_idx - 1, [m_idx] + cw)
    circ += X.on(m_idx, [m_idx - 1] + cw)

    # Levels 2..k: CCRY(2*arccos(sqrt(l/m))) on deeper qubits
    for lv in range(2, k + 1):
        angle_l = 2.0 * np.arccos(np.sqrt(float(lv) / m_val))
        circ += X.on(m_idx, [m_idx - lv] + cw)
        circ += RY(angle_l).on(m_idx - lv, [m_idx, m_idx - lv + 1] + cw)
        circ += X.on(m_idx, [m_idx - lv] + cw)

    return circ


def build_controlled_dicke_prep(m_val, k, ctrl_wires, ctrl_vals):
    """
    Build a controlled Dicke state |D_k^m> preparation circuit.

    Prepares the equal superposition of all m-bit strings with Hamming
    weight k, controlled on the weight register qubits matching ctrl_vals.

    The preparation uses the inductive decomposition:
        Stage 1: SCS(m, k), SCS(m-1, k), ..., SCS(k+1, k)  [descending]
        Stage 2: SCS(k, k-1), SCS(k-1, k-2), ..., SCS(2, 1)  [refinement]

    Args:
        m_val: number of qubits in the error register.
        k: target Hamming weight.
        ctrl_wires: list of control qubit indices (from weight register).
        ctrl_vals: list of control values (0 or 1) for each control qubit.

    Returns:
        circ: Circuit that prepares the controlled Dicke state.
    """
    circ = Circuit()
    cw_list = list(ctrl_wires)

    # Apply X gates for zero-valued controls (bit-flip-apply-unflip pattern)
    for cw, cv in zip(cw_list, ctrl_vals):
        if cv == 0:
            circ += X.on(cw)

    # Initialize all error qubits in the SCS range to |1>
    for wire_idx in range(m_val - k + len(cw_list), m_val + len(cw_list)):
        circ += X.on(wire_idx, cw_list)

    # Stage 1: descending SCS from m to k+1
    for i in reversed(range(k + 1, m_val + 1)):
        circ += build_controlled_scs(i, k, cw_list)

    # Stage 2: refinement SCS from k down to 2
    for i in reversed(range(2, k + 1)):
        circ += build_controlled_scs(i, i - 1, cw_list)

    # Restore control qubits
    for cw, cv in zip(cw_list, ctrl_vals):
        if cv == 0:
            circ += X.on(cw)

    return circ
