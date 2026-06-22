# pylint: disable=unused-argument, unused-variable, invalid-name, consider-using-enumerate, unnecessary-list-index-lookup, wrong-import-position, broad-exception-caught, use-dict-literal, no-else-return, superfluous-parens
"""

未知态制备电路模块。

Unknown state preparation circuits for DQI using MindQuantum.

本模块实现了 Qiskit 参考实现中的 WeightedUnaryEncoding 和 UnkStatePreparation
门到 MindQuantum 的翻译。

核心功能：
  - build_weighted_unary_encoding: 权重一元编码，制备 Σ_l w[l] |l⟩
  - build_scs2/scs3/scsnk: SCS (Split & Cyclic Shift) 门
  - build_unk_state_preparation: 未知态制备（两阶段递推分解）
  - 受控变体：用于 Dicke 态制备的受控版本

注意：MindQuantum 的 RY 门默认 ctrl_state="1"（与 Qiskit 一致）。
X 门也默认 ctrl_state="1"。当期望的控制态为 |1⟩ 时，不需要 X-flip 包装。
"""

import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X, RY


def build_weighted_unary_encoding(weights, qubit_register):
    """
    Build circuit for WeightedUnaryEncoding: creates sum_l w[l] |l>.

    Uses a chain of RY/CRY gates:
        RY(beta_0) on qubit 0
        CRY(beta_1) controlled on qubit 0, targeting qubit 1
        CRY(beta_2) controlled on qubit 1, targeting qubit 2
        ...

    The rotation angles are:
        beta_l = 2 * arccos(sqrt(w[l]^2 / (1 - sum_{j<l} w[j]^2)))

    Args:
        weights: ndarray of weights (normalized).
        qubit_register: list of qubit indices (length >= len(weights)).

    Returns:
        circ: WeightedUnaryEncoding circuit.
    """
    circ = Circuit()
    w = np.asarray(weights, dtype=float)
    n_qubits = len(qubit_register)

    # Compute rotation angles
    betas = []
    for k_idx, _ in enumerate(w):
        denom = 1.0 - np.sum(w[:k_idx] ** 2)
        ratio = (w[k_idx] ** 2) / denom if denom > 1e-12 else 0.0
        value = min(np.sqrt(max(ratio, 0.0)), 1.0)
        beta = 2.0 * np.arccos(value)
        betas.append(beta)

    # Apply RY on first qubit
    if n_qubits > 0 and len(betas) > 0 and betas[0] > 1e-12:
        circ += RY(betas[0]).on(qubit_register[0])

    # Apply CRY on subsequent qubits
    for i in range(1, min(n_qubits, len(betas))):
        if np.isnan(betas[i]) or betas[i] < 1e-12:
            continue
        # CRY(beta_i) controlled on qubit i-1, targeting qubit i
        # MindQuantum default ctrl_state for RY is "1" (matching Qiskit)
        circ += RY(betas[i]).on(qubit_register[i], qubit_register[i - 1])

    return circ


def build_scs2(n_val, qubit_indices):
    """
    Build SCS2 gate for 2 qubits.

    SCS2(n): CNOT(1→0), CRY(2*arccos(1/sqrt(n)) ctrl=0→1), CNOT(1→0)

    Matches the Qiskit SCS2Gate definition:
      - CNOT(ctrl=q1, tgt=q0)
      - CRY(theta, ctrl=q0, tgt=q1) with ctrl_state="1"
      - CNOT(ctrl=q1, tgt=q0)

    Args:
        n_val: parameter n for rotation angle.
        qubit_indices: [q0, q1].

    Returns:
        circ: SCS2 circuit.
    """
    circ = Circuit()
    q0, q1 = qubit_indices[0], qubit_indices[1]

    # CNOT(q1 → q0)
    circ += X.on(q0, q1)

    # CRY(theta, ctrl=q0, target=q1) with ctrl_state="1"
    # MindQuantum default ctrl_state for RY is "1" (matching Qiskit)
    theta = 2.0 * np.arccos(1.0 / np.sqrt(n_val))
    circ += RY(theta).on(q1, q0)

    # CNOT(q1 → q0)
    circ += X.on(q0, q1)

    return circ


def build_scs3(ell_val, n_val, qubit_indices):
    """
    Build SCS3 gate for 3 qubits.

    SCS3(ell, n): CNOT(2→0), CCY(2*arccos(sqrt(ell/n)) ctrl=[1,2]→0), CNOT(2→0)

    Args:
        ell_val: parameter ell for rotation angle.
        n_val: parameter n for rotation angle.
        qubit_indices: [q0, q1, q2] where q0 is target, q1,q2 are controls.

    Returns:
        circ: SCS3 circuit.
    """
    circ = Circuit()
    q0, q1, q2 = qubit_indices[0], qubit_indices[1], qubit_indices[2]

    # CNOT(q2 → q0)
    circ += X.on(q0, q2)

    # CCRY(theta, ctrl=[q1,q2], target=q0) with ctrl_state="11"
    # MindQuantum default ctrl_state for RY is "1" (matching Qiskit)
    theta = 2.0 * np.arccos(np.sqrt(ell_val / n_val))
    circ += RY(theta).on(q0, [q1, q2])

    # CNOT(q2 → q0)
    circ += X.on(q0, q2)

    return circ


def build_scsnk(n_val, k_val, qubit_indices):
    """
    Build composite SCS gate for k+1 qubits.

    SCS_{n,k} = SCS2(n) on [0,1] + SCS3(ell,n) on [0,ell-1,ell] for ell=2..k

    Args:
        n_val: parameter n for rotation angles.
        k_val: parameter k determining number of qubits.
        qubit_indices: list of k+1 qubit indices.

    Returns:
        circ: SCS_{n,k} circuit.
    """
    circ = Circuit()

    # SCS2 on first two qubits
    circ += build_scs2(n_val, qubit_indices[:2])

    # SCS3 on qubits [0, ell-1, ell] for ell = 2..k
    for ell in range(2, k_val + 1):
        circ += build_scs3(
            ell, n_val, [qubit_indices[0], qubit_indices[ell - 1], qubit_indices[ell]]
        )

    return circ


def build_unk_state_preparation(n_val, k_val, qubit_register):
    """
    Build UnkStatePreparation circuit.

    Stage 1: SCS_{ell,k} on qubits [n-ell .. n-ell+k] for ell = n, n-1, ..., k+1
    Stage 2: SCS_{ell,ell-1} on qubits [n-ell .. n-1] for ell = k, k-1, ..., 2

    Args:
        n_val: number of qubits.
        k_val: parameter k.
        qubit_register: list of qubit indices (length >= n_val).

    Returns:
        circ: UnkStatePreparation circuit.
    """
    circ = Circuit()

    # Stage 1: descending from n to k+1
    for ell in range(n_val, k_val, -1):
        start = n_val - ell
        indices = [qubit_register[start + i] for i in range(k_val + 1)]
        circ += build_scsnk(ell, k_val, indices)

    # Stage 2: refinement from k down to 2
    for ell in range(k_val, 1, -1):
        start = n_val - ell
        indices = [qubit_register[start + i] for i in range(ell)]
        circ += build_scsnk(ell, ell - 1, indices)

    return circ


def build_controlled_unk_state_preparation(
    n_val, k_val, qubit_register, ctrl_register, ctrl_vals
):
    """
    Build controlled UnkStatePreparation circuit.

    Same as UnkStatePreparation but controlled on ctrl_register matching ctrl_vals.

    Args:
        n_val: number of qubits in the target register.
        k_val: parameter k.
        qubit_register: list of target qubit indices.
        ctrl_register: list of control qubit indices.
        ctrl_vals: list of control values (0 or 1).

    Returns:
        circ: Controlled UnkStatePreparation circuit.
    """
    circ = Circuit()

    # Flip zero-valued controls
    for cw, cv in zip(ctrl_register, ctrl_vals):
        if cv == 0:
            circ += X.on(cw)

    # Stage 1: descending from n to k+1
    for ell in range(n_val, k_val, -1):
        start = n_val - ell
        indices = [qubit_register[start + i] for i in range(k_val + 1)]
        circ += build_controlled_scsnk(ell, k_val, indices, ctrl_register)

    # Stage 2: refinement from k down to 2
    for ell in range(k_val, 1, -1):
        start = n_val - ell
        indices = [qubit_register[start + i] for i in range(ell)]
        circ += build_controlled_scsnk(ell, ell - 1, indices, ctrl_register)

    # Restore zero-valued controls
    for cw, cv in zip(ctrl_register, ctrl_vals):
        if cv == 0:
            circ += X.on(cw)

    return circ


def build_controlled_scsnk(n_val, k_val, qubit_indices, ctrl_register):
    """
    Build controlled composite SCS gate.

    Args:
        n_val: parameter n.
        k_val: parameter k.
        qubit_indices: list of target qubit indices.
        ctrl_register: list of control qubit indices.

    Returns:
        circ: Controlled SCS_{n,k} circuit.
    """
    circ = Circuit()
    cw = list(ctrl_register)

    # SCS2 on first two qubits
    circ += build_controlled_scs2(n_val, qubit_indices[:2], cw)

    # SCS3 on qubits [0, ell-1, ell] for ell = 2..k
    for ell in range(2, k_val + 1):
        circ += build_controlled_scs3(
            ell,
            n_val,
            [qubit_indices[0], qubit_indices[ell - 1], qubit_indices[ell]],
            cw,
        )

    return circ


def build_controlled_scs2(n_val, qubit_indices, ctrl_register):
    """
    Build controlled SCS2 gate.

    Args:
        n_val: parameter n.
        qubit_indices: [q0, q1] target qubits.
        ctrl_register: list of control qubit indices.

    Returns:
        circ: Controlled SCS2 circuit.
    """
    circ = Circuit()
    q0, q1 = qubit_indices[0], qubit_indices[1]
    cw = list(ctrl_register)

    # CNOT(q1 → q0) controlled on ctrl_register
    circ += X.on(q0, [q1] + cw)

    # CRY(theta, ctrl=q0, target=q1) with ctrl_state="1" and additional controls
    # MindQuantum default ctrl_state for RY is "1" (matching Qiskit)
    theta = 2.0 * np.arccos(1.0 / np.sqrt(n_val))
    circ += RY(theta).on(q1, [q0] + cw)

    # CNOT(q1 → q0) controlled on ctrl_register
    circ += X.on(q0, [q1] + cw)

    return circ


def build_controlled_scs3(ell_val, n_val, qubit_indices, ctrl_register):
    """
    Build controlled SCS3 gate.

    Args:
        ell_val: parameter ell.
        n_val: parameter n.
        qubit_indices: [q0, q1, q2] target qubits.
        ctrl_register: list of control qubit indices.

    Returns:
        circ: Controlled SCS3 circuit.
    """
    circ = Circuit()
    q0, q1, q2 = qubit_indices[0], qubit_indices[1], qubit_indices[2]
    cw = list(ctrl_register)

    # CNOT(q2 → q0) controlled on ctrl_register
    circ += X.on(q0, [q2] + cw)

    # CCRY(theta, ctrl=[q1,q2], target=q0) with ctrl_state="11" and additional controls
    # MindQuantum default ctrl_state for RY is "1" (matching Qiskit)
    theta = 2.0 * np.arccos(np.sqrt(ell_val / n_val))
    circ += RY(theta).on(q0, [q1, q2] + cw)

    # CNOT(q2 → q0) controlled on ctrl_register
    circ += X.on(q0, [q2] + cw)

    return circ
