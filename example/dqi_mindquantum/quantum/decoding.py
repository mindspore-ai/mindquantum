# pylint: disable=unused-argument, unused-variable, invalid-name, consider-using-enumerate, unnecessary-list-index-lookup, wrong-import-position, broad-exception-caught, use-dict-literal, no-else-return, superfluous-parens
"""

DQI 量子电路的解码策略模块。

Decoding strategies for DQI quantum circuits.

本模块提供两种解码策略：
  1. LUT (Lookup Table) 解码器：建立综合征→错误模式的查找表
     - 门数复杂度：O(m · 2^n)，指数级
     - 适用于小规模问题 (n ≤ 15)
  2. BP1 (Belief Propagation) 解码器：量子相干的 Gallager 硬判决解码
     - 门数复杂度：O(T · m · d)，多项式级
     - 适用于大规模问题

解码器的作用 (Role of decoder):
  在 DQI 电路中，解码器负责利用综合征信息将 error register 反计算为 |0⟩。
  这一步骤确保了 Hadamard 变换后的相位相干性，是 DQI 算法的关键。
"""

import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X



def _compute_and_gates(circ, syndrome_register, constraint_mat, aux_start):
    """Compute AND gates for connected syndrome qubits."""
    m, n = constraint_mat.shape
    and_qubits = []
    aux_idx = aux_start
    for j in range(m):
        connected = [i for i in range(n) if constraint_mat[j, i] == 1]
        if not connected:
            and_qubits.append(None)
            continue
        and_qubit = aux_idx
        aux_idx += 1
        and_qubits.append(and_qubit)
        ctrl_qubits = [syndrome_register[i] for i in connected]
        circ += X.on(and_qubit, ctrl_qubits)
    return circ, and_qubits, aux_idx


def build_bp1_decode(_constraint_mat, syndrome_register, error_register, aux_start=0):
    """
    Build a quantum coherent BP1 (belief propagation) decoder circuit.

    Implements one iteration of hard-decision Gallager-style belief
    propagation as a quantum circuit. For each error qubit:
         1. Compute AND of connected syndrome qubits
         2. Store in flip register
         3. Apply flip to error register
         4. Uncompute flip register (disentangle aux from error)
         5. Uncompute AND gates
         6. Recompute syndrome from corrected error

    All auxiliary qubits are properly uncomputed to |0> so that
    unmeasured qubits do not destroy phase coherence during sampling.

    Args:
        constraint_mat: ndarray of shape (m, n), constraint matrix.
        syndrome_register: list of syndrome qubit indices (length n).
        error_register: list of error qubit indices (length m).
        aux_start: starting index for auxiliary qubits.

    Returns:
        circ: BP1 decoding circuit.
        aux_used: number of auxiliary qubits used.
    """
    m, n = _constraint_mat.shape
    circ = Circuit()
    flip_register = list(range(aux_start, aux_start + m))
    and_start = aux_start + m

    circ, and_qubits, aux_idx = _compute_and_gates(
        circ, syndrome_register, _constraint_mat, and_start
    )

    # Store flip candidates, apply flips, then uncompute
    for j in range(m):
        if and_qubits[j] is not None:
            circ += X.on(flip_register[j], and_qubits[j])
    for j in range(m):
        circ += X.on(error_register[j], flip_register[j])
    for j in range(m):
        if and_qubits[j] is not None:
            circ += X.on(flip_register[j], and_qubits[j])

    # Uncompute AND gates
    for j in range(m):
        if and_qubits[j] is not None:
            connected = [i for i in range(n) if _constraint_mat[j, i] == 1]
            ctrl_qubits = [syndrome_register[i] for i in connected]
            circ += X.on(and_qubits[j], ctrl_qubits)

    # Recompute syndrome from corrected error
    for i in range(n):
        for j in range(m):
            if _constraint_mat[j, i] == 1:
                circ += X.on(syndrome_register[i], error_register[j])

    aux_used = aux_idx - aux_start
    return circ, aux_used


def syndrome_lookup_table(constraint_mat_t):
    """
    Build a lookup table mapping each syndrome to the minimum-weight
    error pattern that produces it.

    The table maps: syndrome (length n) -> error (length m)
    where error = argmin_{y: B^T y = syndrome} |y|.

    Args:
        constraint_mat_t: ndarray of shape (n, m), transpose of constraint matrix.

    Returns:
        table: list of [syndrome, error] pairs, sorted by syndrome.
    """
    _, m = constraint_mat_t.shape
    syn_dict = {}

    for i in range(2**m):
        err_str = format(i, f"0{m}b")
        err_vec = np.array([int(b) for b in err_str])
        syn_vec = constraint_mat_t.dot(err_vec) % 2
        syn_str = "".join(map(str, syn_vec.astype(int)))

        if syn_str not in syn_dict:
            syn_dict[syn_str] = err_str
        else:
            existing_hw = sum(int(b) for b in syn_dict[syn_str])
            if int(np.sum(err_vec)) < existing_hw:
                syn_dict[syn_str] = err_str

    table = []
    for s, e in syn_dict.items():
        table.append([[int(b) for b in s], [int(b) for b in e]])
    table.sort(key=lambda x: int("".join(map(str, x[0])), 2))
    return table


def _add_multi_ctrl_x(circ, target, controls, control_values):
    """
    Add a multi-controlled X gate with specified control values.
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


def build_lut_decode(decoding_table, error_register, syndrome_register):
    """
    Build a LUT (Lookup Table) decoding circuit.

    For each (syndrome, error) pair in the table, applies multi-controlled
    X gates to flip the corresponding error qubits.

    Args:
        decoding_table: list of [syndrome, error] pairs from syndrome_lookup_table.
        error_register: list of error qubit indices (length m).
        syndrome_register: list of syndrome qubit indices (length n).

    Returns:
        circ: LUT decoding circuit.
    """
    circ = Circuit()
    for syndrome, error in decoding_table:
        for i in range(len(error)):
            if error[i] == 1:
                circ = _add_multi_ctrl_x(
                    circ, error_register[i], syndrome_register, syndrome
                )
    return circ


def build_quantum_bp1_circuit(
    constraint_mat, syndrome_register, error_register, max_iter=1, aux_start=0
):
    """
    Build a quantum coherent BP1 decoder circuit.
    Args:
        constraint_mat: ndarray of shape (m, n), constraint matrix.
        syndrome_register: list of syndrome qubit indices.
        error_register: list of error qubit indices.
        max_iter: number of BP1 iterations (default 1).
        aux_start: starting index for auxiliary qubits.
    Returns:
        circ: Quantum BP1 circuit.
        aux_register: list of auxiliary qubit indices used.
    """
    m, n = constraint_mat.shape
    circ = Circuit()
    aux_register = list(range(aux_start, aux_start + m))

    for t in range(max_iter):
        circ = _apply_bp1_iteration(circ, constraint_mat, syndrome_register, error_register, (aux_register, m, n))
        if max_iter > 1 and t < max_iter - 1:
            aux_register = [r + m for r in aux_register]

    return circ, aux_register


def _apply_bp1_iteration(circ, constraint_mat, syndrome_register, error_register, aux_info):
    """Apply one iteration of BP1 decoding."""
    aux_register, m, n = aux_info
    for j in range(m):
        connected = [i for i in range(n) if constraint_mat[j, i] == 1]
        if not connected:
            continue
        ctrl_qubits = [syndrome_register[i] for i in connected]
        if len(ctrl_qubits) > 1:
            circ += X.on(aux_register[j], ctrl_qubits)
        elif len(ctrl_qubits) == 1:
            circ += X.on(aux_register[j], ctrl_qubits[0])

    for j in range(m):
        circ += X.on(error_register[j], aux_register[j])

    for j in range(m):
        connected = [i for i in range(n) if constraint_mat[j, i] == 1]
        for i in connected:
            circ += X.on(syndrome_register[i], aux_register[j])

    return circ
