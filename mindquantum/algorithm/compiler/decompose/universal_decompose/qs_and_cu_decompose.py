# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Quantum Shannon Decomposition."""
# pylint: disable=invalid-name
from typing import List

import numpy as np
from scipy import linalg

from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import BarrierGate, QuantumGate
from mindquantum.utils.f import is_power_of_two

from .. import utils
from ..fixed_decompose import ccx_decompose
from ..utils import optimize_circuit
from .one_qubit_decompose import euler_decompose
from .two_qubit_decompose import abc_decompose


def cu_decompose(gate: QuantumGate, with_barrier: bool = False) -> Circuit:
    """
    Decompose arbitrary-dimension controlled-U gate.

    This gate has :math:`m` control qubits and :math:`n` target qubits.
    When recursively calling the function itself, :math:`m` decreases, while :math:`n` holds constant.

    Args:
        gate (:class:`~.core.gates.QuantumGate`): instance of quantum gate.
        with_barrier (bool): whether add :class:`~.core.gates.BarrierGate` into
            decomposed circuit. Default: False.

    Returns:
        :class:`~.core.circuit.Circuit`, composed of 1-qubit gates and CNOT gates.

    Examples:
        >>> import mindquantum as mq
        >>> from mindquantum.algorithm.compiler import cu_decompose
        >>> from scipy.stats import unitary_group
        >>> cqs = [0, 2, 4, 5]
        >>> tqs = [1, 6]
        >>> m = len(cqs)
        >>> n = len(tqs)
        >>> u = unitary_group.rvs(2 ** n, random_state=123)
        >>> g = mq.UnivMathGate('U', u).on(tqs, cqs)
        >>> circ = cu_decompose(g)
        >>> num_cnot = len([g for g in circ if isinstance(g, mq.XGate) and len(g.ctrl_qubits)==1])
        >>> print('total gate number: {}, CNOT number: {}'.format(len(circ), num_cnot))
        total gate number: 632, CNOT number: 314
    """
    m = len(gate.ctrl_qubits)
    n = len(gate.obj_qubits)

    if m == 0:
        return qs_decompose(gate, with_barrier)

    if m == 1:
        if n == 1:
            # normal 2-qubit controlled-U gate
            if isinstance(gate, gates.XGate):
                return Circuit() + gate
            return abc_decompose(gate)
        if n > 1:
            # 1 control qubits, 2+ target qubits
            cu = utils.controlled_unitary_matrix(gate.matrix())
            return qs_decompose(gates.UnivMathGate('CU', cu).on(gate.ctrl_qubits + gate.obj_qubits), with_barrier)

    if m == 2 and isinstance(gate, gates.XGate) and n == 1:
        # Toffoli gate
        return ccx_decompose(gate)[0]

    v = linalg.sqrtm(gate.matrix())
    vh = np.conj(np.transpose(v))
    cqs_1st, cq_2nd = gate.ctrl_qubits[:-1], gate.ctrl_qubits[-1]
    circ = Circuit()
    circ += cu_decompose(gates.UnivMathGate('V', v).on(gate.obj_qubits, cq_2nd), with_barrier)
    circ += cu_decompose(gates.X.on(cq_2nd, cqs_1st))
    circ += cu_decompose(gates.UnivMathGate('Vh', vh).on(gate.obj_qubits, cq_2nd), with_barrier)
    circ += cu_decompose(gates.X.on(cq_2nd, cqs_1st))
    circ += cu_decompose(gates.UnivMathGate('V', v).on(gate.obj_qubits, cqs_1st), with_barrier)
    return circ


def qs_decompose(gate: QuantumGate, with_barrier: bool = False) -> Circuit:
    r"""
    Quantum Shannon decomposition for arbitrary-dimension unitary gate.

    The number of CNOT gates in the decomposed circuit is:

    .. math::

        O(4^n)

    For more detail, please refer to `Synthesis of Quantum
    Logic Circuits <https://arxiv.org/abs/quant-ph/0406176>`_.

    Args:
        gate (:class:`~.core.gates.QuantumGate`): instance of quantum gate.
        with_barrier (bool): whether add barriers into decomposed circuit.

    Returns:
        :class:`~.core.circuit.Circuit`, composed of 1-qubit gates and CNOT gates.

    Examples:
        >>> import mindquantum as mq
        >>> from mindquantum.algorithm.compiler.decompose import qs_decompose
        >>> from scipy.stats import unitary_group
        >>> tqs = [1,2,3,6] # arbitrary qubit index order is OK
        >>> n = len(tqs) # qubit number
        >>> u = unitary_group.rvs(2 ** n, random_state=123)
        >>> g = mq.UnivMathGate('U', u).on(tqs)
        >>> circ = qs_decompose(g)
        >>> num_cnot =  len([g for g in circ if isinstance(g, mq.XGate) and len(g.ctrl_qubits)==1])
        >>> print('total gate number: {}, CNOT number: {}'.format(len(circ), num_cnot))
        total gate number: 412, CNOT number: 180
    """
    if gate.ctrl_qubits:
        raise ValueError(f'{gate} is a controlled gate. Use cu_decompose() instead.')
    n = gate.n_qubits

    if n == 1:
        return euler_decompose(gate, basis='u3', with_phase=False)

    (u1, u2), rads, (v1h, v2h) = linalg.cossin(gate.matrix(), separate=True, p=2 ** (n - 1), q=2 ** (n - 1))
    rads *= 2
    circ_left = demultiplex_pair(v1h, v2h, gate.obj_qubits[1:], gate.obj_qubits[0], with_barrier)
    circ_middle = demultiplex_pauli('Y', gate.obj_qubits[0], gate.obj_qubits[1:], *rads)
    circ_right = demultiplex_pair(u1, u2, gate.obj_qubits[1:], gate.obj_qubits[0], with_barrier)
    if with_barrier:
        return optimize_circuit(circ_left + BarrierGate() + circ_middle + BarrierGate() + circ_right)
    return optimize_circuit(circ_left + circ_middle + circ_right)


def demultiplex_pair(u1: np.ndarray, u2: np.ndarray, tqs: List[int], cq: int, with_barrier: bool = False) -> Circuit:
    """
    Decompose a multiplexor defined by a pair of unitary matrices operating on the same subspace.

    Args:
        u1 (numpy.ndarray): applied if the control qubit is |0>
        u2 (numpy.ndarray): applied if the control qubit is |1>
        tqs (List[int]): target qubit indices
        cq (int): control qubit index
        with_barrier (bool): whether add barriers into decomposed circuit.

    Returns:
        Circuit, composed of 1-qubit gates and CNOT gates.
    """
    if u1.shape != u2.shape:
        raise ValueError(f'Input matrices have different dimensions: {u1.shape}, {u2.shape}.')
    tqs = tqs.copy()
    u1u2h = u1 @ u2.conj().T
    if np.allclose(u1u2h, u1u2h.conj().T):  # is hermitian
        eigvals, v = linalg.eigh(u1u2h)
        eigvals = eigvals.astype(np.complex128)
    else:
        evals, v = linalg.schur(u1u2h, output='complex')
        eigvals = np.diag(evals)
    dvals = np.sqrt(eigvals)
    rads = 2 * np.angle(dvals.conj())
    w = np.diag(dvals) @ v.conj().T @ u2
    circ_left = qs_decompose(gates.UnivMathGate('W', w).on(tqs))
    circ_middle = demultiplex_pauli('Z', cq, tqs, *rads)
    circ_right = qs_decompose(gates.UnivMathGate('V', v).on(tqs))
    if with_barrier:
        return circ_left + BarrierGate() + circ_middle + BarrierGate() + circ_right
    return circ_left + circ_middle + circ_right


def demultiplex_pauli(sigma: str, tq: int, cqs: List[int], *args, permute_cnot: bool = False) -> Circuit:
    """
    Decompose a Pauli-rotation (RY or RZ) multiplexor defined by 2^(n-1) rotation angles.

    Args:
        sigma (str): Axis of rotation Pauli matrix, 'Y' or 'Z'.
        tq (int): target qubit index
        cqs (List[int]): control qubit indices
        *args: 2^(n-1) rotation angles in which n-1 is the length of `cqs`
        permute_cnot (bool): whether permute positions of CNOT gates

    Returns:
        Circuit, composed of 1-qubit Pauli-rotation gates and CNOT gates.
    """
    if not is_power_of_two(len(args)) or len(args) < 2:
        raise ValueError('Number of angle parameters is illegal (should be power of 2 and no less than 2)')
    if len(args) != 2 ** len(cqs):
        raise ValueError(f'Number of angle parameters ({len(args)}) does not coincide with control qubits ({len(cqs)})')
    n = int(np.log2(len(args))) + 1
    cqs = cqs.copy()
    circ = Circuit()
    if n == 2:
        circ += getattr(gates, f'R{sigma.upper()}')((args[0] + args[1]) / 2).on(tq)
        circ += gates.X.on(tq, cqs[0])
        circ += getattr(gates, f'R{sigma.upper()}')((args[0] - args[1]) / 2).on(tq)
        circ += gates.X.on(tq, cqs[0])
        if permute_cnot:
            circ.append(circ.pop(0))
    elif n == 3:
        (s0, s1), (t0, t1) = _cal_demultiplex_rads(args)
        cq_1st = cqs.pop(0)
        cq_2nd = cqs.pop(0)
        circ = Circuit()
        circ += getattr(gates, f'R{sigma.upper()}')(s0.item()).on(tq)
        circ += gates.X.on(tq, cq_2nd)
        circ += getattr(gates, f'R{sigma.upper()}')(s1.item()).on(tq)
        circ += gates.X.on(tq, cq_1st)
        circ += getattr(gates, f'R{sigma.upper()}')(t1.item()).on(tq)
        circ += gates.X.on(tq, cq_2nd)
        circ += getattr(gates, f'R{sigma.upper()}')(t0.item()).on(tq)
        circ += gates.X.on(tq, cq_1st)
    else:
        (s0, s1), (t0, t1) = _cal_demultiplex_rads(args)
        cq_1st = cqs.pop(0)
        cq_2nd = cqs.pop(0)
        circ += demultiplex_pauli(sigma, tq, cqs, *s0)
        circ += Circuit() + gates.X.on(tq, cq_2nd)
        circ += demultiplex_pauli(sigma, tq, cqs, *s1)
        circ += Circuit() + gates.X.on(tq, cq_1st)
        circ += demultiplex_pauli(sigma, tq, cqs, *t1)
        circ += Circuit() + gates.X.on(tq, cq_2nd)
        circ += demultiplex_pauli(sigma, tq, cqs, *t0)
        circ += Circuit() + gates.X.on(tq, cq_1st)
    return circ


def _cal_demultiplex_rads(rads):
    r"""
    Calculate rotation angles for two-level decomposing of a Pauli-rotation multiplexor.

    Args:
        rads: rotation angles representing the original Pauli-rotation multiplexor

    Returns:
        rotation angles after two-level decomposition
    """
    dim = len(rads)
    rads = np.reshape(rads, [2, 2, int(dim / 2 / 2)])
    p0 = (rads[0, 0, :] + rads[1, 0, :]) / 2
    p1 = (rads[0, 1, :] + rads[1, 1, :]) / 2
    l0 = (rads[0, 0, :] - rads[1, 0, :]) / 2
    l1 = (rads[0, 1, :] - rads[1, 1, :]) / 2
    return ((p0 + p1) / 2, (p0 - p1) / 2), ((l0 + l1) / 2, (l0 - l1) / 2)
