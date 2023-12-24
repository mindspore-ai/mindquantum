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
"""Two-qubit gate decomposition."""
# pylint: disable=invalid-name
from math import pi, sqrt

import numpy as np
from scipy import linalg

from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import QuantumGate

from .. import utils
from ..fixed_decompose import crx_decompose, cry_decompose, crz_decompose
from ..utils import (
    M_DAG,
    A,
    M,
    is_tensor_prod,
    kron_decomp,
    kron_factor_4x4_to_2x2s,
    optimize_circuit,
    params_abc,
    params_u3,
    params_zyz,
)


def tensor_product_decompose(gate: QuantumGate, return_u3: bool = True) -> Circuit:
    """
    Tensor product decomposition of a 2-qubit gate.

    Args:
        gate (:class:`~.core.gates.QuantumGate`): 2-qubit gate composed by tensor product.
        return_u3 (bool): return gates in form of :class:`~.core.gates.U3` if ``True``, otherwise
            return :class:`~.core.gates.UnivMathGate`. Default: ``True``.

    Returns:
        :class:`~.core.circuit.Circuit`, including two single-qubit gates.

    Examples:
        >>> import numpy as np
        >>> import mindquantum as mq
        >>> from mindquantum.algorithm.compiler.decompose import tensor_product_decompose
        >>> g = mq.UnivMathGate('XY', np.kron(mq.X.matrix(), mq.Y.matrix())).on([0, 1])
        >>> print(mq.Circuit() + g)
              ┏━━━━┓
        q0: ──┨    ┠───
              ┃    ┃
              ┃ XY ┃
        q1: ──┨    ┠───
              ┗━━━━┛
        >>> circ_decomposed = tensor_product_decompose(g)
        >>> print(circ_decomposed)
              ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        q0: ──┨ U3(θ=π, φ=π/2, λ=-π/2) ┠───
              ┗━━━━━━━━━━━━━━━━━━━━━━━━┛
              ┏━━━━━━━━━━━━━━━━━━━┓
        q1: ──┨ U3(θ=π, φ=0, λ=0) ┠────────
              ┗━━━━━━━━━━━━━━━━━━━┛
    """
    if len(gate.obj_qubits) != 2 or gate.ctrl_qubits:
        raise ValueError(f'{gate} is not a 2-qubit gate with designated qubits')
    if not is_tensor_prod(gate.matrix()):
        raise ValueError(f'{gate} is not a tensor-product unitary gate.')
    u0, u1 = kron_decomp(gate.matrix())
    circ = Circuit()
    if return_u3:
        circ += gates.U3(*params_u3(u0)).on(gate.obj_qubits[0])  # pylint: disable=no-value-for-parameter
        circ += gates.U3(*params_u3(u1)).on(gate.obj_qubits[1])  # pylint: disable=no-value-for-parameter
    else:
        circ += gates.UnivMathGate('U0', u0).on(gate.obj_qubits[0])
        circ += gates.UnivMathGate('U1', u1).on(gate.obj_qubits[1])
    return optimize_circuit(circ)


def abc_decompose(gate: QuantumGate, return_u3: bool = True) -> Circuit:
    """
    Decompose two-qubit controlled gate via ABC decomposition.

    Args:
        gate (:class:`~.core.gates.QuantumGate`): quantum gate with 1 control bit and 1 target bit.
        return_u3 (bool): return gates in form of :class:`~.core.gates.U3` if ``True``, otherwise
            return :class:`~.core.gates.UnivMathGate`. Default: ``True``.

    Returns:
        :class:`~.core.circuit.Circuit`, including at most 2 CNOT gates and 4 single-qubit gates.

    Examples:
        >>> import mindquantum as mq
        >>> from mindquantum.algorithm.compiler.decompose import abc_decompose
        >>> from scipy.stats import unitary_group
        >>> g = mq.UnivMathGate('U', unitary_group.rvs(2, random_state=123)).on(1, 0)
        >>> print(mq.Circuit() + g)
        q0: ────■─────
                ┃
              ┏━┻━┓
        q1: ──┨ U ┠───
              ┗━━━┛
        >>> circ_decomposed = abc_decompose(g)
        >>> print(circ_decomposed)
                                                                          ┏━━━━━━━━━━━━┓
        q0: ───────────────────■──────────────────────────────────────■───┨ RZ(1.1469) ┠────────────────────
                               ┃                                      ┃   ┗━━━━━━━━━━━━┛
              ┏━━━━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        q1: ──┨ RZ(2.6016) ┠─┨╺╋╸┠─┨ U3(θ=1.1043, φ=π, λ=-0.6572) ┠─┨╺╋╸┠─┨ U3(θ=1.1043, φ=-5.086, λ=0) ┠───
              ┗━━━━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """
    if len(gate.ctrl_qubits) != 1 or len(gate.obj_qubits) != 1:
        raise ValueError(f'{gate} is not a two-qubit controlled gate with designated qubits')
    if isinstance(gate, gates.RX):
        return crx_decompose(gate)[0]
    if isinstance(gate, gates.RY):
        return cry_decompose(gate)[0]
    if isinstance(gate, gates.RZ):
        return crz_decompose(gate)[0]

    cq = gate.ctrl_qubits[0]
    tq = gate.obj_qubits[0]
    _, (_, phi, lam) = params_zyz(gate.matrix())
    alpha, (a, b, c) = params_abc(gate.matrix())
    circ = Circuit()
    if return_u3:
        # regardless global phases
        circ += gates.RZ((lam - phi) / 2).on(tq)
        circ += gates.X.on(tq, cq)
        circ += gates.U3(*params_u3(b)).on(tq)  # pylint: disable=no-value-for-parameter
        circ += gates.X.on(tq, cq)
        circ += gates.U3(*params_u3(a)).on(tq)  # pylint: disable=no-value-for-parameter
        circ += gates.RZ(alpha).on(cq)
    else:
        circ += gates.UnivMathGate('C', c).on(tq)
        circ += gates.X.on(tq, cq)
        circ += gates.UnivMathGate('B', b).on(tq)
        circ += gates.X.on(tq, cq)
        circ += gates.UnivMathGate('A', a).on(tq)
        circ += gates.PhaseShift(alpha).on(cq)
    return optimize_circuit(circ)


# pylint: disable=too-many-locals
def kak_decompose(gate: QuantumGate, return_u3: bool = True) -> Circuit:
    r"""
    KAK decomposition (CNOT basis) of an arbitrary two-qubit gate.

    For more detail, please refer to `An Introduction to Cartan's KAK Decomposition for QC
    Programmers <https://arxiv.org/abs/quant-ph/0406176>`_.

    Args:
        gate (:class:`~.core.gates.QuantumGate`): 2-qubit quantum gate.
        return_u3 (bool): return gates in form of :class:`~.core.gates.U3` if ``True``, otherwise
            return :class:`~.core.gates.UnivMathGate`. Default: ``True``.

    Returns:
        :class:`~.core.circuit.Circuit`, including at most 3 CNOT gates and 6 single-qubit gates.

    Examples:
        >>> import mindquantum as mq
        >>> from mindquantum.algorithm.compiler.decompose import kak_decompose
        >>> from scipy.stats import unitary_group
        >>> g = mq.UnivMathGate('U', unitary_group.rvs(4, random_state=123)).on([0, 1])
        >>> print(mq.Circuit() + g)
              ┏━━━┓
        q0: ──┨   ┠───
              ┃   ┃
              ┃ U ┃
        q1: ──┨   ┠───
              ┗━━━┛
        >>> circ_decomposed = kak_decompose(g)
        >>> print(circ_decomposed)
              ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓       ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        q0: ──┨ U3(θ=2.2601, φ=-3.602, λ=2.4907) ┠───■───┨ U3(θ=π/2, φ=-0.2573, λ=-π) ┠───■───↯─
              ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   ┃   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   ┃
              ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  ┏━┻━┓
        q1: ──┨ U3(θ=1.846, φ=-2.9209, λ=0.5375) ┠─┨╺╋╸┠─┨ U3(θ=0, φ=-0.19, λ=-0.19) ┠──┨╺╋╸┠─↯─
              ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  ┗━━━┛
              ┏━━━━━━━━━━━━━━━━━━━━━┓             ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        q0: ──┨ U3(θ=π/2, φ=0, λ=π) ┠─────────■───┨ U3(θ=2.273, φ=-1.8708, λ=0.7431) ┠───
              ┗━━━━━━━━━━━━━━━━━━━━━┛         ┃   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
              ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        q1: ──┨ U3(θ=0, φ=0.358, λ=0.358) ┠─┨╺╋╸┠─┨ U3(θ=2.7317, φ=1.8583, λ=0.6685) ┠───
              ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """
    if len(gate.obj_qubits) != 2 or gate.ctrl_qubits:
        raise ValueError(f'{gate} is not an arbitrary 2-qubit gate with designated qubits')
    pauli_i = gates.I.matrix()
    pauli_x = gates.X.matrix()
    pauli_z = gates.Z.matrix()

    # construct a new matrix replacing U
    u_su4 = M_DAG @ utils.remove_glob_phase(gate.matrix()) @ M  # ensure the decomposed object is in SU(4)
    ur = np.real(u_su4)  # real part of u_su4
    ui = np.imag(u_su4)  # imagine part of u_su4

    # simultaneous SVD decomposition
    (q_left, q_right), (dr, di) = utils.simult_svd(ur, ui)
    d = dr + 1j * di

    _, a0, a1 = kron_factor_4x4_to_2x2s(M @ q_left @ M_DAG)
    _, b0, b1 = kron_factor_4x4_to_2x2s(M @ q_right.T @ M_DAG)

    k = linalg.inv(A) @ np.angle(np.diag(d))
    h1, h2, h3 = -k[1:]

    u0 = 1j / sqrt(2) * (pauli_x + pauli_z) @ linalg.expm(-1j * (h1 - pi / 4) * pauli_x)
    v0 = -1j / sqrt(2) * (pauli_x + pauli_z)
    u1 = linalg.expm(-1j * h3 * pauli_z)
    v1 = linalg.expm(1j * h2 * pauli_z)
    w = (pauli_i - 1j * pauli_x) / sqrt(2)

    # list of operators
    rots1 = [b0, u0, v0, a0 @ w]  # rotation gate on idx1
    rots2 = [b1, u1, v1, a1 @ w.conj().T]

    idx1, idx2 = gate.obj_qubits
    circ = Circuit()
    if return_u3:
        circ += gates.U3(*params_u3(rots1[0])).on(idx1)  # pylint: disable=no-value-for-parameter
        circ += gates.U3(*params_u3(rots2[0])).on(idx2)  # pylint: disable=no-value-for-parameter
        circ += gates.X.on(idx2, idx1)
        circ += gates.U3(*params_u3(rots1[1])).on(idx1)  # pylint: disable=no-value-for-parameter
        circ += gates.U3(*params_u3(rots2[1])).on(idx2)  # pylint: disable=no-value-for-parameter
        circ += gates.X.on(idx2, idx1)
        circ += gates.U3(*params_u3(rots1[2])).on(idx1)  # pylint: disable=no-value-for-parameter
        circ += gates.U3(*params_u3(rots2[2])).on(idx2)  # pylint: disable=no-value-for-parameter
        circ += gates.X.on(idx2, idx1)
        circ += gates.U3(*params_u3(rots1[3])).on(idx1)  # pylint: disable=no-value-for-parameter
        circ += gates.U3(*params_u3(rots2[3])).on(idx2)  # pylint: disable=no-value-for-parameter
    else:
        circ += gates.UnivMathGate('B0', rots1[0]).on(idx1)
        circ += gates.UnivMathGate('B1', rots2[0]).on(idx2)
        circ += gates.X.on(idx2, idx1)
        circ += gates.UnivMathGate('U0', rots1[1]).on(idx1)
        circ += gates.UnivMathGate('U1', rots2[1]).on(idx2)
        circ += gates.X.on(idx2, idx1)
        circ += gates.UnivMathGate('V0', rots1[2]).on(idx1)
        circ += gates.UnivMathGate('V1', rots2[2]).on(idx2)
        circ += gates.X.on(idx2, idx1)
        circ += gates.UnivMathGate('W0', rots1[3]).on(idx1)
        circ += gates.UnivMathGate('W1', rots2[3]).on(idx2)
    return optimize_circuit(circ)
