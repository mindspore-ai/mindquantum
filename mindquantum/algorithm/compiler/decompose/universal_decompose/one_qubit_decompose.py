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
"""One-qubit gate decomposition."""

import numpy as np
from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import QuantumGate, U3, RX, RZ
from mindquantum.utils.type_value_check import _check_input_type

from ..utils import params_u3, params_zyz

optional_basis = ['zyz', 'u3']


def euler_decompose(gate: QuantumGate, basis: str = 'zyz', with_phase: bool = True) -> Circuit:
    """
    One-qubit Euler decomposition.

    Currently only support 'zyz' and 'u3' decomposition.

    Args:
        gate (QuantumGate): single-qubit quantum gate.
        basis (str): decomposition basis, can be one of ``'zyz'`` or ``'u3'``. Default: ``'zyz'``.
        with_phase (bool): whether return global phase in form of a :class:`~.core.gates.GlobalPhase` gate.

    Returns:
        :class:`~.core.circuit.Circuit`, quantum circuit after Euler decomposition.
    """
    if len(gate.obj_qubits) != 1 or gate.ctrl_qubits:
        raise ValueError(f'{gate} is not a single-qubit gate with designated qubit for Euler decomposition')
    basis = basis.lower()
    tq = gate.obj_qubits[0]
    circ = Circuit()
    if basis == 'zyz':
        alpha, (theta, phi, lamda) = params_zyz(gate.matrix())
        circ += gates.RZ(lamda).on(tq)
        circ += gates.RY(theta).on(tq)
        circ += gates.RZ(phi).on(tq)
        if with_phase:
            circ += gates.GlobalPhase(-alpha).on(tq)
    elif basis == 'u3':
        phase, (theta, phi, lamda) = params_u3(gate.matrix(), return_phase=True)
        circ += gates.U3(theta, phi, lamda).on(tq)
        if with_phase:
            circ += gates.GlobalPhase(-phase).on(tq)
    else:
        raise ValueError(f'{basis} is not a supported decomposition method of {optional_basis}')
    return circ


def u3_decompose(gate: U3, method: str = 'standard'):
    """
    Decompose a U3 gate into a sequence of Z-X-Z-X-Z rotations.

    The decomposition follows one of two methods:
    1. standard: U3(θ,φ,λ) = Rz(φ)Rx(-π/2)Rz(θ)Rx(π/2)Rz(λ)
    2. alternative: U3(θ,φ,λ) = Rz(φ)Rx(π/2)Rz(π-θ)Rx(π/2)Rz(λ-π)

    When any rotation angle is a constant value and equals to 0, the corresponding RZ gate will be omitted.

    Args:
        gate (U3): The U3 gate to be decomposed.
        method (str): The decomposition method to use, either 'standard' or 'alternative'. Default: 'standard'

    Returns:
        Circuit: A quantum circuit implementing the U3 gate using ZXZXZ sequence.

    Raises:
        ValueError: If the method is not 'standard' or 'alternative'.
    """
    _check_input_type('gate', U3, gate)
    if method not in ['standard', 'alternative']:
        raise ValueError("method must be either 'standard' or 'alternative'")

    theta, phi, lamda = gate.theta, gate.phi, gate.lamda
    qubits = gate.obj_qubits
    circ = Circuit()

    if method == 'standard':
        if not (lamda.is_const() and np.isclose(lamda.const, 0.0, atol=1e-8)):
            circ += RZ(lamda).on(qubits[0])

        circ += RX(np.pi / 2).on(qubits[0])

        if not (theta.is_const() and np.isclose(theta.const, 0.0, atol=1e-8)):
            circ += RZ(theta).on(qubits[0])

        circ += RX(-np.pi / 2).on(qubits[0])

        if not (phi.is_const() and np.isclose(phi.const, 0.0, atol=1e-8)):
            circ += RZ(phi).on(qubits[0])
    else:  # alternative method
        if not (lamda.is_const() and np.isclose(lamda.const - np.pi, 0.0, atol=1e-8)):
            circ += RZ(lamda - np.pi).on(qubits[0])

        circ += RX(np.pi / 2).on(qubits[0])

        if not (theta.is_const() and np.isclose(np.pi - theta.const, 0.0, atol=1e-8)):
            circ += RZ(np.pi - theta).on(qubits[0])

        circ += RX(np.pi / 2).on(qubits[0])

        if not (phi.is_const() and np.isclose(phi.const, 0.0, atol=1e-8)):
            circ += RZ(phi).on(qubits[0])

    return circ
