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

from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import QuantumGate

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
