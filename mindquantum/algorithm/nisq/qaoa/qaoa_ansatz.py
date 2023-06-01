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

# pylint: disable=duplicate-code

"""QAOA ansatz."""

from mindquantum.core.circuit import (
    CPN,
    UN,
    Circuit,
    decompose_single_term_time_evolution,
)
from mindquantum.core.gates import RX, H
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_value_should_not_less,
)

from .._ansatz import Ansatz


class QAOAAnsatz(Ansatz):
    r"""
    The QAOA ansatz.

    For more detail, please refer to `A Quantum Approximate Optimization
    Algorithm <https://arxiv.org/abs/1411.4028.pdf>`_.

    .. math::

        U(\beta, \gamma) = e^{-\beta_pH_b}e^{-\gamma_pH_c}
        \cdots e^{-\beta_0H_b}e^{-\gamma_0H_c}H^{\otimes n}

    Args:
        ham (QubitOperator): The hamiltonian structure.
        depth (int): The depth of QAOA ansatz. Default: ``1``.

    Examples:
        >>> from mindquantum.algorithm.nisq import QAOAAnsatz
        >>> from mindquantum.core.operators import QubitOperator
        >>> ham = QubitOperator('Z0 Z1', 2) + QubitOperator('Z2 Z1', 1) + QubitOperator('Z0 Y2', 0.5)
        >>> qaoa = QAOAAnsatz(ham, 1)
        >>> qaoa.circuit[:11]
        q0: ──H────●────────────────────●──
                   │                    │
        q1: ──H────X────RZ(4*beta_0)────X──

        q2: ──H────────────────────────────
        >>> qaoa.hamiltonian
        2 [Z0 Z1] +
        0.5 [Z0 Y2] +
        1 [Z1 Z2]
    """

    def __init__(self, ham, depth=1):
        """Initialize a QAOAAnsatz object."""
        _check_int_type('depth', depth)
        _check_value_should_not_less('depth', 1, depth)
        _check_input_type('hamiltonian', QubitOperator, ham)
        self.ham = ham  # QubitOperator object
        self.depth = depth  # depth of QAOA ansatz
        super().__init__('QAOA', self.hamiltonian.n_qubits, ham, depth)

    @property
    def hamiltonian(self):
        """
        Get the hamiltonian.

        Returns:
            Hamiltonian, hamiltonian of the problem.
        """
        return Hamiltonian(self.ham)  # Hamiltonian object

    def _build_hc(self, ham):
        """Build hc circuit."""
        ham = ham.real
        circ = Circuit()
        for h in ham.terms:
            if h:
                circ += decompose_single_term_time_evolution(h, {'beta': ham.terms[h]})
        return circ

    def _build_hb(self, circ):
        """Build hc circuit."""
        return Circuit([RX('alpha').on(i) for i in circ.all_qubits.keys()])

    def _implement(self, ham, depth):  # pylint: disable=arguments-differ
        """Implement of QAOA ansatz."""
        hc = self._build_hc(ham)
        hb = self._build_hb(hc)
        self._circuit = UN(H, hc.all_qubits.keys())
        for current_depth in range(depth):
            self._circuit += CPN(hc, {'beta': f'beta_{current_depth}'})
            self._circuit += CPN(hb, {'alpha': f'alpha_{current_depth}'})
