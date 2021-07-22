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
"""Max-2-SAT ansatz."""

from math import copysign as sign
from mindquantum.ansatz import Ansatz
from mindquantum.circuit import Circuit, CPN, UN, TimeEvolution
from mindquantum.gate import H, RX
from mindquantum.ops import QubitOperator


def _get_clause_act_qubits_num(clauses):
    """Get qubits number."""
    return len(_get_clause_act_qubits(clauses))


def _get_clause_act_qubits(clauses):
    """Get all acted qubits."""
    qubits = set()
    for clause in clauses:
        clause = list(map(abs, clause))
        clause = [i - 1 for i in clause]
        qubits |= set(clause)
    qubits = list(qubits)
    return sorted(qubits)


def _check_clause(clauses):
    """check clause"""
    if not isinstance(clauses, list):
        raise TypeError(f"clauses requires a list, but get {type(clauses)}")
    for clause in clauses:
        if not isinstance(clause, tuple):
            raise TypeError(f"clause requires a tuple, but get {type(clause)}")
        if len(clause) != 2:
            raise ValueError(f"each clause must contain two integers, but get {len(clause)}")
        if 0 in clause:
            raise  ValueError("clause must contain non-zero integers, but get 0")


class Max2SATAnsatz(Ansatz):
    r"""
    The Max-2-SAT ansatz. For more detail,
    please refers to https://arxiv.org/pdf/1906.11259.pdf.

    .. math::

        U(\beta, \gamma) = e^{-\beta_pH_b}e^{-\gamma_pH_c}
        \cdots e^{-\beta_0H_b}e^{-\gamma_0H_c}H^{\otimes n}

    Where,

    .. math::

        H_b = \sum_{i\in n}X_{i}, H_c = \sum_{l\in m}P(l)

    Here :math:`n` is the number of Boolean variables and :math:`m` is the number
    of total clauses and :math:`P(l)` is rank-one projector.

    Args:
        graph (list[tuple[int]]): The Max-2-SAT structure. Every element of list
            is a clause represented by a tuple with length two. The element of
            tuple must be non-zero integer. For example, (2, -3) stands for clause
            :math:`x_2\lor\lnot x_3`.
        depth (int): The depth of Max-2-SAT ansatz. Default: 1.

    Examples:
        >>> from mindquantum.ansatz import Max2SATAnsatz
        >>> clauses = [(1, 2), (2, -3)]
        >>> max2sat = Max2SATAnsatz(clauses, 2)
        >>> max2sat.circuit
        H(0)
        H(1)
        H(2)
        RZ(0.25*beta_0|0)
        RZ(0.5*beta_0|1)
        X(1 <-: 0)
        RZ(0.5*beta_0|1)
        X(1 <-: 0)
        RZ(-0.25*beta_0|2)
        X(2 <-: 1)
        RZ(-0.5*beta_0|2)
        X(2 <-: 1)
        RX(alpha_0|0)
        RX(alpha_0|1)
        RX(alpha_0|2)
        RZ(0.25*beta_1|0)
        RZ(0.5*beta_1|1)
        X(1 <-: 0)
        RZ(0.5*beta_1|1)
        X(1 <-: 0)
        RZ(-0.25*beta_1|2)
        X(2 <-: 1)
        RZ(-0.5*beta_1|2)
        X(2 <-: 1)
        RX(alpha_1|0)
        RX(alpha_1|1)
        RX(alpha_1|2)

        >>> max2sat.hamiltonian
        0.5 [] +
        0.25 [Z0] +
        0.25 [Z0 Z1] +
        0.5 [Z1] +
        -0.25 [Z1 Z2] +
        -0.25 [Z2]
    """
    def __init__(self, clauses, depth=1):
        if not isinstance(depth, int):
            raise TypeError(f"depth requires a int, but get {type(depth)}")
        if depth <= 0:
            raise ValueError(f"depth must be greater than 0, but get {depth}.")
        _check_clause(clauses)
        super(Max2SATAnsatz, self).__init__('Max2SAT',
                                            _get_clause_act_qubits_num(clauses),
                                            clauses, depth)
        self.clauses = clauses
        self.depth = depth

    def _build_hc(self, clauses):
        """Build hc circuit."""
        ham = QubitOperator()
        for clause in clauses:
            ham += (sign(1, clause[0]) * QubitOperator(f'Z{abs(clause[0]) - 1}', 'beta') +
                    sign(1, clause[1]) * QubitOperator(f'Z{abs(clause[1]) - 1}', 'beta') +
                    sign(1, clause[0]) * sign(1, clause[1]) *
                    QubitOperator(f'Z{abs(clause[0]) - 1} Z{abs(clause[1]) - 1}', 'beta')
                    ) / 4
        return TimeEvolution(ham).circuit

    def _build_hb(self, clauses):
        """Build hb circuit."""
        circ = Circuit(
            [RX('alpha').on(i) for i in _get_clause_act_qubits(clauses)])
        return circ

    @property
    def hamiltonian(self):
        """
        Get the hamiltonian of this max 2 sat problem.

        Returns:
            QubitOperator, hamiltonian of this max 2 sat problem.
        """
        qo = QubitOperator()
        for clause in self.clauses:
            qo += (QubitOperator('') +
                   sign(1, clause[0]) * QubitOperator(f'Z{abs(clause[0]) - 1}') +
                   sign(1, clause[1]) * QubitOperator(f'Z{abs(clause[1]) - 1}') +
                   sign(1, clause[0]) * sign(1, clause[1]) *
                   QubitOperator(f'Z{abs(clause[0]) - 1} Z{abs(clause[1]) - 1}')
                   ) / 4
        return qo

    def _implement(self, clauses, depth):
        """Implement of max 2 sat ansatz."""
        self._circuit = UN(H, _get_clause_act_qubits(clauses))
        for d in range(depth):
            self._circuit += CPN(self._build_hc(clauses), {'beta': f'beta_{d}'})
            self._circuit += CPN(self._build_hb(clauses),
                                 {'alpha': f'alpha_{d}'})
