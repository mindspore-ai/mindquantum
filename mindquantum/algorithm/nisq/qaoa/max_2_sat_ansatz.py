# -*- coding: utf-8 -*-
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
import numpy as np
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.circuit.utils import CPN
from mindquantum.simulator import Simulator
from mindquantum.utils.type_value_check import _check_int_type
from mindquantum.utils.type_value_check import _check_value_should_between_close_set
from mindquantum.utils.type_value_check import _check_input_type
from mindquantum.core.gates import H, RX
from mindquantum.core.operators import TimeEvolution, QubitOperator
from .._ansatz import Ansatz


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
    """check clauses"""
    if not isinstance(clauses, list):
        raise TypeError(f"clauses requires a list, but get {type(clauses)}")
    for clause in clauses:
        if not isinstance(clause, tuple):
            raise TypeError(f"clause requires a tuple, but get {type(clause)}")
        if len(clause) != 2:
            raise ValueError(f"each clause must contain two integers, but get {len(clause)}")
        if 0 in clause:
            raise ValueError("clause must contain non-zero integers, but get 0")


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
        clauses (list[tuple[int]]): The Max-2-SAT structure. Every element of list
            is a clause represented by a tuple with length two. The element of
            tuple must be non-zero integer. For example, (2, -3) stands for clause
            :math:`x_2\lor\lnot x_3`.
        depth (int): The depth of Max-2-SAT ansatz. Default: 1.

    Examples:
        >>> from mindquantum.algorithm.nisq.qaoa import Max2SATAnsatz
        >>> clauses = [(2, -3)]
        >>> max2sat = Max2SATAnsatz(clauses, 1)
        >>> max2sat.circuit
        q1: ──H─────RZ(0.5*beta_0)────●───────────────────────●────RX(alpha_0)──
                                      │                       │
        q2: ──H────RZ(-0.5*beta_0)────X────RZ(-0.5*beta_0)────X────RX(alpha_0)──

        >>> max2sat.hamiltonian
        0.25 [] +
        0.25 [Z1] +
        -0.25 [Z1 Z2] +
        -0.25 [Z2]
        >>> sats = max2sat.get_sat(4, np.array([4, 1]))
        >>> sats
        ['001', '000', '011', '010']
        >>> for i in sats:
        >>>     print(f'sat value: {max2sat.get_sat_value([i])}')
        sat value: 1
        sat value: 0
        sat value: 2
        sat value: 1
    """
    def __init__(self, clauses, depth=1):
        if not isinstance(depth, int):
            raise TypeError(f"depth requires a int, but get {type(depth)}")
        if depth <= 0:
            raise ValueError(f"depth must be greater than 0, but get {depth}.")
        _check_clause(clauses)
        super(Max2SATAnsatz, self).__init__('Max2SAT', _get_clause_act_qubits_num(clauses), clauses, depth)
        self.clauses = clauses
        self.depth = depth

    def _build_hc(self, clauses):
        """Build hc circuit."""
        ham = QubitOperator()
        for clause in clauses:
            ham += (sign(1, clause[0]) * QubitOperator(f'Z{abs(clause[0]) - 1}', 'beta') +
                    sign(1, clause[1]) * QubitOperator(f'Z{abs(clause[1]) - 1}', 'beta')) / 4
            ham += (sign(1, clause[0]) * sign(1, clause[1]) *
                    QubitOperator(f'Z{abs(clause[0]) - 1} Z{abs(clause[1]) - 1}', 'beta')) / 4
        return TimeEvolution(ham).circuit

    def _build_hb(self, clauses):
        """Build hb circuit."""
        circ = Circuit([RX('alpha').on(i) for i in _get_clause_act_qubits(clauses)])
        return circ

    @property
    def hamiltonian(self):
        """
        Get the hamiltonian of this max-2-sat problem.

        Returns:
            QubitOperator, hamiltonian of this max-2-sat problem.
        """
        qo = QubitOperator()
        for clause in self.clauses:
            qo += (QubitOperator('') + sign(1, clause[0]) * QubitOperator(f'Z{abs(clause[0]) - 1}') +
                   sign(1, clause[1]) * QubitOperator(f'Z{abs(clause[1]) - 1}') + sign(1, clause[0]) *
                   sign(1, clause[1]) * QubitOperator(f'Z{abs(clause[0]) - 1} Z{abs(clause[1]) - 1}')) / 4
        return qo

    def get_sat(self, max_n, weight):
        """
            Get the strings of this max-2-sat problem.

            Args:
                max_n (int): how many strings you want.
                weight (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]): parameter
                    value for Max-2-SAT ansatz.

            Returns:
                list, a list of strings.
        """
        _check_int_type('max_n', max_n)
        _check_value_should_between_close_set('max_n', 1, 1 << self._circuit.n_qubits, max_n)
        sim = Simulator('projectq', self._circuit.n_qubits)
        sim.apply_circuit(self._circuit, weight)
        qs = sim.get_qs()
        idxs = np.argpartition(np.abs(qs), -max_n)[-max_n:]
        strings = [bin(i)[2:].zfill(self._circuit.n_qubits)[::-1] for i in idxs]
        return strings

    def get_sat_value(self, string):
        """
        Get the sat values for given strings.
        The string is a str that satisfies all the clauses of the given max-2-sat problem.

        Args:
            string (str): a string of the max-2-sat problem consided.

        Returns:
            int, sat_value under the given string.
        """
        _check_input_type('string', str, string)
        sat_value = string.count('1')
        return sat_value

    def _implement(self, clauses, depth):
        """Implement of Max-2-SAT ansatz."""
        self._circuit = UN(H, _get_clause_act_qubits(clauses))
        for d in range(depth):
            self._circuit += CPN(self._build_hc(clauses), {'beta': f'beta_{d}'})
            self._circuit += CPN(self._build_hb(clauses), {'alpha': f'alpha_{d}'})
