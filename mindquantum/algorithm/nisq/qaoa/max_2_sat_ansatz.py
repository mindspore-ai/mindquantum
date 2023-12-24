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

from mindquantum.core.circuit import CPN, UN, Circuit
from mindquantum.core.gates import RX, H
from mindquantum.core.operators import QubitOperator, TimeEvolution
from mindquantum.simulator import Simulator
from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_value_should_between_close_set,
)

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
    """Check clauses."""
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
    The Max-2-SAT ansatz.

    For more detail, please refer to `Reachability Deficits
    in Quantum Approximate Optimization <https://arxiv.org/abs/1906.11259.pdf>`_.

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
        depth (int): The depth of Max-2-SAT ansatz. Default: ``1``.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.nisq import Max2SATAnsatz
        >>> clauses = [(2, -3)]
        >>> max2sat = Max2SATAnsatz(clauses, 1)
        >>> max2sat.circuit
              ┏━━━┓ ┏━━━━━━━━━━━━━━━━┓                                  ┏━━━━━━━━━━━━━┓
        q1: ──┨ H ┠─┨ RZ(1/2*beta_0) ┠────■─────────────────────────■───┨ RX(alpha_0) ┠───
              ┗━━━┛ ┗━━━━━━━━━━━━━━━━┛    ┃                         ┃   ┗━━━━━━━━━━━━━┛
              ┏━━━┓ ┏━━━━━━━━━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━━━━┓
        q2: ──┨ H ┠─┨ RZ(-1/2*beta_0) ┠─┨╺╋╸┠─┨ RZ(-1/2*beta_0) ┠─┨╺╋╸┠─┨ RX(alpha_0) ┠───
              ┗━━━┛ ┗━━━━━━━━━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━━━━┛
        >>> max2sat.hamiltonian
        1/4 [] +
        1/4 [Z1] +
        -1/4 [Z1 Z2] +
        -1/4 [Z2]
        >>> sats = max2sat.get_sat(4, np.array([4, 1]))
        >>> sats
        ['001', '000', '011', '010']
        >>> for i in sats:
        ...     print(f'sat value: {max2sat.get_sat_value(i)}')
        sat value: 1
        sat value: 0
        sat value: 2
        sat value: 1
    """

    def __init__(self, clauses, depth=1):
        """Initialize a Max2SATAnsatz object."""
        if not isinstance(depth, int):
            raise TypeError(f"depth requires a int, but get {type(depth)}")
        if depth <= 0:
            raise ValueError(f"depth must be greater than 0, but get {depth}.")
        _check_clause(clauses)
        super().__init__('Max2SAT', _get_clause_act_qubits_num(clauses), clauses, depth)
        self.clauses = clauses
        self.depth = depth

    def _build_hc(self, clauses):
        """Build hc circuit."""
        ham = QubitOperator()
        for clause in clauses:
            ham += (
                sign(1, clause[0]) * QubitOperator(f'Z{abs(clause[0]) - 1}', 'beta')
                + sign(1, clause[1]) * QubitOperator(f'Z{abs(clause[1]) - 1}', 'beta')
            ) / 4
            ham += (
                sign(1, clause[0])
                * sign(1, clause[1])
                * QubitOperator(f'Z{abs(clause[0]) - 1} Z{abs(clause[1]) - 1}', 'beta')
            ) / 4
        return TimeEvolution(ham.real).circuit

    def _build_hb(self, clauses):
        """Build hb circuit."""
        return Circuit([RX('alpha').on(i) for i in _get_clause_act_qubits(clauses)])

    @property
    def hamiltonian(self):
        """
        Get the hamiltonian of this max-2-sat problem.

        Returns:
            QubitOperator, hamiltonian of this max-2-sat problem.
        """
        qubit_operator = QubitOperator()
        for clause in self.clauses:
            qubit_operator += (
                QubitOperator('')
                + sign(1, clause[0]) * QubitOperator(f'Z{abs(clause[0]) - 1}')
                + sign(1, clause[1]) * QubitOperator(f'Z{abs(clause[1]) - 1}')
                + sign(1, clause[0])
                * sign(1, clause[1])
                * QubitOperator(f'Z{abs(clause[0]) - 1} Z{abs(clause[1]) - 1}')
            ) / 4
        return qubit_operator

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
        sim = Simulator('mqvector', self._circuit.n_qubits)
        sim.apply_circuit(self._circuit, weight)
        state = sim.get_qs()
        idxs = np.argpartition(np.abs(state), -max_n)[-max_n:]
        return [bin(i)[2:].zfill(self._circuit.n_qubits)[::-1] for i in idxs]

    def get_sat_value(self, string):
        """
        Get the sat values for given strings.

        The string is a str that satisfies all the clauses of the given max-2-sat problem.

        Args:
            string (str): a string of the max-2-sat problem considered.

        Returns:
            int, sat_value under the given string.
        """
        _check_input_type('string', str, string)
        return string.count('1')

    def _implement(self, clauses, depth):  # pylint: disable=arguments-differ
        """Implement of Max-2-SAT ansatz."""
        self._circuit = UN(H, _get_clause_act_qubits(clauses))
        for depth_idx in range(depth):
            self._circuit += CPN(self._build_hc(clauses), {'beta': f'beta_{depth_idx}'})
            self._circuit += CPN(self._build_hb(clauses), {'alpha': f'alpha_{depth_idx}'})
