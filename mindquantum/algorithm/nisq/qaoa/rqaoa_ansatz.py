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

"""RQAOA ansatz."""

import numpy as np

from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator
from mindquantum.utils.type_value_check import _check_input_type

from .qaoa_ansatz import QAOAAnsatz


def _check_rqaoa_ham(ham):
    """Check hamiltonian of rqaoa."""
    _check_input_type('hamiltonian', QubitOperator, ham)
    for h in ham:
        _check_input_type('hamiltonian', QubitOperator, h)
        # further investigation is needed to answer whether reducing variable on higher order terms is beneficial
        # therefore we focus on quadratic Hamiltonians currently
        if h.count_gates() not in [2, 0]:
            raise ValueError("Only quadratic hamiltonian is supported in RQAOA.")


def _check_rqaoa_eliminate_input(f, sigma, v):
    """Check input for EliminateVariable of rqaoa."""
    if sigma not in [-1, 1]:
        raise ValueError("Sigma should be 1 or -1.")
    QubitOperator(f)
    if v is not None:
        QubitOperator((v,))
        if v not in f:
            raise ValueError("v need be in f.")


def _check_rqaoa_var_set(var_set):
    """Check the variable set of rqaoa."""
    _check_input_type('Variable set', dict, var_set)
    for k in var_set:
        QubitOperator((k,))
        if var_set[k] not in [-1, 1]:
            raise ValueError("The value of variable should be 1 or -1.")


class RQAOAAnsatz(QAOAAnsatz):
    r"""
    The RQAOA ansatz.

    For more detail, please refer to `Obstacles to State Preparation and
    Variational Optimization from Symmetry Protection <https://arxiv.org/pdf/1910.08980.pdf>`_.

    Args:
        ham (QubitOperator): The hamiltonian structure.
        p (int): The depth of QAOA ansatz. Default: ``1``.

    Examples:
        >>> from mindquantum.algorithm.nisq import RQAOAAnsatz
        >>> from mindquantum.core.operators import QubitOperator
        >>> ham = QubitOperator('Z0 Z1', 2) + QubitOperator('Z2 Z1', 1) + QubitOperator('Z0 Z2', 0.5)
        >>> ra = RQAOAAnsatz(ham, 1)
        >>> ra.all_variables                # All parameters(as Pauli Operators)
        [(0, 'Z'), (1, 'Z'), (2, 'Z')]
        >>> hams = ra.m_hamiltonians        # Hamiltonian for Measuring M
        >>> hams
        [1.0 [Z0 Z1] , 1.0 [Z1 Z2] , 1.0 [Z0 Z2] ]
        >>> n, p, m = ra.get_subproblem()   # convert the Hamiltonian to optimization problem
        >>> n     # number of variables
        3
        >>> p     # optimzation problem([variable index], weight)
        [((0, 1), 2), ((1, 2), 1), ((0, 2), 0.5)]
        >>> m     # variable mapping relation {index: Pauli operator}
        {0: (0, 'Z'), 1: (1, 'Z'), 2: (2, 'Z')}
        >>> f = ((1, 'Z'), (2, 'Z'))     # reduce variable
        >>> v = f[1]                     # variable to reduce
        >>> sigma = -1                   # reduce relation
        >>> ra.eliminate_single_variable(f, sigma, v) # reduce variable
        >>> ra.hamiltonian               # Hamiltonian
        -1 [] +
        1.5 [Z0 Z1]
        >>> ra.restricted_set            # view the restricted set
        [((2, 'Z'), ((1, 'Z'),), -1)]
        >>> ra.translate({(0, 'Z'):-1, (1, 'Z'):1})   # recover original problem's solution by the subproblem's solution
        {(0, 'Z'): -1, (1, 'Z'): 1, (2, 'Z'): -1}
        >>> ra = RQAOAAnsatz(ham, 1)
        >>> pr = {'beta_0': -0.4617199, 'alpha_0': 0.6284928}
        >>> ra.eliminate_variable(pr, 1) # reduce variable
        -- eliminated variable: Z1
        -- correlated variable: Z0
        -- σ: -1
        >>> ham = QubitOperator('Z0 Z1', 2) + QubitOperator('Z2 Z1', 1) + QubitOperator('Z0 Z3', 0.5)
        >>> ra = RQAOAAnsatz(ham, 1)
        >>> ra.ham
        2 [Z0 Z1] +
        0.5 [Z0 Z3] +
        1 [Z1 Z2]
        >>> f = ((0, 'Y'), (1, 'Z'), (3, 'Z'))
        >>> v = (3, 'Z')
        >>> sigma = 1
        >>> ra.eliminate_single_variable(f, sigma, v, True)
        -- eliminated variable: Z3
        -- correlated variable: Y0*Z1
        -- σ: 1
        >>> ra.ham                                          # please modify _check_rqaoa_ham to support higher order terms
        0.5j [X0 Z1] +
        2 [Z0 Z1] +
        1 [Z1 Z2]
    """

    def __init__(self, ham, p=1):
        """Initialize a RQAOAAnsatz object."""
        _check_rqaoa_ham(ham)
        super().__init__(ham, p)
        self.name = 'RQAOA'
        self._xi = []  # the restricted set

    @property
    def restricted_set(self):
        """
        Get the current restricted set of RQAOA.

        Returns:
            list[tuple], the restricted set.
        """
        return self._xi

    @property
    def all_variables(self):
        """
        Get all variables in the current hamiltonian.

        Returns:
            list[tuple[int, str]], variables.
        """
        terms = set()
        for k in self.ham.terms.keys():
            terms |= set(k)
        return sorted(list(terms))

    @property
    def variables_number(self):
        """
        Get the number of variables.

        Returns:
            int, number of variables.
        """
        return len(self.all_variables)

    @property
    def m_hamiltonians(self):
        """
        Get the hamiltonian for each M.

        Returns:
            list[Hamiltonian], hamiltonians of the problem.
        """
        mhs = []
        for h in self.ham:
            h = list(h.terms.keys())[0]
            # abandon constant term and primary term(no correlation)
            if len(h) < 2:
                continue
            mhs.append(Hamiltonian(QubitOperator(h)))
        return mhs

    def get_subproblem(self):
        """
        Transform hamiltonian into variable optimization problem.

        Returns:
            int, number of variables.
            list[tuple], list of variable indexes and weights.
            dict[int:tuple], mapping of index and variable.
        """
        terms = self.all_variables
        term_num = len(terms)
        mapping = dict()
        for i in range(term_num):
            mapping[terms[i]] = i
        subproblem = list()
        hams = self.ham.terms
        for h in hams:
            subproblem.append((tuple([mapping.get(k, -1) for k in h]), hams[h]))
        mapping = dict(zip(mapping.values(), mapping.keys()))
        return term_num, subproblem, mapping

    def eliminate_single_variable(self, f, sigma, v=None, show_process=False):
        """
        Eliminate single variable in hamiltonian.

        Args:
            f (tuple[tuple]): The corresponding variables.
            sigma (int): Correlation between variables f.
            v (tuple): Eliminated variable. Select randomly from f by default.
            show_process (bool): Whether to show the process of eliminating variables. Default: ``False``.
        """
        _check_rqaoa_eliminate_input(f, sigma, v)
        hams = self.ham.terms
        new_ham = 0
        if v is None:
            v = f[np.random.randint(0, 2)]
        # Invalid elimination
        if v not in self.all_variables:
            return  # quit to avoid contaminating the restricted set
        for h in list(hams.keys()):
            if v in h:
                new_ham += QubitOperator(tuple(set(h) ^ set(f)), sigma * hams[h])
            else:
                new_ham += QubitOperator(h, hams[h])
        self.ham = new_ham
        self._update()
        self._xi.append((v, tuple({v} ^ set(f)), sigma))
        if show_process:
            xi = self._xi[-1]
            print(f'-- eliminated variable: {xi[0][1]}{xi[0][0]}')
            pp = '*'.join([f'{x[1]}{x[0]}' for x in xi[1]])
            print('-- correlated variable: ' + pp)
            print(f'-- σ: {xi[2]}')

    def translate(self, var_set):
        """
        Translate the solution of the subproblem into the solution
        of the complete problem over the restricted set.

        Args:
            var_set (dict[tuple:int]): The solution of the subproblem.

        Returns:
            dict[tuple:int], the solution of the complete problem .
        """
        _check_rqaoa_var_set(var_set)
        for xi in self._xi[::-1]:
            val = xi[2]
            for u in xi[1]:
                try:
                    val *= var_set[u]
                except KeyError:
                    # unconnected variable, default: 1
                    var_set[u] = 1
            var_set[xi[0]] = val
        return var_set

    def eliminate_variable(self, weight, show_process=False):
        """
        For more detail, please refer to article.

        Args:
            weight (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]): parameter
                value for QAOA ansatz.
            show_process (bool): Whether to show the process of eliminating variables. Default: ``False``.
        """
        _check_input_type('The flag of showing process', (bool, int), show_process)
        hams = self.m_hamiltonians
        sim = Simulator('mqvector', self._circuit.n_qubits)
        sim.apply_circuit(self._circuit, pr=weight)
        m = list(map(sim.get_expectation, hams))
        i = np.abs(m).argmax()
        sigma = int(np.sign(m[i].real))
        f = hams[i].ham_termlist[0][0]
        self.eliminate_single_variable(f, sigma, None, show_process)

    def _update(self):
        """Update the circuit with new hamiltonian."""
        self.n_qubits = self.hamiltonian.n_qubits
        self._implement(self.ham, self.depth)
