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

from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.simulator import Simulator
from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_value_should_not_less,
)

from .qaoa_ansatz import QAOAAnsatz

def _check_rqaoa_ham(ham):
    """Check hamiltonian of rqaoa."""
    _check_input_type('hamiltonian', QubitOperator, ham)
    for h in ham:
        _check_input_type('hamiltonian', QubitOperator, h)
        # 对高次项消元是否有益仍待研究
        # 暂时限制为仅包含二次项的哈密顿量
        if h.count_gates() not in [2, 0]:
            raise ValueError("Only quadratic Hamiltonian is supported in RQAOA.")

def _check_rqaoa_eliminate_input(f, sigma, v):
    """Check input for EliminateVariable of rqaoa."""
    if sigma not in [-1, 1]:
        raise ValueError("Sigma should be 1 or -1.")
    QubitOperator(f)
    if v is not None:
        QubitOperator((v,))
        if v not in f:
            raise ValueError("v need be in f.")

def _check_rqaoa_translate_input(var_set):
    """Check input for Translate of rqaoa."""
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
        p (int): The depth of QAOA ansatz. Default: 1.

    Examples:
        >>> from mindquantum.algorithm.nisq import RQAOAAnsatz
        >>> from mindquantum.core.operators import QubitOperator
        >>> ham = QubitOperator('Z0 Z1', 2) + QubitOperator('Z2 Z1', 1) + QubitOperator('Z0 Z2', 0.5)
        >>> ra = RQAOAAnsatz(ham, 1)
        >>> ra.all_variables                # 所有变量(泡利算符形式)
        [(0, 'Z'), (1, 'Z'), (2, 'Z')]
        >>> hams = ra.m_hamiltonians        # 测量M所需的哈密顿量
        >>> hams
        [1.0 [Z0 Z1] , 1.0 [Z1 Z2] , 1.0 [Z0 Z2] ]
        >>> n, p, m = ra.get_subproblem()   # 将哈密顿量转换为优化问题
        >>> n     # 变量数量
        3
        >>> p     # 优化问题([变量索引], 权重)
        [([0, 1], 2), ([1, 2], 1), ([0, 2], 0.5)]
        >>> m     # 变量映射关系{索引:泡利算符} 
        {0: (0, 'Z'), 1: (1, 'Z'), 2: (2, 'Z')}
        >>> f = ((1, 'Z'), (2, 'Z'))   # 约束变量
        >>> v = f[1]                   # 待消除变量
        >>> sigma = -1                 # 约束关系
        >>> ra.eliminate_single_variable(f, sigma, v) # 消除变量
        >>> ra.hamiltonian             # 哈密顿量
        -1 [] +
        1.5 [Z0 Z1]
        >>> ra.restricted_set          # 查看已有约束集
        [((2, 'Z'), ((1, 'Z'),), -1)]
        >>> ra.translate({(0, 'Z'):-1, (1, 'Z'):1})   # 根据子问题求解结果逆推问题的解
        {(0, 'Z'): -1, (1, 'Z'): 1, (2, 'Z'): -1}
        >>> ra = RQAOAAnsatz(ham, 1)
        >>> pr = {'beta_0': -0.4617199, 'alpha_0': 0.6284928}
        >>> ra.eliminate_variable(pr, 1) # 消除变量
        -- eliminated variable: Z1
        -- correlated variable: Z0
        -- σ: -1
    """

    def __init__(self, ham, p=1):
        """Initialize a RQAOAAnsatz object."""
        _check_int_type('depth', p)
        _check_value_should_not_less('depth', 1, p)
        _check_rqaoa_ham(ham)
        super().__init__(ham, p)
        self.name = 'RQAOA'
        self.Xi = [] # the restricted set

    @property
    def restricted_set(self):
        """
        Get the current restricted set of RQAOA.

        Returns:
            list[tuple], the restricted set.
        """
        return self.Xi

    @property
    def all_variables(self):
        """
        Get all variables in the current hamiltonian.

        Returns:
            list[tuple[int, str]], variables.
        """
        terms = list()
        for k in self.ham.terms.keys():
            terms += list(k)
        terms = list(set(terms))
        return sorted(terms)

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
            # 舍去常数项和一次项(无相关性)
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
            subproblem.append(([mapping[k] for k in h], hams[h]))
        mapping = dict(zip(mapping.values(), mapping.keys()))
        return term_num, subproblem, mapping

    def eliminate_single_variable(self, f, sigma, v=None):
        """
        Eliminate single variable in hamiltonian.

        Args:
            f (tuple[tuple]): The corresponding variables.
            sigma (int): Correlation between variables f.
            v (tuple): Eliminated variable. Select randomly from f by default.
        """
        _check_rqaoa_eliminate_input(f, sigma, v)
        hams = self.ham.terms
        new_ham = 0
        if v is None:
            v = f[np.random.randint(0, 2)]
        for h in list(hams.keys()):
            if v in h:
                new_ham += QubitOperator(tuple(set(h)^set(f)), sigma*hams[h])
            else:
                new_ham += QubitOperator(h, hams[h])
        self.ham = new_ham
        self._update()
        self.Xi.append((v, tuple(set((v,))^set(f)), sigma))

    def translate(self, var_set):
        """
        Translate the solution of the subproblem into the solution
        of the complete problem over the restricted set.

        Args:
            var_set (dict[tuple:int]): The solution of the subproblem.

        Returns:
            dict[tuple:int], the solution of the complete problem .
        """
        _check_rqaoa_translate_input(var_set)
        for xi in self.Xi[::-1]:
            val = xi[2]
            for u in xi[1]:
                try:
                    val *= var_set[u]
                except KeyError:
                    # 无关联变量
                    var_set[u] = 1
            var_set[xi[0]] = val
        return var_set

    def eliminate_variable(self, weight, show_process=False):
        """
        For more detail, please refer to article.

        Args:
            weight (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]): parameter
                    value for QAOA ansatz.
            show_process (bool): Whether to show the process of eliminating variables. Default: True.
        """
        hams = self.m_hamiltonians
        #sim = Simulator('mqvector', self._circuit.n_qubits)
        sim = Simulator('projectq', self._circuit.n_qubits)
        sim.apply_circuit(self._circuit, pr=pr)
        M = list(map(sim.get_expectation, hams))
        i = np.abs(M).argmax()
        sigma = int(np.sign(M[i].real))
        f = hams[i].ham_termlist[0][0]
        self.eliminate_single_variable(f, sigma)
        if show_process:
            xi = self.Xi[-1]
            print(f'-- eliminated variable: {xi[0][1]}{xi[0][0]}')
            pp = '*'.join([f'{x[1]}{x[0]}' for x in xi[1]])
            print('-- correlated variable: '+pp)
            print(f'-- σ: {xi[2]}')

    def _update(self):
        """Update the circuit with new hamiltonian."""
        self.n_qubits = self.hamiltonian.n_qubits
        self._implement(self.ham, self.depth)
