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

"""MaxCutRQAOA ansatz."""

from mindquantum.core.operators import QubitOperator
from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_value_should_not_less,
)

from .rqaoa_ansatz import RQAOAAnsatz


def _check_graph(graph):
    """Check graph."""
    _check_input_type('graph', list, graph)
    for edge in graph:
        _check_input_type('edge', tuple, edge)
        if len(edge) != 2:
            raise ValueError("Edge should be composed of vertex set and weight.")
        _check_input_type('nodes', tuple, edge[0])
        if len(edge[0]) not in [0, 2]:
            raise ValueError("The number of vertices forming the edge should be 2.")
        for node in edge[0]:
            _check_int_type('node', node)
            _check_value_should_not_less('node', 0, node)
        if len(edge[0]) == 2:
            if edge[0][0] == edge[0][1]:
                raise ValueError("Edge should have different nodes.")
        _check_input_type('Weight of edges', (int, float), edge[1])


def _check_enum_method(method):
    """Check method of enum."""
    if method not in [max, min]:
        raise ValueError("Enumeration method needs to be selected from max or min.")


def _check_partition(partition):
    """Check partition."""
    _check_input_type('partition', list, partition)
    for node in partition:
        _check_int_type('node', node)
        _check_value_should_not_less('node', 0, node)


class MaxCut:
    r"""
    Some algorithms for MaxCut problem.
    """

    def enum(self, graph, method=max):
        """
        Solving MaxCut problem with enum.

        Args:
            graph (list[tuple]): The graph structure. Every element of graph
                is a edge that constructed by two nodes and one weight.
                For example, `[((0, 1), 1), ((1, 2), -1)]` means the graph has
                three nodes which are `0` , `1` and `2` with one edge connect
                between node `0` and node `1` with weight `1` and another
                connect between node `1` and node `2` with weight `-1`.
            method (function): ``max`` or ``min``. Default: ``max``.

        Returns:
            float, cut size.
            list[int], partition.
        """
        _check_graph(graph)
        _check_enum_method(method)
        nodes = set()
        for n, _ in graph:
            nodes |= set(n)
        n = len(nodes)
        return self._enum(graph, list(nodes), method, (0, []), n, (n + 1) // 2, [])

    def get_cut_value(self, graph, partition):
        """
        Get the cut value for given partitions.

        Args:
            graph (list[tuple]): The graph structure. Every element of graph
                is a edge that constructed by two nodes and one weight.
                For example, `[((0, 1), 1), ((1, 2), -1)]` means the graph has
                three nodes which are `0` , `1` and `2` with one edge connect
                between node `0` and node `1` with weight `1` and another
                connect between node `1` and node `2` with weight `-1`.
            partition (list[int]): a partition of the graph considered.

        Returns:
            float, cut size.
        """
        _check_graph(graph)
        _check_partition(partition)
        return self._cut_value(graph, partition)

    @staticmethod
    def _cut_value(g, s):
        """Get the cut value."""
        s = set(s)
        cut = 0
        for n, w in g:
            if len(set(n) & s) == 1:
                cut += w
        return cut

    def _enum(self, g, nodes, m, cut, u, c, s):
        """Solving MaxCut problem with enum."""
        if c > 0:
            for v in range(u):
                s_ = [*s, nodes[v]]
                cut = m(cut, (self._cut_value(g, s_), s_), key=lambda x: x[0])
                cut = m(cut, self._enum(g, nodes, m, cut, v, c - 1, s_), key=lambda x: x[0])
        return cut


class MaxCutRQAOAAnsatz(RQAOAAnsatz, MaxCut):
    r"""
    The RQAOA ansatz for MaxCut problem.

    For more detail, please refer to `Obstacles to State Preparation and
    Variational Optimization from Symmetry Protection <https://arxiv.org/pdf/1910.08980.pdf>`_.

    Args:
            graph (list[tuple]): The graph structure. Every element of graph
                is a edge that constructed by two nodes and one weight.
                For example, `[((0, 1), 1), ((1, 2), -1)]` means the graph has
                three nodes which are `0` , `1` and `2` with one edge connect
                between node `0` and node `1` with weight `1` and another
                connect between node `1` and node `2` with weight `-1`.
            nc (int): Lower threshold of the number of hamiltonian variables.
                Default: ``8``.
            p (int): The depth of QAOA ansatz. Default: ``1``.

    Examples:
        >>> from import MaxCutRQAOAAnsatz
        >>> graph = [((0, 1), 1), ((1, 2), 1), ((0, 2), 0.5), ((2, 3), 2)]
        >>> mcra = MaxCutRQAOAAnsatz(graph, nc=3)
        >>> mcra.ham                             # Hamiltonian
        1 [Z0 Z1] +
        0.5 [Z0 Z2] +
        1 [Z1 Z2] +
        2 [Z2 Z3]
        >>> mcra.get_result()                    # get brute-force enumeration results
        (4, [[0, 2], [1, 3]])
        >>> mcra.enum(graph, method=max)         # apply brute-force enumeration method on the graph
        (4, [2, 0])
        >>> mcra.get_cut_value(graph, [2, 0])    # calculate max-cut value of the graph
        4
        >>> from mindquantum.framework import MQAnsatzOnlyLayer
        >>> from mindquantum.simulator import Simulator
        >>> import mindspore.nn as nn
        >>> import mindspore as ms
        >>> import numpy as np
        >>> ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        >>> circ = mcra.circuit                  # QAOA circuit for the Hamiltonian
        >>> ham = mcra.hamiltonian               # the Hamiltonian
        >>> sim = Simulator('mqvector', circ.n_qubits)                 # create simulator with 'mqvector' backend
        >>> grad_ops = sim.get_expectation_with_grad(ham, circ)        # get expectation and gradient operators for the variational quantum circuit
        >>> net = MQAnsatzOnlyLayer(grad_ops)                          # generate ansatz to train
        >>> opti = nn.Adam(net.trainable_params(), learning_rate=0.05) # decide all trainable parameters in the model, and use Adam optimizer with learning rate 0.05
        >>> train_net = nn.TrainOneStepCell(net, opti)                 # train the model one step
        >>> e = [train_net() for i in range(30)]                       # train 30 steps
        >>> pr = dict(zip(circ.params_name, net.weight.asnumpy()))     # get model parameters
        >>> mcra.one_step_rqaoa(pr, 1)           # reduce variables according to QAOA optimization results
        -- eliminated variable: Z3
        -- correlated variable: Z2
        -- σ: -1
        True
        >>> mcra.all_variables                   # display all variables in the current Hamiltonian. Z3 is reduced.
        [(0, 'Z'), (1, 'Z'), (2, 'Z')]
        >>> mcra.restricted_set                  # the restricted set contains necessary information to recover Z3 from Z2
        [((3, 'Z'), ((2, 'Z'),), -1)]
        >>> mcra.get_result()                    # apply brute-force enumeration method on the reduced Hamiltonian of MaxCut problem, and recover the complete solution using restricted set.
        (4, [[1, 3], [0, 2]])
        >>> mcra.ham                             # current Hamiltonian
        -2 [] +
        1 [Z0 Z1] +
        0.5 [Z0 Z2] +
        1 [Z1 Z2]
        >>> mcra.one_step_rqaoa(pr, 1)           # number of variables in current Hamiltonian is less than nc, so reducing variables becomes impossible. This can be a criteria for quitting the loop.
        False
        >>> f, v, sigma = ((1, 'Z'), (2, 'Z')), (1, 'Z'), 1
        >>> mcra.eliminate_single_variable(f, sigma, v)  # However, we can still call internal method of RQAOAAnsatz to reduce variables directly
        >>> mcra.ham
        -1 [] +
        1.5 [Z0 Z2]
    """

    def __init__(self, graph, nc=8, p=1):
        """Initialize a MaxCutRQAOAAnsatz object."""
        _check_int_type('nc', nc)
        _check_value_should_not_less('nc', 3, nc)
        ham = self.get_ham_from_graph(graph)
        super().__init__(ham, p)
        self.graph = graph  # graph of maxcut problem
        self.nc = nc  # number of variables threshold

    def get_ham_from_graph(self, graph):
        """
        Get hamiltonian from graph.

        Args:
            graph (list[tuple]): The graph structure. Every element of graph
                is a edge that constructed by two nodes and one weight.
                For example, `[((0, 1), 1), ((1, 2), -1)]` means the graph has
                three nodes which are `0` , `1` and `2` with one edge connect
                between node `0` and node `1` with weight `1` and another
                connect between node `1` and node `2` with weight `-1`.

        Returns:
            QubitOperator, hamiltonian of the graph.
        """
        _check_graph(graph)
        ham = 0
        for nodes, weight in graph:
            if nodes:
                ham += QubitOperator(f'Z{nodes[0]} Z{nodes[1]}', weight)
            else:
                ham += QubitOperator('', weight)
        return ham

    def one_step_rqaoa(self, weight, show_process=False):
        """
        Eliminate one hamiltonian variable with RQAOA.

        Args:
            weight (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]): parameter
                value for QAOA ansatz.
            show_process (bool): Whether to show the process of eliminating variables. Default: False.

        Returns:
            bool, flag of running state.
        """
        if self.variables_number <= self.nc:
            return False
        self.eliminate_variable(weight, show_process)
        return True

    def get_result(self):
        """
        Get the final result of MaxCut problem solved by RQAOA.

        Returns:
            float, cut value under the partition.
            list[list[int]], a partition of the graph.
        """
        _, graph, mapping = self.get_subproblem()
        cut_value, partition = self.enum(graph)
        var_set = dict()
        for i in mapping:
            var_set[mapping[i]] = -1 if i in partition else 1
        var_set = self.translate(var_set)
        left, right = [], []
        for v in var_set:
            tag = var_set.get(v, 0)
            if tag == -1:
                left.append(v[0])
            elif tag == 1:
                right.append(v[0])
        cut_value = self.get_cut_value(self.graph, left)
        return cut_value, [left, right]
