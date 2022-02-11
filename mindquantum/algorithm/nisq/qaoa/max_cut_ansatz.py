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
"""MaxCut ansatz."""

import numpy as np
from mindquantum.core.gates import H, RX, ZZ
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.operators import QubitOperator
from mindquantum.core.circuit.utils import CPN
from mindquantum.simulator import Simulator
from mindquantum.utils.type_value_check import _check_int_type
from mindquantum.utils.type_value_check import _check_value_should_between_close_set
from mindquantum.utils.type_value_check import _check_input_type
from mindquantum.utils.type_value_check import _check_value_should_not_less
from .._ansatz import Ansatz


def _get_graph_act_qubits_num(graph):
    """Get qubits number."""
    return len(_get_graph_act_qubits(graph))


def _get_graph_act_qubits(graph):
    """Get all acted qubits."""
    nodes = set({})
    for node in graph:
        nodes |= set({i for i in node})
    nodes = list(nodes)
    return sorted(nodes)


def _check_graph(graph):
    """check graph"""
    _check_input_type('graph', list, graph)
    for edge in graph:
        _check_input_type('edge', tuple, edge)
        for node in edge:
            _check_int_type('node', node)
            _check_value_should_not_less('node', 0, node)


class MaxCutAnsatz(Ansatz):
    r"""
    The MaxCut ansatz. For more detail,
    please refers to https://arxiv.org/pdf/1411.4028.pdf.

    .. math::

        U(\beta, \gamma) = e^{-\beta_pH_b}e^{-\gamma_pH_c}
        \cdots e^{-\beta_0H_b}e^{-\gamma_0H_c}H^{\otimes n}

    Where,

    .. math::

        H_b = \sum_{i\in n}X_{i}, H_c = \sum_{(i,j)\in C}Z_iZ_j

    Here :math:`n` is the set of nodes and :math:`C` is the set of
    edges of the graph.

    Args:
        graph (list[tuple[int]]): The graph structure. Every element of graph
            is a edge that constructed by two nodes.
        depth (int): The depth of max cut ansatz. Default: 1.

    Examples:
        >>> from mindquantum.algorithm.nisq.qaoa import MaxCutAnsatz
        >>> graph = [(0, 1), (1, 2), (0, 2)]
        >>> maxcut = MaxCutAnsatz(graph, 1)
        >>> maxcut.circuit
        q0: ──H────ZZ(beta_0)──────────────────ZZ(beta_0)────RX(alpha_0)──
                       │                           │
        q1: ──H────ZZ(beta_0)────ZZ(beta_0)────────┼─────────RX(alpha_0)──
                                     │             │
        q2: ──H──────────────────ZZ(beta_0)────ZZ(beta_0)────RX(alpha_0)──

        >>> maxcut.hamiltonian
        1.5 [] +
        -0.5 [Z0 Z1] +
        -0.5 [Z0 Z2] +
        -0.5 [Z1 Z2]
        >>> maxcut.hamiltonian
        >>> partitions = maxcut.get_partition(5, np.array([4, 1]))
        >>> for i in partitions:
        >>>     print(f'partition: left: {i[0]}, right: {i[1]}, cut value: {maxcut.get_cut_value(i)}')
        partition: left: [2], right: [0, 1], cut value: 2
        partition: left: [0, 1], right: [2], cut value: 2
        partition: left: [0], right: [1, 2], cut value: 2
        partition: left: [0, 1, 2], right: [], cut value: 0
        partition: left: [], right: [0, 1, 2], cut value: 0
    """

    def __init__(self, graph, depth=1):
        _check_int_type('depth', depth)
        _check_value_should_not_less('depth', 1, depth)
        _check_graph(graph)
        super(MaxCutAnsatz, self).__init__('MaxCut', _get_graph_act_qubits_num(graph), graph, depth)
        self.graph = graph
        self.all_node = set()
        for edge in self.graph:
            for node in edge:
                self.all_node.add(node)
        self.depth = depth

    def _build_hc(self, graph):
        """Build hc circuit."""
        circ = Circuit()
        for node in graph:
            circ += ZZ('beta').on(node)
        return circ

    def _build_hb(self, graph):
        """Build hb circuit."""
        circ = Circuit([RX('alpha').on(i) for i in _get_graph_act_qubits(graph)])
        return circ

    @property
    def hamiltonian(self):
        """
        Get the hamiltonian of this max cut problem.

        Returns:
            QubitOperator, hamiltonian of this max cut problem.
        """
        qo = QubitOperator('', 0)
        for node in self.graph:
            qo += (QubitOperator('') - QubitOperator(f'Z{node[0]} Z{node[1]}')) / 2
        return qo

    def get_partition(self, max_n, weight):
        """
        Get the partitions of this max-cut problem.

        Args:
            max_n (int): how many partitions you want.
            weight (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]): parameter
                value for max-cut ansatz.

        Returns:
            list, a list of partitions.
        """
        _check_int_type('max_n', max_n)
        _check_value_should_between_close_set('max_n', 1, 1 << self._circuit.n_qubits, max_n)
        sim = Simulator('projectq', self._circuit.n_qubits)
        sim.apply_circuit(self._circuit, weight)
        qs = sim.get_qs()
        idxs = np.argpartition(np.abs(qs), -max_n)[-max_n:]
        partitions = [bin(i)[2:].zfill(self._circuit.n_qubits)[::-1] for i in idxs]
        res = []
        for partition in partitions:
            left = []
            right = []
            for i, j in enumerate(partition):
                if j == '0':
                    left.append(i)
                else:
                    right.append(i)
            res.append([left, right])
        return res

    def get_cut_value(self, partition):
        """
        Get the cut values for given partitions. The partition is a list that contains two lists,
        each list contains the nodes of the given graph.

        Args:
            partition (list): a partition of the graph consided.

        Returns:
            int, cut_value under the given partition.
        """
        _check_input_type('partition', list, partition)
        if len(partition) != 2:
            raise ValueError(f"Partition of max-cut problem only need two parts, but get {len(partition)} parts")
        for part in partition:
            _check_input_type('each part of partition', list, part)
        all_node = set()
        for part in partition:
            for node in part:
                all_node.add(node)
        if all_node != self.all_node:
            raise ValueError(f"Invalid partition, partition nodes are different with given graph.")
        cut_value = 0
        for edge in self.graph:
            node_left, node_right = edge
            for n in edge:
                if n not in partition[0] and n not in partition[1]:
                    raise ValueError(f'Invalid partition, node {n} not in partition.')
                if n in partition[0] and n in partition[1]:
                    raise ValueError(f'Invalid partition, node {n} in both side of cut.')
            if (node_left in partition[0] and node_right in partition[1]
                    or node_left in partition[1] and node_right in partition[0]):
                cut_value += 1
        return cut_value

    def _implement(self, graph, depth):
        """Implement of max cut ansatz."""
        self._circuit = UN(H, _get_graph_act_qubits(graph))
        for d in range(depth):
            self._circuit += CPN(self._build_hc(graph), {'beta': f'beta_{d}'})
            self._circuit += CPN(self._build_hb(graph), {'alpha': f'alpha_{d}'})
