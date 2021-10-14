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

from mindquantum.gate import H, RX, ZZ
from mindquantum.circuit import Circuit, CPN, UN
from mindquantum.ops import QubitOperator
from ._ansatz import Ansatz


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
    if not isinstance(graph, list):
        raise TypeError(f"graph requires a list, but get {type(graph)}")
    for edge in graph:
        if not isinstance(edge, tuple):
            raise TypeError(f"edge requires a tuple, but get {type(edge)}")
        for node in edge:
            if not isinstance(node, int):
                raise TypeError(f"node requires a int, but get {type(node)}")
            if node < 0:
                raise ValueError(
                    f"node requires a positive number, but get {node}")


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
        >>> from mindquantum.ansatz import MaxCutAnsatz
        >>> graph = [(0, 1), (1, 2), (0, 2)]
        >>> maxcut = MaxCutAnsatz(graph, 2)
        >>> maxcut.circuit
        H(0)
        H(1)
        H(2)
        ZZ(beta_0|0 1)
        ZZ(beta_0|1 2)
        ZZ(beta_0|0 2)
        RX(alpha_0|0)
        RX(alpha_0|1)
        RX(alpha_0|2)
        ZZ(beta_1|0 1)
        ZZ(beta_1|1 2)
        ZZ(beta_1|0 2)
        RX(alpha_1|0)
        RX(alpha_1|1)
        RX(alpha_1|2)

        >>> maxcut.hamiltonian
        1.5 [] +
        -0.5 [Z0 Z1] +
        -0.5 [Z0 Z2] +
        -0.5 [Z1 Z2]
    """
    def __init__(self, graph, depth=1):
        if not isinstance(depth, int):
            raise TypeError(f"depth requires a int, but get {type(depth)}")
        if depth <= 0:
            raise ValueError(f"depth must be greater than 0, but get {depth}.")
        _check_graph(graph)
        super(MaxCutAnsatz, self).__init__('MaxCut',
                                           _get_graph_act_qubits_num(graph),
                                           graph, depth)
        self.graph = graph
        self.depth = depth

    def _build_hc(self, graph):
        """Build hc circuit."""
        circ = Circuit()
        for node in graph:
            circ += ZZ('beta').on(node)
        return circ

    def _build_hb(self, graph):
        """Build hb circuit."""
        circ = Circuit(
            [RX('alpha').on(i) for i in _get_graph_act_qubits(graph)])
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
            qo += (QubitOperator('') -
                   QubitOperator(f'Z{node[0]} Z{node[1]}')) / 2
        return qo

    def _implement(self, graph, depth):
        """Implement of max cut ansatz."""
        self._circuit = UN(H, _get_graph_act_qubits(graph))
        for d in range(depth):
            self._circuit += CPN(self._build_hc(graph), {'beta': f'beta_{d}'})
            self._circuit += CPN(self._build_hb(graph),
                                 {'alpha': f'alpha_{d}'})
