# -*- coding: utf-8 -*-
"""
[1] https://doi.org/10.48550/arXiv.2106.13304

@NoEvaa
"""
import numpy as np
from mindquantum.core import QubitOperator, Hamiltonian
from mindquantum.simulator import Simulator
from src.ansatz_mpo import generate_ansatz_mpo
class MBEEdge:
    """
    MBE edge.
    """
    def __init__(self, nodes, weight):
        self.nodes = nodes
        self.weight = weight
    def get_loss(self, et, grad):
        """Get loss."""
        f1, g1 = et[self.nodes[0][0]][self.nodes[0][1]]
        f2, g2 = et[self.nodes[1][0]][self.nodes[1][1]]
        f = self.weight * np.tanh(f1) * np.tanh(f2)
        if grad:
            g = (1 - np.tanh(f1) ** 2) * g1 * np.tanh(f2)
            g += (1 - np.tanh(f2) ** 2) * g2 * np.tanh(f1)
            return f, self.weight * g
        return f, 0
class MBEGraph:
    """
    MBE graph.
    """
    def __init__(self, n, g):
        self.n = n
        self.build_graph(g)
    def build_graph(self, g):
        """Build graph."""
        self.graph = []
        m = 2 * self.n
        for i in g:
            if len(i) == 2:
                i.append(1)
            if len(i) != 3:
                raise
            if i[2] <= 1e-8:
                continue
            if i[0] == i[1]:
                raise ValueError('Invalid edge definition.')
            if max(i[0], i[1]) > m:
                raise ValueError('Node out of bounds.')
            nodes = [[i[0] // self.n, i[0] % self.n],
                     [i[1] // self.n, i[1] % self.n]]
            self.graph.append(MBEEdge(nodes, i[2]))
    def get_loss(self, et, grad):
        """Get loss."""
        f, g = 0, 0
        for i in self.graph:
            f_, g_ = i.get_loss(et, grad)
            f += f_
            g += g_
        return f, g
class MBELoss:
    """
    MBE loss.

    Args:
        n (int): Nodes of graph.
        depth (int): Depth of circuit.
    """
    def __init__(self, n, depth):
        self.n = n
        self.n_q = (n + 1) // 2
        self.circ = generate_ansatz_mpo(self.n_q, depth)
        self.build_grad_ops()
        self.graph = None
    def build_grad_ops(self):
        """Build grad_ops."""
        sim = Simulator('projectq', self.n_q)
        self.ops_z, self.ops_x = [], []
        for i in range(self.n_q):
            ham = Hamiltonian(QubitOperator(f'Z{i}'))
            self.ops_z.append(sim.get_expectation_with_grad(ham, self.circ))
            ham = Hamiltonian(QubitOperator(f'X{i}'))
            self.ops_x.append(sim.get_expectation_with_grad(ham, self.circ))
    def set_graph(self, g):
        """Set graph."""
        self.graph = MBEGraph(self.n_q, g)
    def get_loss(self, w, grad=False):
        """
        Get loss.

        Args:
            w (np.ndarray): Weights.
            grad (bool): Return gradient or not.
        """
        if self.graph is None:
            raise ValueError('Please execute `set_graph` first!')
        fz, fx = [], []
        for gz in self.ops_z:
            fz.append(gz(w))
        for gx in self.ops_x:
            fx.append(gx(w))
        return self.graph.get_loss([fz, fx], grad)
    def measure(self, w):
        """Measure"""
        fz, fx = [], []
        for gz in self.ops_z:
            fz.append(gz(w)[0])
        for gx in self.ops_x:
            fx.append(gx(w)[0])
        return (np.squeeze(np.real(fz)),
                np.squeeze(np.real(fx)))
