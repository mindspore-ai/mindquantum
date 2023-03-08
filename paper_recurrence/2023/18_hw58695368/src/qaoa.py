# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, ZZ, RX
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator
import mindspore.nn as nn
import mindspore as ms
import numpy as np
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
from src.config import sim_backend, qaoa_level, qaoa_step, qaoa_lr

def qaoa(g, w_key='weight', depth=None, step=None, lr=None, monitor=0):
    """
    Quantum Approximate Optimization Algorithm for MaxCut problem.

    Args:
        g (networkx.Graph): The graph structure.
        w_key (str): The key name of edge weight.
        depth (int): The depth of max cut ansatz.
        step (int): Training Steps.
        lr (float): Learning rate.
        monitor (int): Print training process.
    """
    depth = depth or qaoa_level
    step = step or qaoa_step
    lr = lr or qaoa_lr
    circ = _build_ansatz(g, depth, w_key)
    ham = Hamiltonian(_build_ham(g, w_key))
    sim = Simulator(sim_backend, circ.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    net = MQAnsatzOnlyLayer(grad_ops)
    opti = nn.Adam(net.trainable_params(), learning_rate=lr)
    train_net = nn.TrainOneStepCell(net, opti)
    for i in range(step):
        c = train_net()
        if monitor:
            if i % monitor == 0:
                print("train step:", i, ", expectation:", c)
    pr = dict(zip(circ.params_name, net.weight.asnumpy()))
    return circ, pr

def correlate_edges(g, circ, pr):
    """For every edge e∈E compute Me."""
    hams, E = _build_ham_me(g)
    sim = Simulator(sim_backend, circ.n_qubits)
    sim.apply_circuit(circ, pr=pr)
    return list(map(sim.get_expectation, hams)), E

def get_partition(g, circ, pr, max_n=1):
    """Get the partitions of this max-cut problem."""
    sim = Simulator(sim_backend, circ.n_qubits)
    sim.apply_circuit(circ, pr=pr)
    qs = sim.get_qs()
    qs = qs[:len(qs)>>1]
    idxs = np.argpartition(np.abs(qs), -max_n)[-max_n:]
    partitions = [bin(i)[2:].zfill(circ.n_qubits)[::-1] for i in idxs]
    nodes = list(g.nodes)
    res = []
    for p in partitions:
        r = dict()
        for i in nodes:
            r[i] = 2 * int(p[i]) - 1
        res.append(r)
    return res

def get_expectation(g, circ, pr, w_key='weight'):
    ham = Hamiltonian(_build_ham(g, w_key))
    sim = Simulator(sim_backend, circ.n_qubits)
    sim.apply_circuit(circ, pr=pr)
    return sim.get_expectation(ham)

def _build_hc(g, para, w_key='weight'):
    hc = Circuit()
    E = g.edges
    for i in E:
        hc += ZZ({para:E[i].get(w_key, 1)}).on(i)
    return hc
def _build_hb(g, para):
    hb = Circuit()
    for i in g.nodes:
        hb += RX(para).on(i)
    return hb
def _build_ansatz(g, p, w_key='weight'):
    """QAOA circuit."""
    circ = UN(H, g.nodes)
    for i in range(p):
        circ += _build_hc(g, f'g{i}', w_key)
        circ += _build_hb(g, f'b{i}')
    return circ
def _build_ham(g, w_key='weight'):
    """QAOA hamiltonian."""
    ham = QubitOperator()
    E = g.edges
    for i in E:
        ham += QubitOperator(f'Z{i[0]} Z{i[1]}') * E[i].get(w_key, 1)
    return ham
def _build_ham_me(g):
    """Hamiltonian used to compute Me."""
    hams = []
    E = list(g.edges)
    for e in E:
        hams.append(Hamiltonian(QubitOperator(f'Z{e[0]} Z{e[1]}')))
    return hams, E
