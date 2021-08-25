from mindquantum.gate import Hamiltonian, H, ZZ, RX
from mindquantum.circuit import Circuit, StateEvolution, UN
from mindquantum.nn import MindQuantumAnsatzOnlyLayer
from mindquantum.ops import QubitOperator
import networkx as nx
import mindspore.nn as nn
import numpy as np
import matplotlib.pyplot as plt

g = nx.Graph()
nx.add_path(g, [0, 1])
nx.add_path(g, [1, 2])
nx.add_path(g, [2, 3])
nx.add_path(g, [3, 4])
nx.add_path(g, [0, 4])
nx.add_path(g, [0, 2])
nx.draw(g, with_labels=True, font_weight='bold')
plt.show()


def build_hc(g, para):
    hc = Circuit()
    for i in g.edges:
        hc += ZZ(para).on(i)
    return hc


def build_hb(g, para):
    hc = Circuit()
    for i in g.nodes:
        hc += RX(para).on(i)
    return hc


def build_ham(g):
    hc = QubitOperator()
    for i in g.edges:
        hc += QubitOperator(f'Z{i[0]} Z{i[1]}')
    return hc


def build_ansatz(g, p):
    c = Circuit()
    for i in range(p):
        c += build_hc(g, f'g{i}')
        c += build_hb(g, f'b{i}')
    return c


def show_amp(state):
    amp = np.abs(state)**2
    n_qubits = int(np.log2(len(amp)))
    labels = [bin(i)[2:].zfill(n_qubits) for i in range(len(amp))]
    plt.bar(labels, amp)
    plt.xticks(rotation=45)
    plt.show()


p = 4
ham = Hamiltonian(build_ham(g))
ansatz = build_ansatz(g, p)
init_state_circ = UN(H, g.nodes)

net = MindQuantumAnsatzOnlyLayer(ansatz.para_name, init_state_circ + ansatz,
                                 ham)
opti = nn.Adam(net.trainable_params(), learning_rate=0.05)
train_net = nn.TrainOneStepCell(net, opti)

for i in range(600):
    if i % 10 == 0:
        print("train step:", i, ", cut:", (len(g.edges) - train_net()) / 2)

pr = dict(zip(ansatz.para_name, net.weight.asnumpy()))

print(StateEvolution(init_state_circ + ansatz).final_state(pr, ket=True))
state = StateEvolution(init_state_circ + ansatz).final_state(pr)
show_amp(state)
