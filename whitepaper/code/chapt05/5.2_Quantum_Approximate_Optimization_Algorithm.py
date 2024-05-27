# 5.2_Quantum_Approximate_Optimization_Algorithm.py

import networkx as nx
import numpy as np
import mindspore.nn as nn
from mindquantum.core import Circuit, Hamiltonian, UN
from mindquantum.core.operators import QubitOperator
from mindquantum.simulator import Simulator
from mindquantum.core.gates import RX, Rzz, H
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.algorithm.nisq import MaxCutAnsatz

# Generate graph structure
g = nx.Graph()
nx.add_path(g, [0, 1])
nx.add_path(g, [1, 2])
nx.add_path(g, [2, 3])
nx.add_path(g, [3, 4])
nx.add_path(g, [0, 4])
nx.add_path(g, [0, 2])
nx.draw(g, with_labels=True, font_weight="bold")


# Build Hamiltonian for Max-Cut problem
def build_ham(g):
    hc = QubitOperator()
    for i in g.edges:
        hc += QubitOperator(f"Z{i[0]} Z{i[1]}")
    return hc


# Construct QAOA ansatz
def build_hc(g, para):
    hc = Circuit()
    for i in g.edges:
        hc += Rzz(para).on(i)
    return hc


def build_hb(g, para):
    hc = Circuit()
    for i in g.nodes:
        hc += RX(para).on(i)
    return hc


def build_ansatz(g, p):
    c = Circuit()
    for i in range(p):
        c += build_hc(g, f"g{i}")
        c += build_hb(g, f"b{i}")
    return c


# Set layer parameter
p = 4
ham = Hamiltonian(build_ham(g))
ansatz = build_ansatz(g, p)
init_state_circ = UN(H, g.nodes)

# Simulation
import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

total_circuit = init_state_circ + ansatz
sim = Simulator("mqvector", total_circuit.n_qubits)
grad_ops = sim.get_expectation_with_grad(ham, total_circuit)
net = MQAnsatzOnlyLayer(grad_ops)
opti = nn.Adam(net.trainable_params(), learning_rate=0.05)
train_net = nn.TrainOneStepCell(net, opti)

for i in range(600):
    if i % 10 == 0:
        print("train step:", i, ", cut:", (len(g.edges) - train_net()) / 2)

# Use MaxCutAnsatz from MindQuantum
graph = [(0, 1), (1, 2), (0, 2)]
p = 1  # layer
maxcut = MaxCutAnsatz(graph, p)
print(maxcut.circuit)

# Print Hamiltonian
print(maxcut.hamiltonian)

# Get partition and cut value
partitions = maxcut.get_partition(5, np.array([4, 1]))
for i in partitions:
    print(
        f"partition: left: {i[0]}, right: {i[1]}, cut value: {maxcut.get_cut_value(i)}"
    )
