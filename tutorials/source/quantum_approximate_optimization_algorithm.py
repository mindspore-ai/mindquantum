#   Copyright 2022 <Huawei Technologies Co., Ltd>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# pylint: disable=redefined-outer-name,invalid-name

"""Example of the quantum approximate optimization algorithm."""

import matplotlib.pyplot as plt
import mindspore as ms
import networkx as nx
import numpy as np
from mindspore import nn

from mindquantum.core import RX, UN, ZZ, Circuit, H, Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator

g = nx.Graph()
nx.add_path(g, [0, 1])
nx.add_path(g, [1, 2])
nx.add_path(g, [2, 3])
nx.add_path(g, [3, 4])
nx.add_path(g, [0, 4])
nx.add_path(g, [0, 2])
nx.draw(g, with_labels=True, font_weight='bold')


def build_hc(g, para):
    """Build an HC circuit."""
    hc = Circuit()
    for i in g.edges:
        hc += ZZ(para).on(i)
    return hc


def build_hb(g, para):
    """Build an HB circuit."""
    hc = Circuit()
    for i in g.nodes:
        hc += RX(para).on(i)
    return hc


def build_ansatz(g, p):
    """Build an ansatz circuit."""
    c = Circuit()
    for i in range(p):
        c += build_hc(g, f'g{i}')
        c += build_hb(g, f'b{i}')
    return c


def build_ham(g):
    """Build a circuit for the hamiltonian."""
    hc = QubitOperator()
    for i in g.edges:
        hc += QubitOperator(f'Z{i[0]} Z{i[1]}')
    return hc


p = 4
ham = Hamiltonian(build_ham(g))
ansatz = build_ansatz(g, p)
init_state_circ = UN(H, g.nodes)


ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

circ = init_state_circ + ansatz
sim = Simulator('projectq', circ.n_qubits)
grad_ops = sim.get_expectation_with_grad(ham, circ)
net = MQAnsatzOnlyLayer(grad_ops)
opti = nn.Adam(net.trainable_params(), learning_rate=0.05)
train_net = nn.TrainOneStepCell(net, opti)

for i in range(600):
    if i % 10 == 0:
        print("train step:", i, ", cut:", (len(g.edges) - train_net()) / 2)

pr = dict(zip(ansatz.params_name, net.weight.asnumpy()))
print(circ.get_qs(pr=pr, ket=True))


def show_amp(state):
    """Show the amplitude of a quantum state."""
    amp = np.abs(state) ** 2
    n_qubits = int(np.log2(len(amp)))
    labels = [bin(i)[2:].zfill(n_qubits) for i in range(len(amp))]
    plt.bar(labels, amp)
    plt.xticks(rotation=45)
    plt.show()


state = circ.get_qs(pr=pr)
show_amp(state)
