import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from mindquantum import *
import mindspore as ms
from mindquantum.ansatz import MaxCutAnsatz
from mindquantum.nn import MindQuantumAnsatzOnlyLayer
from mindquantum.gate import Hamiltonian

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="CPU")





graph = [(1, 2, 3, 11), (0, 3, 6, 8, 9, 10, 11), (0, 6, 8, 10, 13), (0, 1, 12, 14), (7, 9),(6, 7, 12),(1, 2, 5, 7),(4, 5, 6, 10, 11, 12),(1, 2, 10, 13, 14),(1, 4, 14),(1, 2, 7, 8, 14),(0, 1, 7),(3, 5, 7, 14),(2, 8),(3, 8, 9, 10, 12)]
depth = 3
maxcut = MaxCutAnsatz(graph, depth)
circuit=maxcut.circuit
sim = Simulator('projectq', circuit.n_qubits)
grad_ops = sim.get_expectation_with_grad(Hamiltonian(-maxcut.hamiltonian), circuit)
net = MQAnsatzOnlyLayer(grad_ops)
from mindspore.nn import TrainOneStepCell
from mindspore.nn import Adam

opti = Adam(net.trainable_params(), learning_rate=0.1)
train_net = TrainOneStepCell(net, opti)
for i in range(100):
    res = train_net()
    if i % 10 == 0:
        print(f"step: {i}, res: {res}")