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
"""Benchmark for QAOA with MindQuantum"""
import time
import os
from _parse_args import parser
args = parser.parse_args()
os.environ['OMP_NUM_THREADS'] = str(args.omp_num_threads)
import numpy as np
from mindquantum.ops import QubitOperator
import mindspore.context as context
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Model
from mindspore.train.callback import LossMonitor
from mindquantum import Hamiltonian
from mindquantum import Circuit
from mindquantum import RX, X, RZ, H
from mindquantum.circuit import UN
from mindquantum.nn import MindQuantumLayer


def circuit_qaoa(p):
    circ = Circuit()
    circ += UN(H, n)
    for layer in range(p):
        for (u, v) in E:
            circ += X.on(v, u)
            circ += RZ('gamma_{}'.format(layer)).on(v)
            circ += X.on(v, u)
        for v in V:
            circ += RX('beta_{}'.format(layer)).on(v)
    return circ


n = 12
V = range(n)
E = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 6),
     (6, 7), (7, 8), (3, 8), (3, 9), (4, 9), (0, 10), (10, 11), (3, 11)]
p = 4
ITR = 120
LR = 0.1

ham = QubitOperator()
for (v, u) in E:
    ham += QubitOperator('Z{} Z{}'.format(v, u), -1.0)
ham = Hamiltonian(ham)

circ = circuit_qaoa(p)
ansatz_name = circ.parameter_resolver().para_name
net = MindQuantumLayer(['null'], ansatz_name, RX('null').on(0) + circ, ham)
train_loader = ds.NumpySlicesDataset({
    'x': np.array([[0]]).astype(np.float32),
    'y': np.array([0]).astype(np.float32)
}).batch(1)


class Loss(nn.MSELoss):
    """Loss"""
    def construct(self, base, target):
        return self.get_loss(-base)


context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
net_loss = Loss()
net_opt = nn.Adam(net.trainable_params(), learning_rate=LR)
model = Model(net, net_loss, net_opt)
t0 = time.time()
model.train(ITR, train_loader, callbacks=[LossMonitor()])
t1 = time.time()
print('Total time for mindquantum :{}'.format(t1 - t0))
