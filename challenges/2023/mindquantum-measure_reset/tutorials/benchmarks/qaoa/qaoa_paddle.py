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

# pylint: disable=redefined-outer-name,invalid-name,too-few-public-methods,duplicate-code

"""Benchmark for QAOA with PaddlePaddle Quantum."""

import time

import numpy as np
import paddle
from paddle_quantum.circuit import UAnsatz

n = 12
V = range(n)
E = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 0),
    (0, 3),
    (1, 4),
    (2, 6),
    (6, 7),
    (7, 8),
    (3, 8),
    (3, 9),
    (4, 9),
    (0, 10),
    (10, 11),
    (3, 11),
]

h_d_list = []
for (u, v) in E:
    h_d_list.append([-1.0, f"z{u},z{v}"])


def circuit_qaoa(p, gamma, beta):
    """Generate a QAOA circuit."""
    cir = UAnsatz(n)
    cir.superposition_layer()
    for layer in range(p):
        for (u, v) in E:
            cir.cnot([u, v])
            cir.rz(gamma[layer], v)
            cir.cnot([u, v])

        for v in V:
            cir.rx(beta[layer], v)

    return cir


class Net(paddle.nn.Layer):
    """Net class."""

    def __init__(self, p, dtype="float64"):
        """Initialize a Net object."""
        super().__init__()

        self.p = p
        self.gamma = self.create_parameter(
            shape=[self.p],
            default_initializer=paddle.nn.initializer.Uniform(low=0.0, high=2 * np.pi),
            dtype=dtype,
            is_bias=False,
        )
        self.beta = self.create_parameter(
            shape=[self.p],
            default_initializer=paddle.nn.initializer.Uniform(low=0.0, high=2 * np.pi),
            dtype=dtype,
            is_bias=False,
        )

    def forward(self):
        """Forward algorithm method."""
        cir = circuit_qaoa(self.p, self.gamma, self.beta)
        cir.run_state_vector()
        loss = -cir.expecval(h_d_list)

        return loss, cir


p = 4
ITR = 120
LR = 0.1
SEED = 1024

paddle.seed(SEED)

net = Net(p)
opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())
t0 = time.time()
for itr in range(1, ITR + 1):
    loss, cir = net()
    loss.backward()
    opt.minimize(loss)
    opt.clear_grad()
    if itr % 10 == 0:
        print("iter:", itr, "  loss:", f"{loss.numpy():.4f}")
t1 = time.time()
print(f'Total time for paddle quantum :{t1 - t0}')
