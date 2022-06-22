# -*- coding: utf-8 -*-
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

"""Example of running a quantum neural network."""

import mindspore as ms
import numpy as np
from mindspore.nn import Adam, TrainOneStepCell

from mindquantum.core import RX, RY, RZ, Circuit, H, Hamiltonian, QubitOperator
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator

encoder = Circuit()
encoder += H.on(0)
encoder += RX(f'alpha{0}').on(0)
encoder += RY(f'alpha{1}').on(0)
encoder += RZ(f'alpha{2}').on(0)
encoder = encoder.no_grad()
encoder.summary()
encoder

alpha0, alpha1, alpha2 = 0.2, 0.3, 0.4
state = encoder.get_qs(pr={'alpha0': alpha0, 'alpha1': alpha1, 'alpha2': alpha2}, ket=True)
print(state)

ansatz = Circuit()
ansatz += RX(f'theta{0}').on(0)
ansatz += RY(f'theta{1}').on(0)
ansatz

theta0, theta1 = 0, 0
state = ansatz.get_qs(pr=dict(zip(ansatz.params_name, [theta0, theta1])), ket=True)
print(state)
encoder.as_encoder()
ansatz.as_ansatz()
circuit = encoder + ansatz
circuit


ham = Hamiltonian(QubitOperator('Z0', -1))
print(ham)

encoder_names = encoder.params_name
ansatz_names = ansatz.params_name

print('encoder_names = ', encoder.params_name, '\nansatz_names =', ansatz.params_name)

# 导入Simulator模块

sim = Simulator('projectq', circuit.n_qubits)

grad_ops = sim.get_expectation_with_grad(ham, circuit)

encoder_data = np.array([[alpha0, alpha1, alpha2]]).astype(np.float32)

ansatz_data = np.array([theta0, theta1]).astype(np.float32)

measure_result, encoder_grad, ansatz_grad = grad_ops(encoder_data, ansatz_data)

print('Measurement result: ', measure_result)
print('Gradient of encoder parameters: ', encoder_grad)
print('Gradient of ansatz parameters: ', ansatz_grad)


ms.set_seed(1)
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

QuantumNet = MQLayer(grad_ops)
QuantumNet


opti = Adam(QuantumNet.trainable_params(), learning_rate=0.5)
net = TrainOneStepCell(QuantumNet, opti)

for i in range(200):
    res = net(ms.Tensor(encoder_data))
    if i % 10 == 0:
        print(i, ': ', res)

theta0, theta1 = QuantumNet.weight.asnumpy()

print(QuantumNet.weight.asnumpy())

pr = {'alpha0': alpha0, 'alpha1': alpha1, 'alpha2': alpha2, 'theta0': theta0, 'theta1': theta1}
state = circuit.get_qs(pr=pr, ket=True)

print(state)

state = circuit.get_qs(pr=pr)
fid = np.abs(np.vdot(state, [1, 0])) ** 2

print(fid)
