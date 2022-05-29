# -*- coding: utf-8 -*-
import numpy as np
import mindquantum as mq
from mindquantum.core import Circuit
from mindquantum.core import H, RX, RY, RZ

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

from mindquantum.core import QubitOperator
from mindquantum.core import Hamiltonian

ham = Hamiltonian(QubitOperator('Z0', -1))
print(ham)

encoder_names = encoder.params_name
ansatz_names = ansatz.params_name

print('encoder_names = ', encoder.params_name, '\nansatz_names =', ansatz.params_name)

# 导入Simulator模块
from mindquantum.simulator import Simulator

sim = Simulator('projectq', circuit.n_qubits)

grad_ops = sim.get_expectation_with_grad(ham, circuit)

encoder_data = np.array([[alpha0, alpha1, alpha2]]).astype(np.float32)

ansatz_data = np.array([theta0, theta1]).astype(np.float32)

measure_result, encoder_grad, ansatz_grad = grad_ops(encoder_data, ansatz_data)

print('Measurement result: ', measure_result)
print('Gradient of encoder parameters: ', encoder_grad)
print('Gradient of ansatz parameters: ', ansatz_grad)

from mindquantum.framework import MQLayer
import mindspore as ms

ms.set_seed(1)
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

QuantumNet = MQLayer(grad_ops)
QuantumNet

from mindspore import nn
from mindspore.nn import Adam, TrainOneStepCell

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
fid = np.abs(np.vdot(state, [1, 0]))**2

print(fid)
