import numpy as np
import mindquantum as mq
from mindquantum.gate import H, X, Y, RY, RX

print('Gate name: ', Y)
print('Gate matrix: \n', Y.matrix())

ry = RY('a')
ry.matrix({'a': 0.5})

eng = mq.engine.CircuitEngine()
qubits = eng.allocate_qureg(3)
H | qubits[0]
X | (qubits[0], qubits[1])
RY('p1') | qubits[2]
encoder = eng.circuit
print(encoder)
encoder.summary()

from mindquantum.engine import circuit_generator


@circuit_generator(3)
def encoder(qubits):
    H | qubits[0]
    X | (qubits[0], qubits[1])
    RY('p1') | qubits[2]


print(encoder)
encoder.summary()

@circuit_generator(3, prefix='encoder')
def encoder(qubits, prefix):
    H | qubits[0]
    X | (qubits[0], qubits[1])
    RY(prefix + '_1') | qubits[2]


print(encoder)
encoder.summary()

from mindquantum import Circuit

encoder = Circuit()
encoder += H.on(0)
encoder += X.on(1, 0)
encoder += RY('p1').on(2)
print(encoder)
encoder.summary()

from mindquantum.ops import QubitOperator


@circuit_generator(2)
def encoder(qubits):
    RY('a') | qubits[0]
    RY('b') | qubits[1]


@circuit_generator(2)
def ansatz(qubits):
    X | (qubits[0], qubits[1])
    RX('p1') | qubits[0]
    RX('p2') | qubits[1]


ham = mq.Hamiltonian(QubitOperator('Z1'))
encoder_names = ['a', 'b']
ansatz_names = ['p1', 'p2']

from mindquantum.nn import generate_pqc_operator
from mindspore import Tensor
from mindspore import context

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

pqc = generate_pqc_operator(encoder_names, ansatz_names, encoder + ansatz, ham)
encoder_data = Tensor(np.array([[0.1, 0.2]]).astype(np.float32))
ansatz_data = Tensor(np.array([0.3, 0.4]).astype(np.float32))
measure_result, encoder_grad, ansatz_grad = pqc(encoder_data, ansatz_data)
print('Measurement result: ', measure_result.asnumpy())
print('Gradient of encoder parameters: ', encoder_grad.asnumpy())
print('Gradient of ansatz parameters: ', ansatz_grad.asnumpy())

encoder.no_grad()
pqc = generate_pqc_operator(encoder_names, ansatz_names, encoder + ansatz, ham)
measure_result, encoder_grad, ansatz_grad = pqc(encoder_data, ansatz_data)
print('Measurement result: ', measure_result.asnumpy())
print('Gradient of encoder parameters: ', encoder_grad.asnumpy())
print('Gradient of ansatz parameters: ', ansatz_grad.asnumpy())
