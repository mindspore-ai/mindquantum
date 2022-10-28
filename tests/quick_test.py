# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Basic functionality test module."""

# NB: This file is mainly used to perform quick installation tests when building Python binary wheels

import numpy as np

from mindquantum.core import RX, RY, Hamiltonian, QubitOperator, X
from mindquantum.engine import circuit_generator
from mindquantum.simulator import Simulator


@circuit_generator(2)
def encoder(qubits):
    """Generate an encoder circuit."""
    # pylint: disable=expression-not-assigned
    RY('a') | qubits[0]
    RY('b') | qubits[1]


@circuit_generator(2)
def ansatz(qubits):
    """Generate an Ansatz."""
    # pylint: disable=expression-not-assigned,pointless-statement
    X | (qubits[0], qubits[1])
    RX('p1') | qubits[0]
    RX('p2') | qubits[1]


ham = Hamiltonian(QubitOperator('Z1'))
encoder_names = ['a', 'b']
ansatz_names = ['p1', 'p2']

total_circuit = encoder.as_encoder() + ansatz.as_ansatz()
sim = Simulator('mqvector', total_circuit.n_qubits)
grad_ops = sim.get_expectation_with_grad(ham, total_circuit)
encoder_data = np.array([[0.1, 0.2]])
ansatz_data = np.array([0.3, 0.4])
measure_result, encoder_grad, ansatz_grad = grad_ops(encoder_data, ansatz_data)
print('Measurement result: ', measure_result)
print('Gradient of encoder parameters: ', encoder_grad)
print('Gradient of ansatz parameters: ', ansatz_grad)
