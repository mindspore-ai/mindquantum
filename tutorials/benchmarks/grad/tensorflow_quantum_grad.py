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

# pylint: disable=redefined-outer-name,too-few-public-methods,protected-access,duplicate-code

"""Benchmark for gradient calculation of tensorflow quantum."""

import time

import cirq
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
import tqdm
from _parse_args import parser

args = parser.parse_args()
tf.config.threading.set_intra_op_parallelism_threads(args.omp_num_threads)


def convert_to_circuit(image):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit


class CircuitLayerBuilder:
    """CircuitLayerBuilder class."""

    def __init__(self, data_qubits, readout):
        """Initialize a CircuitLayerBuilder object."""
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        """Add a layer to this instance."""
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout) ** symbol)


def create_quantum_model():
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(4, 4)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)  # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(data_qubits=data_qubits, readout=readout)

    # Then add layers (experiment by adding more).
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)


n_qubits = 17
data = np.load('./mnist_resize.npz')
x_train_bin, y_train_nocon, x_test_bin, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
x_train_circ = [convert_to_circuit(x) for x in x_train_bin]
x_test_circ = [convert_to_circuit(x) for x in x_test_bin]

model_circuit, model_readout = create_quantum_model()
names = sorted(model_circuit._parameter_names_())
init = np.random.random(len(names))[None, :].astype(np.float32)
values_tensor = tf.convert_to_tensor(init)
for c in x_train_circ:
    c.append(model_circuit)

expectation_calculation = tfq.layers.Expectation(differentiator=tfq.differentiators.Adjoint())

t0 = time.time()
eval_time = []
for circuit in tqdm.tqdm(x_train_circ[: args.num_sampling]):
    eval_time.append(time.time())
    with tf.GradientTape() as g:
        g.watch(values_tensor)
        exact_outputs = expectation_calculation(
            model_circuit, operators=model_readout, symbol_names=names, symbol_values=values_tensor
        )
    g.gradient(exact_outputs, values_tensor)
    eval_time[-1] = time.time() - eval_time[-1]
eval_time = np.sort(eval_time[1:])
t1 = time.time()
print(f"Eval grad mean time:{eval_time[1:-1].mean()}")
print(f"Total time:{t1 - t0}")
