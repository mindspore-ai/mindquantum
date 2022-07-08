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

# pylint: disable=too-few-public-methods,duplicate-code

"""Benchmakr for mnist classification of tensorflow quantum."""

import time

import cirq
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
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
x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

model_circuit, model_readout = create_quantum_model()
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        tfq.layers.PQC(model_circuit, model_readout),
    ]
)
y_train_hinge = 2.0 * y_train_nocon - 1.0
y_test_hinge = 2.0 * y_test - 1.0


def hinge_accuracy(y_true, y_pred):
    """Hinge accuracy calculation function."""
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)


NUM_EXAMPLES = args.num_sampling
x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]

model.compile(loss=tf.keras.losses.Hinge(), optimizer=tf.keras.optimizers.Adam(), metrics=[hinge_accuracy])

t0 = time.time()
qnn_history = model.fit(x_train_tfcirc_sub, y_train_hinge_sub, batch_size=args.batchs, epochs=3, verbose=1)
t1 = time.time()
print(f"Total time: {t1 - t0}")
