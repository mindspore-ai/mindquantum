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

# pylint: disable=invalid-name,too-few-public-methods,duplicate-code
"""Benchmark for mnist classification with MindQuantum."""

import os
import time

import mindspore as ms
import mindspore.dataset as ds
import numpy as np
from _parse_args import parser
from mindspore import Tensor, nn, ops

from mindquantum.core import RX, Circuit, H, Hamiltonian, QubitOperator, Rxx, Rzz, X, Z
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator

args = parser.parse_args()
os.environ['OMP_NUM_THREADS'] = str(args.omp_num_threads)

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


class Hinge(nn.MSELoss):
    """Hinge loss."""

    def __init__(self, reduction='mean'):
        """Initialize a Hinge object."""
        super().__init__(reduction)
        self.maximum = ops.Maximum()
        self.mul = ops.Mul()
        self.zero = Tensor(np.array([0]).astype(np.float32))

    def construct(self, base, target):
        """Construct a Hinge node (?)."""
        x = 1 - self.mul(base.squeeze(), target.squeeze())
        x = self.maximum(x, self.zero)
        return self.get_loss(x)


def encoder_circuit_builder(n_qubits_range, prefix='encoder'):
    """
    RX encoder circuit.

    Returns:
        Circuit
    """
    c = Circuit()
    for i in n_qubits_range:
        c += RX(f'{prefix}_{i}').on(i)
    return c


class CircuitLayerBuilder:
    """Build ansatz layer."""

    def __init__(self, data_qubits, readout):
        """Initialize a CircuitLayerBuild object."""
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        """Add one layer to circuit."""
        for i, qubit in enumerate(self.data_qubits):
            symbol = f"{prefix}-{i}"
            circuit.append(gate({symbol: np.pi / 2}).on([qubit, self.readout]))


def create_quantum_model(n_qubits):
    """
    Create QNN.

    Returns:
        tuple
    """
    data_qubits = range(1, n_qubits)
    readout = 0
    c = Circuit()

    c = c + X.on(readout) + H.on(readout)
    builder = CircuitLayerBuilder(data_qubits=data_qubits, readout=readout)
    builder.add_layer(c, Rxx, 'xx1')
    builder.add_layer(c, Rzz, 'zz1')
    c += H.on(readout)
    return c, Z.on(readout)


def binary_encoder(image, n_qubits=None):
    """
    Input a binary image into data supported by RX encoder.

    Returns:
        numbers.Number
    """
    values = np.ndarray.flatten(image)
    if n_qubits is None:
        n_qubits = len(values)
    return values[:n_qubits] * np.pi


def generate_dataset(data_file_path, n_qubits, sampling_num, batch_num, eval_size_num):
    """
    Generate train and test dataset.

    Returns:
        Dataset
    """
    data = np.load(data_file_path)
    x_train_bin, y_train_nocon, x_test_bin, y_test_nocon = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    x_train_encoder = np.array([binary_encoder(x, n_qubits - 1) for x in x_train_bin]).astype(np.float32)
    x_test_encoder = np.array([binary_encoder(x, n_qubits - 1) for x in x_test_bin]).astype(np.float32)
    y_train_nocon = np.array(y_train_nocon).astype(np.int32)[:, None]
    y_test_nocon = np.array(y_test_nocon).astype(np.int32)[:, None]
    y_train_nocon = y_train_nocon * 2 - 1
    y_test_nocon = y_test_nocon * 2 - 1
    train = ds.NumpySlicesDataset(
        {"image": x_train_encoder[:sampling_num], "label": y_train_nocon[:sampling_num]}, shuffle=False
    ).batch(batch_num)

    test = ds.NumpySlicesDataset(
        {"image": x_test_encoder[:eval_size_num], "label": y_test_nocon[:eval_size_num]}, shuffle=False
    ).batch(eval_size_num)
    return train, test


def train_loop(model, dataset, loss_fn, optimizer):
    """Define train loop."""

    def forward_fn(data, label):
        """Define forward function."""
        pred = model(data)
        loss = loss_fn(pred, label)
        return loss, pred

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        """Define train step."""
        (loss, _), grads = grad_fn(data, label)
        return ops.depend(loss, optimizer(grads))

    size = dataset.get_dataset_size()
    model.set_train()
    batch_begin = time.time()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)
        if batch % 1 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {float(loss):>7f}  [{current:>3d}/{size:>3d}]  Time: {time.time() - batch_begin:>4f}")
            batch_begin = time.time()


def test_loop(model, dataset, loss_fn):
    """Test loop."""
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss = loss_fn(pred, label).asnumpy()
        correct += ((pred > 0) == (label > 0)).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    correct = float(correct)
    test_loss = float(test_loss)
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    n = 17
    num_sampling = args.num_sampling
    eval_size = 100
    batchs = args.batchs
    parallel_worker = args.parallel_worker
    epochs = 3
    file_path = './mnist_resize.npz'
    train_loader, test_loader = generate_dataset(file_path, n, num_sampling, batchs, eval_size)
    ansatz, read_out = create_quantum_model(n)
    encoder_circuit = encoder_circuit_builder(range(1, n))
    encoder_circuit.no_grad()
    encoder_names = encoder_circuit.params_name
    ansatz_names = ansatz.params_name
    ham = Hamiltonian(QubitOperator('Z0'))

    circ = encoder_circuit.as_encoder() + ansatz.as_ansatz()
    sim = Simulator('mqvector', circ.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, circ, parallel_worker=parallel_worker)
    mql = MQLayer(grad_ops, 'normal')

    net_loss = Hinge()
    net_opt = nn.Adam(mql.trainable_params())
    t0 = time.time()
    for epoc in range(epochs):
        print(f"Epoch {epoc+1}\n-------------------------------")
        train_loop(mql, train_loader, net_loss, net_opt)
    t1 = time.time()
    test_loop(mql, test_loader, net_loss)
    print(
        f"\nNum sampling:{args.num_sampling}\nBatchs:{args.batchs}\n"
        f"Parallel worker:{args.parallel_worker}\n"
        f"\nOMP THREADS:{args.omp_num_threads}\nTotal time: {t1 - t0}s"
    )
