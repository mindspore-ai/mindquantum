# -*- coding: utf-8 -*-
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

"""Benchmark for mnist classification with MindQuantum."""

import os
import time

import mindspore as ms
import mindspore.dataset as ds
import numpy as np
from _parse_args import parser
from mindspore import Model, Tensor, nn
from mindspore.ops import operations as ops
from mindspore.train.callback import Callback

from mindquantum.core import RX, XX, ZZ, Circuit, H, Hamiltonian, QubitOperator, X, Z
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator

args = parser.parse_args()
os.environ['OMP_NUM_THREADS'] = str(args.omp_num_threads)

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


class FPSMonitor(Callback):
    """A fps monitor."""

    def __init__(self, forget_first_n=2):
        """Initialize a FPSMonitor object."""
        super(FPSMonitor, self).__init__()
        self.forget_first_n = forget_first_n
        self.step = 0
        self.times = np.array([])

    def step_begin(self, run_context):
        """Step initialization method."""
        run_context.original_args()
        self.step += 1
        if self.step > self.forget_first_n:
            self.times = np.append(self.times, time.time())

    def step_end(self, run_context):
        """Step finalization method."""
        run_context.original_args()
        if self.times.size > 0:
            self.times[-1] = time.time() - self.times[-1]
            print("\rAverage: {:.6}ms/step".format(self.times.mean() * 1000), end="")


class Hinge(nn.MSELoss):
    """Hinge loss."""

    def __init__(self, reduction='mean'):
        """Initialize a Hinge object."""
        super(Hinge, self).__init__(reduction)
        self.maximum = ops.Maximum()
        self.mul = ops.Mul()
        self.zero = Tensor(np.array([0]).astype(np.float32))

    def construct(self, base, target):
        """Construct a Hinge node (?)."""
        x = 1 - self.mul(base, target)
        x = self.maximum(x, self.zero)
        return self.get_loss(x)


class MnistNet(nn.Cell):
    """Net for mnist dataset."""

    def __init__(self, net):
        """Initialize a MnistNet object."""
        super(MnistNet, self).__init__()
        self.net = net

    def construct(self, x):
        """Construct a MnistNet node (?)."""
        x = self.net(x)
        return x


def encoder_circuit_builder(n_qubits_range, prefix='encoder'):
    """
    RX encoder circuit.

    Returns:
        Circuit
    """
    c = Circuit()
    for i in n_qubits_range:
        c += RX('{}_{}'.format(prefix, i)).on(i)
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
            symbol = prefix + '-' + str(i)
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
    builder.add_layer(c, XX, 'xx1')
    builder.add_layer(c, ZZ, 'zz1')
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

    circ = encoder_circuit + ansatz
    sim = Simulator('projectq', circ.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, circ, None, None, encoder_names, ansatz_names, parallel_worker)
    mql = MQLayer(grad_ops, 'normal')

    mnist_net = MnistNet(mql)
    net_loss = Hinge()
    net_opt = nn.Adam(mnist_net.trainable_params())
    model = Model(mnist_net, net_loss, net_opt)
    fps = FPSMonitor(5)
    t0 = time.time()
    model.train(epochs, train_loader, callbacks=[fps])
    t1 = time.time()
    print(
        "\nNum sampling:{}\nBatchs:{}\nParallel worker:{}\nOMP THREADS:{}\nTotal time: {}s".format(
            args.num_sampling, args.batchs, args.parallel_worker, args.omp_num_threads, t1 - t0
        )
    )
    res = np.array([])
    for train_x, train_y in train_loader:
        y_pred = mnist_net(train_x)
        res = np.append(res, (train_y.asnumpy() > 0) == (y_pred.asnumpy() > 0))
    print('Acc: {}'.format(np.mean(res)))
