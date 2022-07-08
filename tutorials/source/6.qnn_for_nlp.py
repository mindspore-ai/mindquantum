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

"""Example of running QNN for NLP."""

# pylint: disable=redefined-outer-name,invalid-name,too-few-public-methods,unused-argument


import time

import matplotlib.pyplot as plt
import mindspore as ms
import mindspore.dataset as ds
import numpy as np
from mindspore import Model, Tensor, context, nn, ops
from mindspore.train.callback import LossMonitor

from mindquantum import RX, RY, UN, Circuit, H, Hamiltonian, X
from mindquantum.core import QubitOperator
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator


def generate_word_dict_and_sample(corpus, window=2):
    """Generate a word dictionary and some samples."""
    all_words = corpus.split()
    word_set = list(set(all_words))
    word_set.sort()
    word_dict = {w: i for i, w in enumerate(word_set)}
    sampling = []
    for index, _ in enumerate(all_words[window:-window]):
        around = []
        for i in range(index, index + 2 * window + 1):
            if i != index + window:
                around.append(all_words[i])
        sampling.append([around, all_words[index + window]])
    return word_dict, sampling


word_dict, sample = generate_word_dict_and_sample("I love natural language processing")
print(word_dict)
print('word dict size: ', len(word_dict))
print('samples: ', sample)
print('number of samples: ', len(sample))


def generate_encoder_circuit(n_qubits, prefix=''):
    """Generate an encoder circuit."""
    if len(prefix) != 0 and prefix[-1] != '_':
        prefix += '_'
    circ = Circuit()
    for i in range(n_qubits):
        circ += RX(prefix + str(i)).on(i)
    return circ.as_encoder()


generate_encoder_circuit(3, prefix='e')


n_qubits = 3
label = 2
label_bin_global = bin(label)[-1:1:-1].ljust(n_qubits, '0')
label_array_global = np.array([int(i) * np.pi for i in label_bin_global]).astype(np.float32)
encoder = generate_encoder_circuit(n_qubits, prefix='e')
encoder_params_names = encoder.params_name

print("Label is: ", label)
print("Binary label is: ", label_bin_global)
print("Parameters of encoder is: \n", np.round(label_array_global, 5))
print("Encoder circuit is: \n", encoder)
print("Encoder parameter names are: \n", encoder_params_names)

state = encoder.get_qs(pr=dict(zip(encoder_params_names, label_array_global)))
amp = np.round(np.abs(state) ** 2, 3)

print("Amplitude of quantum state is: \n", amp)
print("Label in quantum state is: ", np.argmax(amp))


def generate_train_data(sample, word_dict):
    """Generate some training data."""
    n_qubits = np.int(np.ceil(np.log2(1 + max(word_dict.values()))))
    data_x = []
    data_y = []
    for around, center in sample:
        data_x.append([])
        for word in around:
            label = word_dict[word]
            label_bin = bin(label)[-1:1:-1].ljust(n_qubits, '0')
            label_array = [int(i) * np.pi for i in label_bin]
            data_x[-1].extend(label_array)
        data_y.append(word_dict[center])
    return np.array(data_x).astype(np.float32), np.array(data_y).astype(np.int32)


generate_train_data(sample, word_dict)


def generate_ansatz_circuit(n_qubits, layers, prefix=''):
    """Generate some training ansatz circuit."""
    if len(prefix) != 0 and prefix[-1] != '_':
        prefix += '_'
    circ = Circuit()
    for layer in range(layers):
        for i in range(n_qubits):
            circ += RY(f"{prefix + layer}_{i}").on(i)
        for i in range(layer % 2, n_qubits, 2):
            if i < n_qubits and i + 1 < n_qubits:
                circ += X.on(i + 1, i)
    return circ


generate_ansatz_circuit(5, 2, 'a')


def generate_embedding_hamiltonian(dims, n_qubits):
    """Generate an embedded hamiltonian."""
    hams = []
    for i in range(dims):
        s = ''
        for j, k in enumerate(bin(i + 1)[-1:1:-1]):
            if k == '1':
                s = f"{s}Z{j} "
        hams.append(Hamiltonian(QubitOperator(s)))
    return hams


generate_embedding_hamiltonian(5, 5)


def q_embedding(num_embedding, embedding_dim, window, layers, n_threads):
    """QEmbedding."""
    n_qubits = int(np.ceil(np.log2(num_embedding)))
    hams = generate_embedding_hamiltonian(embedding_dim, n_qubits)
    circ = Circuit()
    circ = UN(H, n_qubits)
    encoder_param_name = []
    ansatz_param_name = []
    for w in range(2 * window):
        encoder = generate_encoder_circuit(n_qubits, f"Encoder_{w}")
        ansatz = generate_ansatz_circuit(n_qubits, layers, f"Ansatz_{w}")
        encoder.no_grad()
        circ += encoder
        circ += ansatz
        encoder_param_name.extend(encoder.params_name)
        ansatz_param_name.extend(ansatz.params_name)
    grad_ops = Simulator('projectq', circ.n_qubits).get_expectation_with_grad(hams, circ, parallel_worker=n_threads)
    return MQLayer(grad_ops)


class CBOW(nn.Cell):
    """CBOW class."""

    # pylint: disable=too-many-arguments

    def __init__(self, num_embedding, embedding_dim, window, layers, n_threads, hidden_dim):
        """Initialize a CBOW object."""
        super().__init__()
        self.embedding = q_embedding(num_embedding, embedding_dim, window, layers, n_threads)
        self.dense1 = nn.Dense(embedding_dim, hidden_dim)
        self.dense2 = nn.Dense(hidden_dim, num_embedding)
        self.relu = ops.ReLU()

    def construct(self, x):
        """Construct a CBOW(?)."""
        embed = self.embedding(x)
        out = self.dense1(embed)
        out = self.relu(out)
        out = self.dense2(out)
        return out


class LossMonitorWithCollection(LossMonitor):
    """LossMonitorWithCollection class."""

    # pylint: disable=attribute-defined-outside-init

    def __init__(self, per_print_times=1):
        """Initialize a LossMonitorWithCollection object."""
        super().__init__(per_print_times)
        self.loss = []

    def begin(self, run_context):
        """Begin method."""
        self.begin_time = time.time()

    def end(self, run_context):
        """End method."""
        self.end_time = time.time()
        print(f'Total time used: {self.end_time - self.begin_time}')

    def epoch_begin(self, run_context):
        """Epoch begin method."""
        self.epoch_begin_time = time.time()

    def epoch_end(self, run_context):
        """Epoch end method."""
        cb_params = run_context.original_args()
        self.epoch_end_time = time.time()
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print('')

    def step_end(self, run_context):
        """Step end finalizer method."""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError(
                f"epoch: {cb_params.cur_epoch_num} step: {cur_step_in_epoch}. Invalid loss, terminating training."
            )
        self.loss.append(loss)
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print(
                (
                    f"\repoch: {cb_params.cur_epoch_num:>3} step: {cur_step_in_epoch:>3} "
                    f"time: {(time.time() - self.epoch_begin_time):5.5f}, loss is  {loss:5.5f}"
                ),
                flush=True,
                end='',
            )


context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
corpus = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells."""

ms.set_seed(42)
window_size = 2
embedding_dim = 10
hidden_dim = 128
word_dict, sampling = generate_word_dict_and_sample(corpus, window=window_size)
train_x, train_y = generate_train_data(sampling, word_dict)

train_loader = ds.NumpySlicesDataset({"around": train_x, "center": train_y}, shuffle=False).batch(3)
net = CBOW(len(word_dict), embedding_dim, window_size, 3, 4, hidden_dim)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
loss_monitor = LossMonitorWithCollection(500)
model = Model(net, net_loss, net_opt)
model.train(350, train_loader, callbacks=[loss_monitor], dataset_sink_mode=False)


plt.plot(loss_monitor.loss, '.')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.show()

net.embedding.weight.asnumpy()


class CBOWClassical(nn.Cell):
    """CBOWClassical class."""

    def __init__(self, num_embedding, embedding_dim, window, hidden_dim):
        """Initialize a CBOWClassical object."""
        super().__init__()
        self.dim = 2 * window * embedding_dim
        self.embedding = nn.Embedding(num_embedding, embedding_dim, True)
        self.dense1 = nn.Dense(self.dim, hidden_dim)
        self.dense2 = nn.Dense(hidden_dim, num_embedding)
        self.relu = ops.ReLU()
        self.reshape = ops.Reshape()

    def construct(self, x):
        """Construct a CBOWClassical(?)."""
        embed = self.embedding(x)
        embed = self.reshape(embed, (-1, self.dim))
        out = self.dense1(embed)
        out = self.relu(out)
        out = self.dense2(out)
        return out


train_x = []
train_y = []
for sample in sampling:
    around, center = sample
    train_y.append(word_dict[center])
    train_x.append([])
    for j in around:
        train_x[-1].append(word_dict[j])
train_x = np.array(train_x).astype(np.int32)
train_y = np.array(train_y).astype(np.int32)
print("train_x shape: ", train_x.shape)
print("train_y shape: ", train_y.shape)

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

train_loader = ds.NumpySlicesDataset({"around": train_x, "center": train_y}, shuffle=False).batch(3)
net = CBOWClassical(len(word_dict), embedding_dim, window_size, hidden_dim)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
loss_monitor = LossMonitorWithCollection(500)
model = Model(net, net_loss, net_opt)
model.train(350, train_loader, callbacks=[loss_monitor], dataset_sink_mode=False)


plt.plot(loss_monitor.loss, '.')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.show()
