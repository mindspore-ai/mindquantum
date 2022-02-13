# -*- coding: utf-8 -*-
import time
import numpy as np
from mindquantum.core import QubitOperator
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import nn
from mindspore.train.callback import LossMonitor
from mindspore import Model
from mindquantum.framework import MQLayer
from mindquantum import Hamiltonian, Circuit, RX, RY, X, H, UN


def GenerateWordDictAndSample(corpus, window=2):
    all_words = corpus.split()
    word_set = list(set(all_words))
    word_set.sort()
    word_dict = {w: i for i, w in enumerate(word_set)}
    sampling = []
    for index, word in enumerate(all_words[window:-window]):
        around = []
        for i in range(index, index + 2 * window + 1):
            if i != index + window:
                around.append(all_words[i])
        sampling.append([around, all_words[index + window]])
    return word_dict, sampling


word_dict, sample = GenerateWordDictAndSample("I love natural language processing")
print(word_dict)
print('word dict size: ', len(word_dict))
print('samples: ', sample)
print('number of samples: ', len(sample))


def GenerateEncoderCircuit(n_qubits, prefix=''):
    if len(prefix) != 0 and prefix[-1] != '_':
        prefix += '_'
    circ = Circuit()
    for i in range(n_qubits):
        circ += RX(prefix + str(i)).on(i)
    return circ


GenerateEncoderCircuit(3, prefix='e')

from mindquantum.simulator import Simulator
from mindspore import context
from mindspore import Tensor

n_qubits = 3
label = 2
label_bin = bin(label)[-1:1:-1].ljust(n_qubits, '0')
label_array = np.array([int(i) * np.pi for i in label_bin]).astype(np.float32)
encoder = GenerateEncoderCircuit(n_qubits, prefix='e')
encoder_params_names = encoder.params_name

print("Label is: ", label)
print("Binary label is: ", label_bin)
print("Parameters of encoder is: \n", np.round(label_array, 5))
print("Encoder circuit is: \n", encoder)
print("Encoder parameter names are: \n", encoder_params_names)

state = encoder.get_qs(pr=dict(zip(encoder_params_names, label_array)))
amp = np.round(np.abs(state)**2, 3)

print("Amplitude of quantum state is: \n", amp)
print("Label in quantum state is: ", np.argmax(amp))


def GenerateTrainData(sample, word_dict):
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


GenerateTrainData(sample, word_dict)


def GenerateAnsatzCircuit(n_qubits, layers, prefix=''):
    if len(prefix) != 0 and prefix[-1] != '_':
        prefix += '_'
    circ = Circuit()
    for l in range(layers):
        for i in range(n_qubits):
            circ += RY(prefix + str(l) + '_' + str(i)).on(i)
        for i in range(l % 2, n_qubits, 2):
            if i < n_qubits and i + 1 < n_qubits:
                circ += X.on(i + 1, i)
    return circ


GenerateAnsatzCircuit(5, 2, 'a')


def GenerateEmbeddingHamiltonian(dims, n_qubits):
    hams = []
    for i in range(dims):
        s = ''
        for j, k in enumerate(bin(i + 1)[-1:1:-1]):
            if k == '1':
                s = s + 'Z' + str(j) + ' '
        hams.append(Hamiltonian(QubitOperator(s)))
    return hams


GenerateEmbeddingHamiltonian(5, 5)


def QEmbedding(num_embedding, embedding_dim, window, layers, n_threads):
    n_qubits = int(np.ceil(np.log2(num_embedding)))
    hams = GenerateEmbeddingHamiltonian(embedding_dim, n_qubits)
    circ = Circuit()
    circ = UN(H, n_qubits)
    encoder_param_name = []
    ansatz_param_name = []
    for w in range(2 * window):
        encoder = GenerateEncoderCircuit(n_qubits, 'Encoder_' + str(w))
        ansatz = GenerateAnsatzCircuit(n_qubits, layers, 'Ansatz_' + str(w))
        encoder.no_grad()
        circ += encoder
        circ += ansatz
        encoder_param_name.extend(encoder.params_name)
        ansatz_param_name.extend(ansatz.params_name)
    grad_ops = Simulator('projectq',
                         circ.n_qubits).get_expectation_with_grad(hams, circ, None, None, encoder_param_name,
                                                                  ansatz_param_name, n_threads)
    return MQLayer(grad_ops)


class CBOW(nn.Cell):

    def __init__(self, num_embedding, embedding_dim, window, layers, n_threads, hidden_dim):
        super(CBOW, self).__init__()
        self.embedding = QEmbedding(num_embedding, embedding_dim, window, layers, n_threads)
        self.dense1 = nn.Dense(embedding_dim, hidden_dim)
        self.dense2 = nn.Dense(hidden_dim, num_embedding)
        self.relu = ops.ReLU()

    def construct(self, x):
        embed = self.embedding(x)
        out = self.dense1(embed)
        out = self.relu(out)
        out = self.dense2(out)
        return out


class LossMonitorWithCollection(LossMonitor):

    def __init__(self, per_print_times=1):
        super(LossMonitorWithCollection, self).__init__(per_print_times)
        self.loss = []

    def begin(self, run_context):
        self.begin_time = time.time()

    def end(self, run_context):
        self.end_time = time.time()
        print('Total time used: {}'.format(self.end_time - self.begin_time))

    def epoch_begin(self, run_context):
        self.epoch_begin_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        self.epoch_end_time = time.time()
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print('')

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        self.loss.append(loss)
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("\repoch: %+3s step: %+3s time: %5.5s, loss is %5.5s" %
                  (cb_params.cur_epoch_num, cur_step_in_epoch, time.time() - self.epoch_begin_time, loss),
                  flush=True,
                  end='')


import mindspore as ms
from mindspore import context
from mindspore import Tensor

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
word_dict, sample = GenerateWordDictAndSample(corpus, window=window_size)
train_x, train_y = GenerateTrainData(sample, word_dict)

train_loader = ds.NumpySlicesDataset({"around": train_x, "center": train_y}, shuffle=False).batch(3)
net = CBOW(len(word_dict), embedding_dim, window_size, 3, 4, hidden_dim)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
loss_monitor = LossMonitorWithCollection(500)
model = Model(net, net_loss, net_opt)
model.train(350, train_loader, callbacks=[loss_monitor], dataset_sink_mode=False)

import matplotlib.pyplot as plt

plt.plot(loss_monitor.loss, '.')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.show()

net.embedding.weight.asnumpy()


class CBOWClassical(nn.Cell):

    def __init__(self, num_embedding, embedding_dim, window, hidden_dim):
        super(CBOWClassical, self).__init__()
        self.dim = 2 * window * embedding_dim
        self.embedding = nn.Embedding(num_embedding, embedding_dim, True)
        self.dense1 = nn.Dense(self.dim, hidden_dim)
        self.dense2 = nn.Dense(hidden_dim, num_embedding)
        self.relu = ops.ReLU()
        self.reshape = ops.Reshape()

    def construct(self, x):
        embed = self.embedding(x)
        embed = self.reshape(embed, (-1, self.dim))
        out = self.dense1(embed)
        out = self.relu(out)
        out = self.dense2(out)
        return out


train_x = []
train_y = []
for i in sample:
    around, center = i
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

import matplotlib.pyplot as plt

plt.plot(loss_monitor.loss, '.')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.show()
