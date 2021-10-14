import os

os.environ['OMP_NUM_THREADS'] = '2'
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np


def generate_train_and_test(split=0.8, shuffle=True):
    iris = datasets.load_iris()
    data = iris.data[:100, :].astype(np.float32)
    data = preprocessing.minmax_scale(data) * 2 - 1
    label = np.zeros(100).astype(int)
    label[50:] = 1
    return train_test_split(data, label, train_size=split, shuffle=True)


train_x, test_x, train_y, test_y = generate_train_and_test()
print('train sample and feature shape: ', train_x.shape)

import matplotlib.pyplot as plt

feature_name = {
    0: 'speal length',
    1: 'speal width',
    2: 'petal length',
    3: 'petal width'
}
axs = plt.figure(figsize=(18, 18)).subplots(4, 4)
for i in range(4):
    for j in range(4):
        if i != j:
            ax = axs[i][j]
            ax.scatter(train_x[:, i], train_x[:, j], c=train_y)
            ax.set_xlabel(feature_name[i])
            ax.set_ylabel(feature_name[j])
plt.show()

from mindquantum import H, RZ, RX, X, Circuit


def encoder(n):
    c = Circuit([H.on(i) for i in range(n)])
    for i in range(n):
        c += RZ(f'x{i}').on(i)
    for i in range(n - 1):
        c += X.on(i + 1, i)
        c += RZ(f'x{i},{i+1}').on(i + 1)
        c += X.on(i + 1, i)
    return c


enc = encoder(4).no_grad()
enc

from mindquantum.ansatz import HardwareEfficientAnsatz
from mindquantum import X

ans = HardwareEfficientAnsatz(4,
                              single_rot_gate_seq=[RX],
                              entangle_gate=X,
                              depth=3).circuit
ans.summary()

from mindquantum.ops import QubitOperator
from mindquantum import Hamiltonian

hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [0, 1]]
hams

from mindquantum.nn import MindQuantumLayer

pqc = MindQuantumLayer(enc.para_name,
                       ans.para_name,
                       enc + ans,
                       hams,
                       n_threads=5)
pqc

import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore as ms
from mindspore.train.callback import Callback


class StepAcc(Callback):
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.acc = []

    def step_end(self, run_context):
        self.acc.append(
            self.model.eval(self.test_loader, dataset_sink_mode=False)['Acc'])


class DataPrep(ms.nn.Cell):
    def __init__(self):
        super(DataPrep, self).__init__()
        self.concat = ms.ops.Concat(axis=1)
        self.pi = np.pi

    def construct(self, x):
        y = (self.pi - x[:, :-1]) * (self.pi - x[:, 1:])
        y = self.concat((x, y))
        return y


class QuantumNet(ms.nn.Cell):
    def __init__(self, pqc):
        super(QuantumNet, self).__init__()
        self.dp = DataPrep()
        self.pqc = pqc

    def construct(self, x):
        x = self.dp(x)
        x = self.pqc(x)
        return x


batch = 5
train_loader = ds.NumpySlicesDataset({
    'feats': train_x,
    'labs': train_y
},
                                     shuffle=False).batch(batch)
test_loader = ds.NumpySlicesDataset({
    'feats': test_x,
    'labs': test_y
}).batch(batch)
net = QuantumNet(pqc)
loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
opti = ms.nn.Adam(net.trainable_params(), learning_rate=1e-1)
monitor = ms.train.callback.LossMonitor(16)
model = ms.Model(net, loss, opti, metrics={'Acc': ms.nn.Accuracy()})
acc = StepAcc(model, test_loader)
model.train(10,
            train_loader,
            callbacks=[monitor, acc],
            dataset_sink_mode=False)

plt.plot(acc.acc)
plt.title('acc')
plt.xlabel('step')
plt.ylabel('acc')
plt.show()

predict = np.argmax(ms.ops.Softmax()(model.predict(ms.Tensor(test_x))), axis=1)
corr = model.eval(test_loader, dataset_sink_mode=False)
print(corr)
