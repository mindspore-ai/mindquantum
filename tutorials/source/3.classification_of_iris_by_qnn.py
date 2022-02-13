# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets

iris_dataset = datasets.load_iris()

print(iris_dataset.data.shape)
print(iris_dataset.feature_names)
print(iris_dataset.target_names)
print(iris_dataset.target)
print(iris_dataset.target.shape)
X = iris_dataset.data[:100, :].astype(np.float32)
X_feature_names = iris_dataset.feature_names
y = iris_dataset.target[:100].astype(int)
y_target_names = iris_dataset.target_names[:2]

print(X.shape)
print(X_feature_names)
print(y_target_names)
print(y)
print(y.shape)

import matplotlib.pyplot as plt

feature_name = {0: 'sepal length', 1: 'sepal width', 2: 'petal length', 3: 'petal width'}
axes = plt.figure(figsize=(23, 23)).subplots(4, 4)

colormap = {0: 'r', 1: 'g'}
cvalue = [colormap[i] for i in y]

for i in range(4):
    for j in range(4):
        if i != j:
            ax = axes[i][j]
            ax.scatter(X[:, i], X[:, j], c=cvalue)
            ax.set_xlabel(feature_name[i], fontsize=22)
            ax.set_ylabel(feature_name[j], fontsize=22)
plt.show()

alpha = X[:, :3] * X[:, 1:]
X = np.append(X, alpha, axis=1)
print(X.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

print(X_train.shape)
print(X_test.shape)

import mindquantum as mq
from mindquantum.core import Circuit
from mindquantum.core import UN
from mindquantum.core import H, X, RZ

encoder = Circuit()

encoder += UN(H, 4)
for i in range(4):
    encoder += RZ(f'alpha{i}').on(i)
for j in range(3):  #j = 0, 1, 2
    encoder += X.on(j + 1, j)
    encoder += RZ(f'alpha{j+4}').on(j + 1)
    encoder += X.on(j + 1, j)

encoder = encoder.no_grad()
encoder.summary()
encoder

from mindquantum.algorithm import HardwareEfficientAnsatz
from mindquantum.core import RY

ansatz = HardwareEfficientAnsatz(4, single_rot_gate_seq=[RY], entangle_gate=X, depth=3).circuit
ansatz.summary()
ansatz

circuit = encoder + ansatz
circuit.summary()
circuit

from mindquantum.core import QubitOperator
from mindquantum.core import Hamiltonian

hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [2, 3]]
print(hams)

import mindspore as ms
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)
sim = Simulator('projectq', circuit.n_qubits)
grad_ops = sim.get_expectation_with_grad(hams,
                                         circuit,
                                         encoder_params_name=encoder.params_name,
                                         ansatz_params_name=ansatz.params_name,
                                         parallel_worker=5)
QuantumNet = MQLayer(grad_ops)
QuantumNet

from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Adam, Accuracy
from mindspore import Model
from mindspore.dataset import NumpySlicesDataset
from mindspore.train.callback import Callback, LossMonitor

loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
opti = Adam(QuantumNet.trainable_params(), learning_rate=0.1)

model = Model(QuantumNet, loss, opti, metrics={'Acc': Accuracy()})

train_loader = NumpySlicesDataset({'features': X_train, 'labels': y_train}, shuffle=False).batch(5)
test_loader = NumpySlicesDataset({'features': X_test, 'labels': y_test}).batch(5)


class StepAcc(Callback):

    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.acc = []

    def step_end(self, run_context):
        self.acc.append(self.model.eval(self.test_loader, dataset_sink_mode=False)['Acc'])


monitor = LossMonitor(16)

acc = StepAcc(model, test_loader)

model.train(20, train_loader, callbacks=[monitor, acc], dataset_sink_mode=False)

plt.plot(acc.acc)
plt.title('Statistics of accuracy', fontsize=20)
plt.xlabel('Steps', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.show()

from mindspore import ops, Tensor

predict = np.argmax(ops.Softmax()(model.predict(Tensor(X_test))), axis=1)
correct = model.eval(test_loader, dataset_sink_mode=False)

print("预测分类结果：", predict)
print("实际分类结果：", y_test)

print(correct)
