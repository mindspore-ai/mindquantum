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

"""Example of running a QNN iris classification."""

import matplotlib.pyplot as plt
import mindspore as ms
import numpy as np
from mindspore import Model, Tensor, ops
from mindspore.dataset import NumpySlicesDataset
from mindspore.nn import Accuracy, Adam, SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import Callback, LossMonitor
from sklearn import datasets
from sklearn.model_selection import train_test_split

from mindquantum.algorithm import HardwareEfficientAnsatz
from mindquantum.core import RY, RZ, UN, Circuit, H, Hamiltonian, QubitOperator, X
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator

iris_dataset = datasets.load_iris()

print(iris_dataset.data.shape)
print(iris_dataset.feature_names)
print(iris_dataset.target_names)
print(iris_dataset.target)
print(iris_dataset.target.shape)

XX = iris_dataset.data[:100, :].astype(np.float32)
XX_feature_names = iris_dataset.feature_names
y = iris_dataset.target[:100].astype(int)
y_target_names = iris_dataset.target_names[:2]

print(XX.shape)
print(XX_feature_names)
print(y_target_names)
print(y)
print(y.shape)


feature_name = {0: 'sepal length', 1: 'sepal width', 2: 'petal length', 3: 'petal width'}
axes = plt.figure(figsize=(23, 23)).subplots(4, 4)

colormap = {0: 'r', 1: 'g'}
cvalue = [colormap[i] for i in y]

for i in range(4):
    for j in range(4):
        if i != j:
            ax = axes[i][j]
            ax.scatter(XX[:, i], XX[:, j], c=cvalue)
            ax.set_xlabel(feature_name[i], fontsize=22)
            ax.set_ylabel(feature_name[j], fontsize=22)
plt.show()

alpha = XX[:, :3] * XX[:, 1:]
XX = np.append(XX, alpha, axis=1)
print(XX.shape)


X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=0.2, random_state=0, shuffle=True)

print(X_train.shape)
print(X_test.shape)


encoder = Circuit()

encoder += UN(H, 4)
for i in range(4):
    encoder += RZ(f'alpha{i}').on(i)
for j in range(3):  # j = 0, 1, 2
    encoder += X.on(j + 1, j)
    encoder += RZ(f'alpha{j+4}').on(j + 1)
    encoder += X.on(j + 1, j)

encoder = encoder.no_grad()
encoder.summary()
encoder


ansatz = HardwareEfficientAnsatz(4, single_rot_gate_seq=[RY], entangle_gate=X, depth=3).circuit
ansatz.summary()
ansatz
encoder.as_encoder()
ansatz.as_ansatz()
circuit = encoder + ansatz
circuit.summary()
circuit

hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [2, 3]]
print(hams)


ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)
sim = Simulator('projectq', circuit.n_qubits)
grad_ops = sim.get_expectation_with_grad(hams, circuit, parallel_worker=5)
QuantumNet = MQLayer(grad_ops)
QuantumNet


loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
opti = Adam(QuantumNet.trainable_params(), learning_rate=0.1)

model = Model(QuantumNet, loss, opti, metrics={'Acc': Accuracy()})

train_loader = NumpySlicesDataset({'features': X_train, 'labels': y_train}, shuffle=False).batch(5)
test_loader = NumpySlicesDataset({'features': X_test, 'labels': y_test}).batch(5)


class StepAcc(Callback):
    """Step accumulator class."""

    def __init__(self, model, test_loader):
        """Initialize a StepAcc object."""
        self.model = model
        self.test_loader = test_loader
        self.acc = []

    def step_end(self, run_context):
        """Mark the end of an accumulation."""
        self.acc.append(self.model.eval(self.test_loader, dataset_sink_mode=False)['Acc'])


monitor = LossMonitor(16)

acc = StepAcc(model, test_loader)

model.train(20, train_loader, callbacks=[monitor, acc], dataset_sink_mode=False)

plt.plot(acc.acc)
plt.title('Statistics of accuracy', fontsize=20)
plt.xlabel('Steps', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.show()


predict = np.argmax(ops.Softmax()(model.predict(Tensor(X_test))), axis=1)
correct = model.eval(test_loader, dataset_sink_mode=False)

print("预测分类结果：", predict)
print("实际分类结果：", y_test)

print(correct)
