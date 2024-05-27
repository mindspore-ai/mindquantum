# 5.1_Quantum_Neural_Network.py

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mindquantum import *
import mindspore as ms
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator
from mindspore import LossMonitor, Model
from mindspore.nn import SoftmaxCrossEntropyWithLogits, Adam, Accuracy
from mindspore.dataset import NumpySlicesDataset
from mindspore.train import Callback
from mindspore import ops

# Data initialization
iris_dataset = datasets.load_iris()

x = iris_dataset.data[:100, :].astype(np.float32)
x_feature_names = iris_dataset.feature_names
y = iris_dataset.target[:100].astype(int)
y_target_names = iris_dataset.target_names[:2]

# Scatter plot to visualize the data
feature_name = {
    0: "sepal length",
    1: "sepal width",
    2: "petal length",
    3: "petal width",
}
axes = plt.figure(figsize=(23, 23)).subplots(4, 4)

colormap = {0: "r", 1: "g"}
cvalue = [colormap[i] for i in y]

for i in range(4):
    for j in range(4):
        if i != j:
            ax = axes[i][j]
            ax.scatter(x[:, i], x[:, j], c=cvalue)
            ax.set_xlabel(feature_name[i], fontsize=22)
            ax.set_ylabel(feature_name[j], fontsize=22)

plt.show()

# Data preprocessing
alpha = x[:, :3] * x[:, 1:]
x = np.append(x, alpha, axis=1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0, shuffle=True
)

# Building the encoder circuit
encoder = Circuit()

encoder += UN(H, 4)
for i in range(4):
    encoder += RZ(f"alpha{i}").on(i)
for j in range(3):
    encoder += X.on(j + 1, j)
    encoder += RZ(f"alpha{j + 4}").on(j + 1)
    encoder += X.on(j + 1, j)

encoder = encoder.no_grad()
encoder.svg()

# Building the ansatz circuit
ansatz = HardwareEfficientAnsatz(
    4, single_rot_gate_seq=[RY], entangle_gate=X, depth=3
).circuit
ansatz.svg()

# Combining the encoder and ansatz into a full quantum circuit
circuit = encoder.as_encoder() + ansatz.as_ansatz()
circuit.svg()

# Constructing Hamiltonians
hams = [Hamiltonian(QubitOperator(f"Z{i}")) for i in [2, 3]]
for h in hams:
    print(h)

# Constructing the quantum neural network
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)
sim = Simulator("mqvector", circuit.n_qubits)
grad_ops = sim.get_expectation_with_grad(hams, circuit, parallel_worker=5)
QuantumNet = MQLayer(grad_ops)
print(QuantumNet)

# Training setup
loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
opti = Adam(QuantumNet.trainable_params(), learning_rate=0.1)

model = Model(QuantumNet, loss, opti, metrics={"Acc": Accuracy()})

train_loader = NumpySlicesDataset(
    {"features": x_train, "labels": y_train}, shuffle=False
).batch(5)
test_loader = NumpySlicesDataset({"features": x_test, "labels": y_test}).batch(5)


class StepAcc(Callback):
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.acc = []

    def step_end(self, run_context):
        self.acc.append(
            self.model.eval(self.test_loader, dataset_sink_mode=False)["Acc"]
        )


monitor = LossMonitor(16)
acc = StepAcc(model, test_loader)

model.train(20, train_loader, callbacks=[monitor, acc], dataset_sink_mode=False)

# Plotting the accuracy
plt.plot(acc.acc)
plt.title("Statistics of accuracy", fontsize=20)
plt.xlabel("Steps", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.show()

# Prediction on the testing set
predict = np.argmax(ops.Softmax()(model.predict(ms.Tensor(x_test))), axis=1)
correct = model.eval(test_loader, dataset_sink_mode=False)

print("Predicted classification result:", predict)
print("Actual classification result:", y_test)
print(correct)
