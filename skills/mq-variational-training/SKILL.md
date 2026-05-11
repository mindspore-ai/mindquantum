---
name: mq-variational-training
description: "Build and train variational quantum algorithms (VQE, QAOA, QML, QNN) with MindQuantum. Covers the encoder/ansatz circuit pattern, get_expectation_with_grad, gradient-based optimization with SciPy or MindSpore MQLayer, and hybrid quantum-classical training loops. Use whenever the user wants to train a parameterized quantum circuit, run VQE, implement QAOA, build a quantum neural network, compute quantum gradients, use MQLayer, optimize circuit parameters, or do any hybrid quantum-classical machine learning with MindQuantum."
---

# Variational Training with MindQuantum

This skill covers common workflows for training parameterized quantum circuits, from circuit design through optimization to result extraction.

## The MindQuantum Variational Pipeline

Common MindQuantum variational workflows use this pipeline:

```text
Circuit Design → Hamiltonian → Simulator.get_expectation_with_grad → Optimization Loop → Results
     │                │                      │                              │
  encoder +        QubitOperator →       GradOpsWrapper              SciPy or MindSpore
  ansatz           Hamiltonian           encoder+ansatz: f,g_enc,g_ans
                                       ansatz-only: f,g
```

## Pattern 1: SciPy Optimization (No MindSpore Required)

Use this pattern when the objective is a scalar expectation value and you want a plain NumPy/SciPy optimization loop.

```python
import numpy as np
from scipy.optimize import minimize
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, RY, RX, CNOT
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.simulator import Simulator

# 1. Build ansatz (no encoder — pure variational)
n_qubits = 4
ansatz = Circuit()
for i in range(n_qubits):
    ansatz += RY(f"p{i}").on(i)
for i in range(n_qubits - 1):
    ansatz += CNOT.on(i + 1, i)
for i in range(n_qubits):
    ansatz += RY(f"q{i}").on(i)

# 2. Define Hamiltonian
ham = Hamiltonian(
    QubitOperator("Z0 Z1", -1.0)
    + QubitOperator("Z1 Z2", -1.0)
    + QubitOperator("Z2 Z3", -1.0)
    + QubitOperator("X0", -0.5)
)

# 3. Create gradient operator
sim = Simulator("mqvector", n_qubits)
grad_ops = sim.get_expectation_with_grad(ham, ansatz)


# 4. Wrap for SciPy (value + gradient)
def fun(params):
    f, g = grad_ops(params)
    return np.real(f)[0, 0], np.real(g)[0, 0]


# 5. Optimize
x0 = np.random.uniform(-np.pi, np.pi, len(ansatz.params_name))
result = minimize(fun, x0, method="BFGS", jac=True)
print(f"Ground state energy: {result.fun:.6f}")
```

## Pattern 2: MindSpore MQLayer Training

Use this pattern when a parameterized quantum circuit is part of a MindSpore model. Requires MindSpore.

```python
import numpy as np
import mindspore as ms
from mindspore import nn
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RY, CNOT
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator

ms.set_device("CPU")
ms.set_context(mode=ms.PYNATIVE_MODE)

# 1. Encoder: data → quantum state
encoder = Circuit()
for i in range(4):
    encoder += RY(f"x{i}").on(i)
encoder.as_encoder()

# 2. Ansatz: trainable weights
ansatz = Circuit()
for i in range(4):
    ansatz += RY(f"w{i}").on(i)
for i in range(3):
    ansatz += CNOT.on(i + 1, i)
ansatz.as_ansatz()

circuit = encoder + ansatz

# 3. Hamiltonian and gradient operator
ham = Hamiltonian(QubitOperator("Z0"))
sim = Simulator("mqvector", circuit.n_qubits)
grad_ops = sim.get_expectation_with_grad(ham, circuit)

# 4. Create MQLayer (acts as a MindSpore nn.Cell)
qnet = MQLayer(grad_ops)

# 5. Standard MindSpore training.
# MQLayer returns the circuit expectation; wrap it with a loss cell for supervised tasks.
opti = nn.Adam(qnet.trainable_params(), learning_rate=0.1)
train_net = nn.TrainOneStepCell(qnet, opti)

# Training loop
for epoch in range(100):
    encoder_data = ms.Tensor(np.random.uniform(0, np.pi, (1, 4)).astype(np.float32))
    loss = train_net(encoder_data)
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {float(loss.asnumpy().mean()):.4f}")

# 6. Extract trained parameters
print(dict(zip(ansatz.params_name, qnet.weight.asnumpy())))
```

## Pattern 3: MQAnsatzOnlyLayer (No Encoder Data)

For VQE and QAOA where there is no classical input data:

```python
from mindquantum.framework import MQAnsatzOnlyLayer

# Circuit has only ansatz parameters (no encoder)
grad_ops = sim.get_expectation_with_grad(ham, ansatz_circuit)
net = MQAnsatzOnlyLayer(grad_ops)

opti = nn.Adam(net.trainable_params(), learning_rate=0.05)
train_net = nn.TrainOneStepCell(net, opti)

for i in range(300):
    loss = train_net()
    if i % 50 == 0:
        print(f"Step {i}: E = {loss.asnumpy():.6f}")
```

## VQE Workflow

```python
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, RY, RX, CNOT
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.simulator import Simulator
import numpy as np
from scipy.optimize import minimize

# Hamiltonian for H2 (simplified)
ham_str = (
    QubitOperator("", -0.8)
    + QubitOperator("Z0", 0.17)
    + QubitOperator("Z1", 0.17)
    + QubitOperator("Z0 Z1", 0.16)
    + QubitOperator("X0 X1", 0.04)
    + QubitOperator("Y0 Y1", 0.04)
)
ham = Hamiltonian(ham_str)

# Hardware-efficient ansatz
ansatz = Circuit()
ansatz += RY("a0").on(0)
ansatz += RY("a1").on(1)
ansatz += CNOT.on(1, 0)
ansatz += RY("a2").on(0)
ansatz += RY("a3").on(1)

sim = Simulator("mqvector", 2)
grad_ops = sim.get_expectation_with_grad(ham, ansatz)


def energy_and_grad(params):
    f, g = grad_ops(params)
    return np.real(f)[0, 0], np.real(g)[0, 0]


x0 = np.zeros(4)
result = minimize(energy_and_grad, x0, method="BFGS", jac=True)
print(f"VQE Energy: {result.fun:.6f} Ha")
```

## QAOA Workflow

```python
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, Rzz, RX
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.simulator import Simulator
import networkx as nx
import numpy as np
from scipy.optimize import minimize

# 1. Problem: Max-Cut on a graph
g = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
n = g.number_of_nodes()

# 2. Cost Hamiltonian from edges
ham = QubitOperator()
for u, v in g.edges:
    ham += QubitOperator(f"Z{u} Z{v}")

# 3. QAOA circuit
p = 3  # number of layers
init = Circuit(UN(H, range(n)))
ansatz = Circuit()
for layer in range(p):
    for u, v in g.edges:
        ansatz += Rzz(f"g{layer}").on([u, v])
    for node in range(n):
        ansatz += RX(f"b{layer}").on(node)

circuit = init + ansatz

# 4. Optimize
sim = Simulator("mqvector", n)
grad_ops = sim.get_expectation_with_grad(Hamiltonian(ham), circuit)


def cost(params):
    f, g = grad_ops(params)
    return np.real(f)[0, 0], np.real(g)[0, 0]


result = minimize(cost, np.random.uniform(-np.pi, np.pi, 2 * p), method="BFGS", jac=True)

# 5. Extract solution
pr = dict(zip(circuit.params_name, result.x))
sim.reset()
sim.apply_circuit(circuit, pr)
state = sim.get_qs()
probs = np.abs(state) ** 2
best = np.argmax(probs)
print(f"Most probable bitstring: {bin(best)[2:].zfill(n)}, Cost: {result.fun:.4f}")
```

## QML Classification Workflow

```python
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RY, CNOT
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator
import mindspore as ms
from mindspore import nn
import numpy as np

ms.set_device("CPU")
ms.set_context(mode=ms.PYNATIVE_MODE)

n_features = 4
n_qubits = 4

# Encoder: amplitude encoding via rotations
encoder = Circuit()
for i in range(n_features):
    encoder += RY(f"f{i}").on(i)
encoder.as_encoder()

# Ansatz: entangling layers
ansatz = Circuit()
for i in range(n_qubits):
    ansatz += RY(f"w{i}").on(i)
for i in range(n_qubits - 1):
    ansatz += CNOT.on(i + 1, i)
for i in range(n_qubits):
    ansatz += RY(f"v{i}").on(i)

# Measurement: Z expectation as prediction
ham = Hamiltonian(QubitOperator("Z0"))
sim = Simulator("mqvector", n_qubits)
grad_ops = sim.get_expectation_with_grad(ham, encoder + ansatz)


# Hybrid model: quantum layer inside classical network
class HybridQNN(nn.Cell):
    def __init__(self):
        super().__init__()
        self.qnn = MQLayer(grad_ops)
        self.dense = nn.Dense(1, 2)  # map expectation → 2 classes

    def construct(self, x):
        q_out = self.qnn(x)
        return self.dense(q_out)


model = HybridQNN()
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
opti = nn.Adam(model.trainable_params(), learning_rate=0.05)
train_net = nn.TrainOneStepCell(nn.WithLossCell(model, loss_fn), opti)

# Train on data
# train_features: [batch, 4], train_labels: [batch]
# for epoch in range(20):
#     train_net(train_features, train_labels)
```

## Optimizer Notes

| Scenario | API-compatible options |
|----------|------------------------|
| SciPy with gradients from `get_expectation_with_grad` | `BFGS`, `L-BFGS-B`, or another SciPy method accepting `jac=True` |
| SciPy without using gradients | Gradient-free SciPy methods such as `Nelder-Mead` or `COBYLA` |
| MindSpore training | MindSpore optimizers such as `nn.Adam` or `nn.SGD` over `MQLayer` / `MQAnsatzOnlyLayer` trainable parameters |

## Barren Plateau Awareness

MindQuantum exposes `ansatz_variance` for checking gradient variance of a selected parameter.

```python
from mindquantum.algorithm.nisq import ansatz_variance

# Check one trainable parameter before training
var = ansatz_variance(
    ansatz,
    ham,
    focus=ansatz.params_name[0],
    init_batch=100,
    sim=sim,
)
print(var)
```

## Reference Files

| File | When to Read |
|------|-------------|
| `reference/ansatz-catalog.md` | Choosing between built-in ansätze (HEA, UCCSD, QAOA, StronglyEntangling, etc.) |
