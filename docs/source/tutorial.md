# First experience

## Build parameterized quantum circuit

The below example shows how to build a parameterized quantum circuit.

```python
from mindquantum import *
import numpy as np

encoder = Circuit().h(0).rx({'a0': 2}, 0).ry('a1', 1)
print(encoder)
print(encoder.get_qs(pr={'a0': np.pi / 2, 'a1': np.pi / 2}, ket=True))
```

Then you will get,

```bash
      ┏━━━┓ ┏━━━━━━━━━━┓
q0: ──┨ H ┠─┨ RX(2*a0) ┠───
      ┗━━━┛ ┗━━━━━━━━━━┛
      ┏━━━━━━━━┓
q1: ──┨ RY(a1) ┠───────────
      ┗━━━━━━━━┛

-1/2j¦00⟩
-1/2j¦01⟩
-1/2j¦10⟩
-1/2j¦11⟩
```

In jupyter notebook, we can just call `svg()` of any circuit to display the circuit in svg picture (`dark` and `light` mode are also supported).

```python
circuit = (qft(range(3)) + BarrierGate(True)).measure_all()
circuit.svg()
```

<img src="https://gitee.com/mindspore/mindquantum/raw/master/docs/circuit_svg.png" alt="Circuit SVG" width="600"/>

## Train quantum neural network

```python
ansatz = CPN(encoder.hermitian(), {'a0': 'b0', 'a1': 'b1'})
sim = Simulator('mqvector', 2)
ham = Hamiltonian(-QubitOperator('Z0 Z1'))
grad_ops = sim.get_expectation_with_grad(
    ham,
    encoder.as_encoder() + ansatz.as_ansatz(),
)

import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
net = MQLayer(grad_ops)
encoder_data = ms.Tensor(np.array([[np.pi / 2, np.pi / 2]]))
opti = ms.nn.Adam(net.trainable_params(), learning_rate=0.1)
train_net = ms.nn.TrainOneStepCell(net, opti)
for i in range(100):
    train_net(encoder_data)
print(dict(zip(ansatz.params_name, net.trainable_params()[0].asnumpy())))
```

The trained parameters are,

```bash
{'b1': 1.5720831, 'b0': 0.006396801}
```

# Tutorials

1. Basic usage

    - [Variational Quantum Circuit](https://mindspore.cn/mindquantum/docs/en/master/parameterized_quantum_circuit.html)

2. Variational quantum algorithm

    - [Quantum Approximate Optimization Algorithm](https://mindspore.cn/mindquantum/docs/en/master/quantum_approximate_optimization_algorithm.html)
    - [The Application of Quantum Neural Network in NLP](https://mindspore.cn/mindquantum/docs/en/master/qnn_for_nlp.html)
    - [VQE Application in Quantum Chemistry Computing](https://mindspore.cn/mindquantum/docs/en/master/vqe_for_quantum_chemistry.html)
