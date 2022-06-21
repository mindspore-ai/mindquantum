# MindQuantum

[![API](https://badg.vercel.app/badge/MQ/API?scale=1.3&color=red)](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.html)
[![Tutorial](https://badg.vercel.app/badge/MQ/Tutorial?scale=1.3&color=red)](https://mindspore.cn/mindquantum/docs/en/master/parameterized_quantum_circuit.html)
[![Tutorial](https://badg.vercel.app/gitee/open-issues/mindspore/mindquantum?scale=1.3)](https://gitee.com/mindspore/mindquantum/issues)
[![Tutorial](https://badg.vercel.app/gitee/stars/mindspore/mindquantum?scale=1.3&color=purple)](https://gitee.com/mindspore/mindquantum)
[![Tutorial](https://badg.vercel.app/gitee/forks/mindspore/mindquantum?scale=1.3&color=purple)](https://gitee.com/mindspore/mindquantum)
[![Release](https://badg.vercel.app/gitee/release/mindspore/mindquantum?scale=1.3)](https://gitee.com/mindspore/mindquantum/releases)
[![LICENSE](https://badg.vercel.app/gitee/license/mindspore/mindquantum?scale=1.3)](https://github.com/mindspore-ai/mindquantum/blob/master/LICENSE)
[![PRs Welcome](https://badg.vercel.app/badge/PRs/Welcome?scale=1.3)](https://gitee.com/mindspore/mindquantum/pulls)
[![Documentation Status](https://readthedocs.org/projects/mindquantum/badge/?version=latest&style=flat)](https://mindquantum.readthedocs.io/en/latest/?badge=latest)

[查看中文](./README_CN.md)

<!-- TOC --->

- [MindQuantum](#mindquantum)
    - [What is MindQuantum](#what-is-mindquantum)
    - [First experience](#first-experience)
        - [Build parameterized quantum circuit](#build-parameterized-quantum-circuit)
        - [Train quantum neural network](#train-quantum-neural-network)
    - [Tutorials](#tutorials)
    - [API](#api)
    - [Installation](#installation)
        - [Confirming System Environment Information](#confirming-system-environment-information)
        - [Install by Source Code](#install-by-source-code)
        - [Install by pip](#install-by-pip)
            - [Install MindSpore](#install-mindspore)
            - [Install MindQuantum](#install-mindquantum)
    - [Verifying Successful Installation](#verifying-successful-installation)
    - [Install with Docker](#install-with-docker)
    - [Note](#note)
    - [Quick Start](#quick-start)
    - [Docs](#docs)
    - [Community](#community)
        - [Governance](#governance)
    - [Contributing](#contributing)
    - [How to cite](#how-to-cite)
    - [License](#license)

<!-- /TOC -->

## What is MindQuantum

MindQuantum is a general quantum computing framework developed by [MindSpore](https://www.mindspore.cn/en) and [HiQ](https://hiq.huaweicloud.com/), that can be used to build and train different quantum neural networks. Thanks to the powerful algorithm of quantum software group of Huawei and High-performance automatic differentiation ability of MindSpore, MindQuantum can efficiently handle problems such as  quantum machine learning, quantum chemistry simulation, and quantum optimization, which provides an efficient platform for researchers, teachers and students to quickly design and verify quantum machine learning algorithms.
<img src="docs/MindQuantum-architecture_EN.png" alt="MindQuantum Architecture" width="600"/>

## First experience

### Build parameterized quantum circuit

The below example shows how to build a parameterized quantum circuit.

```python
from mindquantum import *
import numpy as np
encoder = Circuit().h(0).rx({'a0': 2}, 0).ry('a1', 1)
print(encoder)
print(encoder.get_qs(pr={'a0': np.pi/2, 'a1': np.pi/2}, ket=True))
```

Then you will get,

```bash
q0: ────H───────RX(2*a0)──

q1: ──RY(a1)──────────────

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

### Train quantum neural network

```python
ansatz = CPN(encoder.hermitian(), {'a0': 'b0', 'a1': 'b1'})
sim = Simulator('projectq', 2)
ham = Hamiltonian(-QubitOperator('Z0 Z1'))
grad_ops = sim.get_expectation_with_grad(ham,
                                         encoder + ansatz,
                                         encoder_params_name=encoder.params_name,
                                         ansatz_params_name=ansatz.params_name)

import mindspore as ms
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
net = MQLayer(grad_ops)
encoder_data = ms.Tensor(np.array([[np.pi/2, np.pi/2]]))
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

## Tutorials

1. Basic usage

    - [Variational Quantum Circuit](https://mindspore.cn/mindquantum/docs/en/master/parameterized_quantum_circuit.html)
    - [Initial experience of quantum neural network](https://www.mindspore.cn/mindquantum/docs/en/master/initial_experience_of_quantum_neural_network.html)
    - [Advanced gradient calculation of variational quantum circuits](https://www.mindspore.cn/mindquantum/docs/en/master/get_gradient_of_PQC_with_mindquantum.html)

2. Variational quantum algorithm
    - [Classification of iris by quantum neural network](https://www.mindspore.cn/mindquantum/docs/en/master/classification_of_iris_by_qnn.html)
    - [Quantum Approximate Optimization Algorithm](https://mindspore.cn/mindquantum/docs/en/master/quantum_approximate_optimization_algorithm.html)
    - [The Application of Quantum Neural Network in NLP](https://mindspore.cn/mindquantum/docs/en/master/qnn_for_nlp.html)
    - [VQE Application in Quantum Chemistry Computing](https://mindspore.cn/mindquantum/docs/en/master/vqe_for_quantum_chemistry.html)

3. GENERAL QUANTUM ALGORITHM
    - [Quantum Phase Estimation algorithm](https://www.mindspore.cn/mindquantum/docs/en/master/quantum_phase_estimation.html)
    - [Grover search algorithm based on MindQuantum](https://www.mindspore.cn/mindquantum/docs/en/master/grover_search_algorithm.html)
    - [Shor’s algorithm based on MindQuantum](https://www.mindspore.cn/mindquantum/docs/en/master/shor_algorithm.html)

## API

For more API, please refers to [MindQuantum API](https://www.mindspore.cn/mindquantum/docs/en/master/mindquantum.core.html)

## Installation

### Confirming System Environment Information

- The hardware platform should be CPU with avx2 supported.
- Refer to [MindQuantum Installation Guide](https://www.mindspore.cn/install/en), install MindSpore, version 1.4.0 or later is required.
- See [setup.py](https://gitee.com/mindspore/mindquantum/blob/master/setup.py) for the remaining dependencies.

### Install by Source Code

1.Download Source Code from Gitee

```bash
cd ~
git clone https://gitee.com/mindspore/mindquantum.git
```

2.Compiling MindQuantum

```bash
cd ~/mindquantum
bash build.sh
cd output
pip install mindquantum-*.whl
```

### Install by pip

#### Install MindSpore

Please refers to [MindSpore installation guide](https://www.mindspore.cn/install) to install MindSpore that at least 1.4.0 version.

#### Install MindQuantum

- Linux

```bash
pip install https://hiq.huaweicloud.com/download/mindquantum/newest/linux/mindquantum-master-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- Windows

```bash
pip install https://hiq.huaweicloud.com/download/mindquantum/newest/windows/mindquantum-master-cp37-cp37m-win_amd64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- MacOSX

```bash
pip install https://hiq.huaweicloud.com/download/mindquantum/newest/macosx/mindquantum-master-cp37-cp37m-macosx_10_13_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - Change `cp37-cp37m` to `cp38-cp38` or `cp39-cp39` according to your python version.
> - When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [setup.py](https://gitee.com/mindspore/mindquantum/blob/master/setup.py)). In other cases, you need to manually install dependency items.

## Verifying Successful Installation

Successfully installed, if there is no error message such as No module named 'mindquantum' when execute the following command:

```bash
python -c 'import mindquantum'
```

## Install with Docker

Mac or Windows users can install MindQuantum through Docker. Please refer to [Docker installation guide](./install_with_docker_en.md)

## Note

Please set the parallel core number before running MindQuantum scripts. For example, if you want to set the parallel core number to 4, please run the command below:

```bash
export OMP_NUM_THREADS=4
```

For large servers, please set the number of parallel kernels appropriately according to the size of the model to achieve optimal results.

## Building binary wheels

If you would like to build some binary wheels for redistribution, please have a look to our [binary wheel building guide](./build_binary_wheels_en.md)

## Quick Start

For more details about how to build a parameterized quantum circuit and a quantum neural network and how to train these models, see the [MindQuantum Tutorial](https://www.mindspore.cn/mindquantum/docs/en/master/index.html).

## Docs

More details about installation guide, tutorials and APIs, please see the [User Documentation](https://gitee.com/mindspore/docs/blob/master/README.md).

## Community

### Governance

Check out how MindSpore Open Governance [works](<https://gitee.com/mindspore/community/blob/master/governance.md>).

## Contributing

Welcome contributions. See our [Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md) for more details.

## How to cite

When using MindQuantum for research, please cite:

```bash
@misc{mq_2021,
    author      = {MindQuantum Developer},
    title       = {MindQuantum, version 0.6.0},
    month       = {March},
    year        = {2021},
    url         = {https://gitee.com/mindspore/mindquantum}
}
```

## License

[Apache License 2.0](LICENSE)
