# MindQuantum

[查看中文](./README_CN.md)

<!-- TOC --->

- [What is MindQuantum](#what-is-mindquantum)
- [Installation](#installation)
    - [Confirming System Environment Information](#confirming-system-environment-information)
    - [Install by Source Code](#install-by-source-code)
    - [Install by pip](#install-by-pip)
        - [Install MindSpore](#install-mindspore)
        - [Install MindQuantum](#install-mindquantum)
- [Verifying Successful Installation](#verifying-successful-installation)
- [Install with Docker](#install-with-docker)
- [Note](#Note)
- [Quick Start](#quick-start)
- [Docs](#docs)
- [Community](#community)
    - [Governance](#governance)
- [Contributing](#contributing)
- [License](#license)

<!-- /TOC -->

## What is MindQuantum

MindQuantum is a quantum machine learning framework developed by [MindSpore](https://www.mindspore.cn/en) and [HiQ](https://hiq.huaweicloud.com/), that can be used to build and train different quantum neural networks. Thanks to the powerful algorithm of quantum software group of Huawei and High-performance automatic differentiation ability of MindSpore, MindQuantum can efficiently handle problems such as quantum chemical simulation and quantum approximation optimization with [TOP1](https://gitee.com/mindspore/mindquantum/tree/master/tutorials/benchmarks) performance, which provides an efficient platform for researchers, teachers and students to quickly design and verify quantum machine learning algorithms.

<img src="docs/MindQuantum-architecture_EN.png" alt="MindQuantum Architecture" width="600"/>

## Installation

### Confirming System Environment Information

- The hardware platform should be Linux CPU with avx supported.
- Refer to [MindQuantum Installation Guide](https://www.mindspore.cn/install/en), install MindSpore, version 1.2.0 or later is required.
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

```bash
pip install https://hiq.huaweicloud.com/download/mindspore/cpu/x86_64/mindspore-1.3.0-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### Install MindQuantum

```bash
pip install https://hiq.huaweicloud.com/download/mindquantum/any/mindquantum-0.2.0-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [setup.py](https://gitee.com/mindspore/mindquantum/blob/master/setup.py)). In other cases, you need to manually install dependency items.

## Verifying Successful Installation

Successfully installed, if there is no error message such as No module named 'mindquantum' when execute the following command:

```bash
python -c 'import mindquantum'
```

## Install with Docker

Mac or Windows users can install MindQuantum through Docker. Please refer to [Docker installation guide](./install_with_docker.md)

## Note

Please set the parallel core number before running MindQuantum scripts. For example, if you want to set the parallel core number to 4, please run the command below:

```bash
export OMP_NUM_THREADS=4
```

For large servers, please set the number of parallel kernels appropriately according to the size of the model to achieve optimal results.

## Quick Start

For more details about how to build a parameterized quantum circuit and a quantum neural network and how to train these models, see the [MindQuantum Tutorial](https://www.mindspore.cn/mindquantum/docs/en/master/index.html).

## Docs

More details about installation guide, tutorials and APIs, please see the [User Documentation](https://gitee.com/mindspore/docs/blob/master/README.md).

## Community

### Governance

Check out how MindSpore Open Governance [works](<https://gitee.com/mindspore/community/blob/master/governance.md>).

## Contributing

Welcome contributions. See our [Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md) for more details.

## License

[Apache License 2.0](LICENSE)
