# MindQuantum

[View English](./README.md)

<!-- TOC --->

- [MindQuantum介绍](#mindquantum介绍)
- [初体验](#初体验)
    - [搭建参数化量子线路](#搭建参数化量子线路)
    - [训练量子神经网络](#训练量子神经网络)
- [API](#api)
- [安装教程](#安装教程)
    - [确认系统环境信息](#确认系统环境信息)
    - [pip安装](#pip安装)
        - [安装MindSpore](#安装mindspore)
        - [安装MindQuantum](#安装mindquantum)
    - [源码安装](#源码安装)
- [验证是否成功安装](#验证是否成功安装)
- [Docker安装](#docker安装)
- [注意事项FAQ](#注意事项faq)
- [快速入门](#快速入门)
- [文档](#文档)
- [社区](#社区)
    - [治理](#治理)
- [贡献](#贡献)
- [许可证](#许可证)

<!-- /TOC -->

## MindQuantum介绍

MindQuantum是基于华为开源自研AI框架MindSpore开发的高性能量子-经典混合计算框架，能高效的生成多种变分量子线路，支持量子模拟、量子组合优化、量子机器学习等NISQ算法，性能达到业界[领先水平](https://gitee.com/mindspore/mindquantum/tree/master/tutorials/benchmarks)。结合HiQ量子计算云平台，MindQuantum可以作为广大的科研人员、老师和学生快速设计和体验量子计算的高效解决方案。

<img src="docs/MindQuantum-architecture.png" alt="MindQuantum Architecture" width="600"/>

## 初体验

### 搭建参数化量子线路

通过如下示例可便捷搭建参数化量子线路

```python
from mindquantum import *
import numpy as np
encoder = Circuit().h(0).rx({'a0': 2}, 0).ry('a1', 1)
print(encoder)
print(encoder.get_qs(pr={'a0': np.pi/2, 'a1': np.pi/2}, ket=True))
```

你将得到

```bash
q0: ────H───────RX(2*a0)──

q1: ──RY(a1)──────────────

-1/2j¦00⟩
-1/2j¦01⟩
-1/2j¦10⟩
-1/2j¦11⟩
```

### 训练量子神经网络

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

训练得到参数为

```bash
{'b1': 1.5720831, 'b0': 0.006396801}
```

## API

对于上述示例所涉及API和其他更多用法，请查看MindQuantum API文档[文档链接](https://www.mindspore.cn/mindquantum/api/zh-CN/master/index.html)

## 安装教程

### 确认系统环境信息

- 硬件平台确认为Linux系统下的CPU，并支持avx指令集。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装，要求至少1.2.0版本。
- 其余依赖请参见[setup.py](https://gitee.com/mindspore/mindquantum/blob/master/setup.py)

### pip安装

#### 安装MindSpore

请根据MindSpore官网[安装指南](https://www.mindspore.cn/install)，安装1.3.0及以上版本的MindSpore。

#### 安装MindQuantum

```bash
pip install https://hiq.huaweicloud.com/download/mindquantum/any/mindquantum-0.2.0-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindQuantum安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindquantum/blob/master/setup.py)），其余情况需自行安装。

### 源码安装

1. 从代码仓下载源码

    ```bash
    cd ~
    git clone https://gitee.com/mindspore/mindquantum.git
    ```

2. 编译MindQuantum

    Linux系统下请确保安装好CMake >= 3.18.3，然后运行如下命令：

    ```bash
    cd ~/mindquantum
    bash build.sh
    ```

    Windows系统下请确保安装好MinGW-W64和CMake >= 3.18.3，然后运行如下命令：

    ```bash
    cd mindquantum
    ./build.bat -G "MinGW Makefiles"
    ```

3. 安装编译好的whl包

    进入output目录，通过`pip`命令安装编译好的mindquantum的whl包。

## 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindquantum'`，则说明安装成功。

```bash
python -c 'import mindquantum'
```

## Docker安装

通过Docker也可以在Mac系统或者Windows系统中使用Mindquantum。具体参考[Docker安装指南](./install_with_docker.md).

## 注意事项FAQ

运行代码前请设置量子模拟器运行时并行内核数，例如设置并行内核数为4，可运行如下代码：

```bash
export OMP_NUM_THREADS=4
```

对于大型服务器，请根据模型规模合理设置并行内核数以达到最优效果。

更多注意事项请查看[FAQ页面](https://gitee.com/mindspore/mindquantum/blob/master/tutorials/0.frequently_asked_questions.ipynb)。

## 快速入门

关于如何快速搭建参数化量子线路和量子神经网络，并进行训练，请点击查看[MindQuantum使用教程](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/index.html)

## 文档

有关安装指南、教程和API的更多详细信息，请参阅[用户文档](https://gitee.com/mindspore/docs)。

## 社区

### 治理

查看MindSpore如何进行[开放治理](https://gitee.com/mindspore/community/blob/master/governance.md)。

## 贡献

欢迎参与贡献。更多详情，请参阅我们的[贡献者Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md)。

## 许可证

[Apache License 2.0](LICENSE)
