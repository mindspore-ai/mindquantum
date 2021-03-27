# MindQuantum

[View English](./README.md)

<!-- TOC --->

- [MindQuantum介绍](#mindquantum介绍)
- [安装教程](#安装教程)
    - [确认系统环境信息](#确认系统环境信息)
    - [源码安装](#源码安装)
    - [pip安装](#pip安装)
- [验证是否成功安装](#验证是否成功安装)
- [注意事项](#注意事项)
- [快速入门](#快速入门)
- [文档](#文档)
- [社区](#社区)
    - [治理](#治理)
- [贡献](#贡献)
- [许可证](#许可证)

<!-- /TOC -->

## MindQuantum介绍

MindQuantum是结合MindSpore和HiQ开发的量子机器学习框架，支持多种量子神经网络的训练和推理。得益于华为HiQ团队的量子计算模拟器和MindSpore高性能自动微分能力，MindQuantum能够高效处理量子机器学习、量子化学模拟和量子优化等问题，性能达到业界[TOP1](https://gitee.com/mindspore/mindquantum/tree/master/tutorials/benchmarks)，为广大的科研人员、老师和学生提供了快速设计和验证量子机器学习算法的高效平台。

<img src="docs/MindQuantum-architecture.png" alt="MindQuantum Architecture" width="600"/>

## 安装教程

### 确认系统环境信息

- 硬件平台确认为Linux系统下的CPU，并支持avx指令集。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装，要求至少1.2.0版本。
- 其余依赖请参见[setup.py](https://gitee.com/mindspore/mindquantum/blob/master/setup.py)

### 源码安装

1. 从代码仓下载源码

    ```bash
    cd ~
    git clone https://gitee.com/mindspore/mindquantum.git
    ```

2. 编译安装MindQuantum

    ```bash
    cd ~/mindquantum
    python setup.py install --user
    ```

### pip安装

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindQuantum/x86_64/mindquantum-{version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindQuantum安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindquantum/blob/master/setup.py)），其余情况需自行安装。
> - `{version}`表示MindQuantum版本号，例如下载0.1.0版本MindQuantum时，`{version}`应写为0.1.0。  

## 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindquantum'`，则说明安装成功。

```bash
python -c 'import mindquantum'
```
## 注意事项

运行代码前请设置量子模拟器运行时并行内核数，例如设置并行内核数为4，可运行如下代码：

```bash
export OMP_NUM_THREADS=4
```

对于大型服务器，请根据模型规模合理设置并行内核数以达到最优效果。

## 快速入门

关于如何快速搭建参数化量子线路和量子神经网络，并进行训练，请点击查看[MindQuantum使用教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/quantum_neural_network.html)

## 文档

有关安装指南、教程和API的更多详细信息，请参阅[用户文档](https://gitee.com/mindspore/docs)。

## 社区

### 治理

查看MindSpore如何进行[开放治理](https://gitee.com/mindspore/community/blob/master/governance.md)。

## 贡献

欢迎参与贡献。更多详情，请参阅我们的[贡献者Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md)。

## 许可证

[Apache License 2.0](LICENSE)
