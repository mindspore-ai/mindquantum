# MindQuantum

[View English](./README.md)

<!-- TOC --->

- [MindQuantum介绍](#mindquantum介绍)
- [安装教程](#安装教程)
    - [确认系统环境信息](#确认系统环境信息)
    - [pip安装](#pip安装)
        - [安装MindSpore](#安装mindspore)
        - [安装MindQuantum](#安装mindquantum)
    - [源码安装](#源码安装)
- [API](#api)
- [验证是否成功安装](#验证是否成功安装)
- [Docker安装](#docker安装)
- [注意事项](#注意事项)
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

## 安装教程

### 确认系统环境信息

- 硬件平台确认为Linux系统下的CPU，并支持avx指令集。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装，要求至少1.2.0版本。
- 其余依赖请参见[setup.py](https://gitee.com/mindspore/mindquantum/blob/master/setup.py)

### pip安装

#### 安装MindSpore

```bash
pip install https://hiq.huaweicloud.com/download/mindspore/cpu/x86_64/mindspore-1.3.0-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 请根据本机的python版本选择合适的安装包，如本机为python 3.7，则可将上面命令中的`cp38-cp38`修改为`cp37-cp37m`。

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

2. 编译安装MindQuantum

    ```bash
    cd ~/mindquantum
    python setup.py install --user
    ```

## API

MindQuantum API文档请查看[文档链接](https://www.mindspore.cn/mindquantum/api/zh-CN/master/index.html)

## 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindquantum'`，则说明安装成功。

```bash
python -c 'import mindquantum'
```

## Docker安装

通过Docker也可以在Mac系统或者Windows系统中使用Mindquantum。具体参考[Docker安装指南](./install_with_docker.md).

## 注意事项

运行代码前请设置量子模拟器运行时并行内核数，例如设置并行内核数为4，可运行如下代码：

```bash
export OMP_NUM_THREADS=4
```

对于大型服务器，请根据模型规模合理设置并行内核数以达到最优效果。

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
