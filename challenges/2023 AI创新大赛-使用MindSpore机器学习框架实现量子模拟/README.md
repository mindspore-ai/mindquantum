# QuSmoke

基于 Mindspore 开发的量子模拟器。

## 背景描述

在NISQ阶段，变分量子算法是最有可能具有使用价值的算法。在变分量子算法里，我们需要学习量子线路中的参数，并使得线路的测量结果跟目标解决，因此我们需要利用梯度下降的算法来完成学习任务。而mindspore架构具有自动微分的能力，因此我们想要利用mindspore中的tensor作为基本数据类型、利用mindspore中的各种算子来完成量子模拟任务。此时在mindspore框架下，整个量子算法流程都是可微分的，能够达成量子机器学习的任务。

## 问题描述

- mindspore机器学习框架使用便捷，具有高度的自定义性，且多平台适用。
- 考虑利用mindspore来完成量子模拟器，并自动利用mindspore架构完成量子线路的梯度计算。

## 硬件平台

- CPU
- GPU
- Ascend

## 软件平台

- Python >= 3.7
- Mindspore 2.0

## 功能

- 实现常见量子逻辑门，包括
  - 单量子比特门：`H`，`X`，`Y`，`Z`，`RX`，`RY`，`RZ`，`T`，`SWAP`，`ISWAP`，`U1`，`U2`，`U3`
  - 多量子比特门：所有单量子比特门受控形式，支持单比特或多比特控制量子门。例如可通过单比特或两比特控制 `X` 门实现 `CNOT` 或 `Toffoli` 门。
- 量子线路幅值计算、实现哈密顿量测量等功能。
- 利用 Mindspore 的自动微分实现变分量子算法，能直接适用 Mindspore 的 `nn.Adam` 等优化器进行梯度更新。

## 特点

- 基于 Mindspore 开发，量子线路采用深度学习的架构，将每个量子逻辑门类比经典神经网络层，用户可以像使用经典神经网络一样进行量子线路设计。
- 支持 Mindspore 静态图。
- 接口尽量与 mindquantum 保持一致，降低从 mindquantum 到 qusmoke 的学习成本。

## 文件结构

其文件主要包括 `qusmoke/` 文件夹和 `demo/` 文件夹，`qusmoke/` 下主要包括模拟器开发功能，`demo/` 为提供的一些案例教程。

```python
.
│  README.md  
│
├─qusmoke/
│      circuit.py               # 线路模块
│      define.py                # 全局参数定义
│      expect.py                # 哈密顿量/期望值模块
│      gates.py                 # 量子逻辑门模块
│      operations.py            # 针对复数进行的操作模块
│      utils.py                 # 辅助功能函数
│
└─demo/
        demo_basic.ipynb                          # 基本操作
        demo_classification_of_iris_by_qnn.ipynb  # 鸢尾花二分类
        demo_qaoa_for_maxcut.ipynb                # QAOA 解决最大割问题
        demo_qnn_for_nlp.ipynb                    # QNN 用于自然语言处理
```

## 展望

- 由于 Mindspore 目前自动微分不支持 8 维度以上张量，因此开发的量子模拟器目前最多支持 8 个量子比特的线路。
- 目前机器训练数据不支持批并行处理，但可以利用 Mindspore 的灵活性扩展。

## 参考 GIT 项目

[1] [昇思MindSpore](https://gitee.com/mindspore/mindspore)
[2] [Minspore/mindquantum](https://gitee.com/mindspore/mindquantum)
