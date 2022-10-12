<TOC>

# QCNN

量子卷积神经网络Quantum Convolutional Neural Network(QCNN)。

[复现论文](https://arxiv.org/abs/2111.05076)："An Application of Quantum Machine Learning on Quantum Correlated Systems: Quantum Convolutional Neural Network as a Classifier for Many-Body Wavefunctions from the Quantum Variational Eigensolver"

## 数据集

- [TFI_chain](https://storage.googleapis.com/download.tensorflow.org/data/quantum/spin_systems/TFI_chain.zip)是TFQ提供的一个广泛用于模型测试的数据集，称为一维横向场Ising(TFIM)模型量子数据集。此数据集包含边界条件为closed且N = [4, 8, 12, 16]的四组数据，每组数据包含81个数据点。[参考文档](https://tensorflow.google.cn/quantum/api_docs/python/tfq/datasets/tfi_chain)

- 数据集大小：392 MB

- 数据格式：NPY文件及TXT文件
    - 注：数据在`dataset.py`中处理。

## 环境要求

- 硬件（CPU）
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
    - [MindQuantum](https://gitee.com/mindspore/mindquantum)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindQuantum教程](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/index.html)

- 第三方库

```bash
pip install numpy
pip install sklearn
```

## 快速入门

1. 下载TFI_chain数据集。

2. 执行训练程序。数据集准备完成后，按照如下步骤开始训练：

    ```text
   # 训练
    python train.py
    ```

4. 执行评估程序。训练结束后，按照如下步骤启动评估：

   ```bash
   # 评估
   python eval.py
   ```

## 脚本说明

### 脚本和样例代码

```shell
qcnn
├── README.md                           # 模型说明文档
├── requirements.txt                    # 依赖说明文件
├── eval.py                             # 精度验证脚本
├── src                                 # 模型定义源码目录
│   ├── QCNNet.py                       # 模型结构定义
│   ├── ansatz_qcnn.py                  # 量子拟设定义
│   ├── loss.py                         # 损失函数定义
│   └── dataset.py                      # 数据集处理定义
└── train.py                            # 训练脚本
```

## 训练过程

### 准备

- 下载数据集并置于`qcnn`目录下。

### 训练

- 运行`train.py`开始训练。将会对N = 4, 8, 12情况数据集进行训练。

- 训练过程中每训练5步将会打印一次损失值。

### 训练结果

- 训练checkpoint将被保存在`model_N{N}.ckpt`中。

## 评估过程

### 评估

- 运行`eval.py`进行评估。将会对N = 4, 8, 12情况数据集进行评估。

## 性能

### 训练性能

| 参数                  | QCNN(N = 4)                     | QCNN(N = 8)                     | QCNN(N = 12)                    |
| -------------------   | ------------------------------- | ------------------------------- | ------------------------------- |
| 模型版本              | V1                              | V1                              | V1                              |
| 资源                  | Ubuntu 18.04.5 LTS              | Ubuntu 18.04.5 LTS              | Ubuntu 18.04.5 LTS              |
| 上传日期              | 2022-06-28                      | 2022-06-28                      | 2022-06-28                      |
| MindSpore版本         | 1.6.0                           | 1.6.0                           | 1.6.0                           |
| MindQuantum版本       | 0.5.0                           | 0.5.0                           | 0.5.0                           |
| 数据集                | TFI_chain                       | TFI_chain                       | TFI_chain                       |
| 训练参数              | epoch=5，batch_size=60          | epoch=4，batch_size=30          | epoch=1，batch_size=30          |
| 优化器                | Adam                            | Adam                            | Adam                            |
| 损失函数              | SoftMarginLoss                  | SoftMarginLoss                  | SoftMarginLoss                  |
| 参数（M）             | 63                              | 147                             | 231                             |
| 脚本                  | [link](https://gitee.com/NoEvaa/zero/tree/master/NPark/NGPark/mindspore/qcnn/)                      |

### 评估性能

| 参数                  | QCNN(N = 4)                     | QCNN(N = 8)                     | QCNN(N = 12)                    |
| --------------------- | ------------------------------- | ------------------------------- | ------------------------------- |
| 模型版本              | V1                              | V1                              | V1                              |
| 资源                  | Ubuntu 18.04.5 LTS              | Ubuntu 18.04.5 LTS              | Ubuntu 18.04.5 LTS              |
| 上传日期              | 2022-06-28                      | 2022-06-28                      | 2022-06-28                      |
| MindSpore版本         | 1.6.0                           | 1.6.0                           | 1.6.0                           |
| MindQuantum版本       | 0.5.0                           | 0.5.0                           | 0.5.0                           |
| 数据集                | TFI_chain                       | TFI_chain                       | TFI_chain                       |
| 批次大小              | 81                              | 81                              | 81                              |
| 输出                  | Acc                             | Acc                             | Acc                             |
| 精确度                | 0.9876543209876543              | 0.9876543209876543              | 0.9876543209876543              |
| 推理模型              | 287B（.ckpt文件）               | 624B（.ckpt文件）               | 960B（.ckpt文件）               |

## 随机情况说明

使用train.py中的随机种子进行权重初始化。

在train.py中随机分割测试集和训练集。

### 贡献者

* [NoEvaa](https://gitee.com/NoEvaa)

## ModelZoo 主页

请浏览官方[主页](https://gitee.com/mindspore/models)。
