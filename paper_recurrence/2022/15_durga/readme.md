# 利用MindQuantum实现量子卷积神经网络求解量子多体问题

## 项目介绍

赛题十五：利用MindQuantum实现量子卷积神经网络求解量子多体问题

论文：
[An Application of Quantum Machine Learning on Quantum Correlated Systems: Quantum Convolutional Neural Network as a Classifier for Many-Body Wavefunctions from the Quantum Variational Eigensolver](https://arxiv.org/abs/2111.05076)

复现要求：
基于MindQuantum实现图4中的量子卷积神经网络，并在N=4、8、12的情况下实现对顺磁性和铁磁性的分类，精度要求达到90%以上

## 数据集
数据可以通过tersorflow quantum获得，有比特数N=4,8,12几组数据，每组数据有81个数据点，对应序参数g在[0.2, 1.8]之间，以0.02的间隔采样。量子卷积神经网络需要对加载在量子比特上的信息实现卷积和池化操作，最终根据量子态实现顺磁性和铁磁性的分类预测。量子卷积和池化电路参见main.ipynb中的示例。

数据可以通过以下代码获取，或者采用本仓库的tfi_chain文件夹。
```python
import tensorflow_quantum as tfq
tfq.datasets.tfi_chain(qubits, boundary_condition='closed', data_dir=None)
```

## QCNN的主要结构及训练结果

`QCNN`的初始化会先完成 据加载和预处理 以及量子电路生成。初始化参数包括：`qubits`：系统比特数，`learning_rate`：学习率，`epoch`：训练次数，`batch`：训练数据块大小，`opt`：是否优化电路。详细参见`QCNN`的[API文档](./doc/build/html/qcnn.html)
```python
qcnn = QCNN(qubits=4, learning_rate=0.001, epoch=8, batch=8, opt = False)
```

### 数据加载和预处理

`QCNN.load_data()`可以加载原始数据，由于原始数据点数不够多，参考文献的做法，我们也进行了插值来扩充数据，`QCNN.data_ext()`函数可以实现。我们将数据扩充了10倍，即约800个数据，其中640个作为训练集，160个作为验证集。


### 量子电路生成

量子电路主要包括两部分：encoder 和 ansatz，encoder用于将数据进行编码加载到量子电路上，不可以训练。ansatz部分是量子卷积神经网络的核心部分，包括卷积电路和池化电路。`QCNN.gen_encoder()`用于实现encoder电路，`QCNN.gen_qcnn_ansatz()`用于实现ansatz电路，其中卷积电路由`QCNN.q_convolution`实现，池化电路由`QCNN.q_pooling`实现

#### 4 qubits encoder电路

```python
encoder = qcnn.gen_encoder()
print(encoder)

            q0: ──H────ZZ(alpha_0_0)────RX(alpha_0_1)──────────────────────────────────────ZZ(alpha_0_0)────ZZ(alpha_1_0)────RX(alpha_1_1)──────────────────────────────────────ZZ(alpha_1_0)───────────────────
                             │                                                                   │                │                                                                   │
            q1: ──H────ZZ(alpha_0_0)────ZZ(alpha_0_0)────RX(alpha_0_1)───────────────────────────┼──────────ZZ(alpha_1_0)────ZZ(alpha_1_0)────RX(alpha_1_1)───────────────────────────┼─────────────────────────
                                              │                                                  │                                 │                                                  │
            q2: ──H─────────────────────ZZ(alpha_0_0)────ZZ(alpha_0_0)────RX(alpha_0_1)──────────┼───────────────────────────ZZ(alpha_1_0)────ZZ(alpha_1_0)────RX(alpha_1_1)──────────┼─────────────────────────
                                                               │                                 │                                                  │                                 │
            q3: ──H──────────────────────────────────────ZZ(alpha_0_0)─────────────────────ZZ(alpha_0_0)────RX(alpha_0_1)─────────────────────ZZ(alpha_1_0)─────────────────────ZZ(alpha_1_0)────RX(alpha_1_1)──

```

#### 2 qubits 卷积电路
采用文献中的卷积方案。
```python
conv = qcnn.q_convolution('0',[0,1])
print(conv)

            q0: ──RX(cov_0_0)────RY(cov_0_1)────RZ(cov_0_2)────XX(cov_0_6)────YY(cov_0_7)────ZZ(cov_0_8)────RX(cov_0_9)─────RY(cov_0_10)────RZ(cov_0_11)──
                                                                    │              │              │
            q1: ──RX(cov_0_3)────RY(cov_0_4)────RZ(cov_0_5)────XX(cov_0_6)────YY(cov_0_7)────ZZ(cov_0_8)────RX(cov_0_11)────RY(cov_0_12)────RZ(cov_0_13)──
```

#### 2 qubits 池化电路
采用文献中的池化方案。
```python
pool = qcnn.q_pooling('0',[0,1])
print(pool)
            q0: ──RX(p_0_0)────RY(p_0_1)────RZ(p_0_2)────●────────────────────────────────────────────
                                                         │
            q1: ──RX(p_0_3)────RY(p_0_4)────RZ(p_0_5)────X────RZ(-p_0_5)────RY(-p_0_4)────RX(-p_0_3)──
```

任意偶数比特的卷积ansatz 可以通过2 qubits 卷积电路和2 qubits 池化电路来构成。

### 电路优化

考虑到卷积电路和池化电路相连的时候，有大量的`RX`，`RY`，`RZ`门会重复，因此针对池化电路可以将`RX`，`RY`，`RZ`门去掉（最后一个池化电路输出除外）。

优化后ansatz电路参数对比如下：


| Qubits     | Total number of gates | Total number of gates (optimized)  | Parameter gates  | Parameter gates  (optimized) |
| :---:       |    :----:   |          :---: |  :---: |                :---: | 
| 4      | 75      | 51   | 72 | 48|
| 8   | 175     | 115   | 168 | 108|
| 12   | 275     | 179   | 264 | 168|

优化后的参数门减少了33%~36%。QCNN只需在初始化时将opt参数设置为True，即可产生优化后的电路。

### 量子卷积神经网络训练及结果

| Qubits     | 4 | 8  | 12  | 
| :---:       |    :----:   |          :---: |  :---: | 
| Accuracy      | 1      | 1   | 0.99| 
| Accuracy(optimized)   | 1     | 1   | 0.99 |
| training time（s）   | 9690     | 25370   | 35909 |
| training time（s, optimized）   | 6830     | 15668   | 25365 |

优化后的训练时间减少了30~38%左右，而分类精度保持与原始方案一样。对于N=4，8的情况时，能够100%正确分类，对于N=12时，分类准确度为99% (最好情况)。

## 创新点总结
- 实现了一个通用化的量子卷积神经网络的类`QCNN`，可以产生不同量子比特位数的量子神经网络，并可以进行训练；量子电路的产生使用了分块模式设计，主要包括`gen_encoder`，`q_convolution`，`q_pooling`三个模块。在此基础上实现了比原文章更好的分类精度。
- 针对量子电路进行了优化，特别是简化了`q_pooling`使用的量子门，使得参数门减少约33%~36%，在没有降低分类精度的情况下，提升了性能，训练时间减少了30%~38%。

## 相关信息

### 代码目录结构说明
```bash
.                    <--根目录
├── main.ipynb       <--项目代码的演示说明，包含结果验证
├── readme.md        <--说明文档
├── src              <--源代码目录
│   ├── qcnn.py      <--实现QCNN类和量子神经网络训练的py文件
├── tfi_chain        <--训练所需要的数据目录，也可以从tensorflow quantum上下载
├── res              <--训练结果保存所在目录
├── doc              <--API文档目录
```
### 测试版本
```
mindquantum==0.6.0rc1
mindspore==1.8.1
numpy==1.22.3
scikit_learn==1.1.2
```

### 作者
- rongge ketyxu@126.com
- levitan levitan@msn.cn

### API 文档
QCNN类的[API](./doc/build/html/qcnn.html)

### 自验报告参见本文件夹内的 [main.ipynb](main.ipynb)