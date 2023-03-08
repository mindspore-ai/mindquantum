## IMAGE COMPRESSION AND CLASSIFICATION USING QUBITS AND QUANTUM DEEP LEARNING

本篇文章基于FRQI编码方法提出了一种新的编码方式，并应用于图像的二分类中。复现内容主要包含如下：
1. 数据的预处理
2. 编码线路的实现
3. 网络结构
4. 训练模型
5. 结果展示

### 1. 数据的预处理
文中使用MNIST中的3和6，下采样到8x8和16x16两种分辨率，最后二值化。最终剩下大约12000个训练样本，2000个测试样本（论文中是1100个测试样本）。
之后，把训练测试数据混合，进行10-fold交叉验证。

### 2. 编码线路的实现

文中使用了两种编码方式，第一种是FRQI，第二种是由FQRI改进的压缩方式。
1. FRQI使用前n个qubits编码位置，最后一个编码颜色。
2. 在FRQI的基础上，把最后两个qubits压缩到编码颜色的qubit上。

### 3. 网络结构

文中给出了两种量子神经网络结构，并实现了一个简单的经典网络进行对比：

1. Color-Readout-Alternating-Double-Layer (CRADL)
2. Color-Readout-Alternating-Mixed-Layer (CRAML)
3. 经典ANN, 结构：64-1-1，256-1-1

其中，CRADL用于本篇文章的QNN。

### 4. 训练模型
论文中共做了五组实验，分别是：
1. qnn1, 8x8, 6 qubits no compression, 72 params 12 层
2. qnn2, 8x8, 4 qubits with compression, 64 params 16 层 
3. 8x8, classical 64-1-1 with bias 67 params(如果按照论文使用65个参数，没有bias分类结果很差)

4. 16x16, 6 qubits with compression, 252 params 42 层
5. 16x16, classical 256-1-1 with bias 259 params(同上，原文为257个参数)

训练过程在train.py中，采用了"mqvector"加速。运行python train.py 得到作图数据。

在train.py的line 24 25 选择 loss，line 189修改保存文件名。
1. 作图的数据被存储在"validation.npy"中, 此时使用的hinge loss。
2. 作图的数据被存储在"validation_mse.npy"中, 此时使用的mse loss。

### 5. 结果展示

复现论文中Figure 6, 图中展示了10折交叉验证在测试集上的准确率及其标准差。本次复现实现了两种loss的结果。

1. 使用hinge loss时，和论文相似度较差。
2. 使用mse loss时，和论文结果相似度更高。（在main.ipynb的最后）