# QCNN模型自验报告

## 1. 模型简介

该模型使用QCNN对多体波函数进行分类。

抽象到具体的分类问题，就是对波函数进行二分类。量子比特数目确定的情况下，编码线路是给定的。每一个样本输入是encoder中的参数，输出是-1或1，分别代表顺磁相和铁磁相。分类网络使用的是QCNN。

### 1.1. 网络模型结构简介

QCNN主要包含卷积层和池化层，如果把conv和pooling看成是一个模块的话，每个模块共包含21个参数。下面分别介绍。

### 1.1.1 Conv层

结构如图：

![image-20220715104914171](/readme.assets/image-20220715104914171.png)

1. 分别对qubits按顺序作用RX，RY，RZ门
2. 在qubits上按顺序作用ZZ，YY，XX门
3. 再分别对qubits按顺序作用RX，RY，RZ门

其中每个门的参数各不相同。有15个参数门，15个参数。

### 1.1.2 Pooling层

结构如图：

![image-20220715105336377](/readme.assets/image-20220715105336377.png)

1. 分别对qubits按顺序作用RX，RY，RZ门
2. 在两个qubits上作用CX门，确定哪一个qubit在pooling后继续前向传播
3. 在保留的qubits上按顺序作用RZ，RY，RX门，注意此时参数与前面作用在这个qubit上的互为相反数

其中有些参数门公用同一个参数。有9个参数门，6个参数。



### 1.2. 数据集

### 1.2.1 原始数据集

数据集使用的是tensorflow-quantum中内置的[tfi_chain](https://tensorflow.google.cn/quantum/api_docs/python/tfq/datasets/tfi_chain?hl=zh-cn)数据集。数据集中共包含81个数据，每条数据对本任务有用的数据为

1. gamma：取值范围[0.2, 1.8]。确定样本为顺磁相和铁磁相，在训练过程中被二值化为-1和1
2. params：是encoder中的参数。不同qubits数目对应的参数数目不同

以4-qubit系统为例，展示样本encoder的线路：

![image-20220715110434933](/readme.assets/image-20220715110434933.png)

1. 每个qubits作用H门
2. ZZ门按顺序两两作用在所有qubits上形成环，之后在每个qubits上作用RX门。注意此时的参数ZZ门都相同，RX门也都相同
3. 作用2中相同的门，此时参数采用新的参数

### 1.2.2 数据集增强

使用线性插值方法对样本中的gamma和params做了插值，增加样本个数。相邻的两个样本点中再插入11个点，这样样本数量增加到了961。



### 1.3. 代码提交地址

https://gitee.com/richybai/qcnn



## 2.   代码目录结构说明

```
QCNN
├── data			 	# 存储原始的tfi_chain数据
│   ├── 12qbsdata.npy	# 12qubits数据
│   ├── 4qbsdata.npy	#  4qubits数据
│   └── 8qbsdata.npy	#  8qubits数据
├── readme.md			# 说明文档
├── requirements.txt	# 代码依赖项
├── src					# 模型定义源码目录
│   ├── config.py		# 模型的配置参数
│   ├── data_gene.py	# 由原始tfi_chain数据线性插值生成数据的文件
│   └── QCNN.py			# QCNN模型代码
└── main.py				# 训练测试代码
```



## 3.   自验结果

### 3.1. 自验环境

- 硬件环境：win10 cpu
- 包版本：
  1. mindquantum==0.6.0
  2. mindspore==1.7.0
  3. numpy==1.23.0

### 3.2. 训练超参数

1. `batch_size = 4`
2. `epochs = 20`
3. `learning_rate = 0.001`
4. `loss = MSELoss()`
5. `optimizer = Adam()`
6. 并行度未设置
7. 训练测试比4：1。

### 3.3. 训练

在训练前，需要使用`data_gene.py`线性插值生成数据。

`cd`到`src`目录下直接**`python data_gene.py`**即可。

#### 3.3.1. 启动训练脚本

每次训练前，需要到`config.py`文件夹下修改参数，需要修改的有：

	1. `n_qubits`：指定代码数据的量子比特数，可以为 4 8 12
 	2. `use_additional_data`：指定是否使用线性插值后的数据，可以为 `True`or`False`
 	3. `random_seed`：指定随机数种子。

运行代码后会打印如下信息：

	1. config of training
 	2. summary of encoder
 	3. data information
 	4. summary of ansatz(QCNN)
 	5. training and testing information

会在`result_{n_qubits}`文件夹下保存若干模型参数以及loss和accuarcy数据。

开始训练时，`cd`到根目录，使用命令**`python main.py`**

下面以（4，True，33）为例，展示代码输出：

1. ```config of training
   config of training------------
   qubits numbers : 4
   additional data: True
   random seed    : 33
   
   batch size     : 4
   repeat sise    : 1
   learning rate  : 0.001
   epochs         : 20
   ```

2. ```
   summary of encoder:
   ========================Circuit Summary========================
   |Total number of gates  : 20.                                 |
   |Parameter gates        : 16.                                 |
   |with 4 parameters are  : theta_0, theta_2, theta_1, theta_3. |
   |Number qubit of circuit: 4                                   |
   ===============================================================
   parameters of encoder:  ['theta_0', 'theta_2', 'theta_1', 'theta_3']
   ```

3. ```
   total data number: 960
   params in one sample: 4
   
   train sample: 768, x_train shape: (768, 4), y_train shape: (768, 1)
   test  sample: 768, x_test  shape: (192, 4), y_test  shape: (192, 1)
   ```

4. ```
   summary of ansatz(QCNN):
   ==========================Circuit Summary===========================
   |Total number of gates  : 75.      								   |
   |Parameter gates        : 72.                                      |
   |with 63 parameters are : l1c1_px0, l1c1_px1, l1c1_py0, ...		   |
   |Number qubit of circuit: 4                                        |
   ====================================================================
   ```

5. ```
   begin training:  --------------
   epoch:  1, training loss:   1.0602, accuracy: 0.849, testing loss: 0.614144, accuracy: 0.8854
   epoch:  2, training loss: 0.490626, accuracy: 0.9193, testing loss: 0.400552, accuracy: 0.9583
   epoch:  3, training loss: 0.420938, accuracy: 0.9284, testing loss: 0.387441, accuracy: 0.9583
   epoch:  4, training loss: 0.415707, accuracy: 0.931, testing loss: 0.385453, accuracy: 0.9583
   epoch:  5, training loss:  0.41375, accuracy: 0.9323, testing loss: 0.383884, accuracy: 0.9583
   epoch:  6, training loss:  0.41196, accuracy: 0.9349, testing loss: 0.382288, accuracy: 0.9583
   epoch:  7, training loss: 0.410165, accuracy: 0.9362, testing loss: 0.380624, accuracy: 0.9583
   epoch:  8, training loss: 0.408331, accuracy: 0.9375, testing loss: 0.378879, accuracy: 0.9583
   epoch:  9, training loss: 0.406446, accuracy: 0.9388, testing loss: 0.377072, accuracy: 0.9583
   epoch: 10, training loss:  0.40454, accuracy: 0.9414, testing loss: 0.375264, accuracy: 0.9583
   epoch: 11, training loss: 0.402684, accuracy: 0.9427, testing loss: 0.373557, accuracy: 0.9583
   epoch: 12, training loss: 0.400981, accuracy: 0.9427, testing loss: 0.372066, accuracy: 0.9583
   epoch: 13, training loss:  0.39952, accuracy: 0.9427, testing loss: 0.370861, accuracy: 0.9635
   epoch: 14, training loss: 0.398338, accuracy: 0.944, testing loss: 0.369944, accuracy: 0.9635
   epoch: 15, training loss: 0.397416, accuracy: 0.944, testing loss:  0.36927, accuracy: 0.9635
   epoch: 16, training loss: 0.396701, accuracy: 0.9453, testing loss: 0.368772, accuracy: 0.9635
   epoch: 17, training loss: 0.396134, accuracy: 0.9466, testing loss: 0.368387, accuracy: 0.9635
   epoch: 18, training loss: 0.395664, accuracy: 0.9466, testing loss: 0.368072, accuracy: 0.9635
   epoch: 19, training loss: 0.395256, accuracy: 0.9466, testing loss: 0.367796, accuracy: 0.9635
   epoch: 20, training loss: 0.394889, accuracy: 0.9466, testing loss: 0.367544, accuracy: 0.9688
   ```

#### 3.3.2. 训练精度结果

| additional data | qubits | final acc | best acc |
| :-------------: | :----: | :-------: | :------: |
|      False      |   4    |    1.0    |   1.0    |
|      False      |   8    |    1.0    |   1.0    |
|      False      |   12   |  0.9375   |   1.0    |
|      True       |   4    |  0.9688   |  0.9688  |
|      True       |   8    |  0.9688   |   1.0    |
|      True       |   12   |   0.974   |  0.9948  |

## 4.   参考资料

### 4.1. 参考论文

Wrobel N, Baul A, Moreno J, et al. An Application of Quantum Machine Learning on Quantum Correlated Systems: Quantum Convolutional Neural Network as a Classifier for Many-Body Wavefunctions from the Quantum Variational Eigensolver[J]. arXiv preprint arXiv:2111.05076, 2021.
