# 黑客松赛题

量子神经网络具有表达能力强、搜索空间大等特点，在特定算法上能够超越经典的神经网络。在本题中，选手探索利用量子经典混合的神经网络来完成手写字体识别任务。

## 赛题要求：

- 利用MindQuantum和MindSpore来完成量子经典混合神经网络的搭建、训练和推理
- 整个混合神经网络中，待训练参数只能在量子神经网络里，且只能是构成量子神经网络的参数化量子线路中的待训练参数，经典框架只能提供激活函数、优化器等不含可训练参数的模块，否则视为无效。
- 选手可以在HiQ量子计算云平台上编程，也可以根据代码仓readme文档中的指令，通过pip安装MindQuantum在本地进行开发。

## 数据集：

- 数据集为经过压缩后的mnist手写体中的0和1，每张图片已经从原始的28x28像素压缩为4x4像素，为项目中的`train.npy`
- 提供给选手的样本为5000个
- 测试集样本为800个（不对外公布，项目中的`test.npy`仅供参考，非真实测试数依据）

##使用的主要方法：

1.分别使用QNN与QCNN建模进行测试
由于输入的图片大小为：4*4，经过观察发现，最后一位全部是0，故准备在量子线路中最初只传入15
个参数。
QNN：先搭建编码线路：使用IQP编码，使用8量子比特，传入15个参数进行编码。
再搭建Ansatz:使用HardwareEfiicientAnsatz:mindquantum.ansatz.HardwareEfficientAnsatz(n_qubits, single_rot_gate_seq, entangle_gate=X, entangle_mapping="linear”, depth=1)，
括号中的n_qubits表示ansatz需要作用的量子比特总数，
single_rot_gate_seq表示一开始每一位量子比特执行的参数门，
同时后面需要执行的参数门也固定了，只是参数不同，
entangle_gate=X表示执行的纠缠门为X，
entangle_mapping=”linear”表示纠缠门将作用于每对相邻量子比特，
depth表示黑色虚线框内的量子门需要重复的次数。
QCNN：先搭建编码线路：使用IQP编码，使用8量子比特，传入15个参数进行编码。
再搭建Ansatz:即搭建量子卷积层与量子池化层。
量子卷积层部分：
def qconv(i,j,params):
    qconv_circuit=Circuit()
    #i，j表示作用位，params表示参数的个数
    X=[]
    for p in range(params):
        X.append(p)
    qconv_circuit+=RY({f'q_{X[0]}':1}).on(i)
    qconv_circuit+=RY({f'q_{X[1]}':1}).on(j)
    qconv_circuit+=RZ({f'q_{X[2]}':1}).on(i,j)
    qconv_circuit+=RY({f'q_{X[3]}':1}).on(i)
    qconv_circuit+=RY({f'q_{X[4]}':1}).on(j)
    qconv_circuit+=RZ({f'q_{X[5]}':1}).on(j,i)
    return qconv_circuit
量子池化层部分:
def qpool(i,j):
    qpool_circuit=Circuit()
    qpool_circuit+=RZ({f'p_{i}':1}).on(j,i)
    qpool_circuit+=X.on(i)
    qpool_circuit+=RX({f'p_{j}':1}).on(j,i)
    qpool_circuit+=X.on(i)
    return qpool_circuit
量子池化层线路中线路的作用功能相当于trace掉第i个量子比特位，与经典神经网络的maxpooling等
将2*2变成1*1差不多
搭建线路：卷积与池化交替
两层卷积层：8个qocnv，1层池化层：4个qpool,两层卷积层：4个qconv,1层池化层:2个qpool,
再加1个qconv
2.调参：
QNN：在小样本数据集上：取440个数据集，对第三位和第四位量子比特执行测量，
训练8个批次，就可以达到91%以上精确度。且最终的测试集得分为0.9338。
将训练好的模型保存到model.ckpt文件中。
QCNN：对2000个样本进行训练：对第三位和第七位量子比特执行测量，
训练5个批次，就可以达到91%以上精确度。且最终的测试集得分为0.9325。
将训练好的模型保存到model.ckpt文件中。

## 项目代码结构

```bash
.                    <--根目录，提交时将整个项目打包成hackathon01_队长姓名_队长联系电话.zip
├── eval.py          <--举办方用来测试选手的模型的脚本，无需修改
├── readme.md        <--说明文档
├── src              <--源代码目录，选手的所有开发源代码都应该放入该文件夹类
│   ├── hybrid.py    <--模型父类，选手的模型需继承自该类，无需修改
│   ├── main.py      <--选手的模型调用入口，该文件为参考代码，无有意义训练结果
│   ├── model.ckpt   <--整个模型训练好的参数
│   └── train.npy    <--训练集
└── test.npy         <--测试集，仅供参考，数据内容无意义
```

##项目思考与未来改进方向：
1.编码方式修改：不仅仅使用IQP编码，还可以使用Amplitude encoding、qubit encoding、
dense qubit encoding、Hybrid encoding等。
2.Ansatz修改：不仅仅使用HardwareEfiicientAnsatz，量子卷积层还可以使用其他线路。
3.QCNN可以在一定程度上有效解决Barren Plateau现象。

##参考文献

[1] Hur T, Kim L, Park D K. Quantum convolutional neural network for classical data classification[J]. Quantum Machine Intelligence, 2022, 4(1): 1-18.