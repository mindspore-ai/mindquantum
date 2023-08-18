import numpy as np                                        # 导入numpy库并简写为np
from sklearn.model_selection import train_test_split                                                   # 导入train_test_split函数，用于对数据集进行划分
# pylint: disable=W0104
from mindquantum.core import Circuit                 # 导入Circuit模块，用于搭建量子线路
from mindquantum.core import UN                      # 导入UN模块
from mindquantum.core import H, X, RZ                # 导入量子门H, X, RZ
from mindquantum.algorithm import HardwareEfficientAnsatz                                           # 导入HardwareEfficientAnsatz
from mindquantum.core import RY                                                                     # 导入量子门RY
from mindquantum.core import QubitOperator                     # 导入QubitOperator模块，用于构造泡利算符
from mindquantum.core import Hamiltonian                       # 导入Hamiltonian模块，用于构建哈密顿量
import mindspore as ms                                                                         # 导入mindspore库并简写为ms
from mindquantum.framework import MQLayer                                                      # 导入MQLayer
from mindquantum.simulator import Simulator
from mindspore.nn import SoftmaxCrossEntropyWithLogits                         # 导入SoftmaxCrossEntropyWithLogits模块，用于定义损失函数
from mindspore.nn import Adam, Accuracy                                        # 导入Adam模块和Accuracy模块，分别用于定义优化参数，评估预测准确率
from mindspore import Model                                                    # 导入Model模块，用于建立模型
from mindspore.dataset import NumpySlicesDataset                               # 导入NumpySlicesDataset模块，用于创建模型可以识别的数据集
from mindspore.train.callback import Callback, LossMonitor                     # 导入Callback模块和LossMonitor模块，分别用于定义回调函数和监控损失
import os
from mindspore.train.callback import Callback, LossMonitor 
os.environ['OMP_NUM_THREADS'] = '2'
import matplotlib.pylab as plt
import mindspore.dataset as ds
from mindspore import ops, Tensor 

train_dataset = np.load('/home/user/mindquantum/mindquantum/hackathon01/src/train.npy',allow_pickle=True)[0]
x = np.array(train_dataset['train_x'][:])
y = np.array(train_dataset['train_y'][:])
x1=x.reshape(x.shape[0], -1)
#alpha = x1[:, :15] * x1[:, 1:]           # 每一个样本中，利用相邻两个特征值计算出一个参数，即每一个样本会多出15个参数（因为有16个特征值），并储存在alpha中
#x1 = np.append(x1, alpha, axis=1)       # 在axis=1的维度上，将alpha的数据值添加到X的特征值中
y1=y.astype(np.int32)
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.02, random_state=0, shuffle=True) # 将数据集划分为训练集和测试集

#encoder = Circuit()                                                          #初始化量子门
#encoder += UN(H, 16)                                                         # H门作用在每1位量子比特                                                       
#for i in range(16):                                                       # i = 0, 1, 2, 3...15
#        encoder += RZ(f'alpha{i}').on(i)                                         # RZ(alpha_i)门作用在第i位量子比特
#for j in range(15):                                                       # j = 0, 1, 2...14
#        encoder += X.on(j+1, j)                                                  #X门作用在第j+1位量子比特，受第j位量子比特控制
#        encoder += RZ(f'alpha{j+16}').on(j+1)                                     # RZ(alpha_{j+16})门作用在第0位量子比特
#        encoder += X.on(j+1, j)                                                  # X门作用在第j+1位量子比特，受第j位量子比特控制
        
#encoder = encoder.no_grad() 

encoder = Circuit()                                                          #初始化量子门
encoder += UN(H, 8)                                                         # H门作用在每1位量子比特                                                       
for i in range(8):                                                       # i = 0, 1, 2, 3...15
        encoder += RZ(f'alpha{i}').on(i)   
encoder += X.on(0, 7)
for j in range(7):                                   # j = 0, 1, 2
    encoder += X.on(j+1, j)  
for k in range(8): 
                        # X门作用在第j+1位量子比特，受第j位量子比特控制
    encoder += RZ(f'alpha{k+8}').on(k)
encoder += X.on(0, 7)
for l in range(7): 
            # RZ(alpha_{j+4})门作用在第0位量子比特
    encoder += X.on(l+1, l)                          # X门作用在第j+1位量子比特，受第j位量子比特控制                                          # RZ(alpha_i)门作用在第i位量子比特
    
encoder = encoder.no_grad() 
#circ += UN(X, [1, 3, 5, 7], [0, 2, 4, 6])
#circ += UN(X, [2, 4, 6], [1, 3, 5])
#encoder = add_prefix(circ, 'e1') + add_prefix(circ, 'e2')
#ansatz = add_prefix(circ, 'a1')
ansatz = HardwareEfficientAnsatz(8, single_rot_gate_seq=[RY], entangle_gate=X, depth=3).circuit  # 通过HardwareEfficientAnsatz搭建Ansatz
circuit = encoder + ansatz                                                                     # 完整的量子线路由Encoder和Ansatz组成
ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [6,7]]                                   # 分别对第14位和第15位量子比特执行泡利Z算符测量，且将系数都设为1，构建对应的哈密顿量
sim = Simulator('projectq', circuit.n_qubits)
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)                                                                                 # 设置生成随机数的种子
sim = Simulator('projectq', circuit.n_qubits)

grad_ops = sim.get_expectation_with_grad(
            ham,
            circuit,
            encoder_params_name=encoder.params_name,
            ansatz_params_name=ansatz.params_name,
            parallel_worker=5)

QuantumNet = MQLayer(grad_ops)

loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')            # 通过SoftmaxCrossEntropyWithLogits定义损失函数，sparse=True表示指定标签使用稀疏格式，reduction='mean'表示损失函数的降维方法为求平均值
opti = Adam(QuantumNet.trainable_params(), learning_rate=0.0001)                  # 0.0001-0.001通过Adam优化器优化Ansatz中的参数，需要优化的是Quantumnet中可训练的参数，学习率设为0.1
#opti = Adam(QuantumNet.trainable_params()) 
model = Model(QuantumNet, loss, opti, metrics={'Acc': Accuracy()})             # 建立模型：将MindQuantum构建的量子机器学习层和MindSpore的算子组合，构成一张更大的机器学习网络

train_loader = ds.NumpySlicesDataset({'features': X_train, 'labels': y_train}, shuffle=False).batch(10) # 通过NumpySlicesDataset创建训练样本的数据集，shuffle=False表示不打乱数据，batch(5)表示训练集每批次样本点有5个
test_loader = ds.NumpySlicesDataset({'features': X_test, 'labels': y_test}).batch(10)                   # 通过NumpySlicesDataset创建测试样本的数据集，batch(5)表示测试集每批次样本点有5个

#class StepAcc(Callback):                                                        # 定义一个关于每一步准确率的回调函数
#    def __init__(self, model, test_loader):
#        self.model = model
#        self.test_loader = test_loader
#        self.acc = []
#
#    def step_end(self, run_contex
#        self.acc.append(self.model.eval(self.test_loader, dataset_sink_mode=False)['Acc'])

monitor = LossMonitor(490)                                                       # 监控训练中的损失，每16步打印一次损失值

#acc = StepAcc(model, test_loader)                                               # 使用建立的模型和测试样本计算预测的准确率

model.train(15, train_loader, callbacks=monitor, dataset_sink_mode=False)# 将上述建立好的模型训练20次
        
#plt.plot(acc.acc)
#plt.title('Statistics of accuracy', fontsize=20)
#plt.xlabel('Steps', fontsize=20)
#plt.ylabel('Accuracy', fontsize=20)

predict = np.argmax(ops.Softmax()(model.predict(Tensor(X_test))),axis=1)  # 使用建立的模型和测试样本，得到测试样本预测的分类
correct = model.eval(test_loader, dataset_sink_mode=False)   
print("预测分类结果：", predict)                                              # 对于测试样本，打印预测分类结果
print("实际分类结果：", y_test)                                               # 对于测试样本，打印实际分类结果                # 计算测试样本应用训练好的模型的预测准确率
print(correct)

