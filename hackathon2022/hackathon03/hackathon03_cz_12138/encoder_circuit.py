from mindquantum import *
from mindquantum.algorithm import HardwareEfficientAnsatz
from mindquantum.core import RY   
from mindquantum.core import QubitOperator                    
from mindquantum.core import Hamiltonian
import mindspore as ms                                                                         
from mindquantum.framework import MQLayer                                                    
from mindquantum.simulator import Simulator
from mindspore.nn import SoftmaxCrossEntropyWithLogits                       
from mindspore.nn import Adam, Accuracy                                      
from mindspore import Model                                                   
from mindspore.dataset import NumpySlicesDataset                               

#from mindspore import Callback, LossMonitor 
import os
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import datasets     
from mindspore.train.callback import Callback, LossMonitor                        


os.environ['OMP_NUM_THREADS'] = '2' 
loadData_train_x = np.load('train_x.npy',allow_pickle=True)
loadData_train_y = np.load('train_y.npy',allow_pickle=True)
loadData_test_x = np.load('test_x.npy',allow_pickle=True)
S = {}
S['data'] = loadData_train_x
S['target'] = loadData_train_y

x = S['data'][:800, :]
y = S['target'][:800, :]
for i in range(len(x)):
    for j in range(len(x[0])):
        x[i][j] = np.abs(x[i][j])
for i in range(len(y)):
    for j in range(len(y[0])):
        y[i][j] = np.abs(y[i][j])
        y[i][j] = np.abs(y[i][j])
x = x[:800, :].astype(np.float64) 
y = y[:800, :].astype(np.float64) 

def generate_encoder():
    n_qubits = 3

    enc_layer = Circuit()
    for i in range(n_qubits):
        enc_layer += U3(f'a{i}', f'b{i}', f'c{i}', i)

    coupling_layer = UN(X, [1, 2, 0], [0, 1, 2])

    encoder = Circuit()
    for i in range(2):
        encoder += add_prefix(enc_layer, f'l{i}')
        encoder += coupling_layer
    encoder = encoder.no_grad()

    return encoder

def generate_Ansatz():
    ansatz = HardwareEfficientAnsatz(3, single_rot_gate_seq=[RY], entangle_gate=X, depth=3).circuit
    return ansatz


encoder = generate_encoder()

ansatz = generate_Ansatz()

circuit = encoder + ansatz
hams = [Hamiltonian(QubitOperator(f'Z{i} Z{i}')) for i in [0,1,2]]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)                                                                                 # 设置生成随机数的种子
sim = Simulator('projectq', circuit.n_qubits)
grad_ops = sim.get_expectation_with_grad(hams,
                                         circuit,
                                         None,
                                         None,
                                         encoder.params_name,
                                         ansatz.params_name,
                                         parallel_worker=5)
QuantumNet = MQLayer(grad_ops)
ms.set_seed(1)                                     # 设置生成随机数的种子
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')            # 通过SoftmaxCrossEntropyWithLogits定义损失函数，sparse=True表示指定标签使用稀疏格式，reduction='mean'表示损失函数的降维方法为求平均值
opti = Adam(QuantumNet.trainable_params(), learning_rate=0.1)                  # 通过Adam优化器优化Ansatz中的参数，需要优化的是Quantumnet中可训练的参数，学习率设为0.1

model = Model(QuantumNet, loss, opti)             # 建立模型：将MindQuantum构建的量子机器学习层和MindSpore的算子组合，构成一张更大的机器学习网络

train_loader = NumpySlicesDataset({'features': X_train, 'labels': y_train}, shuffle=False).batch(5) # 通过NumpySlicesDataset创建训练样本的数据集，shuffle=False表示不打乱数据，batch(5)表示训练集每批次样本点有5个
test_loader = NumpySlicesDataset({'features': X_test, 'labels': y_test}).batch(5)                   # 通过NumpySlicesDataset创建测试样本的数据集，batch(5)表示测试集每批次样本点有5个



monitor = LossMonitor(16)                                                       # 监控训练中的损失，每16步打印一次损失值
                                       # 使用建立的模型和测试样本计算预测的准确率

model.train(20, train_loader, callbacks=[monitor], dataset_sink_mode=False)# 将上述建立好的模型训练20次