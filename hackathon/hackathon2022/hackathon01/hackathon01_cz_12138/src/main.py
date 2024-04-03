import os
from mindspore.train.callback import Callback, LossMonitor

os.environ['OMP_NUM_THREADS'] = '2'
from hybrid import HybridModel
from hybrid import project_path
import numpy as np
import mindspore as ms
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model
#from mindspore.train.callback import LossMonitor
from mindquantum import *
from mindspore.nn import Adam, Accuracy  # 导入Adam模块和Accuracy模块，分别用于定义优化参数，评估预测准确率

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
from mindquantum.algorithm import HardwareEfficientAnsatz
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import mindspore.dataset as ds
from mindspore import ops, Tensor


class Main(HybridModel):
    def __init__(self):
        super().__init__()
        self.dataset = self.build_dataset(self.origin_x, self.origin_y,
                                          25)  #加载xunlian数据
        #self.train_set_x_orig = np.array(self.origin_x) # 从训练数据中提取出图片的特征数据
        #self.train_set_y_orig = np.array(self.origin_y) # 从训练数据中提取出图片的标签数据
        #X=self.train_set_x_orig.reshape(self.origin_x.shape[0], -1)
        #alpha = X[:, :15] * X[:, 1:]
        # X = np.append(X, alpha, axis=1)
        #self.train_x=X.reshape(X.shape[0], -1).T
        #self.train_y= self.train_set_y_orig
        #self.dataset_test = self.build_dataset_test(self.origin_x, self.origin_y, 10)
        self.qnet = MQLayer(self.build_grad_ops())  #搭建量子神经网络
        self.model = self.build_model()  #搭建经典神经网络
        self.checkpoint_name = os.path.join(project_path, "model.ckpt")

    def build_dataset(self, x, y, batch=None):

        #train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
        X = x.reshape(x.shape[0], -1)
        #alpha = X[:, :15] * X[:, 1:]
        #X = np.append(X, alpha, axis=1)       # 在axis=1的维度上，将alpha的数据值添加到X的特征值中
        Y = y.astype(np.int32)
        #  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True) # 将数据集划分为训练集和测试集
        train = ds.NumpySlicesDataset({"image": X, "label": Y}, shuffle=False)
        #  train = ds.NumpySlicesDataset({'features': X_train, 'labels': y_train.astype(np.int32)},shuffle=False)
        #  test = ds.NumpySlicesDataset({'features': X_test, 'labels': y_test})

        if batch is not None:
            train = train.batch(batch)
        return train

    #def build_dataset_test(self, x, y, batch=None):

    #train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
    ##    X=x.reshape(x.shape[0], -1)
    #   alpha = X[:, :15] * X[:, 1:]
    #    X = np.append(X, alpha, axis=1)       # 在axis=1的维度上，将alpha的数据值添加到X的特征值中
    #    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True) # 将数据集划分为训练集和测试集
    #    train = ds.NumpySlicesDataset(
    #       {
    #
    #            "image": X.reshape((X.shape[0], -1)),
    #           "label": y.astype(np.int32)
    #       },
    #       shuffle=False)
    #   train = ds.NumpySlicesDataset({'features': X_train, 'labels': y_train})
    #   test = ds.NumpySlicesDataset({'features': X_test, 'labels': y_test})
    #
    #   if batch is not None:
    #       test = test.batch(batch)
    #   return test

    def build_grad_ops(self):

        encoder = Circuit()  #初始化量子门
        encoder += UN(H, 8)  # H门作用在每1位量子比特
        for i in range(8):  # i = 0, 1, 2, 3...15
            encoder += RZ(f'alpha{i}').on(i)
        encoder += X.on(0, 7)
        for j in range(7):  # j = 0, 1, 2
            encoder += X.on(j + 1, j)
        for k in range(8):
            # X门作用在第j+1位量子比特，受第j位量子比特控制
            encoder += RZ(f'alpha{k+8}').on(k)
        encoder += X.on(0, 7)
        for l in range(7):
            # RZ(alpha_{j+4})门作用在第0位量子比特
            encoder += X.on(
                l + 1, l
            )  # X门作用在第j+1位量子比特，受第j位量子比特控制                                          # RZ(alpha_i)门作用在第i位量子比特

        encoder = encoder.no_grad()

        #circ += UN(X, [1, 3, 5, 7], [0, 2, 4, 6])
        #circ += UN(X, [2, 4, 6], [1, 3, 5])
        #encoder = add_prefix(circ, 'e1') + add_prefix(circ, 'e2')
        #ansatz = add_prefix(circ, 'a1')
        ansatz = HardwareEfficientAnsatz(
            8, single_rot_gate_seq=[RY], entangle_gate=X,
            depth=12).circuit  # 通过HardwareEfficientAnsatz搭建Ansatz
        circuit = encoder.as_encoder() + ansatz  # 完整的量子线路由Encoder和Ansatz组成
        ham = [Hamiltonian(QubitOperator(f'Z{i}'))
               for i in [6, 7]]  # 分别对第14位和第15位量子比特执行泡利Z算符测量，且将系数都设为1，构建对应的哈密顿量
        sim = Simulator('mqvector', circuit.n_qubits)
        grad_ops = sim.get_expectation_with_grad(ham,
                                                 circuit,
                                                 parallel_worker=5)
        return grad_ops

    def build_model(self):
        #损失函数
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(
            sparse=True
        )  # 通过SoftmaxCrossEntropyWithLogits定义损失函数，sparse=True表示指定标签使用稀疏格式，reduction='mean'表示损失函数的降维方法为求平均值

        #adam下降
        self.opti = ms.nn.Adam(self.qnet.trainable_params(
        ))  # 通过Adam优化器优化Ansatz中的参数，需要优化的是Quantumnet中可训练的参数，学习率设为0.1
        self.metrics = {'Acc': Accuracy()}
        self.model = Model(self.qnet, self.loss, self.opti, self.metrics)
        return self.model

    def train(self):

        self.model.train(
            10, self.dataset,
            callbacks=LossMonitor(200))  # 监控训练中的损失，每64步打印一次损失值,10个人epoch

    # 对训练数据集进行预测

    #pred_train = self.predict(self.train_x)
    # print("预测准确率是: "  + str(np.sum((pred_train == self.train_y) / self.train_x.shape[1])))

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name)
        print("saved")

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        predict = np.argmax(ops.Softmax()(self.model.predict(
            ms.Tensor(test_x))),
                            axis=1)
        predict = predict.flatten() > 0
        return predict


# def test(self):
#    origin_test_x =np.array(self.dataset_test['features'][:])
#    origin_test_y =np.array(self.dataset_test['labels'][:])

#     predict = self.predict(origin_test_x)
#     acc = np.mean(predict == origin_test_y)
#     print(f"Acc: {acc}")
