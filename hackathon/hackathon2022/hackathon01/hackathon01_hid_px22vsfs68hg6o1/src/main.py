import os

os.environ['OMP_NUM_THREADS'] = '2'
from hybrid import HybridModel
from hybrid import project_path
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
import mindspore.context as context
import mindspore.dataset as ds
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Adam, Accuracy
from mindspore import Model
from mindspore.train.callback import Callback, LossMonitor
from mindquantum import *

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
'''
#另一种方法：QCNN，需重新训练
def qconv(i,j,params):
    qconv_circuit=Circuit()
    #i，j表示作用位，params表示参数的个数,k表示第几个qconv门，方便设置参数名
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
def qpool(i,j):
    qpool_circuit=Circuit()
    qpool_circuit+=RZ({f'p_{i}':1}).on(j,i)
    qpool_circuit+=X.on(i)
    qpool_circuit+=RX({f'p_{j}':1}).on(j,i)
    qpool_circuit+=X.on(i)
    return qpool_circuit
'''


class Main(HybridModel):
    def __init__(self):
        super().__init__()
        self.dataset = self.build_dataset(self.origin_x, self.origin_y, 10)
        self.qnet = MQLayer(self.build_grad_ops())
        self.model = self.build_model()
        #self.train()
        #self.plot()
        self.checkpoint_name = os.path.join('./src', "model.ckpt")
        #self.export_trained_parameters()

    def build_dataset(self, x, y, batch=5):
        #加载数据集
        train_data = np.load("./src/train.npy", allow_pickle=True)[0]
        #对数据集的格式进行处理
        train_x1 = np.squeeze(train_data['train_x'], 3)
        train_y = train_data['train_y']
        '''
        #去掉全0的，数据清洗
        train1=[]
        label1=[]
        for i in range(5000):
            sum = 0
            for j in range(4):
                for k in range(4):
                    if (train_x1[i][j][k] == 0):
                        sum += 1
            if (sum <= 15):
                train1.append(train_x1[i])
                label1.append(train_y[i])
        '''
        x1 = []
        for i in range(4625):
            x1.append(train_x1[i].reshape(1, 16))
        #感觉有部分数据冗余：不需要取那么多。
        x = np.squeeze(np.array(x1[:440]), 1)[:, :15]
        #print(x.shape)
        #print(x)
        #同理，处理y
        y1 = []
        for i in range(4625):
            if (train_y[i] == True):
                y1.append(1)
            else:
                y1.append(0)
        #数据集大小也很影响，最好是40倍数。
        y = np.array(y1[:440])
        #print(y.shape)
        #print(y)
        #划分训练集、验证集
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=0,
                                                            shuffle=True)
        train = ds.NumpySlicesDataset({
            "image": X_train,
            "label": y_train
        },
                                      shuffle=False)
        global test_loader
        test_loader = ds.NumpySlicesDataset({
            "image": X_test,
            "label": y_test
        },
                                            shuffle=False).batch(5)
        if batch is not None:
            train = train.batch(batch)
        return train

    def build_grad_ops(self):
        #encoder:
        encoder = Circuit()  # 初始化量子线路

        encoder += UN(H, 8)  # H门作用在每1位量子比特
        for i in range(8):  # i = 0,1,2,3,4,5,6,7
            encoder += RZ(f'alpha{i}').on(i)  # RZ(alpha_i)门作用在第i位量子比特
        for j in range(7):  # j = 0,1,2,3,4,5,6
            encoder += X.on(j + 1, j)  # X门作用在第j+1位量子比特，受第j位量子比特控制
            encoder += RZ(f'alpha{j + 8}').on(j +
                                              1)  # RZ(alpha_{j+8})门作用在第0位量子比特
            encoder += X.on(j + 1, j)  # X门作用在第j+1位量子比特，受第j位量子比特控制

        encoder = encoder.no_grad(
        )  # Encoder作为整个量子神经网络的第一层，不用对编码线路中的梯度求导数，因此加入no_grad()
        encoder.summary()  # 总结Encoder
        print(encoder)
        #ansatz:
        ansatz = HardwareEfficientAnsatz(
            8, single_rot_gate_seq=[RY], entangle_gate=X,
            depth=7).circuit  # 通过HardwareEfficientAnsatz搭建Ansatz
        ansatz.summary()  # 总结Ansatz
        print(ansatz)
        #all circuit
        '''
        QCNN用3，7位测量
        ansatz += qconv(0, 1, 6)
        print(ansatz)
        ansatz += qconv(2, 3, 6)
        ansatz += qconv(4, 5, 6)
        ansatz += qconv(6, 7, 6)
        ansatz += qconv(1, 2, 6)
        ansatz += qconv(3, 4, 6)
        ansatz += qconv(5, 6, 6)
        ansatz += qconv(7, 1, 6)
        # 一层池化：4个
        ansatz += qpool(0, 1)
        ansatz += qpool(2, 3)
        ansatz += qpool(4, 5)
        ansatz += qpool(6, 7)
        # 两层卷积层：4个
        ansatz += qconv(1, 3, 6)
        ansatz += qconv(5, 7, 6)
        ansatz += qconv(7, 1, 6)
        ansatz += qconv(3, 5, 6)
        # 一组池化：2个
        ansatz += qpool(1, 3)
        ansatz += qpool(5, 7)
        #一个卷积层
        ansatz += qconv(3, 7, 6)
        '''
        circuit = encoder + ansatz  # 完整的量子线路由Encoder和Ansatz组成
        circuit.summary()
        print(circuit)
        #hamiltonian
        #对什么位进行量子比特测量区别也很大
        hams = [Hamiltonian(QubitOperator(f'Z{i}'))
                for i in [3, 4]]  # 分别对第3位和第4位量子比特执行泡利Z算符测量，且将系数都设为1，构建对应的哈密顿量
        print(hams)

        ms.context.set_context(mode=ms.context.PYNATIVE_MODE,
                               device_target="CPU")
        ms.set_seed(1)  # 设置生成随机数的种子
        sim = Simulator('mqvector', circuit.n_qubits)
        grad_ops = sim.get_expectation_with_grad(hams,
                                                 circuit,
                                                 None,
                                                 None,
                                                 encoder.params_name,
                                                 ansatz.params_name,
                                                 parallel_worker=2)
        return grad_ops

    def build_model(self):
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True,
                                                        reduction='mean')
        self.opti = ms.nn.Adam(self.qnet.trainable_params(), learning_rate=0.1)
        self.model = Model(self.qnet,
                           self.loss,
                           self.opti,
                           metrics={'Acc': Accuracy()})
        return self.model

    def train(self):
        class StepAcc(Callback):
            def __init__(self, model, test_loader):
                self.model = model
                self.test_loader = test_loader
                self.acc = []

            def step_end(self, run_context):
                self.acc.append(
                    self.model.eval(self.test_loader,
                                    dataset_sink_mode=False)['Acc'])

        monitor = LossMonitor(8)  # 监控训练中的损失，每8步打印一次损失值
        global acc
        acc = StepAcc(self.model, test_loader)
        #训练批次过多，可能出现过拟合现象，也是可以微调的。
        self.model.train(8,
                         self.dataset,
                         callbacks=[monitor, acc],
                         dataset_sink_mode=False)  # 将上述建立好的模型训练8次
        print("finished!")

    def plot(self):
        plt.plot(acc.acc)
        plt.title('Statistics of accuracy', fontsize=20)
        plt.xlabel('Steps', fontsize=20)
        plt.ylabel('Accuracy', fontsize=20)
        plt.savefig('accuracy.png')

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        print(qnet_weight)
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))[:, :15]
        predict1 = self.model.predict(ms.Tensor(test_x))
        predict1 = np.argmax(ms.ops.Softmax()(predict1), axis=1)
        predict = []
        for i in range(800):
            if (predict1[i] == 1):
                predict.append(True)
            else:
                predict.append(False)
        return predict