from src.QCNN import gene_tfim_encoder, gene_ansatz
from src.config import n_qubits, batch_size, repeat_size, lr, epochs, random_seed, use_additional_data
import numpy as np
import os
import mindspore as ms
from mindspore import nn, ops, save_checkpoint
from mindspore import dataset as ds
from mindquantum import Simulator, Hamiltonian, QubitOperator, MQLayer


print("config of training------------")
print(f"qubits numbers : {n_qubits}")
print(f"additional data: {use_additional_data}")
print(f"random seed    : {random_seed}\n")

print(f"batch size     : {batch_size}")
print(f"repeat sise    : {repeat_size}")
print(f"learning rate  : {lr}")
print(f"epochs         : {epochs}\n\n")

if not os.path.exists(f"./result_{n_qubits}"):
    os.makedirs(f"./result_{n_qubits}")

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(random_seed) 


def create_dataset(data, batch_size=5, repeat_size=1):
    """
    输入zip在一起的x_data和y_label
    输出是 GeneratorDataset对象
    """
    input_data = ds.GeneratorDataset(list(data), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size, drop_remainder=True)
    input_data = input_data.repeat(repeat_size)
    return input_data


class ForwardWithLoss(nn.Cell):
    """定义损失网络"""

    def __init__(self, backbone, loss_fn):
        """实例化时传入前向网络和损失函数作为参数"""
        super(ForwardWithLoss, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, label):
        """连接前向网络和损失函数"""
        out = self.backbone(data)
        return self.loss_fn(out, label)

    def backbone_network(self):
        """要封装的骨干网络"""
        return self.backbone


class TrainOneStep(nn.TrainOneStepCell):
    """定义训练流程"""

    def __init__(self, network, optimizer):
        """参数初始化"""
        super(TrainOneStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, data, label):
        """构建训练过程"""
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads)


class LossAgent(nn.Metric):
    """定义Acc"""

    def __init__(self):
        super(LossAgent, self).__init__()
        self.clear()

    def clear(self):
        self.loss = 0
        self.epoch_num = 0

    def update(self, *loss):
        loss = loss[0].asnumpy()
        self.loss += loss
        self.epoch_num += 1

    def eval(self):
        return self.loss / self.epoch_num


class AccAgent(nn.Metric):
    """定义Acc"""

    def __init__(self):
        super(AccAgent, self).__init__()
        self.clear()

    def clear(self):
        self.correct_num = 0
        self.samples_num = 0

    def update(self, *inputs):
        y_pred = inputs[0].asnumpy()
        y = inputs[1].asnumpy()

        y_predout = np.zeros_like(y_pred)
        y_predout[y_pred < 0] = -1
        y_predout[y_pred > 0] =  1

        self.correct_num += np.sum(y == y_predout)
        self.samples_num += y.shape[0]

    def eval(self):
        return self.correct_num / self.samples_num


if __name__ == "__main__":
    # 把gammas 和 对应的参数dict加载进来
    if use_additional_data:
        gammas, params = np.load(f"./data/{n_qubits}additional_data.npy", allow_pickle=True)
    else:
        gammas, params = np.load(f"./data/{n_qubits}qbsdata.npy", allow_pickle=True)
    # 删除 gamma=1的项
    gammas = np.delete(gammas, len(gammas)//2, 0)
    params = np.delete(params, len(params)//2, 0)

    # 创建encoder之后才知道参数顺序和参数dict如何对应
    encoder = gene_tfim_encoder(n_qubits=n_qubits)
    encoder.no_grad()
    print("\nsummary of encoder:")
    encoder.summary()
    print("parameters of encoder: ", encoder.params_name)

    # 按照parameters的顺序提取每个sample的参数并放在list里
    X_list = []
    for sample in params:
        sample_list = []
        for name in encoder.params_name:
            sample_list.append(sample[name])
        X_list.append(sample_list)
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(gammas)
    Y[Y > 1] = 1
    Y[Y < 1] = -1
    Y = Y[:, np.newaxis].astype(np.int32)

    print(f"\ntotal data number: {len(X)}")
    print(f"params in one sample: {len(X[0])}")
    
    # 随机生成index抽取训练集和测试集
    idx = np.random.permutation(len(params))
    split_point = int(0.8 * len(X))
    x_train, x_test = X[idx[: split_point]], X[idx[split_point: ]]
    y_train, y_test = Y[idx[: split_point]], Y[idx[split_point: ]]
    print()
    print(f"train sample: {len(x_train)}, x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"test  sample: {len(x_test)}, x_test  shape: {x_test.shape }, y_test  shape: {y_test.shape}")
    
    ds_train = create_dataset(zip(x_train, y_train), batch_size=batch_size, repeat_size=repeat_size)
    ds_test = create_dataset(zip(x_test, y_test), batch_size=batch_size, repeat_size=repeat_size)


    ansatz = gene_ansatz(n_qubits=n_qubits)
    print("\nsummary of ansatz(QCNN):")
    ansatz.summary()
    # print(ansatz.params_name)

    circuit = encoder + ansatz
    
    # 哈密顿量，如果结果为0态，得到的是1， 如果结果为1态，得到的是-1
    ham = Hamiltonian(QubitOperator("Z0"))

    simulator = Simulator('projectq', n_qubits=n_qubits)
    # 构建算期望和有关参数梯度的算子
    grad_ops = simulator.get_expectation_with_grad(hams=ham,
                                                circ_right=circuit,
                                                encoder_params_name=encoder.params_name,
                                                ansatz_params_name=ansatz.params_name)
    # grad_ops 输入encoder ansatz参数后，得到三个值，分别是，期望值f，期望关于encoder的导数g，期望关于ansatz的导数h
    # f, g, h = grad_ops(np.random.random([1, 4]), np.random.random([63]))


    QCNN = MQLayer(grad_ops)
    loss = nn.MSELoss(reduction='mean')
    optimizer = nn.Adam(QCNN.trainable_params(), learning_rate=lr)
    
    net_with_loss = ForwardWithLoss(QCNN, loss)
    train_one_step = TrainOneStep(net_with_loss, optimizer)

    acc_agent = AccAgent()
    loss_agent = LossAgent()

    print("\nbegin training:  --------------")
    train_epoch_loss = []
    train_epoch_acc = []
    test_epoch_loss = []
    test_epoch_acc = []
    best_train_loss = 100
    best_train_acc = 0
    best_test_loss = 100
    best_test_acc = 0
    for epoch in range(epochs):
        # training update and loss
        loss_agent.clear()
        for k, data in enumerate(ds_train.create_dict_iterator()):
            # 这里索引出来的label的shape是(5,), 需要是(5, 1)才能和net的输出对的上
            train_one_step(data['data'], data['label'])   # 执行训练，并更新权重
            loss = net_with_loss(data['data'], data['label'])  # 计算损失值
            loss_agent.update(loss)
            # if (k+1) % 15 == 0:
            #     print(f"epoch: {epoch:2}, step: {k+1:3}, loss: {loss.asnumpy():.6}")
        train_loss = loss_agent.eval()
        train_epoch_loss.append(train_loss)
        # training acc
        acc_agent.clear()
        for data in ds_train.create_dict_iterator():
            logits = QCNN(data['data'])
            acc_agent.update(logits, data['label'])
        train_acc = acc_agent.eval()
        train_epoch_acc.append(train_acc)

        # testing loss
        loss_agent.clear()
        for data in ds_test.create_dict_iterator():
            loss = net_with_loss(data['data'], data['label'])  # 计算损失值
            loss_agent.update(loss)
        test_loss = loss_agent.eval()
        test_epoch_loss.append(test_loss)
        # testing acc
        acc_agent.clear()
        for data in ds_test.create_dict_iterator():
            logits = QCNN(data['data'])
            acc_agent.update(logits, data['label'])
        test_acc = acc_agent.eval()
        test_epoch_acc.append(test_acc)
        print(f"epoch: {epoch+1:2}, training loss: {train_loss:8.6}, accuracy: {train_acc:5.4}, testing loss: {test_loss:8.6}, accuracy: {test_acc:5.4}")
        
        if train_loss <= best_train_loss:
            best_train_loss = train_loss
            save_checkpoint(QCNN, f"./result_{n_qubits}/{'interpolation' if use_additional_data else 'origin'}_with_best_train_loss.ckpt")
        if train_acc >= best_train_acc:
            best_train_acc = train_acc
            save_checkpoint(QCNN, f"./result_{n_qubits}/{'interpolation' if use_additional_data else 'origin'}_with_best_train_acc.ckpt")
        if test_loss <= best_test_loss:
            best_test_loss = test_loss
            save_checkpoint(QCNN, f"./result_{n_qubits}/{'interpolation' if use_additional_data else 'origin'}_with_best_test_loss.ckpt")
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(QCNN, f"./result_{n_qubits}/{'interpolation' if use_additional_data else 'origin'}_with_best_test_acc.ckpt")
    loss_acc = {}
    loss_acc["train_loss"] = train_epoch_loss
    loss_acc["train_acc"] = train_epoch_acc
    loss_acc["test_loss"] = test_epoch_loss
    loss_acc["test_acc"] = test_epoch_acc
    np.save(f"./result_{n_qubits}/{'interpolation' if use_additional_data else 'origin'}_loss_acc.npy", loss_acc)
