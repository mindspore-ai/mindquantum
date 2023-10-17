import numpy as np

import mindspore.nn as nn
import mindspore as ms
from mindspore.common.initializer import Normal
from mindvision.engine.callback import LossMonitor

# 构建数据集
import mindspore.dataset as ds
from mindspore.dataset import ImageFolderDataset


def load_dataset(ds_dir, batch_size, class_index):
    dataset = ImageFolderDataset(dataset_dir=ds_dir, \
        class_indexing=class_index)
    load_img_ops = ds.transforms.Compose([
        ds.vision.Decode(),
        ds.vision.Rescale(1. / 255, 0),
        ds.vision.Resize((128, 128)),
        ds.vision.HWC2CHW()
    ])
    dataset = dataset.map(load_img_ops, input_columns='image')
    dataset = dataset.map(
        [lambda x: [x], ds.transforms.TypeCast(np.float32)],
        input_columns='label')
    dataset = dataset.batch(batch_size)
    return dataset


# 构建网络
from mindspore.common.initializer import XavierUniform


class Network(nn.Cell):
    def __init__(self, is_quantum=False):
        super().__init__()
        self._is_quantum = is_quantum
        in_channels = 3
        hidden_dim = 32
        hidden_dim_l2 = 64
        self.net_sequential = nn.SequentialCell(
            nn.Conv2d(in_channels, hidden_dim, 3, weight_init=XavierUniform()),
            nn.BatchNorm2d(hidden_dim),
            nn.MaxPool2d(4, stride=4),
            nn.Conv2d(hidden_dim,
                      hidden_dim_l2,
                      3,
                      weight_init=XavierUniform()),
            nn.BatchNorm2d(hidden_dim_l2),
            nn.MaxPool2d(4, stride=4),
            nn.Flatten(),
            nn.Dense(4096, 128, activation='relu',
                     weight_init=XavierUniform()),
            nn.Dense(128, 2, activation='relu', weight_init=XavierUniform()),
        )
        if is_quantum:
            self.vqc = self.vqc_layer()
        self.sigmoid = nn.Dense(2,
                                1,
                                activation='sigmoid',
                                weight_init=XavierUniform())

    def vqc_layer(self):
        # Embedding Layer构建
        from mindquantum.core.circuit import Circuit
        from mindquantum.core.gates import X, RX, RZ

        encoder = Circuit()
        encoder += RX('x0').on(0)
        encoder += RX('x1').on(1)
        encoder = encoder.no_grad()
        encoder = encoder.as_encoder()
        # Variational Quantum Circuit构建
        ansatz = Circuit()
        for n in range(3):
            ansatz += RX(f'w{n}0').on(0)
            ansatz += RX(f'w{n}1').on(1)
            ansatz += X.on(1, 0)
        ansatz.as_ansatz()

        # 构建哈密顿量
        from mindquantum.core.operators import QubitOperator
        from mindquantum.core.operators import Hamiltonian

        hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [0, 1]]

        # 构建网络算子
        from mindquantum.simulator import Simulator
        from mindquantum.framework import MQLayer
        import mindspore as ms

        circuit = encoder.as_encoder() + ansatz.as_ansatz()
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        ms.set_seed(1)  # 设置生成随机数的种子
        sim = Simulator('mqvector', circuit.n_qubits)
        grad_ops = sim.get_expectation_with_grad(hams, circuit)
        vqc_layer = MQLayer(grad_ops)  # 搭建量子神经网络
        return vqc_layer

    def construct(self, x):
        logits = self.net_sequential(x)
        if self._is_quantum:
            logits = self.vqc(logits)
        logits = self.sigmoid(logits)
        return logits


def train_model(is_quantum):
    # 数据集加载
    class_index = {'normal': 0, 'opacity': 1}
    batch_size = 64
    train_ds_dir = r'./archive/train'
    val_ds_dir = r'./archive/val'

    train_ds = load_dataset(train_ds_dir, batch_size, class_index)
    val_ds = load_dataset(val_ds_dir, batch_size, class_index)

    # 使用Model构建训练网络
    learning_rate = 0.01
    net = Network(is_quantum)
    crit = nn.BCELoss(reduction='mean')
    opt = nn.SGD(net.trainable_params(), learning_rate=learning_rate)
    model = ms.Model(network=net,
                     loss_fn=crit,
                     optimizer=opt,
                     metrics={'Loss': nn.Loss()})

    history_cb = ms.History()
    val_loss = []
    epoch_num = 10
    # 经典模型训练
    for epoch in range(epoch_num):
        model.train(epoch + 1,
                    train_ds,
                    callbacks=[LossMonitor(learning_rate), history_cb],
                    initial_epoch=epoch)
        eval_result = model.eval(val_ds)  # 执行模型评估
        val_loss.append(eval_result['Loss'])
        print(history_cb.epoch, history_cb.history)
        print(eval_result)
    train_loss = history_cb.history['net_output']
    return train_loss, val_loss


import matplotlib.pyplot as plt

if __name__ == '__main__':
    qnet_train_loss = train_model(is_quantum=True)
    cnet_train_loss = train_model(is_quantum=False)

    epoch_num = len(qnet_train_loss[0])
    plt.subplot(121)
    plt.plot(list(range(epoch_num)),
             qnet_train_loss[0],
             label='Quantum Net Train Loss')
    plt.plot(list(range(epoch_num)),
             cnet_train_loss[0],
             label='Classical Net Train Loss')
    plt.subplot(122)
    plt.plot(list(range(epoch_num)),
             qnet_train_loss[1],
             label='Quantum Net Val Loss')
    plt.plot(list(range(epoch_num)),
             cnet_train_loss[1],
             label='Classical Net Val Loss')
