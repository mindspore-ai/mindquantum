import os

os.environ['OMP_NUM_THREADS'] = '2'
from hybrid import HybridModel
from hybrid import project_path
import numpy as np
import mindspore as ms
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model
from mindspore.train.callback import LossMonitor
from mindquantum import *

import collections
# np.random.seed(1000)
# ms.set_seed(1000)
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


def add_CRY_layer(C, num_bit, num_layer):
    for i in range(num_bit):
        if i + 1 < num_bit:
            C += RY(f"cry-{i}-{num_layer}").on(i + 1, i)


def add_CRX_H_RY(C, begin, end, layer):
    for i in range(begin, end):
        if i + 1 <= end:
            # C += RX(f"rx-left-{begin}-{end}-{i}-{layer}").on(i+1, i)
            C += X.on(i + 1, i)
            C += RX(f"rx-left-{begin}-{end}-{i}-{layer}").on(i + 1)
    C += H.on(end)
    C += RY(f"ry-{begin}-{end}").on(end)
    C += H.on(end)
    for i in range(begin, end):
        if i + 1 <= end:
            # C += RX(f"rx-left-{begin}-{end}-{i}-{layer}").on(i+1, i)
            C += X.on(i + 1, i)
            C += RX(f"rx-left-{begin}-{end}-{i}-{layer}").on(i + 1)
    C += H.on(begin)
    C += RZ(f"rz-{begin}-{end}").on(end)
    C += H.on(begin)


def add_RX_layer(C, num_bit, num_layer):

    for i in range(num_bit):
        C += RX(f"theta{num_layer}-{i}").on(i)


def build_encoder(num_bit=4, first=False):
    encoder = Circuit()
    for i in range(num_bit):
        if first:
            encoder += RX(f'alpha0{i}').on(i)
        encoder += RY(f'alpha2{i}').on(i)
        encoder += H.on(i)
    encoder += BarrierGate()
    if first:
        for i in range(num_bit):
            encoder += RX(f'alpha1{i}').on(i)
            encoder += RY(f'alpha3{i}').on(i)
        encoder += BarrierGate()

    return encoder


def build_ansatz(num_bit=4):
    ansatz = Circuit()
    for layer in range(5):
        add_RX_layer(ansatz, num_bit, layer)
        add_CRX_H_RY(ansatz, 0, num_bit, layer)
        add_CRY_layer(ansatz, num_bit, layer)
        ansatz += BarrierGate()

    return ansatz


def build_grad_ops(encoder, ansatz, num_output):
    circuit = encoder.as_encoder() + ansatz

    ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in range(num_output)]

    sim = Simulator('mqvector', circuit.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, circuit, parallel_worker=5)
    return grad_ops


class Net(ms.nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.MQLayer1 = MQLayer(
            build_grad_ops(build_encoder(first=True),
                           build_ansatz(),
                           num_output=4))
        self.MQLayer2 = MQLayer(
            build_grad_ops(build_encoder(first=False),
                           build_ansatz(),
                           num_output=4))
        self.MQLayer3 = MQLayer(
            build_grad_ops(build_encoder(first=False),
                           build_ansatz(),
                           num_output=1))
        self.sigmoid = ms.nn.Sigmoid()
        self.relu = ms.nn.ReLU()

    def construct(self, x):
        x = self.MQLayer1(x)
        x = self.relu(x)
        x = self.MQLayer2(x)
        x = self.relu(x)
        x = self.MQLayer3(x)

        x = x.flatten()
        return self.sigmoid(x)


class Main(HybridModel):
    def __init__(self):
        super().__init__()
        # 父类在初始化的时候就已经把数据集加载进来了, 分别是 self.origin_x 和 self.origin_y
        self.train_loader, self.test_loader = self.build_dataset(
            self.origin_x, self.origin_y, 10)
        self.qnet = Net()  # 返回构建好的net
        self.model = self.build_model()  # 指定优化器, loss 函数等
        self.checkpoint_name = os.path.join(project_path, "MyNet.ckpt")

    def remove_contradicting(self, xs, ys):
        mapping = collections.defaultdict(set)
        orig_x = {}
        for x, y in zip(xs, ys):
            orig_x[tuple(x.flatten())] = x
            mapping[tuple(x.flatten())].add(y)

        new_x = []
        new_y = []
        for flatten_x in mapping:
            x = orig_x[flatten_x]
            labels = mapping[flatten_x]
            if len(labels) == 1:
                new_x.append(x)
                new_y.append(next(iter(labels)))
            else:
                pass

        num_uniq_0 = sum(1 for value in mapping.values()
                         if len(value) == 1 and True in value)
        num_uniq_1 = sum(1 for value in mapping.values()
                         if len(value) == 1 and False in value)
        num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)

        print("Remaining non-contradicting unique images: ", len(new_x))

        return np.array(new_x), np.array(new_y)

    def build_dataset(self, x, y, batch=None):
        x = x.reshape(-1, 16)
        # 把重复的、矛盾的剔除以后，只剩下120个样本
        # new_x, new_y = self.remove_contradicting(x, y)
        new_x, new_y = x, y
        train_x, test_x = new_x[:], new_x[:]
        train_y, test_y = new_y[:], new_y[:]
        train_loader = ds.NumpySlicesDataset(
            {
                "image": train_x,
                "label": train_y.astype(np.float32)
            },
            shuffle=True)
        test_loader = ds.NumpySlicesDataset({
            "image": test_x,
            "label": test_y.astype(np.float32)
        })
        if batch is not None:
            train_loader = train_loader.batch(batch)
            test_loader = test_loader.batch(batch)
        return train_loader, test_loader

    def build_model(self):
        # self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.loss = ms.nn.BCELoss(reduction='mean')
        # self.opti = ms.nn.Momentum(self.qnet.trainable_params(), learning_rate=0.001, momentum=0.9)
        self.opti = ms.nn.Adam(self.qnet.trainable_params(),
                               learning_rate=0.001)
        self.model = Model(self.qnet, self.loss, self.opti)
        return self.model

    def train(self):
        self.model.train(20,
                         self.train_loader,
                         callbacks=LossMonitor(100),
                         dataset_sink_mode=False)

    def export_trained_parameters(self):

        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        predict = self.model.predict(ms.Tensor(test_x))
        predict = predict.asnumpy()

        predict[predict > 0.5] = 1
        predict[predict <= 0.5] = 0
        return predict


if __name__ == "__main__":

    main = Main()
    # print("weight before training: ")
    # for para in main.qnet.trainable_params():
    #     print(para.asnumpy())

    main.train()
    main.export_trained_parameters()

    # print("weight after training: ")
    # for para in main.qnet.trainable_params():
    #     print(para.asnumpy())

    # net = Net()
    # x = ms.Tensor(shape=(1, 16), dtype=ms.dtype.float32, init=ms.common.initializer.One())
    # y = net(x)
    # print(y)