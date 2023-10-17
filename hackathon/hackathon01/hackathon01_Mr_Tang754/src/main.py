import os
import sys
import numpy as np

sys.path.append('./src')  # 加入mypackge的父模块

from mindquantum.framework import MQLayer
from mindquantum.algorithm.library import amplitude_encoder
from mindquantum import Circuit, H, RZ, RX, BarrierGate
from mindquantum.simulator import Simulator
from mindquantum.core import ParameterResolver
from mindquantum.core import QubitOperator
from mindquantum.core import Hamiltonian

import mindspore as ms
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model
from mindspore.train.callback import Callback, LossMonitor
from mindspore.nn import Adam, Accuracy
from mindspore import ops, Tensor  # 导入ops模块和Tensor模块
from hybrid import HybridModel, project_path

os.environ['OMP_NUM_THREADS'] = '2'
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(2)

# The anstatz need optimize 67 parameters, consist three layers, all-to-all entangle


def construct_ansatz():

    layer = 3
    n_qubits = 4
    ansatz = Circuit()

    for k in range(layer):
        for i in range(n_qubits):
            ansatz += RX(f'beta{i + 20*k}').on(i)
            ansatz += RX(f'beta{i + 4 + 20*k}').on(i)

        ansatz += BarrierGate()

        ansatz += RX(f'beta{8  + 20*k}').on(2, 3)
        ansatz += RX(f'beta{9  + 20*k}').on(1, 3)
        ansatz += RX(f'beta{10 + 20*k}').on(0, 3)
        ansatz += RX(f'beta{11 + 20*k}').on(3, 2)
        ansatz += RX(f'beta{12 + 20*k}').on(1, 2)
        ansatz += RX(f'beta{13 + 20*k}').on(0, 2)
        ansatz += RX(f'beta{14 + 20*k}').on(3, 1)
        ansatz += RX(f'beta{15 + 20*k}').on(2, 1)
        ansatz += RX(f'beta{16 + 20*k}').on(0, 1)
        ansatz += RX(f'beta{17 + 20*k}').on(3, 0)
        ansatz += RX(f'beta{18 + 20*k}').on(2, 0)
        ansatz += RX(f'beta{19 + 20*k}').on(1, 0)

        ansatz += BarrierGate()

    for i in range(n_qubits):
        ansatz += RX(f'beta{60 + i}').on(i)
        ansatz += RZ(f'beta{64 + i}').on(i)

    return ansatz


class SaveCallback(Callback):
    def __init__(self, eval_model, ds_eval):
        super(SaveCallback, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        #self.file_name = str('best') + ".ckpt"
        #self.checkpoint_name = os.path.join(project_path, "model.ckpt")
        self.file_name = os.path.join(project_path, "model.ckpt")
        self.test_acc = []
        self.best_acc = 0

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        res = self.model.eval(self.ds_eval, dataset_sink_mode=False)['Acc']
        self.test_acc.append(res)

        if res > self.best_acc:
            self.best_acc = res
            ms.save_checkpoint(save_obj=cb_params.train_network,
                               ckpt_file_name=self.file_name)
            #print("Save the maximum accuracy checkpoint,the accuracy is", self.best_acc)


class Main(HybridModel):
    def __init__(self):
        super().__init__()
        self.n_qubit = 4
        self.n_train = 4200  # 训练样本的总数
        self.n_test = 800  # 测试样本的总数
        # 用Amplitude encoding 方法得到量子数据
        self.train_loader, self.test_loader = self.build_dataset(
            self.origin_x, self.origin_y, self.n_train, self.n_test, 10)
        self.qnet = MQLayer(self.build_grad_ops())
        self.model = self.build_model()
        self.checkpoint_name = os.path.join(project_path, "model.ckpt")

    def encoder_parameters(self, image):

        # 首先将经典的图片矩阵信息转化为列向量，然后进行归一化处理。
        denominator = np.sqrt(sum(image)**2)
        if denominator != 0:
            image = image / denominator
        # 因为Mindquantum的 Amplitude Encoding方法是对于固定的量子比特数目量子线路是一样的
        # 所以我们要得到的就是
        parameters = []
        encoder, parameterResolver = amplitude_encoder(image, self.n_qubit)
        for i, param in parameterResolver.items():
            parameters.append(param)
        parameters = np.array(parameters)
        return parameters

    def build_dataset(self, x, y, n_train, n_test, batch=None):

        images = x.reshape((x.shape[0], -1))
        label = y.astype(np.int32)

        training_images = images[:self.n_train]
        training_labels = label[:self.n_train]

        testing_images = images[self.n_train:self.n_train + self.n_test]
        testing_labels = label[self.n_train:self.n_train + self.n_test]

        training_quantum_data = np.array([
            Main.encoder_parameters(self, image) for image in training_images
        ])

        testing_quantum_data = np.array(
            [Main.encoder_parameters(self, image) for image in testing_images])

        train = ds.NumpySlicesDataset(
            {
                "features": training_quantum_data,
                "labels": training_labels
            },
            shuffle=False).batch(batch)

        test = ds.NumpySlicesDataset(
            {
                "features": testing_quantum_data,
                "labels": testing_labels
            },
            shuffle=False).batch(batch)

        return train, test

    def build_grad_ops(self):

        inital_image = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        inital_quantum_encoder = Main.encoder_parameters(self, inital_image)
        encoder, parameterResolver = amplitude_encoder(inital_quantum_encoder,
                                                       self.n_qubit)

        ansatz = construct_ansatz()

        total_circ = encoder.as_encoder() + ansatz
        hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [1, 2]]
        sim = Simulator('mqvector', total_circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(
            hams,
            total_circ,
            parallel_worker=5)
        return grad_ops

    def build_model(self):
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True,
                                                        reduction='mean')
        self.opti = ms.nn.Adam(self.qnet.trainable_params(),
                               learning_rate=0.10)
        self.model = Model(self.qnet,
                           self.loss,
                           self.opti,
                           metrics={'Acc': Accuracy()})
        return self.model

    def train(self):

        monitor = LossMonitor(470)
        save_acc = SaveCallback(self.model, self.test_loader)
        self.model.train(1,
                         self.train_loader,
                         callbacks=[monitor, save_acc],
                         dataset_sink_mode=False)

    def export_trained_parameters(self):
        # qnet_weight = self.qnet.weight.asnumpy()
        # ms.save_checkpoint(self.qnet, self.checkpoint_name)
        return None

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        testing_quantum_data = np.array(
            [Main.encoder_parameters(self, image) for image in test_x])
        predict = np.argmax(ops.Softmax()(self.model.predict(
            Tensor(testing_quantum_data))),
                            axis=1)
        predict = predict.flatten() > 0
        return predict


main = Main()
main.train()
