import os

os.environ['OMP_NUM_THREADS'] = '4'
from hybrid import HybridModel
from hybrid import project_path

import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import Model
from mindspore.train.callback import LossMonitor
from mindquantum.algorithm import HardwareEfficientAnsatz
from mindquantum import Circuit, X, PhaseShift, Simulator, MQLayer, I

from ansatz_circuit import *
from encoder_circuit import generate_encoder
from loss_fn import MyLoss
from qlayer import *
from myencoder import generate_circuit, generate_value

from mindquantum.core import QubitOperator
from mindquantum.core import Hamiltonian
from mindquantum.core.circuit import dagger

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


class Main(HybridModel):
    def __init__(self):
        super().__init__()
        self.dataset = self.build_dataset(self.origin_x, self.origin_y, 10)

        encoder, paras = generate_encoder()
        encoder = encoder.no_grad()

        self.ham = Hamiltonian(QubitOperator(''))
        self.encoder = encoder
        self.ansatz = self.build_ansatz()

        self.qnet = QLayer(self.build_grad_ops())
        self.model = self.build_model()
        self.checkpoint_name = os.path.join(project_path, "model.ckpt")

    def build_dataset(self, x, y, batch=None):
        train = ds.NumpySlicesDataset(
            {
                "image": x.reshape((x.shape[0], -1)),
                "label": y
            },
            shuffle=False)
        if batch is not None:
            train = train.batch(batch)
        return train

    def build_ansatz(self):
        ansatz = Circuit()
        for i in range(3):
            ansatz += ansatz1(i) + ansatz0(i)
        return ansatz

    def build_grad_ops(self):
        sim = Simulator('projectq', 3)

        circ_left = generate_circuit()
        encoder_y = dagger(circ_left)
        encoder_y.no_grad()
        y_params_name = encoder_y.params_name

        circ_right = self.encoder + self.ansatz + encoder_y

        grad_ops = sim.get_expectation_with_grad(
            hams=self.ham,
            circ_right=circ_right,
            circ_left=Circuit(),
            simulator_left=None,
            encoder_params_name=self.encoder.params_name + y_params_name,
            ansatz_params_name=self.ansatz.params_name,
            parallel_worker=None)

        return grad_ops

    def build_model(self):
        self.loss = MyLoss()
        # self.loss = ms.nn.L1Loss()
        self.opti = ms.nn.Adam(self.qnet.trainable_params(), learning_rate=0.1)
        self.model = Model(self.qnet, self.loss, self.opti)
        return self.model

    def train(self):
        self.model.train(7, self.dataset, callbacks=LossMonitor(40))

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        self.qweight = qnet_weight
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))
        qnet_weight = self.qnet.weight.asnumpy()
        self.qweight = qnet_weight

    def predict(self, origin_test_x):
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))

        qnet_weight = self.qweight
        ans_data = qnet_weight.tolist()
        test_ansatz = self.ansatz
        ans_dict = dict(zip(test_ansatz.params_name, ans_data))

        value = list()
        sim = Simulator('projectq', 3)
        total_circ = self.encoder + test_ansatz

        for i in range(500):
            sim.reset()

            test_x_i = test_x[i]
            test_x_dict = dict(zip(self.encoder.params_name, test_x_i))
            pr = Merge(test_x_dict, ans_dict)
            sim.apply_circuit(total_circ, pr=pr)
            value_i = sim.get_qs()
            value.append(value_i)

        value = np.array(value)

        return value
