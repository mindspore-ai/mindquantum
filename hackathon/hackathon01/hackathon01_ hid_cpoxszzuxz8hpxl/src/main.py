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
from mindspore import ops, Tensor
from mindquantum import *
from ansatz import *

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


class Main(HybridModel):
    def __init__(self):
        super().__init__()
        self.dataset = self.build_dataset(self.origin_x, self.origin_y, 10)
        self.qnet = MQLayer(self.build_grad_ops())
        self.checkpoint_name = os.path.join(project_path, "0.88.ckpt")
        self.model = self.build_model()
    def build_dataset(self, x, y, batch=None):
        train_x = self.get_parameter(x)

        train = ds.NumpySlicesDataset(
            {
                "image": train_x,
                "label": y.astype(np.int32)
            },
            shuffle=False)
        if batch is not None:
            train = train.batch(batch)
        return train

    def build_grad_ops(self):
        encoder1 = encoder(8)
        encoder1 = encoder1.no_grad()
        ansatz1 = ansatz()
        circuit = encoder1 + ansatz1
        hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [3,4]]
        sim = Simulator('projectq', circuit.n_qubits)
        grad_ops = sim.get_expectation_with_grad(
            hams,
            circuit,
            None,
            None,
            encoder1.params_name,
            ansatz1.params_name,
            parallel_worker=5)
        return grad_ops

    def build_model(self):
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.opti = ms.nn.Adam(self.qnet.trainable_params(), 0.05)
        self.model = Model(self.qnet, self.loss, self.opti)
        self.load_trained_parameters()
        return self.model

    def train(self,epoch,step):
        self.model.train(epoch, self.dataset, callbacks=LossMonitor(step))

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,ms.load_checkpoint(self.checkpoint_name))

    def predict(self, test_x) -> float:
        test_x = self.get_parameter(test_x)
        predict = np.argmax(ops.Softmax()(self.model.predict(ms.Tensor(test_x))), axis=1)
        return predict

    def get_angles(self,x):
        new_x = {}
        angles = {}
        inner_angles = {}
        if len(x) > 1:
            for k in range(int(len(x) / 2)):
                new_x[k] = np.sqrt((np.abs(x[2 * k]) ** 2) + (np.abs(x[2 * k + 1]) ** 2))
            inner_angles = self.get_angles(new_x)

            for k in range(len(new_x)):
                if new_x[k] != 0:
                    if x[2 * k] >= 0:
                        angles[k] = 2 * np.arcsin(x[2 * k + 1] / new_x[k])
                    else:
                        angles[k] = 2 * np.pi - 2 * np.arcsin(x[2 * k + 1] / new_x[k])
                else:
                    angles[k] = 0

        angle = list(inner_angles) + list(angles.values())  # angle是字典，开始定义的时候尝试直接定义列表，但是一直报数据索引溢出，故定义为字典
        return angle

    def get_parameter(self,test_x):
        test_x = test_x.reshape(test_x.shape[0],-1)
        new_test_x = np.zeros((test_x.shape[0], 15))
        for i in range(test_x.shape[0]):
            new_test_x[i, :] = np.array(self.get_angles(test_x[i, :]))
        return new_test_x
