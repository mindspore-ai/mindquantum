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

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


class Main(HybridModel):
    def __init__(self):
        super().__init__()
        self.dataset = self.build_dataset(self.origin_x, self.origin_y, 10)
        self.qnet = MQLayer(self.build_grad_ops())
        self.model = self.build_model()
        self.checkpoint_name = os.path.join(project_path, "model.ckpt")

    def build_dataset(self, x, y, batch=None):
        train = ds.NumpySlicesDataset(
            {
                "image": x.reshape((x.shape[0], -1)),
                "label": y.astype(np.int32)
            },
            shuffle=True)
        if batch is not None:
            train = train.batch(batch)
        return train

    def build_grad_ops(self):
        encoder = Circuit()
        for i in range(16):
            encoder += RX(f'alpha{i}').on(i)
            encoder += RZ(f'alpha{i}').on(i)
        encoder = encoder.no_grad()

        ansatz = Circuit()
        for i in range(16):
            ansatz += RY(f'y1-{i}').on(i)
        for i in range(15):
            ansatz += X.on(i+1,i)
        for i in range(16):
            ansatz += RY(f'yy1-{i}').on(i)
        
        for i in range(15):
            ansatz += X.on(i+1,i)
        for i in range(16):
            ansatz += RY(f'yy2-{i}').on(i)

        for i in range(15):
            ansatz += X.on(i+1,i)
        for i in range(16):
            ansatz += RY(f'yy3-{i}').on(i)

        total_circ = encoder + ansatz
        # ham = Hamiltonian(QubitOperator('Z0'))
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [14, 15]]
        sim = Simulator('projectq', total_circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(
            ham,
            total_circ,
            encoder_params_name=encoder.params_name,
            ansatz_params_name=ansatz.params_name,
            parallel_worker=5)
        return grad_ops

    def build_model(self):
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.opti = ms.nn.Adam(self.qnet.trainable_params())
        self.model = Model(self.qnet, self.loss, self.opti)
        return self.model

    def train(self):
        self.model.train(5, self.dataset, callbacks=LossMonitor(10))

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        predict = self.model.predict(ms.Tensor(test_x))
        predict = predict.argmax(1).asnumpy().flatten() > 0
        # predict = predict.asnumpy().flatten() > 0
        return predict

# MyModel = Main()
# MyModel.train()
# MyModel.export_trained_parameters()
