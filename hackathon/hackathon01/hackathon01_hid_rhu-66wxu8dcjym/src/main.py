import os

os.environ['OMP_NUM_THREADS'] = '2'
from hybrid import HybridModel
from hybrid import project_path
import numpy as np
import mindspore as ms
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model
from mindspore.train.callback import LossMonitor, Callback
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
            shuffle=False)
        if batch is not None:
            train = train.batch(batch)
        return train

    def build_grad_ops(self):

        # encoder
        circ = Circuit()
        # 多余的
        for i in range(4):
            circ += RX({f'r2{i}': 1}).on(i)
            circ += RX({f'r2{i}': -1}).on(i)
        for i in range(8):
            circ += RX({f'p{i}': np.pi}).on(i)
        # 多余的
        for i in range(4, 8):
            circ += RX({f'r2{i}': 1}).on(i)
            circ += RX({f'r2{i}': -1}).on(i)

        encoder = add_prefix(circ, 'e')  # + add_prefix(circ, 'e2')

        #ansatz
        circ = Circuit()
        for i in range(8):
            circ += RY(f'a{i}').on(i)
        #circ += UN(X, [1, 3, 5, 7], [0, 2, 4, 6])
        #circ += UN(X, [2, 4, 6], [1, 3, 5])
        ansatz = add_prefix(circ, 'a1')
        total_circ = encoder.as_encoder() + ansatz
        # ham
        ham = QubitOperator('Z0')
        for i in range(1, 8):
            ham += QubitOperator(f'Z{i}')
        ham = [Hamiltonian(ham), Hamiltonian(-ham)]

        sim = Simulator('mqvector', total_circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(ham,
                                                 total_circ,
                                                 parallel_worker=5)
        print('total_circ', total_circ)
        total_circ.summary()
        return grad_ops

    def build_model(self):
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.opti = ms.nn.Adam(self.qnet.trainable_params())
        self.model = Model(self.qnet, self.loss, self.opti)
        return self.model

    def train(self):
        test_loader = self.build_dataset(self.origin_x, self.origin_y, 25)
        acc = StepAcc(self.model, test_loader)
        moniter = LossMonitor(16)
        self.model.train(1,
                         self.dataset,
                         callbacks=[moniter, acc],
                         dataset_sink_mode=False)

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        predict = self.model.predict(ms.Tensor(test_x))
        predict = predict.asnumpy().flatten() > 0
        predict = predict[1::2]
        return predict


class StepAcc(Callback):  # 定义一个关于每一步准确率的回调函数
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.acc = []

    def step_end(self, run_context):
        self.acc.append(
            self.model.eval(self.test_loader, dataset_sink_mode=False)['Acc'])
