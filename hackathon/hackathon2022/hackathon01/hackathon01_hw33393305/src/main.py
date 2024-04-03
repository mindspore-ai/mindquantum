import os

os.environ['OMP_NUM_THREADS'] = '4'
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
            shuffle=False)
        if batch is not None:
            train = train.batch(batch)
        return train

    def build_grad_ops(self):
        # Define encoder
        encoder = Circuit()
        encoder += UN(H, 4)
        for i in range(16):
            x = bin(i)[2:].zfill(4)
            encoder += BarrierGate(True)
            for j in reversed(range(4)):
                if x[j] == '0':
                    encoder += X.on(j)
            encoder += PhaseShift({f'alpha{i}': np.pi}).on(3, [0, 1, 2])
            for j in reversed(range(4)):
                if x[j] == '0':
                    encoder += X.on(j)
        encoder += BarrierGate(True)
        encoder = encoder.no_grad()

        # Define ansatz
        ansatz = HardwareEfficientAnsatz(4,
                                         single_rot_gate_seq=[RY],
                                         entangle_gate=X,
                                         depth=3).circuit
        for i in range(4):
            ansatz += RX(f'theta{i}_2').on(i)

        # Combine encoder and ansatz
        total_circ = encoder.as_encoder() + ansatz

        # Define hamitonian
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [2, 3]]

        # Define simulator
        sim = Simulator('mqvector', total_circ.n_qubits)

        # Calculate gradient
        grad_ops = sim.get_expectation_with_grad(ham,
                                                 total_circ,
                                                 parallel_worker=5)
        return grad_ops

    def build_model(self):
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.opti = ms.nn.Adam(self.qnet.trainable_params(),
                               learning_rate=0.01)
        self.model = Model(self.qnet, self.loss, self.opti)
        return self.model

    def train(self):
        self.model.train(2, self.dataset, callbacks=LossMonitor(10))

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))

        origin_predict = self.model.predict(ms.Tensor(test_x))
        origin_predict = origin_predict.asnumpy().flatten() > 0

        # We measure q2,q3, so each test return 2 result, reshape it to get pair results.
        predict_shaped = origin_predict.reshape(800, 2)

        # Get the bigger one location as predict
        predict = np.argmax(ms.ops.Softmax()(ms.Tensor(predict_shaped,
                                                       ms.float16)),
                            axis=1)

        return predict
