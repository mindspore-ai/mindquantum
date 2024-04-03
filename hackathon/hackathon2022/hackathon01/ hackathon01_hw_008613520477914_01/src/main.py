from hybrid import HybridModel
from hybrid import project_path
import numpy as np
import mindspore as ms
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model
from mindspore.train.callback import LossMonitor
from mindquantum.algorithm import HardwareEfficientAnsatz
from mindquantum import *

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


def gen_enhanced_samples(img_use_feas, lab_use):
    img_use_feas_copy = img_use_feas.copy()
    for i in range(len(img_use_feas)):
        item = img_use_feas[i]
        for pix_id in range(len(item)):
            if item[pix_id] == 0:
                img_use_feas_copy[i,
                                  pix_id] = 0.1 * (np.random.uniform() - 0.95)
            if item[pix_id] == 1:
                img_use_feas_copy[i,
                                  pix_id] = (np.random.uniform() * 0.2 + 0.95)

    img_use_feas = np.concatenate((img_use_feas, img_use_feas_copy), 0)
    lab_use = np.concatenate((lab_use, lab_use), 0)
    return img_use_feas, lab_use


def gen_encoding_cir(rotate_gate):
    circ = Circuit()
    # the default circuit
    for i in range(8):
        circ += rotate_gate(f'p{i}').on(i)
    circ += UN(X, [1, 3, 5, 7], [0, 2, 4, 6])
    circ += UN(X, [2, 4, 6], [1, 3, 5])
    return circ


class Main(HybridModel):
    def __init__(self):
        super().__init__()
        self.dataset = self.build_dataset(self.origin_x, self.origin_y, 16)
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
        circ1 = gen_encoding_cir(RX)
        circ2 = gen_encoding_cir(RY)

        encoder = add_prefix(circ1, 'e1') + add_prefix(circ2, 'e2')
        ansatz = HardwareEfficientAnsatz(8, [RX, RZ, RY], X, 'linear',
                                         4).circuit

        total_circ = encoder.as_encoder() + ansatz
        hams = [Hamiltonian(QubitOperator(f'z{i}')) for i in [6, 7]]

        sim = Simulator('mqvector', total_circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(hams,
                                                 total_circ,
                                                 parallel_worker=5)
        return grad_ops

    def build_model(self):
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.opti = ms.nn.Adam(self.qnet.trainable_params(),
                               learning_rate=0.015)
        self.model = Model(self.qnet, self.loss, self.opti)
        return self.model

    def train(self):
        self.model.train(1, self.dataset, callbacks=LossMonitor())

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        predict = self.model.predict(ms.Tensor(test_x))
        predict_cls = predict.asnumpy().argmax(axis=1)
        pred_bool = predict_cls > 0
        # if the empty pic, predict True
        if_empty = np.sum(test_x, 1) == 0
        pred_bool[if_empty] = True

        return pred_bool
