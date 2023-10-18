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
        self.dataset = self.build_dataset(self.origin_x, self.origin_y, 256)
        self.qnet = MQLayer(self.build_grad_ops(0))
        self.model = self.build_model()
        self.checkpoint_name = os.path.join(project_path, "model.ckpt")

    def build_dataset(self, x, y, batch=None):
        train = ds.NumpySlicesDataset({
                "image": x.reshape((x.shape[0], -1))*np.pi/2,
                "label": y.astype(np.int32)},shuffle=False)
        if batch is not None:
            train = train.batch(batch)
        return train

    def build_grad_ops(self,k):
        encoder = Circuit()
        encoder += UN(H, 8)
        for i in range(8):
            encoder += RZ(f'alpha_a{i}').on(i)
        for j in range(8):
            index_j = (j+1)%8
            encoder += X.on(index_j, j)
            encoder += RZ(f'alpha_b{j+8}').on(index_j)
            encoder += X.on(index_j, j)
        encoder = encoder.no_grad()

        ctrl_seed = [1]
        ansatz = Circuit()
        for j in range(len(ctrl_seed)):
            for i in range(8):
                ansatz += RX('{}beta_a_1{}{}'.format(k,j,i)).on(i)
                ansatz += RZ('{}beta_a_2{}{}'.format(k,j,i)).on(i)
                ansatz += RX('{}beta_a_3{}{}'.format(k,j,i)).on(i)
            for i in range(8):
                index = (i + ctrl_seed[j]) % 8
                ansatz += RX('{}beta_b_1{}{}'.format(k,j,i)).on(index,i)
                ansatz += RZ('{}beta_b_2{}{}'.format(k,j,i)).on(index,i)
                ansatz += RX('{}beta_b_3{}{}'.format(k,j,i)).on(index,i)
        total_circ = encoder.as_encoder() + ansatz
        # Hamiltonian
        ham = [Hamiltonian(QubitOperator(f'Z{i}')+QubitOperator(f'Y{i}')+QubitOperator(f'X{i}')) for i in [3,4]]
        sim = Simulator('mqvector', total_circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(ham,total_circ,
                                                    parallel_worker=5)
        return grad_ops

    def build_model(self):
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction='mean')
        self.opti = ms.nn.Adam(self.qnet.trainable_params(),learning_rate=0.05)
        self.model = Model(self.qnet, self.loss, self.opti)
        return self.model

    def train(self):
        self.model.train(10, self.dataset, callbacks=LossMonitor(10))

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))*np.pi/2
        predict = self.model.predict(ms.Tensor(test_x))
        predict = np.argmax(predict.asnumpy(), axis=1) > 0
        return predict