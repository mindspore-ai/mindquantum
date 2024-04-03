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
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindquantum.framework.operations import MQOps
from mindspore.common.initializer import Tensor

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

INIT = [
    3.2261405e+00, 4.7348962e+00, 6.2831831e+00, 3.1415930e+00, 3.1415925e+00,
    -3.1415927e+00, 2.7386863e+00, 3.5925109e+00, 1.3268498e+00, 4.7391372e+00,
    6.2831535e+00, 3.4613238e-04, 3.1415937e+00, 3.1415927e+00, 3.1415908e+00,
    3.1416035e+00, 3.1418450e+00, 8.2487531e-05, 6.2832031e+00, -3.3914199e-05,
    5.8206611e+00, 5.7924747e+00, 4.7074232e+00, 2.3692575e+00, 6.7947068e+00,
    4.6887016e+00, 6.2831850e+00, 3.1415946e+00, 2.0078316e-05, 3.1416035e+00,
    6.2830667e+00, 3.1414294e+00, 3.1415894e+00, 3.1416073e+00, 3.1508889e+00,
    1.9582824e+00, 4.2198081e+00, 1.5710907e+00, 4.3872066e+00, 5.0926895e+00,
    4.2463574e+00, 2.6224222e+00, 5.6246147e+00, 3.1865094e+00, 4.7120042e+00,
    4.4663334e-03, 5.8422251e+00, 1.1925411e+00, 3.4323020e+00, 2.1616812e+00,
    3.7541652e+00, 1.4462386e+00, 3.1415925e+00, 3.1415918e+00, 3.1415927e+00,
    8.1972331e-08, 3.9027441e+00, 4.1937008e+00, 4.8970876e+00, 6.6975081e-01,
    6.2828259e+00, 1.1213537e-04, 6.2831936e+00, 6.2831850e+00, 4.7073865e+00,
    4.7101002e+00, 3.1416965e+00, 3.1418223e+00, 8.3227060e-06, 3.1415746e+00,
    3.1182301e+00, 3.1977532e+00, 4.2514181e+00, 4.6935992e+00, 3.1665075e+00,
    3.1646855e+00, 6.2831964e+00, 6.2831836e+00, 6.2831845e+00, 3.1415989e+00,
    4.6978717e+00, 4.9313860e+00, 3.1415656e+00, 3.1415644e+00, 1.8927890e-03,
    2.9438283e+00, 6.0461955e+00, -9.1860276e-03, 3.3147871e+00, 4.5034471e+00,
    4.7123895e+00, 3.4007387e+00, 1.2607254e+00, 4.6866302e+00, 1.5703900e+00,
    1.5670323e+00, 3.3652587e+00, 2.9547648e+00, 6.2832260e+00, 4.7445555e+00,
    6.6686954e+00, 3.8643994e+00, -3.9197795e-02, 5.5137334e+00, 1.9697132e+00,
    4.7181635e+00, 4.6874437e+00, 5.8891234e+00, 5.5331278e+00, 1.0264239e+00,
    7.9727334e-01, 1.5338116e+00, 5.8560205e+00, 1.8600461e-01, 1.4906881e+00,
    2.9295988e+00, 2.4033775e+00, 9.5099372e-01, 1.9757155e-01, 4.6108446e+00,
    6.8706268e-01, 3.7696240e+00, 8.7830687e-01, 4.1397185e+00, 4.7734904e+00,
    5.4438396e+00, 1.7602465e+00, 2.8195434e+00, 3.5129263e+00, 1.5773628e+00,
    2.9548094e+00, 4.1143274e+00, 1.3847045e+00, 5.6471372e+00, 1.2441187e+00,
    1.1761650e+00, 1.9633205e+00, 3.9320714e+00, 1.3692541e+00, 1.2609581e+00,
    6.6729838e-01, 1.7555257e+00, 1.4116186e+00, 1.4122249e+00, 1.9598169e+00,
    5.2425246e+00, 1.9030777e+00, 5.6447496e+00, 4.1600776e+00, 5.6751714e+00
]


class Main(HybridModel):
    def __init__(self):
        super().__init__()
        self.dataset = self.build_dataset(self.origin_x, self.origin_y, 50)
        self.qnet = MyHybridCell(3)  #MQLayer(self.build_grad_ops())
        self.model = self.build_model()
        self.checkpoint_name = os.path.join(project_path, "model.ckpt")

    # def build_grad_ops(self):
    #     circ = Circuit()
    #     for i in range(14):
    #         circ = circ+RX(f'p{i}').on(i)
    #     encoder = add_prefix(circ, 'e1')
    #     ansatz = Circuit()
    #     for i in range(14):
    #         ansatz = ansatz+RX(f'a{i}').on(i)+RY(f'b{i}').on(i)+RZ(f'c{i}').on(i)
    #     # for i in range(14):
    #     #     ansatz = ansatz + XX(f'd{i}').on([i, (i + 1) % 14]) + YY(f'e{i}').on([i, (i + 1) % 14]) + ZZ(f'f{i}').on(
    #     #         [i, (i + 1) % 14])
    #     # for i in range(14):
    #     #     ansatz = ansatz + XX(f'g{i}').on([i, (i + 4) % 14]) + YY(f'h{i}').on([i, (i + 4) % 14]) + ZZ(f'i{i}').on(
    #     #         [i, (i + 4) % 14])
    #     total_circ = encoder.as_encoder() + ansatz
    #     ham = Hamiltonian(QubitOperator( f'Y5'))
    #     sim = Simulator('mqvector', total_circ.n_qubits)
    #     grad_ops = sim.get_expectation_with_grad(
    #         ham,
    #         total_circ,
    #         parallel_worker=5)
    #     return grad_ops

    def build_dataset(self, x, y, batch=None):
        train = ds.NumpySlicesDataset(
            {
                "image": x.reshape((x.shape[0], -1)),  #最左侧和最右侧数据都是0，删掉。
                "label": y.astype(np.int32)
            },
            shuffle=False)
        if batch is not None:
            train = train.batch(batch)
        return train

    def build_model(self):
        self.loss = myLoss()
        self.opti = ms.nn.Adam(self.qnet.trainable_params(),
                               learning_rate=0.01)
        self.model = Model(self.qnet, self.loss, self.opti)
        return self.model

    def train(self):
        while True:
            self.model.train(
                5, self.dataset,
                callbacks=LossMonitor(1))  #callbacks=LossMonitor()
            INIT = self.qnet.weight
            self.export_trained_parameters()
            y = self.predict(self.origin_x)
            acc = np.mean(y == self.origin_y)
            print(acc)

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        print(qnet_weight)
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        predict = self.model.predict(ms.Tensor(test_x))
        predict = predict.asnumpy().flatten() > 0
        return predict


class MyHybridCell(ms.nn.Cell):
    def __init__(self, obj):
        super(MyHybridCell, self).__init__()
        self.obj = obj
        op1 = self.build_grad_ops1()
        self.evolution1 = MQOps(op1)
        self.weight_size1 = len(
            self.evolution1.expectation_with_grad.ansatz_params_name)
        init = Tensor(INIT, dtype=ms.float32)
        self.weight = Parameter(
            init, name='ansatz_weight_1'
        )  #initializer('normal', self.weight_size1, dtype=ms.float32)
        self.count = 0

    def construct(self, x):
        if self.count % 50 == 0:
            print(self.weight.asnumpy())
        self.count += 1
        x = self.evolution1(x, self.weight)
        #print('x',x)
        #x = self.evolution2(x, w2)
        return x

    def build_grad_ops1(self):
        circ = Circuit()
        for i in range(16):
            circ = circ + RX({f'p{i}': np.pi}).on(i)
        circ += XGate().on(16)
        encoder = add_prefix(circ, 'e1')
        ansatz = Circuit()
        for i in range(17):
            ansatz = ansatz + RX(f'a1{i}').on(i) + RY(f'a2{i}').on(i)
        ansatz += BarrierGate()
        for i in range(17):
            if i != self.obj:
                ansatz = ansatz + ZZ(f'a3{i}').on([i, self.obj])
        ansatz += BarrierGate()
        for i in range(17):
            ansatz = ansatz + RX(f'a4{i}').on(i) + RY(f'a5{i}').on(i)
        ansatz += BarrierGate()
        for i in range(17):
            if i != self.obj:
                ansatz = ansatz + ZZ(f'a6{i}').on([i, self.obj])
        ansatz += BarrierGate()
        for i in range(17):
            ansatz = ansatz + RX(f'a7{i}').on(i) + RY(f'a8{i}').on(i)
        ansatz += BarrierGate()
        for i in range(17):
            if i != self.obj:
                ansatz = ansatz + ZZ(f'a9{i}').on([i, self.obj])
        ansatz += BarrierGate()
        total_circ = encoder.as_encoder() + ansatz
        ham = Hamiltonian(QubitOperator(f'Z{self.obj}'))
        sim = Simulator('mqvector', total_circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(ham,
                                                 total_circ,
                                                 parallel_worker=5)
        return grad_ops


class myLoss(ms.nn.LossBase):
    def __init__(self, reduction='mean'):
        """Initialize myLoss."""
        super(myLoss, self).__init__(reduction)

    def construct(self, logits, label):
        label = label.reshape((-1, 1))
        #print('label:', label)
        #print('logits', logits)
        x = 1 - (2 * label - 1) * logits
        #print('x:',x)
        return self.get_loss(x)


if __name__ == '__main__':
    #for i in range(17):
    a = Main()
    #a.train()
    a.export_trained_parameters()
    #grad = a.build_grad_ops()
