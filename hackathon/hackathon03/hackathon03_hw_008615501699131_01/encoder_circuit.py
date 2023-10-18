from mindquantum import *
import os

os.environ['OMP_NUM_THREADS'] = '30'
import numpy as np
import mindspore.numpy as mnp
import mindspore.dataset as ds
from mindspore import Model
from mindspore.train.callback import LossMonitor
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from mindquantum.framework.operations import MQOps
from mindspore.common.initializer import Tensor
from scipy.sparse import csr_matrix

ms.set_seed(41)
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
INIT = [
    -1.1725, -0.16865, -0.28183, 0.21415, 0.43777, -1.3743, 0.051727, -1.5713,
    1.4696, -0.78479, 0.010443, -1.5812, 1.5703, 1.4696, 1.672, 1.5703,
    -1.5604, 3.1311, 1.4696, 0.56259, -1.6342, 1.2287, -0.18123, 0.36337,
    2.7782, 1.2287, -2.7715, -1.5643, 1.8428, 0.047711, 1.0912, -1.6168,
    1.5771, 1.8428, -1.5627, 1.5441, -0.59312, 1.2988, 1.5771, -1.5248, 3.0939,
    1.8428, 1.2827, -2.5485, 1.5441, 2.4645, -1.2474, 1.9535, 1.3997, -0.62123,
    -1.5718, 1.7482, -0.011642, 1.7419, 1.9535, -2.1676, -3.13, 1.7482,
    0.79609, 0.25703, 1.136, -2.1287, -1.5037, 1.4002, -0.74857, 2.6407,
    -0.4208, -2.393, 1.4002, -1.6049, 0.49578, -1.3704, 1.3704, -1.5689,
    1.5965, 0.14746, 1.3704, -2.7801, -1.7182, 1.5727, 1.5965, 1.6818, 1.5613,
    -1.4009, 1.28439129684566, 9.76103617672784, 2.9707, 1.4598, -2.177,
    -1.5768, 1.4639, -3.0577, 0.11268, -1.6828, 1.5648, 1.4639, 1.2267, 1.3053,
    -2.8813, -1.7373, 1.9023, -2.9443
]

P = INIT


class Main():
    def __init__(self):
        ms.set_seed(41)
        super().__init__()
        self.dataset = self.build_dataset(50)
        self.qnet = MyHybridCell()  # MQLayer(self.build_grad_ops())
        self.model = self.build_model()
        self.checkpoint_name = "./model.ckpt"

    def build_dataset(self, batch=None):
        x = np.load('train_x.npy', allow_pickle=True)
        y = np.load('train_y.npy', allow_pickle=True)
        y = np.concatenate((np.real(y), np.imag(y)), 1)
        # data = np.concatenate((x,y),1)
        # const = np.ones((len(data),1))
        train = ds.NumpySlicesDataset(
            {
                "data": x.reshape((x.shape[0], -1)),
                "label": y.astype(np.float32)
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
        for i in range(10):
            self.model.train(40, self.dataset, callbacks=LossMonitor(
                1))  # put into self.qnet.weight.asnumpy()

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        #print(qnet_weight)
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        predict = self.model.predict(ms.Tensor(test_x))
        pr = predict.asnumpy()
        pr = pr.astype(np.complex64)
        pr = pr[:, :8] + pr[:, 8:] * 1j
        np.save('test_y.npy', pr)
        return pr


class MyHybridCell(ms.nn.Cell):
    def __init__(self):
        self.N = 4
        super(MyHybridCell, self).__init__()
        self.evolution = MQOps1(self.qcnn_ops())
        self.weight_size = len(
            self.evolution.expectation_with_grad.ansatz_params_name)
        np.random.seed(200)
        init = Tensor(np.array(INIT), dtype=ms.float32)
        self.weight = Parameter(
            init, name='ansatz_weight'
        )  # initializer('normal', self.weight_size1, dtype=ms.float32)
        self.ite = 0

    def construct(self, x):
        # w2 = Parameter(self.weight.asnumpy()[self.weight_size1:].copy())
        # w1 = Parameter(self.weight.asnumpy()[:self.weight_size1].copy())
        # if self.count%1==0:
        #   print(self.weight.asnumpy())
        # self.count+=1
        x = self.evolution(x, self.weight)
        return x

    def generate_encoder(self):
        n_qubits = 3
        enc_layer = sum(
            [U3(f'a{i}', f'b{i}', f'c{i}', i) for i in range(n_qubits)])
        coupling_layer = UN(X, [1, 2, 0], [0, 1, 2])
        encoder = sum([
            add_prefix(enc_layer, f'l{i}') + coupling_layer for i in range(2)
        ])

        ansatz_layer = Circuit()+RZ('P0').on(0)+CNOT(0,1)+RZ('P1').on(0)+CNOT(0,2)+RZ('P2').on(0)+CNOT(0,1)+BARRIER\
        +RZ('P3').on(0)+RZ('P4').on(1)+CNOT(0,2)+RZ(-np.pi/2).on(0)+CNOT(1,2)+RY(np.pi/2).on(0)+RZ('P5').on(1)+CNOT(1,2)+CNOT(0,1)+RZ('P6').on(2)+BARRIER\
        +RZ(np.pi/2).on(0)+RY(np.pi/2).on(0)+RZ(-np.pi/2).on(0)+CNOT(0,2)+RZ(np.pi/2).on(0)+RY(np.pi/2).on(0)+RZ(-np.pi/2).on(0)+BARRIER\
        +CNOT(0,1)+RZ(np.pi).on(0)+RZ('P7').on(1)+RY(np.pi/2).on(0)+RY('P8').on(1)+RZ('P9').on(0)+RZ('P10').on(1)+CNOT(1,0)+BARRIER\
        +RZ('P11').on(1)+RY('P12').on(1)+RZ('P13').on(1)+CNOT(1,2)+RZ('P14').on(1)+BARRIER\
        +RY('P15').on(1)+RZ('P16').on(1)+CNOT(1,0)+RZ(-np.pi/2).on(0)+RZ('P17').on(1)+RY(np.pi/2).on(0)+RY('P18').on(1)+BARRIER\
        +RZ('P19').on(1)+CNOT(0,1)+RZ(-np.pi/2).on(0)+RY(np.pi/2).on(0)+RZ(np.pi/2).on(0)+CNOT(0,2)+BARRIER\
        +RZ(np.pi/2).on(0)+RY(np.pi/2).on(0)+RZ(-np.pi/2).on(0)+CNOT(0,1)+RZ(-np.pi).on(0)+RZ('P20').on(1)+RY(np.pi/2).on(0)+RY('P21').on(1)+BARRIER\
        +RZ('P22').on(0)+RZ('P23').on(1)+RZ(-np.pi/2).on(0)+CNOT(1,2)+RY(np.pi/2).on(0)+RZ('P24').on(1)+RY('P25').on(1)+BARRIER\
        +RZ('P26').on(1)+CNOT(0,1)+RZ(np.pi/2).on(0)+RY(np.pi/2).on(0)+RZ(-np.pi/2).on(0)+CNOT(0,2)+BARRIER\
        +RZ(np.pi/2).on(0)+RY(np.pi/2).on(0)+RZ(-np.pi/2).on(0)+RZ('P27').on(2)+RY('P28').on(2)+RZ('P29').on(2)+CNOT(0,1)+RZ(np.pi).on(0)+RY(np.pi/2).on(0)+BARRIER\
        +RZ('P30').on(0)+CNOT(2,0)+RZ('P31').on(2)+RY('P32').on(2)+RZ('P33').on(2)+CNOT(2,1)+BARRIER\
        +RZ('P34').on(1)+RY('P35').on(1)+RZ('P36').on(1)+RZ('P37').on(2)+RY('P38').on(2)+RZ('P39').on(2)+CNOT(2,0)+RZ(-np.pi/2).on(0)+RZ('P40').on(2)+BARRIER\
        +RY(np.pi/2).on(0)+RY('P41').on(2)+RZ('P42').on(2)+CNOT(1,2)+RZ('P43').on(1)+RY('P44').on(1)+RZ('P45').on(1)+BARRIER\
        +CNOT(0,1)+RZ(-np.pi/2).on(0)+RY(np.pi/2).on(0)+RZ(np.pi/2).on(0)+CNOT(0,2)+BARRIER\
        +RZ(np.pi/2).on(0)+RY(np.pi/2).on(0)+RZ(-np.pi/2).on(0)+RZ('P46').on(2)+RY('P47').on(2)+RZ('P48').on(2)+CNOT(0,1)+RZ(-np.pi).on(0)+CNOT(2,1)+BARRIER\
        +RY(np.pi/2).on(0)+RZ('P49').on(0)+RZ(-np.pi/2).on(0)+RY(np.pi/2).on(0)+RZ('P50').on(1)+RY('P51').on(1)+RZ('P52').on(1)+RZ('P53').on(2)+RY('P54').on(2)+RZ('P55').on(2)+CNOT(1,2) +BARRIER\
        + RZ('P56').on(1) + RY('P57').on(1) + RZ('P58').on(1)+CNOT(0,1)+RZ(-np.pi/2).on(0)+RY(np.pi/2).on(0)+RZ(np.pi/2).on(0)+BARRIER\
        +CNOT(0,2)+RZ('P59').on(0)+RY('P60').on(0)+RZ('P61').on(0)+RZ('P62').on(2)+RY('P63').on(2)+RZ('P64').on(2)+CNOT(0,1)+BARRIER\
        +RZ(-np.pi/2).on(0)+RY('P65').on(0)+RZ('P66').on(0)+RZ(-np.pi/2).on(1)+RY(np.pi/2).on(1)+CNOT(2,0)+BARRIER\
        +RZ('P67').on(2)+RY('P68').on(2)+RZ('P69').on(2)+CNOT(1,2)+RZ(np.pi).on(1) + RY(np.pi/2).on(1) + RZ('P70').on(1)+BARRIER\
        +RZ('P71').on(1) + RY('P72').on(1) + RZ(-np.pi/2).on(1)+CNOT(1,0)+BARRIER\
        +RZ('P73').on(0)+RY('P74').on(0)+RZ('P75').on(0)+RZ(-np.pi/2).on(1)+RY('P76').on(1)+RZ('P77').on(1)+CNOT(0,1) +BARRIER\
        +RZ('P78').on(0) + RY('P79').on(0) + RZ('P80').on(0)+CNOT(0,2)+RZ('P81').on(0) + RY('P82').on(0) + RZ('P83').on(0)+RZ(-np.pi).on(2)+RX(np.pi/2).on(2)+BARRIER\
        +CNOT(0,1)+RZ('P84').on(2)+RX(np.pi/2).on(2)+RZ('P85').on(2)+RZ('P86').on(0)+RZ(-np.pi/2).on(1)+BARRIER\
        +RY('P87').on(0)+RY(np.pi/2).on(1)+RZ('P88').on(0)+CNOT(1,2)+RZ('P89').on(0)+RY(np.pi/2).on(1)+RY('P90').on(0)+RZ('P91').on(1)+BARRIER\
        +RZ('P92').on(0)+CNOT(0,1)+RZ('P93').on(0)+RY('P94').on(0)+RZ('P95').on(0)+CNOT(0,2)+BARRIER\
        +RZ('P96').on(0)+RY('P97').on(0)+RZ('P98').on(0)+CNOT(0,1)+RZ('P99').on(0)+RY('P100').on(0)+RZ('P101').on(0)+BARRIER
        return encoder + ansatz_layer, encoder.params_name, ansatz_layer.params_name

    def qcnn_ops(self):
        ham = []
        for i in range(8):
            a = csr_matrix(([1], ([i], [i])), (8, 8)) * np.sqrt(8)
            ham.append(Hamiltonian(a))
        circ_right, encoder_params_name, ansatz_params_name = self.generate_encoder(
        )
        simulator_right = Simulator('mqvector', 3)
        simulator_left = Simulator('mqvector', 3)
        circ_left = Circuit().h(0).h(1).h(2)
        grad_ops = simulator_right.get_expectation_with_grad(
            ham,
            circ_right,
            circ_left,
            simulator_left,
            encoder_params_name,
            ansatz_params_name,
            parallel_worker=5)
        return grad_ops


class MQOps1(MQOps):
    def __init__(self, expectation_with_grad):
        super(MQOps1, self).__init__(expectation_with_grad)

    def construct(self, enc_data, ans_data):
        enc_data = enc_data.asnumpy()
        ans_data = ans_data.asnumpy()
        f, g_enc, g_ans = self.expectation_with_grad(enc_data, ans_data)
        f = f
        f = ms.Tensor(np.concatenate((np.real(f), np.imag(f)), 1),
                      dtype=ms.float32)
        self.g_enc = np.concatenate((np.real(g_enc), np.imag(g_enc)), 1)
        self.g_ans = np.concatenate((np.real(g_ans), np.imag(g_ans)), 1)
        return f

    def bprop(self, enc_data, ans_data, out, dout):
        dout = dout.asnumpy()
        enc_grad = np.einsum('smp,sm->sp', self.g_enc, dout)
        ans_grad = np.einsum('smp,sm->p', self.g_ans, dout)
        return ms.Tensor(enc_grad,
                         dtype=ms.float32), ms.Tensor(ans_grad,
                                                      dtype=ms.float32)


class myLoss(ms.nn.LossBase):
    def __init__(self, reduction='mean'):
        """Initialize myLoss."""
        super(myLoss, self).__init__(reduction)

    def construct(self, logits, label):
        a = logits[:, :8]
        b = logits[:, 8:]
        c = label[:, :8]
        d = label[:, 8:]
        real = mnp.sum(a * c, 1) + mnp.sum(b * d, 1)
        imag = mnp.sum(a * d, 1) - mnp.sum(b * c, 1)
        e = mnp.power(real, 2) + mnp.power(imag, 2)
        loss = self.get_loss(1 - e)
        return loss


if __name__ == '__main__':
    x = np.load('test_x.npy', allow_pickle=True)
    # y = np.load('train_y.npy', allow_pickle=True)
    a = Main()
    #a.train()
    yy = a.predict(x)
    # acc = np.real(
    #     np.mean([np.vdot(bra, ket) for bra, ket in zip(y, yy)]))
    # print(f"Acc: {acc}")

    # a.load_trained_parameters()
    # x = np.load('test_x.npy', allow_pickle=True)
    # y = a.predict(x)
    # y = y.asnumpy()
    # y = y.astype(np.complex64)
    # y = y[:,:8]+y[:,8:]*1j
    # np.save('test_y.npy',y)
