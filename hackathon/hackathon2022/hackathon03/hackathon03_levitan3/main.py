import os

os.environ['OMP_NUM_THREADS'] = '2'

import numpy as np
import mindspore as ms
from mindspore.nn import Adam, Accuracy
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model
from mindspore.train.callback import LossMonitor, Callback
from mindquantum import *
from mindquantum.algorithm import HardwareEfficientAnsatz
from mindquantum.core import RY, RZ
from encoder_circuit import generate_encoder
import matplotlib.pyplot as plt

from mindspore import ops, Tensor
from sklearn.model_selection import train_test_split

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)


class Main():
    def __init__(self, dep=3, learning_rate=0.5, test_size=0.1):
        super().__init__()
        self.circ = None
        self.dep = dep
        self.learning_rate = learning_rate
        test_size = test_size
        self._raw_x = np.load('train_x.npy', allow_pickle=True)
        self._raw_y = np.load('train_y.npy', allow_pickle=True)
        self._raw_test_x = np.load('test_x.npy', allow_pickle=True)
        self.X, self.Y = self.trans_data(self._raw_x, self._raw_y)
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=0, shuffle=False)
        X_test[:, 18:] = 0
        self.train_x = X_train
        self.train_y = Y_train
        self.test_x = X_test
        self.test_y = self._raw_y[-int(test_size * len(self._raw_y)):]

        self.qnet = MQLayer(self.build_grad_ops())
        self.net = self.bulid_net()
        self.weight_name = os.path.join("weight.npy")
        self.test_y_name = os.path.join("test_y.npy")

    def trans_data(self, X, Y):
        trans_x = []
        trans_y = []
        for i, item in enumerate(Y):
            am, ths = self.convert_complex(item)
            encoder, parameterResolver = amplitude_encoder(np.abs(am), 3)
            am_pa = list(parameterResolver.values())
            th_pa = [
                ths[7] - ths[3], ths[5] - ths[1], ths[6] - ths[2],
                ths[4] - ths[0], (ths[2] + ths[6] - ths[0] - ths[4]) / 2,
                (ths[3] + ths[7] - ths[1] - ths[5]) / 2,
                ((ths[3] + ths[7] + ths[1] + ths[5]) / 2 -
                 (ths[0] + ths[2] + ths[4] + ths[6]) / 2) / 2
            ]
            trans_x.append(list(X[i]) + th_pa[::-1] + am_pa[::-1])
            trans_y.append([3])
        return np.array(trans_x).astype(np.float32), np.array(trans_y).astype(
            np.float32)

    def set_phase_para(self, circ, ths):
        theta = [
            ths[7] - ths[3], ths[5] - ths[1], ths[6] - ths[2], ths[4] - ths[0],
            (ths[2] + ths[6] - ths[0] - ths[4]) / 2,
            (ths[3] + ths[7] - ths[1] - ths[5]) / 2,
            ((ths[3] + ths[7] + ths[1] + ths[5]) / 2 -
             (ths[0] + ths[2] + ths[4] + ths[6]) / 2) / 2
        ]
        paras = dict(zip(circ.parameter_resolver().keys(), theta))
        return paras

    def expform(self, x):
        a = np.abs(x)
        imag = np.imag(x / a)
        real = np.real(x / a)
        th = np.arcsin(imag)
        if imag > 0 and real < 0:
            th = np.pi - th
        if imag < 0 and real < 0:
            th = -np.pi - th
        return a, th

    def convert_complex(self, y):
        amp = []
        the = []
        for item in y:
            a, t = self.expform(item)
            amp.append(a)
            the.append(t)
        return np.array(amp), np.array(the)

    def normal(self, state):
        return state / np.sqrt(np.abs(np.vdot(state, state)))

    def phase_coder(self, pname):
        qubits = 3
        encoder = Circuit()
        encoder += RZ(f'%s_0' % pname).on(2, [0, 1])
        encoder += X.on(1)
        encoder += RZ(f'%s_1' % pname).on(2, [0, 1])
        encoder += X.on(1)
        encoder += X.on(0)
        encoder += RZ(f'%s_2' % pname).on(2, [0, 1])
        encoder += X.on(1)
        encoder += RZ(f'%s_3' % pname).on(2, [0, 1])
        encoder += X.on(1)
        encoder += RZ(f'%s_4' % pname).on(1, 0)
        encoder += X.on(0)
        encoder += RZ(f'%s_5' % pname).on(1, 0)
        encoder += RZ(f'%s_6' % pname).on(0)
        return encoder

    def amplitude_coder(self, pname):
        qubits = 3
        encoder = Circuit()
        encoder += RY(f'%s_0' % pname).on(2)
        encoder += X.on(2)
        encoder += RY(f'%s_1' % pname).on(1, [2])
        encoder += X.on(2)
        encoder += RY(f'%s_2' % pname).on(1, [2])
        encoder += X.on(1)
        encoder += X.on(2)
        encoder += RY(f'%s_3' % pname).on(0, [1, 2])
        encoder += X.on(1)
        encoder += RY(f'%s_4' % pname).on(0, [1, 2])
        encoder += X.on(1)
        encoder += X.on(2)
        encoder += RY(f'%s_5' % pname).on(0, [1, 2])
        encoder += X.on(1)
        encoder += RY(f'%s_6' % pname).on(0, [1, 2])
        return encoder

    def build_grad_ops(self):
        inticoder, paras1 = generate_encoder()
        qubits = 3
        am_encoder, para_dict1 = amplitude_encoder([np.sqrt(1 / 8)] * 8, 3)
        phase_encoder = self.phase_coder('p')
        encoder = inticoder
        encoder = encoder.no_grad()
        print(encoder.summary())

        dep = self.dep
        ansatz = Circuit()

        for i in range(dep):
            ansatz += self.amplitude_coder('ansam_%s' % i)
            ansatz += self.phase_coder('ansth_%s' % i)
        print(ansatz.summary())

        end_coder = phase_encoder.hermitian() + am_encoder.hermitian()
        end_coder = end_coder.no_grad()
        end_paras = end_coder.params_name
        encoder_params_name = paras1 + end_paras
        total_circ = encoder.as_encoder() + ansatz + end_coder
        print(total_circ.parameter_resolver())
        print(encoder_params_name)
        self.circ = total_circ

        ham = [
            Hamiltonian(
                QubitOperator(f'Z0', -1) + QubitOperator(f'Z1', -1) +
                QubitOperator(f'Z2', -1))
        ]
        sim = Simulator('mqvector', total_circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(ham,
                                                 total_circ,
                                                 parallel_worker=16)
        return grad_ops

    def verify_circ(self, ansatz_values, X, Y=None, save=False):
        inticoder, paras1 = generate_encoder()
        qubits = 3
        phase_encoder = self.phase_coder('p')
        encoder = inticoder
        encoder = encoder.no_grad()
        dep = self.dep
        ansatz = Circuit()
        for i in range(dep):
            ansatz += self.amplitude_coder('ansam_%s' % i)
            ansatz += self.phase_coder('ansth_%s' % i)
        total_circ = encoder + ansatz
        para_names = total_circ.params_name
        sim = Simulator('mqvector', total_circ.n_qubits)
        acc_list = []
        y_hat_list = []
        for i, item in enumerate(X):
            trans_x = np.array(list(item) + ansatz_values.tolist())
            pr = dict(zip(para_names, trans_x))
            sim.reset()
            sim.apply_circuit(total_circ, pr)
            y_hat = sim.get_qs()
            y_hat_list.append(y_hat)
            if Y is not None:
                y = Y[i]
                acc = np.real(np.abs(np.vdot(self.normal(y_hat), y)))
                acc_list.append(acc)
        if len(acc_list) > 0:
            print('mean acc: %s' % np.mean(acc_list))
            return np.mean(acc_list)
        if save:
            np.save(self.test_y_name, np.array(y_hat_list))
            print(np.array(y_hat_list))
            print('save to %s' % self.test_y_name)

    def bulid_net(self):
        opti = ms.nn.Adam(self.qnet.trainable_params(),
                          learning_rate=self.learning_rate)
        self.net = ms.nn.TrainOneStepCell(self.qnet, opti)
        return self.net

    def train_net(self, steps):
        for i in range(steps):
            res = self.net(ms.Tensor(self.train_x))
            if i % 10 == 0:
                print(i, ': ', np.mean(np.matrix(res).T))
                ansatz_values = self.qnet.weight.asnumpy()
                self.verify_circ(ansatz_values, self._raw_x[-50:],
                                 self._raw_y[-50:])
        weight = self.qnet.weight.asnumpy()
        print('weight: %s' % weight)
        np.save(self.weight_name, weight)
        print('save to %s' % self.weight_name)
        return weight


if __name__ == '__main__':
    m = Main()
    print(m.circ.summary())
    m.circ.svg().to_file("circ.svg")

    weight = m.train_net(250)
    m.verify_circ(weight, m._raw_x, m._raw_y)
    m.verify_circ(weight, m._raw_test_x, save=True)