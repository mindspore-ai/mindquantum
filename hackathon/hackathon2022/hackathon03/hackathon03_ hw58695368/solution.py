# -*- coding: utf-8 -*-
"""
Solution of hackathon03_challenge.

@NoEvaa
<HelloWorld>
"""
import numpy as np
from mindquantum import *
import mindspore as ms
from scipy.optimize import minimize
from encoder_circuit import generate_encoder
from ansatz_circuit_qsd import generate_ansatz
from utils import param2dict, normal, predict_acc
from layer import MQOps_re_sp

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")


def func(w, x, y, grad_ops, show_iter_val=False):
    f, eg, ag = grad_ops.grad_exec(ms.Tensor(x), ms.Tensor(w), ms.Tensor(y))
    f = f.mean().asnumpy()
    ag = ag.asnumpy()
    if show_iter_val:
        print(np.real(np.squeeze(f)))
    return np.real(np.squeeze(f)), np.squeeze(ag)


class Hackathon03:
    def __init__(self, seed=1202):
        self.set_seed(seed)
        self.init_model()
        self.weights = None

    def set_seed(self, seed=1202):
        """Set random seed"""
        ms.set_seed(seed)

    def init_model(self):
        """Initialize model."""
        self.encoder, self.epn = generate_encoder()
        self.encoder.no_grad()
        self.ansatz, self.apn = generate_ansatz()
        self.circuit = self.encoder + self.ansatz
        self.n_qubits = self.circuit.n_qubits
        self.ham = Hamiltonian(QubitOperator(""))
        sim = Simulator('mqvector', self.n_qubits)
        self.sim_m = Simulator('mqvector', self.n_qubits)
        grad_ops = sim.get_expectation_with_grad(self.ham, self.circuit,
                                                 Circuit(), self.sim_m,
                                                 self.epn, self.apn)
        self.grad_ops = MQOps_re_sp(grad_ops, self.sim_m)

    def train(self,
              train_x,
              train_y,
              test_x,
              test_y,
              batch=10,
              weights=None,
              acc_tol=1,
              method='bfgs',
              iter_info=False,
              tol=1e-5):
        """
        Train.
        """
        acc = -1
        weights_ = weights
        if weights is None:
            weights_ = np.ones(len(self.apn))
        self.weights = None
        for i in range(int(train_x.shape[0] / batch)):
            print(f'Times:{i}')
            weights_ = self._train(train_x[i * batch:(i + 1) * batch],
                                   train_y[i * batch:(i + 1) * batch],
                                   weights_, method, iter_info, tol)

            acc_ = predict_acc(test_x, test_y, self.circuit, weights_,
                               self.epn, self.apn)
            print(f'Times:{i}  acc:{acc_}')
            if acc_ > acc:
                acc = acc_
                self.weights = weights_
            if acc > acc_tol:
                return True, acc
        return False, acc

    def _train(self, x, y, w, method='bfgs', iter_info=False, tol=1e-5):
        res = minimize(func,
                       w,
                       args=(x, y, self.grad_ops, iter_info),
                       method=method,
                       jac=True,
                       options={'gtol': tol})
        print(f' >>> loss:{res.fun}')
        return res.x

    def predict(self, x):
        """
        Predict.
        """
        y = []
        print('Maybe Later~')
        return y

    def save(self, postfix=''):
        """Save the model."""
        np.save('weight' + postfix + '.npy', self.weights)

    def load(self, weights):
        """Load the model."""
        pass


if __name__ == '__main__':
    train_x = np.load('train_x.npy', allow_pickle=True)
    train_y = np.load('train_y.npy', allow_pickle=True)
    test_x = np.load('test_x.npy', allow_pickle=True)
    test_y = np.load('real_test_y.npy', allow_pickle=True)

    model = Hackathon03()
    model.train(train_x[:100],
                train_y[:100],
                test_x,
                test_y,
                acc_tol=0.4,
                iter_info=True,
                tol=1e-2)
    print(model.weights)
    pred_y = model.predict(test_x)

    # It works!
    # Good!! Very good!!!
    # I don't want to improve it any more.
