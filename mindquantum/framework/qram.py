# import numpy as np

# from mindquantum import *
# import mindspore as ms
# ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
# psi_0 = random_state((4, 8), norm_axis=1)

# sim = Simulator('mqvector', 3)
# encoder = Circuit().rx('e0', 0).rx('e1', 0).rx('e2', 1).rx('e3', 2)
# encoder.as_encoder()
# ansatz = Circuit().rx('a', 0).rx('b', 1).rx('c', 2)
# ansatz.as_ansatz()

# ham = Hamiltonian(QubitOperator('Z0 Z1 Z2'))
# grad_ops1 = sim.get_expectation_with_grad(ham, ansatz)

# p0 = np.random.uniform(-3, 3, len(ansatz.params_name))

# def qram_nn(qs,p0, grad_ops, sim):
#     sim.set_qs(qs[0])
#     f, g = grad_ops(p0)
#     for qs_ in qs[1:]:
#         sim.set_qs(qs_)
#         f_, g_ = grad_ops(p0)
#         f = np.append(f, f_, axis=0)
#         g = np.append(g, g_, axis=0)
#     return f, g

# # f, g = qram(psi_0, p0, grad_ops, sim)


# grad_ops2 = sim.get_expectation_with_grad(ham, encoder + ansatz)

# f, ge, ga = grad_ops2(psi_0.T, p0)

import mindspore as ms

# net = MQLayer(grad_ops2)
import numpy as np
from mindspore import context, nn
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations


class QRamOps(nn.Cell):
    def __init__(self, expectation_with_grad):
        super().__init__()
        self.expectation_with_grad = expectation_with_grad
        self.shape_ops = operations.Shape()
        self.g_ans = None
        self.g_enc = None
        self.dim = 1 << self.expectation_with_grad.sim.n_qubits

    def extend_repr(self):
        """Extend string representation."""
        return self.expectation_with_grad.str

    def construct(self, enc_data_r, enc_data_i, ans_data):
        """Construct an MQOps node."""
        enc_data = enc_data_r.asnumpy() + enc_data_i.asnumpy() * 1j
        self.expectation_with_grad.sim.set_qs(enc_data[0])
        fval, g_ans = self.expectation_with_grad(ans_data.asnumpy())
        for qs in enc_data[1:]:
            self.expectation_with_grad.sim.set_qs(qs)
            fval_, g_ans_ = self.expectation_with_grad(ans_data.asnumpy())
            fval = np.append(fval, fval_, axis=0)
            g_ans = np.append(g_ans, g_ans_, axis=0)
        self.g_ans = np.real(g_ans)
        self.g_enc = ms.Tensor(np.zeros(shape=enc_data.shape), dtype=ms.float32)
        return ms.Tensor(np.real(fval), dtype=ms.float32)

    def bprop(self, enc_data_r, enc_data_i, ans_data, out, dout):  # pylint: disable=unused-argument
        """Implement the bprop function."""
        dout = dout.asnumpy()
        ans_grad = np.einsum('smp,sm->p', self.g_ans, dout)
        return self.g_enc, self.g_enc, ms.Tensor(ans_grad, dtype=ms.float32)


class QRamLayer(nn.Cell):  # pylint: disable=too-few-public-methods
    def __init__(self, expectation_with_grad, weight='normal'):
        """Initialize a MQLayer object."""
        super().__init__()
        self.evolution = QRamOps(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        if isinstance(weight, ms.Tensor):
            if weight.ndim != 1 or weight.shape[0] != weight_size:
                raise ValueError(f"Weight init shape error, required ({weight_size}, ), but get {weight.shape}.")
        self.weight = Parameter(initializer(weight, weight_size, dtype=ms.float32), name='ansatz_weight')

    def construct(self, psi_r, psi_i):
        """Construct a MQLayer node."""
        return self.evolution(psi_r, psi_i, self.weight)


from mindspore.nn import SoftmaxCrossEntropyWithLogits

from mindquantum import *

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
psi_0 = random_state((50, 4), norm_axis=1)
psi_r = ms.Tensor(psi_0.real, dtype=ms.complex64)
psi_i = ms.Tensor(psi_0.imag, dtype=ms.complex64)

sim = Simulator('mqvector', 2)
ham = Hamiltonian(QubitOperator('Z0'))
ansatz = HardwareEfficientAnsatz(2, [RX, RY], depth=3).circuit
p_given = np.random.uniform(-3, 3, len(ansatz.params_name))
p_given_tensor = ms.Tensor(p_given, dtype=ms.float32)
y = []
for i in psi_0:
    sim.set_qs(i)
    y.append(sim.get_expectation(ham, ansatz, pr=dict(zip(ansatz.params_name, p_given))).real)
y = ms.Tensor(y, dtype=ms.float32)[:, None]

grad_ops = sim.get_expectation_with_grad(ham, ansatz)

qram = QRamLayer(grad_ops)
opti = ms.nn.Adam(qram.trainable_params(), learning_rate=0.1)
loss_fun = nn.MSELoss()


def forward_fn(psi_r, psi_i, label):
    logits = qram(psi_r, psi_i)
    loss = loss_fun(logits, label)
    return loss, logits


qram.set_train()

import mindspore

grad_fn = mindspore.ops.value_and_grad(forward_fn, None, opti.parameters, has_aux=True)


def train_step(psi_r, psi_i, label):
    (loss, _), grads = grad_fn(psi_r, psi_i, label)
    loss = mindspore.ops.depend(loss, opti(grads))
    return loss


def train(psi_r, psi_i, label, epochs):
    for epoch in range(epochs):
        loss = train_step(psi_r, psi_i, label)
        print(f"epoch: {epoch}, loss: {loss}")


train(psi_r, psi_i, y, 1000)
