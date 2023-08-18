import mindspore as ms
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindquantum import MQOps
from mindspore.ops import operations as P
import numpy as np
from mindquantum import Simulator
from ansatz_circuit import build_ansatz_a
from encoder_circuit import generate_encoder
from mindquantum import Hamiltonian, QubitOperator
# from main import batch
encoder, paras = generate_encoder()
encoder = encoder.no_grad()
ansatz = build_ansatz_a()
total_circ = encoder + ansatz
ham = Hamiltonian(QubitOperator(''))
batch = 10

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


class MyOps(nn.Cell):

    def __init__(self, expectation_with_grad):
        super(MyOps, self).__init__()
        self.expectation_with_grad = expectation_with_grad
        self.shape_ops = P.Shape()

    def extend_repr(self):
        return self.expectation_with_grad.str

    def construct(self, enc_data, ans_data):
        enc_data = enc_data.asnumpy()
        ans_data = ans_data.asnumpy()

        f, g_enc, g_ans = self.expectation_with_grad(enc_data, ans_data)
        f = ms.Tensor(np.real(f), dtype=ms.float32)
        self.g_enc = np.real(g_enc)
        self.g_ans = np.real(g_ans)

        # value = list()
        # sim = Simulator('projectq', 3)

        # ans_dict = dict(zip(ansatz.params_name, ans_data))

        # for i in range(batch):
        #     sim.reset()
        #     enc_data_i = enc_data[i]
        #     enc_dict = dict(zip(encoder.params_name, enc_data_i))
        #     pr = Merge(enc_dict, ans_dict)
        #     # print(pr)
        #     sim.apply_circuit(total_circ, pr=pr)
        #     value_i = sim.get_qs()
        #     value.append(value_i)

        # value = np.array(value)
        # value = ms.Tensor(value, dtype=ms.float32)

        return f

    def bprop(self, enc_data, ans_data, out, dout):
        dout = dout.asnumpy()
        enc_grad = np.einsum('smp,sm->sp', self.g_enc, dout)
        ans_grad = np.einsum('smp,sm->p', self.g_ans, dout)
        return ms.Tensor(-enc_grad,
                         dtype=ms.float32), ms.Tensor(-ans_grad,
                                                      dtype=ms.float32)

class QLayer(nn.Cell):

    def __init__(self, expectation_with_grad, weight='normal'):
        super(QLayer, self).__init__()
        self.evolution = MyOps(expectation_with_grad)
        weight_size = len(
            self.evolution.expectation_with_grad.ansatz_params_name)

        self.weight = Parameter(initializer(weight,
                                            weight_size,
                                            dtype=ms.float32),
                                name='ansatz_weight')

    def construct(self, x):
        return self.evolution(x, self.weight)