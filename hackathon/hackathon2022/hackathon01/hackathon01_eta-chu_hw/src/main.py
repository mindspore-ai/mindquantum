import os
from typing import List

os.environ['OMP_NUM_THREADS'] = '1'
from hybrid import HybridModel
from hybrid import project_path
import numpy as np
import mindspore as ms
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model
from mindspore.train.callback import LossMonitor
from mindquantum import *
from mindspore.nn import Accuracy

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)


def str2gate(circ,
             gate_name,
             param_name,
             obj_qubits,
             gate_type,
             ctrl_qubits=None,
             ctrl=False):
    if not (gate_type in {"single_qubit", "two_qubit"}):
        raise ValueError(
            "The gate_type should be string \"single_qubit\" or \"two_qubit\" "
        )
    if ctrl and (ctrl_qubits is None):
        raise ValueError("We need ctrl qubit")
    if not (gate_name
            in ["CRX", "CRY", "CRZ", "XX", "YY", "ZZ", "RX", "RY", "RZ"]):
        raise ValueError("gate_name must be one of CRX, CRY, ...")

    if gate_type == "two_qubit":

        if ctrl:
            if gate_name == "CRX":
                circ += RX(param_name).on(obj_qubits, ctrl_qubits)
            elif gate_name == "CRY":
                circ += RY(param_name).on(obj_qubits, ctrl_qubits)
            elif gate_name == "CRZ":
                circ += RZ(param_name).on(obj_qubits, ctrl_qubits)

        else:
            if gate_name == "XX":
                circ += XX(param_name).on(obj_qubits)
            elif gate_name == "YY":
                circ += YY(param_name).on(obj_qubits)
            elif gate_name == "ZZ":
                circ += ZZ(param_name).on(obj_qubits)

    elif gate_type == "single_qubit":
        if gate_name == "RX":
            circ += RX(param_name).on(obj_qubits)
        elif gate_name == "RY":
            circ += RY(param_name).on(obj_qubits)
        elif gate_name == "RZ":
            circ += RZ(param_name).on(obj_qubits)

    return circ


def column_rebuild(gatelist, column_index):
    column = Circuit()
    bit = len(gatelist)
    start_index = (column_index - 1) * 2 * bit
    for i in range(bit):
        column = str2gate(column, gatelist[i], f"theta{start_index + i}", i,
                          "single_qubit")

    return column


def block_rebuild(gatelist, block_index):
    block = Circuit()
    bit = len(gatelist)
    start_index = ((block_index * 2) - 1) * bit
    for i in range(bit):

        if gatelist[i] in ["CRX", "CRY", "CRZ"]:
            ctrl = True
            block = str2gate(block, gatelist[i], f"theta{start_index + i}", i,
                             "two_qubit", (i + block_index) % bit, ctrl)
        else:
            block = str2gate(block, gatelist[i], f"theta{start_index + i}",
                             [i, (i + block_index) % bit], "two_qubit")

    return block


def rebuild_circuit(gatelist):
    circ = Circuit()

    if len(gatelist) % 2:
        block_number = (len(gatelist) - 1) / 2
    else:
        block_number = len(gatelist) / 2
    block_number = int(block_number)

    for i in range(block_number):
        column_circ = column_rebuild(gatelist[2 * i], i + 1)
        circ += column_circ
        circ += BarrierGate()

        block_circ = block_rebuild(gatelist[2 * i + 1], i + 1)
        circ += block_circ
        circ += BarrierGate()

    column_circ = column_rebuild(gatelist[-1], block_number + 1)
    circ += column_circ

    return circ


def amplitude_encode_preprocessing(x):
    m = x.shape[0]
    x = x.reshape([m, -1])
    new_origin_x = np.zeros(shape=(m, 8))
    for i in range(m):
        for j in range(0, 16, 2):
            if x[i, j] == x[i, j + 1]:
                if x[i, j] == 0:
                    new_origin_x[i, int(j / 2)] = -(3 / 2) * np.pi
                else:
                    new_origin_x[i, int(j / 2)] = np.pi / 2
            else:
                if x[i, j] == 0:
                    new_origin_x[i, int(j / 2)] = (3 / 2) * np.pi
                else:
                    new_origin_x[i, int(j / 2)] = -np.pi / 2

    return new_origin_x


def rebuild_hams(hamslist, param_coeff, bit=4, block_number=3):
    for i in range(2):
        for j in range(bit):
            if j == 0:
                hams = QubitOperator(hamslist[i][j], param_coeff[i][j])
            else:
                hams += QubitOperator(hamslist[i][j], param_coeff[i][j])

        if i == 0:
            hamiltonian = [Hamiltonian(hams)]
        else:
            hamiltonian += [Hamiltonian(hams)]

    return hamiltonian


class Main(HybridModel):
    def __init__(self):
        super().__init__()
        self.dataset = self.build_dataset(self.origin_x, self.origin_y, 500)
        self.qnet = MQLayer(self.build_grad_ops())
        self.model = self.build_model()
        self.checkpoint_name = os.path.join(project_path, "model.ckpt")
        self.ansatz = self.build_ansatz()
        self.encoder = self.build_encoder()

    def build_dataset(self, x, y, batch=None):
        x = amplitude_encode_preprocessing(x)
        train = ds.NumpySlicesDataset(
            {
                "image": x.reshape((x.shape[0], -1)),
                "label": y.astype(np.int32)
            },
            shuffle=True)
        if batch is not None:
            train = train.batch(batch)
        return train

    def build_ansatz(self):
        gatelist = [['RZ', 'RX', 'RZ', 'RY'], ['XX', 'CRZ', 'XX', 'CRZ'],
                    ['RX', 'RZ', 'RZ', 'RZ'], ['YY', 'CRZ', 'XX', 'YY'],
                    ['RX', 'RX', 'RZ', 'RX'], ['YY', 'YY', 'CRY', 'CRX'],
                    ['RZ', 'RY', 'RX', 'RZ']]
        ansatz = rebuild_circuit(gatelist)
        return ansatz

    def build_encoder(self):
        encoder = Circuit()
        encoder += UN(H, [1, 2, 3])

        encoder += UN(X, [1, 2, 3])
        encoder += RY(f"alpha0").on(0, [1, 2, 3])
        encoder += UN(X, [1, 2, 3])

        encoder += UN(X, [2, 3])
        encoder += RY("alpha1").on(0, [1, 2, 3])
        encoder += UN(X, [1, 2, 3])

        encoder += X.on(3)
        encoder += RY("alpha2").on(0, [1, 2, 3])
        encoder += UN(X, [1, 3])

        encoder += X.on(3)
        encoder += RY("alpha3").on(0, [1, 2, 3])
        encoder += UN(X, [1, 2, 3])

        encoder += RY("alpha4").on(0, [1, 2, 3])
        encoder += UN(X, [1, 2])

        encoder += X.on(2)
        encoder += RY("alpha5").on(0, [1, 2, 3])
        encoder += UN(X, [1, 2])

        encoder += RY("alpha6").on(0, [1, 2, 3])
        encoder += X.on(1)
        encoder += RY("alpha7").on(0, [1, 2, 3])

        return encoder

    def build_grad_ops(self):
        ansatz = self.build_ansatz()
        encoder = self.build_encoder()
        total_circ = encoder.as_encoder() + ansatz
        hamslist = [['X0', 'X1', 'X2', 'Y3'], ['X0', 'Y1', 'X2', 'Y3']]
        coeff = np.array([[
            0.3276474403203895, 0.10726372340486279, 0.06452448848520097,
            0.5005643477895467
        ],
                          [
                              0.19388376821752198, 0.27979629199475337,
                              0.264718577267549, 0.2616013625201757
                          ]])
        ham = rebuild_hams(hamslist, coeff)

        sim = Simulator('mqvector', total_circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(ham,
                                                 total_circ,
                                                 None,
                                                 None,
                                                 parallel_worker=5)
        return grad_ops

    def build_model(self):
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True,
                                                        reduction="mean")
        self.opti = ms.nn.Adam(self.qnet.trainable_params(),
                               learning_rate=0.05)
        self.monitor = LossMonitor(self.dataset.get_dataset_size())
        self.model = Model(self.qnet, self.loss, self.opti)
        return self.model

    def train(self, epoch):
        self.model.train(epoch,
                         self.dataset,
                         callbacks=self.monitor,
                         dataset_sink_mode=False)

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x):
        origin_test_x = amplitude_encode_preprocessing(origin_test_x)
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        predict = self.model.predict(ms.Tensor(test_x))
        result = []
        for i in range(predict.asnumpy().shape[0]):
            if predict.asnumpy()[i, 0] > predict.asnumpy()[i, 1]:
                result.append(0)
            else:
                result.append(1)
        return result