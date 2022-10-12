# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
from mindquantum.core import QubitOperator, Hamiltonian
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator
import mindspore as ms
import mindspore.ops as ops

from src.ansatz_qcnn import generate_qcnn
from src.loss import MySoftMarginLoss

class QCNNet:
    """
    Quantum Convolutional Neural Network.
    """
    def __init__(self, n_qubits, encoder):
        self.n_q = n_qubits
        self.qnet = MQLayer(self.build_grad_ops(encoder))
        self.build_model()
    def build_grad_ops(self, encoder):
        """Build grad_ops."""
        ansatz, qo = generate_qcnn(self.n_q)
        encoder = encoder.no_grad()
        circuit = encoder + ansatz
        ham = Hamiltonian(QubitOperator(f'Z{qo}'))
        sim = Simulator('projectq', circuit.n_qubits)
        grad_ops = sim.get_expectation_with_grad(ham,
                                                 circuit,
                                                 encoder_params_name=encoder.params_name,
                                                 ansatz_params_name=ansatz.params_name,
                                                 parallel_worker=5)
        return grad_ops
    def build_model(self):
        """Build model."""
        self.loss = MySoftMarginLoss()
        self.opti = ms.nn.Adam(self.qnet.trainable_params(), learning_rate=0.1)
        self.model = ms.Model(self.qnet, self.loss, self.opti)
    def train(self, epoch, train_loader, callbacks):
        """
        Train.
        """
        self.model.train(epoch,
                         train_loader,
                         callbacks=callbacks,
                         dataset_sink_mode=False)
    def export_trained_parameters(self, checkpoint_name):
        """Export trained parameters."""
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, checkpoint_name)
    def load_trained_parameters(self, checkpoint_name):
        """Load trained parameters."""
        ms.load_param_into_net(self.qnet, ms.load_checkpoint(checkpoint_name))
    def predict(self, origin_test_x):
        """
        Predict.
        """
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        predict = ops.Sign()(self.qnet(ms.Tensor(test_x)))
        predict = predict.flatten()
        return predict.asnumpy()
