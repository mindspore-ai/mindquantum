# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
os.environ['OMP_NUM_THREADS'] = '2'
from hybrid import HybridModel
from hybrid import project_path
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor   
from mindspore.nn import SoftmaxCrossEntropyWithLogits  
from mindspore.nn import Adam, Accuracy     
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model, load_checkpoint
from mindspore.train.callback import Callback,LossMonitor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindquantum import *
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)

class Main(HybridModel):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = self.build_dataset(self.origin_x, self.origin_y, self.batch_size)
        self.qnet = MQLayer(self.build_grad_ops())
        self.model = self.build_model()
        self.checkpoint_name = os.path.join(project_path, "model.ckpt")
        
    def build_dataset(self, x, y, batch=None):
        train = ds.NumpySlicesDataset(
            {
                "image": x.reshape((x.shape[0], -1)),
                "label": y.astype(np.int32)
            }, shuffle=False) 
        if batch is not None:
            train = train.batch(batch)
        return train

    def build_grad_ops(self):
        
        encoder = Circuit()
        for i in range(8):
            encoder += RX(f'rx{i}').on(i)
            encoder += RY(f'ry{i}').on(i)
        encoder.no_grad()
        
        ansatz = Circuit()
        ansatz += BarrierGate()
        ansatz += conv_circ('c00',0,1)
        ansatz += conv_circ('c01',2,3)
        ansatz += conv_circ('c02',4,5)
        ansatz += conv_circ('c03',6,7)
        ansatz += BarrierGate()
        ansatz += conv_circ('c10',7,0)
        ansatz += conv_circ('c11',1,2)
        ansatz += conv_circ('c12',3,4)
        ansatz += conv_circ('c13',5,6)
        ansatz += BarrierGate()
        ansatz += pool_circ('p00',0,1)
        ansatz += pool_circ('p01',2,3)
        ansatz += pool_circ('p02',4,5)
        ansatz += pool_circ('p03',6,7)
        ansatz += BarrierGate()
        ansatz += conv_circ('c20',1,3)
        ansatz += conv_circ('c21',5,7)
        ansatz += BarrierGate()
        ansatz += conv_circ('c30',7,1)
        ansatz += conv_circ('c31',3,5)
        ansatz += BarrierGate()
        ansatz += pool_circ('p10',1,3)
        ansatz += pool_circ('p11',5,7)
        ansatz += BarrierGate()
        ansatz += conv_circ('c40',3,7)
        
        total_circ = encoder + ansatz
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [3,7]]
        sim = Simulator('projectq', total_circ.n_qubits)
        
        grad_ops = sim.get_expectation_with_grad(
            ham,
            total_circ,
            encoder_params_name=encoder.params_name,
            ansatz_params_name=ansatz.params_name,
            parallel_worker=5)
        return grad_ops

    def build_model(self):
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.opti = ms.nn.Adam(self.qnet.trainable_params())
        self.model = Model(self.qnet, self.loss, self.opti, metrics={'Acc': Accuracy()})
        return self.model

    def train(self, epoch=1):
        self.model.train(epoch, self.dataset, callbacks=[LossMonitor(1)], dataset_sink_mode=False)

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))
        

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        predict = np.argmax(ops.Softmax()(self.model.predict(ms.Tensor(test_x))), axis=1) 
        
        return predict
    
    
def conv_circ(prefix='0', bit_up=0, bit_down=1):
    _circ = Circuit()
    _circ += RY('theta00').on(bit_up)
    _circ += RY('theta01').on(bit_down)
    _circ += X.on(bit_down,bit_up)
    _circ = add_prefix(_circ, prefix)
    return _circ

def pool_circ(prefix='0', bit_up=0, bit_down=1):
    _circ = Circuit()
    _circ += RZ('theta0').on(bit_down,bit_up)
    _circ += X.on(bit_up)
    _circ += RX('theta1').on(bit_down,bit_up)
    _circ += X.on(bit_up)
    _circ = add_prefix(_circ, prefix)
    return _circ
