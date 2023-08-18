import os

os.environ['OMP_NUM_THREADS'] = '2'
from hybrid import HybridModel
from hybrid import project_path
import numpy as np
import mindspore as ms
import mindspore.context as context
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Model, ops, Tensor
from mindspore.train.callback import Callback, LossMonitor
from mindquantum import *

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

np.random.seed(233)
ms.set_seed(22)

class SaveModel(Callback):
    def __init__(self, model, qnet, small_test_loader, test_loader):
        self.model = model
        self.qnet = qnet
        self.small_test_loader = small_test_loader
        self.test_loader = test_loader
        self.bestacc = 0
        self.th1 = 0.89
    
    def step_end(self, run_context):
        acc = self.model.eval(
            self.small_test_loader, dataset_sink_mode=False)['Acc']
        print('th1', self.th1, 'small current best acc:', acc)
        if acc <= self.th1: # too low accurancy
            return
        self.th1 += 0.001
        acc = self.model.eval(
            self.test_loader, dataset_sink_mode=False)['Acc']
        print('big current best acc:', acc)
        if acc > 0.89 and acc > self.bestacc:
            self.bestacc = acc
            self.save_parameters()
        

    def save_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        name = f"model_{str(int(self.bestacc*10000))}.cpppt"
        ms.save_checkpoint(self.qnet, name)



class Main(HybridModel):
    def __init__(self):
        super().__init__()
        self.dataset = self.build_dataset(self.origin_x, self.origin_y, 16)
        self.small_dataset = self.build_small_dataset(self.origin_x, self.origin_y, 16)
        self.qnet = MQLayer(self.build_grad_ops())
        self.model = self.build_model()
        self.checkpoint_name = os.path.join(project_path, "model.ckpt")
        self.model_saver = SaveModel(self.model, self.qnet, self.small_dataset, self.dataset)

    def build_small_dataset(self, x, y, batch=16):
        x = x[-200:]
        y = y[-200:]
        train = ds.NumpySlicesDataset(
            {
                "image": x.reshape((x.shape[0], -1)),
                "label": y.astype(np.int32)
            },
            shuffle=True).batch(batch)
        return train

    def build_dataset(self, x, y, batch=None):
        # only 500 samples to test
        # x = x[:3000]
        # y = y[:3000]

        train = ds.NumpySlicesDataset(
            {
                "image": x.reshape((x.shape[0], -1)),
                "label": y.astype(np.int32)
            },
            shuffle=False)
        if batch is not None:
            train = train.batch(batch)
        return train

    def build_grad_ops(self):
        circ = Circuit()
        for i in range(8):
            circ += RY(f'p{i}').on(i)

        encoder = add_prefix(circ, 'e1') + add_prefix(circ, 'e2')
        ansatz = add_prefix(circ, 'a1') + add_prefix(circ, 'a2') #+ add_prefix(circ, 'a3') 
        total_circ = add_prefix(circ, 'e1') + add_prefix(circ, 'a1')
        for i in range(6):
            total_circ += X.on(6, i)
            total_circ += X.on(7, i)
        total_circ += add_prefix(circ, 'e2') + add_prefix(circ, 'a2')
        for i in range(6):
            total_circ += X.on(6, i)
            total_circ += X.on(7, i)

        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [6,7]]
        sim = Simulator('projectq', total_circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(
            ham,
            total_circ,
            encoder_params_name=encoder.params_name,
            ansatz_params_name=ansatz.params_name,
            parallel_worker=5)
        return grad_ops

    def build_model(self):
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.opti = ms.nn.Adam(self.qnet.trainable_params(), learning_rate=0.1)
        self.model = Model(self.qnet, self.loss, self.opti, metrics={'Acc': nn.Accuracy()})
        return self.model

    def train(self):
        self.model.train(2, self.dataset, callbacks=[LossMonitor(), self.model_saver])

    def export_parameters(self, n):
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name + str(n) + "_.ckpt")

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        predict = np.argmax(ms.ops.Softmax()(self.model.predict(Tensor(test_x))), axis=1)
        predict = predict.flatten() > 0
        return predict


    def get_all_accurancy(self) -> float:
        x = self.origin_x
        y = self.origin_y
        pred_y = self.predict(x)
        acc = np.sum(y == pred_y) / len(x)
        return acc

    def get_accurancy(self) -> float:
        x = self.origin_x[:1000]
        y = self.origin_y[:1000]
        test_y = self.predict(x)

        print(x.shape)
        print(np.array(test_y)[0:5])
        acc = np.sum(y == test_y) / len(x)
        print('Accurancy on train data:', acc)



if __name__ == '__main__':
    m = Main()
    m.load_trained_parameters()
    m.train()
    m.export_trained_parameters()
    # m.get_accurancy()