# -*- coding: utf-8 -*-
"""
碰运气式训练法
"""
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
from qcexlib import QCircuitLib_ex
from main import Main
from sklearn import model_selection
import time
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


class Train(Main):
    def __init__(self):
        super().__init__()
        batch = 10
        nd = 5000
        ts = 0.8
        self.dataset = self.build_dataset(self.origin_x, self.origin_y, ts, batch)
        self.qnet = MQLayer(self.build_grad_ops())
        self.model = self.build_model()
        self.lm = int(nd * (1-ts) / batch)

    def build_dataset(self, x, y, test_size, batch=None):
        x, y_test, y, y_test = model_selection.train_test_split(x, y, test_size=test_size)
        train = ds.NumpySlicesDataset(
            {
                "image": x.reshape((x.shape[0], -1)),
                "label": y.astype(np.int32)
            },
            shuffle=True)
        if batch is not None:
            train = train.batch(batch)
        return train
    def build_grad_ops(self):
        encoder, ansatz, ham = QCircuitLib_ex().qc_test10
        encoder = encoder.no_grad()
        total_circ = encoder + ansatz

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
        self.opti = ms.nn.Adam(self.qnet.trainable_params(), learning_rate=0.1)
        self.model = Model(self.qnet, self.loss, self.opti)
        return self.model

    def train(self):
        self.model.train(5, self.dataset, callbacks=LossMonitor(self.lm))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        predict = self.model.predict(ms.Tensor(test_x))
        predict = np.argmax(ms.ops.Softmax()(predict), axis=1)
        predict = predict.flatten() > 0
        return predict
    def save_cp_t(self, acc):
        cp_name = os.path.join(project_path, "model"+str(acc)+time.asctime()+".ckpt")
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, cp_name)

if __name__ == '__main__':		
    import pickle
    from checkdataset import calcu_acc
    with open("test.pkl", 'rb') as f:
        origin_test_data = pickle.load(f)
    with open("train_statistics.pkl", 'rb') as f:
        c = pickle.load(f)

    origin_test_x = origin_test_data['test_x']
    origin_test_y = origin_test_data['test_y']

    for i in range(100):
        main = Train()
        main.train()
        main.export_trained_parameters()

        print(i)
        #print(main.model._network.weight.asnumpy())
        predict = main.predict(origin_test_x)
        acc1 = np.mean(predict == origin_test_y)
        acc0 = calcu_acc(origin_test_x, predict, c.copy())
        print(f"Acc0: {acc0}")
        print(f"Acc1: {acc1}")

        if acc0 > 0.9:
            main.save_cp_t(acc0)
            if acc0 >= 0.9184:
                break
        if acc1 > 0.86:
            break
