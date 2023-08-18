import os

os.environ['OMP_NUM_THREADS'] = '2'
from hybrid import HybridModel
from hybrid import project_path
import numpy as np
import mindspore as ms
from mindspore.nn import Adam, Accuracy
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model
from mindspore.train.callback import LossMonitor, Callback
from mindquantum import *
from mindquantum.algorithm import HardwareEfficientAnsatz         
from mindquantum.core import RY,RZ 
import matplotlib.pyplot as plt 

from mindspore import ops, Tensor  
from sklearn.model_selection import train_test_split  

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)   

class Main(HybridModel):
    def __init__(self):
        super().__init__()
        self.circ = None
        self.learning_rate=0.012 
        self.epoch = 2  
        self.batch = 128 
        X = self.origin_x.reshape((self.origin_x.shape[0], -1))
        Y = self.origin_y.astype(np.int32)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0, shuffle=True)
        self.train_x = X_train
        self.train_y = Y_train
        self.test_x = X_test
        self.test_y = Y_test
        self.dataset = self.build_dataset(self.train_x, self.train_y, self.batch)
        self.test_dataset = self.build_dataset(self.test_x, self.test_y, self.batch)
        self.qnet = MQLayer(self.build_grad_ops())
        self.model = self.build_model()
        self.checkpoint_name = os.path.join(project_path, "model.ckpt")
        

    def build_dataset(self, x, y, batch=None, shuffle=False):
        train = ds.NumpySlicesDataset(
            {
                "image": x,
                "label": y
            },
            shuffle=shuffle)
        if batch is not None:
            train = train.batch(batch)
        return train

    def build_grad_ops(self):
        encoder = Circuit()                              
        qubits = 8
        encoder += UN(H, qubits)                           
        for i in range(qubits):                                 
            encoder += RZ(f'alpha{i}').on(i)            
        for j in range(qubits-1):                       
            encoder += X.on(j+1, j)                   
            encoder += RZ(f'alpha{j+qubits}').on(j+1)           
            encoder += X.on(j+1, j)                         
        encoder += X.on(0, qubits-1)                        
        encoder += RZ(f'alpha{2*qubits-1}').on(0)            
        encoder += X.on(0, qubits-1) 
        encoder = encoder.no_grad()                     
       
        dep = 3
        ansatz = Circuit() 
        for i in range(dep):
            for j in range(qubits):
                ansatz += RY(f'beta{i*qubits+j}').on(j)
                ansatz += RZ(f'gamma{i*qubits+j}').on(j)
                ansatz += X.on((j+1)%qubits, j%qubits)
                
                ansatz += RY(f'betas{i*qubits+j}').on(j)
                ansatz += RZ(f'gammas{i*qubits+j}').on(j)
                ansatz += X.on((j+4)%qubits, j%qubits)

                ansatz += RY(f'betass{i*qubits+j}').on(j)
                ansatz += RZ(f'gammass{i*qubits+j}').on(j)
                ansatz += X.on((j+3)%qubits, j%qubits)

        for j in range(qubits):
            ansatz += RY(f'beta{dep*qubits+j}').on(j)
            ansatz += RZ(f'gamma{dep*qubits+j}').on(j)

        total_circ = encoder + ansatz
        self.circ = total_circ
      
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [qubits-2, qubits-1]] 
        sim = Simulator('projectq', total_circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(
            ham,
            total_circ,
            encoder_params_name=encoder.params_name,
            ansatz_params_name=ansatz.params_name,
            parallel_worker=16)
        return grad_ops

    def build_model(self):
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction='mean')
        self.opti = ms.nn.Adam(self.qnet.trainable_params(), learning_rate=self.learning_rate)
        self.model = Model(self.qnet, self.loss, self.opti, metrics={'Acc': Accuracy()})
        return self.model

    def train(self, num):
        self.model.train(num, self.dataset, dataset_sink_mode=False)

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        # predict = self.model.predict(ms.Tensor(test_x))
        predict = np.argmax(ops.Softmax()(self.model.predict(Tensor(test_x))), axis=1) 
        predict = predict.flatten() > 0
        # print(predict[:,0])
        # predict = predict.asnumpy().flatten() > 0
        return predict

if __name__ == '__main__':
    minst = Main()
    print(minst.circ.summary())
    # print(minst.circ)

    minst.train(minst.epoch)

    predict = minst.predict(minst.test_x)
    correct = minst.model.eval(minst.test_dataset, dataset_sink_mode=False)                   # 计算测试样本应用训练好的模型的预测准确率

    minst.export_trained_parameters()

    print(correct)
   