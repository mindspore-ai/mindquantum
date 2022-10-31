from base64 import encode
import os

# os.environ['OMP_NUM_THREADS'] = '12'
from abc import ABC, abstractmethod
project_path = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import mindspore as ms
from mindspore.nn import Adam, Accuracy
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model
from mindspore.train.callback import Callback
from mindquantum import *
from mindquantum.algorithm import HardwareEfficientAnsatz         
from mindquantum.core import RY, RZ, RX, XX, YY, ZZ 
import matplotlib.pyplot as plt 

from mindspore import ops, Tensor  
from sklearn.model_selection import train_test_split  

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)   

class QCNN(object):
    '''Quantum Convolutional Neural Network Class

    
    '''
    def __init__(self,qubits, closed=True, learning_rate=0.001, epoch=8, batch=8, opt = False):
        self.circ = None
        self.learning_rate=learning_rate
        self.epoch = epoch 
        self.batch = batch 
        self.qubits = qubits
        self.closed = closed 
        self.opt = opt
        X, Y = self.load_data()
        Xext, Yext = self.data_ext(X, Y, 10)
        X_train, X_test, Y_train, Y_test = train_test_split(Xext, Yext, test_size=0.2, random_state=0, shuffle=True)
        self.train_x = X_train
        self.train_y = Y_train
        self.test_x = X_test
        self.test_y = Y_test
        self.dataset = self.build_dataset(self.train_x, self.train_y, self.batch)
        self.test_dataset = self.build_dataset(self.test_x, self.test_y, self.batch)
        self.qnet = MQLayer(self.build_grad_ops())
        self.model = self.build_model()
        self.checkpoint_name = os.path.join(project_path, "model.ckpt")
    
    def load_data(self):
        '''Load data from local folder: ``tfi_chain/spin_systems/TFI_chain/closed``
        '''
        gamma_list = [i/100.0 for i in range(20,182,2)]
        xlist = []
        ylist = []
        for i in gamma_list:
            f = 'tfi_chain/spin_systems/TFI_chain/closed/%s/%.2f/params.npy'%(self.qubits, i)
            data = np.load(f).T
            xlist.append(data.reshape(-1))
            if i > 1:
                ylist.append(1)
            else:
                ylist.append(0)
        print('Load data finished.')
        return np.array(xlist), np.array(ylist)
    
    def data_ext(self, X, Y, factor=10):
        '''Enhance the original data based on linear interpolation
        
        Args:
            X: array or list, the original data X
            Y: array or list, the original label data Y
            factor: int, data size enhancement factor, default to 10.

        Returns:
            extended data Xext, Yext
        '''
        Xext = []
        Yext = []
        gamma_list = [i/100.0 for i in range(20,182,2)]
        for i in range(1, len(X)):
            dx = (X[i]-X[i-1])/(factor)
            dy = (gamma_list[i]-gamma_list[i-1])/(factor)
            for j in range(factor):
                Xext.append(X[i-1] + j*dx)
                if gamma_list[i-1] + j*dy > 1:
                    Yext.append(1)
                else:
                    Yext.append(0)
        Xext.append(X[-1])
        if gamma_list[i-1] > 1:
            Yext.append(1)
        else:
            Yext.append(0) 
        print('Extended Data shape: %s, %s, factor: %s'%(np.array(Xext).shape, np.array(Yext).shape, factor))
        return np.array(Xext), np.array(Yext)


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

    def gen_encoder(self):
        '''Generate encoder circuit: 
        based on the qubits of ``qcnn``, the ``layers`` is set to ``int(qubits/2)``, each layer of encoder is construct by ``ZZ`` and ``RX`` gates. 
            
        Returns:
            encoder circuit

        Examples:
            >>> import src.qcnn as qcn
            >>> qcnn = qcn.QCNN(qubits=4, learning_rate=0.001, epoch=8, batch=8, opt = False)
            >>> encoder = qcnn.gen_encoder()
            >>> print(encoder)
            q0: ──H────ZZ(alpha_0_0)────RX(alpha_0_1)──────────────────────────────────────ZZ(alpha_0_0)────ZZ(alpha_1_0)────RX(alpha_1_1)──────────────────────────────────────ZZ(alpha_1_0)───────────────────
                             │                                                                   │                │                                                                   │
            q1: ──H────ZZ(alpha_0_0)────ZZ(alpha_0_0)────RX(alpha_0_1)───────────────────────────┼──────────ZZ(alpha_1_0)────ZZ(alpha_1_0)────RX(alpha_1_1)───────────────────────────┼─────────────────────────
                                              │                                                  │                                 │                                                  │
            q2: ──H─────────────────────ZZ(alpha_0_0)────ZZ(alpha_0_0)────RX(alpha_0_1)──────────┼───────────────────────────ZZ(alpha_1_0)────ZZ(alpha_1_0)────RX(alpha_1_1)──────────┼─────────────────────────
                                                               │                                 │                                                  │                                 │
            q3: ──H──────────────────────────────────────ZZ(alpha_0_0)─────────────────────ZZ(alpha_0_0)────RX(alpha_0_1)─────────────────────ZZ(alpha_1_0)─────────────────────ZZ(alpha_1_0)────RX(alpha_1_1)──

        '''
        qubits = self.qubits
        encoder = Circuit()  
        encoder += UN(H, qubits) 
        layers = int(qubits/2)
        for l in range(layers):
            for i in range(qubits-1):
                encoder += ZZ(f'alpha_{l}_0').on([i,i+1])
                encoder += RX(f'alpha_{l}_1').on(i)  
            encoder += ZZ(f'alpha_{l}_0').on([0,qubits-1])
            encoder += RX(f'alpha_{l}_1').on(qubits-1) 
        encoder = encoder.no_grad() 
        return encoder

    def q_convolution(self, label, qubits):
        '''Generate 2 qubits convolution circuit: given by two qubits index, generate the convolution block which is constructed by ``RX,RY,RZ,XX,YY,ZZ`` gates.  
        
        Args:
            label: str, name of the convolution block
            qubits: array or list, the qubits index list of the convolution block

        Returns:
            convolution circuit
        
        Examples:
            >>> conv = qcnn.q_convolution('0',[0,1])
            >>> print(conv)
            q0: ──RX(cov_0_0)────RY(cov_0_1)────RZ(cov_0_2)────XX(cov_0_6)────YY(cov_0_7)────ZZ(cov_0_8)────RX(cov_0_9)─────RY(cov_0_10)────RZ(cov_0_11)──
                                                                    │              │              │
            q1: ──RX(cov_0_3)────RY(cov_0_4)────RZ(cov_0_5)────XX(cov_0_6)────YY(cov_0_7)────ZZ(cov_0_8)────RX(cov_0_11)────RY(cov_0_12)────RZ(cov_0_13)──

        '''
        count = 0
        circ = Circuit()
        for i in range(2):
            circ += RX(f'cov_{label}_{count}').on(qubits[i])
            count+=1
            circ += RY(f'cov_{label}_{count}').on(qubits[i])
            count+=1
            circ += RZ(f'cov_{label}_{count}').on(qubits[i])
            count+=1
        circ += XX(f'cov_{label}_{count}').on(qubits)
        count+=1
        circ += YY(f'cov_{label}_{count}').on(qubits)
        count+=1
        circ += ZZ(f'cov_{label}_{count}').on(qubits)
        count+=1
        for i in range(2):
            circ += RX(f'cov_{label}_{count}').on(qubits[i])
            count+=1
            circ += RY(f'cov_{label}_{count}').on(qubits[i])
            count+=1
            circ += RZ(f'cov_{label}_{count}').on(qubits[i])
        return circ

    def q_pooling(self, label, qubits, last = False):
        '''Generate 2 qubits pooling circuit: given by two qubits index, generate the pooling block which is constructed by ``RX,RY,RZ,X`` gates. 
        When optimized settings are taken (``self.opt`` is set to ``True``), the pooling circuit is optimized.
        
        Args:
            label: str, name of the pooling block
            qubits: array or list, the qubits index list of the pooling block
            last: bool, used to determine whether the pooling block is the last one

        Returns:
            pooling circuit

        Examples:
            >>> pool = qcnn.q_pooling('0',[0,1])
            >>> print(pool)
            q0: ──RX(p_0_0)────RY(p_0_1)────RZ(p_0_2)────●────────────────────────────────────────────
                                                         │
            q1: ──RX(p_0_3)────RY(p_0_4)────RZ(p_0_5)────X────RZ(-p_0_5)────RY(-p_0_4)────RX(-p_0_3)──

        '''
        count = 0
        circ = Circuit()
        if not self.opt:
            for i in range(2):
                circ += RX(f'p_{label}_{count}').on(qubits[i])
                count+=1
                circ += RY(f'p_{label}_{count}').on(qubits[i])
                count+=1
                circ += RZ(f'p_{label}_{count}').on(qubits[i])
                count+=1
        circ += X.on(qubits[1], qubits[0])
        if count != 0:
            count-=1
        if (self.opt and last) or (not self.opt):
            circ += RZ(f'p_{label}_{count}').on(qubits[1]).hermitian()
            count-=1
            circ += RY(f'p_{label}_{count}').on(qubits[1]).hermitian()
            count-=1
            circ += RX(f'p_{label}_{count}').on(qubits[1]).hermitian()
        return circ

    def split_qlist(self, qlist):
        '''Generate the index list of convolution and pooling block: given by the qubits index list, generate the fisrt and the second qubit index for convolution and pooling block.
        
        Args:
            qlist: array or list, the qubits index list

        Returns:
            flist: the list of the fisrt qubit index 
            slist: the list of the second qubit index 
        '''
        n = len(qlist)
        flist = []
        slist = []
        if n>1:
            for i in range(int(n/2)):
                flist.append(qlist[2*i])
                slist.append(qlist[2*i+1])
        if n%2 == 1:
            slist.append(qlist[-1])
        return flist, slist

    def gen_qcnn_ansatz(self):
        '''Generate the ansatz circuit of qcnn: based on the ``q_convolution, q_pooling`` functions, generate ansatz circuit.
        
        Returns:
            circ: ansatz circuit
        '''
        qubits = self.qubits
        assert qubits%2==0
        qlist = list(range(qubits))
        circ = Circuit()
        flist, slist = self.split_qlist(qlist)
        count = 0
        while len(flist)>0:
            if len(flist)==1 and len(slist) == 1:
                last = True
            else:
                last = False
            for i in range(len(flist)):
                q = [flist[i], slist[i]]
                circ += self.q_convolution('blo_%s'%count, q)
                count+=1
                circ += self.q_pooling('blo_%s'%count, q, last)
                count+=1
            flist, slist = self.split_qlist(slist)
        return circ

    def build_grad_ops(self):
        '''Generate the total qcnn circuit, the Hamiltonian operator and build the grad ops wrapper.
        
        Returns:
            grad_ops: the grad ops wrapper
        '''
        encoder = self.gen_encoder()
        ansatz = self.gen_qcnn_ansatz()
        total_circ = encoder + ansatz
        self.circ = total_circ
        qubits = self.qubits
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [int(qubits/2)-1,qubits-1]] 
        sim = Simulator('projectq', total_circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(
            ham,
            total_circ,
            encoder_params_name=encoder.params_name,
            ansatz_params_name=ansatz.params_name,
            parallel_worker=8)
        return grad_ops

    def build_model(self):
        '''Set the loss function, optimizer and build the qcnn model.
        '''
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction='mean')
        self.opti = ms.nn.Adam(self.qnet.trainable_params(), learning_rate=self.learning_rate, beta1=0.9, beta2=0.99)
        self.model = Model(self.qnet, self.loss, self.opti, metrics={'Acc': Accuracy()})
        return self.model

    def train(self, num, callbacks=None):
        '''Training the model.
        
        Args:
            num: int, the epoch of training
            callbacks: list, the list of callbacks
        '''
        self.model.train(num, self.dataset, dataset_sink_mode=False, callbacks=callbacks)

    def export_trained_parameters(self):
        '''Export the parameters of model and save model.
        '''
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        '''Load model parameters by checkpoint data.
        '''
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        predict = np.argmax(ops.Softmax()(self.model.predict(Tensor(test_x))), axis=1) 
        predict = predict.flatten() > 0
        return predict

class StepAcc(Callback):                                                        # 定义一个关于每一步准确率的回调函数
    def __init__(self, model, test_loader, qnet, qubits,opt):
        self.model = model
        self.qnet = qnet
        self.qubits = qubits
        self.opt = opt
        self.test_loader = test_loader
        self.acc = []

    def step_end(self, run_context):
        """
        Record training accuracy and save model at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        self.acc.append(self.model.eval(self.test_loader, dataset_sink_mode=False)['Acc'])
        if self.acc[-1] >0.98:
            print('save model, %s'%self.acc[-1])
            ms.save_checkpoint(self.qnet, "res/model_%.2f_%s_%s.ckpt"%(self.acc[-1],self.qubits, self.opt))

class LossMonitor(Callback):
    
    def __init__(self, per_print_times=1):
        super(LossMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("The argument 'per_print_times' must be int and >= 0, "
                             "but got {}".format(per_print_times))
        self._per_print_times = per_print_times
        self._last_print_time = 0
        self._loss = []
        self._count = 0

    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = float(np.mean(loss.asnumpy()))

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num, cur_step_in_epoch, loss), flush=True)
            self._loss.append([self._count*self._per_print_times, loss])
            self._count += 1

if __name__ == '__main__':
    qubits = 12
    opt = True
    qc = QCNN(qubits, learning_rate=0.01, opt=opt)
    print(qc.circ.summary())
    print(qc.circ)
    monitor = LossMonitor(5)                                                       
    accu = StepAcc(qc.model, qc.test_dataset, qc.qnet, qubits, opt)
    qc.train(6, [monitor,accu])

    correct = qc.model.eval(qc.test_dataset, dataset_sink_mode=False)                  

    print(correct)
    plt.figure()
    plt.plot(accu.acc)
    plt.title('Statistics of accuracy', fontsize=20)
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.grid(ls=":",c='b')
    plt.savefig('acc_%s_%s.png'%(qubits,opt), format='png')
    plt.figure()
    monitor._loss = np.array(monitor._loss)
    plt.plot(monitor._loss[:,0], monitor._loss[:,1])
    plt.title('Statistics of loss', fontsize=20)
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.grid(ls=":",c='b')
    plt.savefig('loss_%s_%s.png'%(qubits, opt), format='png')
