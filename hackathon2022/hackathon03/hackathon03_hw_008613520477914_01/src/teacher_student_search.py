import os

os.environ['OMP_NUM_THREADS'] = '5'
import numpy as np
import mindspore as ms
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model
from mindspore import nn, ops
from mindspore import Parameter
from mindspore.common.initializer import initializer

from mindspore.train.callback import LossMonitor
from mindspore.train.callback import Callback

from mindquantum.algorithm import HardwareEfficientAnsatz
from mindquantum import *
from sklearn.model_selection import train_test_split

import json

#ds.config.set_seed(1)
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

TRAIN_EPOCH = 5
TRAIN_BATCH = 16 
TRAIN_LR = 0.05
TEST_SPLIT_RATIO = 0.2
INI_WEIGHT = 'normal'

MSE_LOSS_RATIO = 0.6 # balance the mse loss and KL divergence loss

QUBIT_NUM = 3 # not tunable



def gen_possible_heuristic_circuit(qnum,dep,option):
    if option == 0:
        cir = HardwareEfficientAnsatz(qnum,[RX,RZ,RY],X,'linear',dep).circuit
    elif option == 1 :
        cir = HardwareEfficientAnsatz(qnum,[RZ,RY],X,'linear',dep).circuit
    elif option == 2:
        cir = HardwareEfficientAnsatz(qnum,[RX,RZ,RY],Z,'linear',dep).circuit
    elif option == 3:
        cir = HardwareEfficientAnsatz(qnum,[RZ,RY],Z,'linear',dep).circuit
    elif option == 4:
        cir = HardwareEfficientAnsatz(qnum,[RX,RZ,RY],X,'all',dep).circuit
    elif option == 5:
        cir = HardwareEfficientAnsatz(qnum,[RX,RZ],X,'all',dep).circuit
    elif option == 6:
        cir = HardwareEfficientAnsatz(qnum,[RX,RZ,RY],Z,'all',dep).circuit
    elif option == 7:
        cir = HardwareEfficientAnsatz(qnum,[RZ,RY],Z,'all',dep).circuit
    elif option == 8: 
        cir = HardwareEfficientAnsatz(qnum,[RX,RZ,RY],X,[(0,1),(1,2),(2,0)],dep).circuit
    elif option == 9:
         cir = HardwareEfficientAnsatz(qnum,[RZ,RY],X,[(0,1),(1,2),(2,0)],dep).circuit
    elif option == 10:
         cir = HardwareEfficientAnsatz(qnum,[RX,RZ,RY],Z,[(0,1),(1,2),(2,0)],dep).circuit
    elif option == 11:
         cir = HardwareEfficientAnsatz(qnum,[RZ,RY],Z,[(0,1),(1,2),(2,0)],dep).circuit
    return cir



class MQLayerDropout(nn.Cell):
    def __init__(self, expectation_with_grad, weight=INI_WEIGHT):
        super(MQLayerDropout, self).__init__()
        self.evolution = MQOps(expectation_with_grad)
        weight_size = len(
            self.evolution.expectation_with_grad.ansatz_params_name)
        self.weight = Parameter(initializer(weight,
                                            weight_size,
                                            dtype=ms.float32), name='ansatz_weight')
        
    def construct(self, x):
        # _, mask = self.dropout(self.weight)
        # self.weight = ops.mul(mask, self.weight)
        return self.evolution(x, self.weight)



def generate_encoder():
    n_qubits = QUBIT_NUM
    enc_layer = Circuit()
    for i in range(n_qubits):
        enc_layer += U3(f'a{i}', f'b{i}', f'c{i}', i)

    coupling_layer = UN(X, [1, 2, 0], [0, 1, 2])

    encoder = Circuit()
    for i in range(2):
        encoder += add_prefix(enc_layer, f'l{i}')
        encoder += coupling_layer
    return encoder, encoder.params_name


class KLDivLoss(nn.LossBase):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.add = ops.add
        self.mul = ops.mul
        self.div = ops.div
        self.log = ops.log
        self.softmax = ops.Softmax(-1)
        

    def construct(self, predict, target):
        target  = self.softmax(target)
        predict = self.softmax(predict)
        term_mid = self.log(self.div(target,predict))
        term_total = ops.reduce_sum(self.mul(target,term_mid),axis=-1)
        loss_val = ops.reduce_mean(term_total)    
        return loss_val
    
    
class TotalLoss(nn.LossBase):
    def __init__(self,use_kl_div=False):
        super(TotalLoss, self).__init__()
        self.mse = ms.nn.MSELoss()
        self.kl_div =  KLDivLoss()
        self.use_kl_div = use_kl_div

    def construct(self, predict, target):

        if not self.use_kl_div:
            # only use the mse loss;
            loss_val = self.mse(predict, target)
        else:
            # sum the kl_div and mse loss togeter;
            loss_val = MSE_LOSS_RATIO*self.mse(predict,target) + \
                        (1-MSE_LOSS_RATIO)*self.kl_div(predict,target) 
        return  loss_val

    
class StepAcc(Callback):                     
    def __init__(self, qnet, combine_circ, test_x, test_y, train_x, train_y):
        self.qnet = qnet
        self.combine_circ = combine_circ
        
        self.test_x = test_x
        self.test_y = test_y
        self.train_x = train_x
        self.train_y = train_y
        
        self.acc = []
        self.count=0

    def step_end(self, run_context):
        self.count += 1
        if self.count % 60 ==0:
            qnet_weights = self.qnet.weight.asnumpy()
            
            # eval on splited test data;
            predicted_states = []
            for item_x in self.test_x:
                params = list(item_x)
                params.extend(list(qnet_weights))
                predicted_states.append(self.combine_circ.get_qs(pr=params))

            predicted_states = np.array(predicted_states) 
            acc_val = eval_acc(predicted_states,self.test_y)
            print('mean acc on tests : ',acc_val)
            
            # eval on the training data;
            predicted_states = []
            for item_x in self.train_x:
                params = list(item_x)
                params.extend(list(qnet_weights))
                predicted_states.append(self.combine_circ.get_qs(pr=params))

            predicted_states = np.array(predicted_states) 
            acc_val = eval_acc(predicted_states,self.train_y)
            print('mean acc on train : ',acc_val)       
            
                
    
# transform state to hamiltonians 
def gen_transformed_hams_from_label(y):
    hamiltonians_features = []
    sim = Simulator('projectq', QUBIT_NUM)
    for i in range(len(y)):
        sim.set_qs(y[i])
        feature_single_state = []
        for i in range(3):
            ham_x = Hamiltonian(QubitOperator(f'X{i}'))
            ham_y = Hamiltonian(QubitOperator(f'Y{i}'))
            ham_z = Hamiltonian(QubitOperator(f'Z{i}'))
            singe_q = [np.real(sim.get_expectation(ham_x)), \
                       np.real(sim.get_expectation(ham_y)),\
                       np.real(sim.get_expectation(ham_z))]
            feature_single_state.extend(singe_q)
        sim.reset()
        hamiltonians_features.append(feature_single_state)
    hamiltonians_features = np.array(hamiltonians_features)
    return hamiltonians_features


class UCirApproximator():
    def __init__(self, depth, circuit_type=0,uc_tea=None):
        super().__init__()
        self.nqubits = QUBIT_NUM
        self.dep = depth
        self.cir_type = circuit_type
        self.uc_teacher = uc_tea
        self.combine_circ, self.encoder, self.ansatz = self.build_circuits()
        self.qnet = MQLayerDropout(self.build_grad_ops())
        self.model = self.build_model()
        saving_name = 'model_dep'+str(self.dep)+'_'+str(self.cir_type)+'.ckpt'
        self.checkpoint_name = os.path.join('./',saving_name)

        

    def build_dataset(self, input_file_x, state_file_y, batch=8):
        self.origin_x = np.load(input_file_x,allow_pickle=True) 
        self.origin_y = np.load(state_file_y,allow_pickle=True)

        self.X_train, self.X_test, self.y_train, self.y_test  = \
        train_test_split(self.origin_x, self.origin_y, test_size = TEST_SPLIT_RATIO)
        
        
        print('building the hams from given states ...')
        hams_features_train = gen_transformed_hams_from_label(self.y_train)
        train_data = ds.NumpySlicesDataset(
            {
                "image": self.X_train,
                "label": hams_features_train
            },
            shuffle=False)
        
        if batch is not None:
            train_data = train_data.batch(batch)
        
        print('dataset had been built ...')
        return train_data, self.X_train, self.X_test
    

    def build_grad_ops(self):
        hams = [Hamiltonian(QubitOperator("X0")),\
                Hamiltonian(QubitOperator("Y0")),\
                Hamiltonian(QubitOperator("Z0")),\
                Hamiltonian(QubitOperator("X1")),\
                Hamiltonian(QubitOperator("Y1")),\
                Hamiltonian(QubitOperator("Z1")),\
                Hamiltonian(QubitOperator("X2")),\
                Hamiltonian(QubitOperator("Y2")),\
                Hamiltonian(QubitOperator("Z2"))]
        
        sim = Simulator('projectq', self.combine_circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(
            hams,
            self.combine_circ,
            encoder_params_name = self.encoder.params_name,
            ansatz_params_name = self.ansatz.params_name,
            parallel_worker = 5)
        return grad_ops
    
 
    def build_circuits(self):
        encoder, _ = generate_encoder()
        qnum = self.nqubits
        dep = self.dep
        ansatz = gen_possible_heuristic_circuit(qnum,dep,self.cir_type)
        print('ansatz information ...')
        print(ansatz.summary())
        combine_circ = encoder + ansatz
        return combine_circ, encoder, ansatz
    

    def build_model(self):
        self.loss = TotalLoss(use_kl_div=True)
        self.opt = ms.nn.Adam(self.qnet.trainable_params(), learning_rate = TRAIN_LR)
        self.model = Model(self.qnet, self.loss, self.opt)
        return self.model

    
    def train(self,input_file_x, state_file_y, continue_from_existed_model=False):
        print('training start ...')
        if continue_from_existed_model:
            print('Training from existed model.ckpt')
            ms.load_param_into_net(self.qnet, ms.load_checkpoint('./model.ckpt'))
            self.model = self.build_model()
        else:
            print('Training from very scrach ...')
        self.train_data, self.X_train, self.X_test = self.build_dataset(input_file_x, state_file_y,TRAIN_BATCH)
        test_acc_callback = StepAcc(self.qnet,self.combine_circ, \
                                    self.X_test,self.y_test, self.X_train,self.y_train)
        self.model.train(TRAIN_EPOCH, self.train_data, callbacks=[LossMonitor(4),test_acc_callback])
        return self.X_test, self.y_test
        
        
    def train_as_student(self,bs,epochs):
        if self.uc_teacher is None:
            print('The teacher has not been assigned...')
            raise

        #  train from teacher
        print('training from my teacher ...')
        ms_train_dataset, x_teach_train, x_teach_test, \
        y_teach_train, y_teach_test = teacher_taught_data(800, self.uc_teacher)
        
        ms_train_dataset = ms_train_dataset.batch(bs)
        test_acc_callback = StepAcc(self.qnet,self.combine_circ, \
                                    x_teach_test, y_teach_test, x_teach_train, y_teach_train)
        self.model.train(epochs, ms_train_dataset, callbacks=[LossMonitor(4),test_acc_callback])
        return 

        
    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    
    def ansatz_predict_with_trained_model(self, test_x, model_file):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(model_file))
        
        # test_x size: 500 by 18
        predicted_states = []
        qnet_weights = self.qnet.weight.asnumpy()
        
        for item_x in test_x:
            params = list(item_x)
            params.extend(list(qnet_weights))
            predicted_states.append(self.combine_circ.get_qs(pr=params))
     
        return np.array(predicted_states)

    
def normal(state):
    return state / np.sqrt(np.abs(np.vdot(state, state)))


def eval_acc(pred_states, true_states):
    acc = np.real(
        np.mean([
            np.abs(np.vdot(normal(bra), ket))
            for bra, ket in zip(pred_states, true_states)
        ]))
    return acc

import time


def gen_rd_x_input(batch_size):
    random_x_input = (np.random.rand(batch_size,18)-0.5)*2*np.pi
    return random_x_input.astype(np.float32)


def teacher_taught_data(num, UC_appro_teacher):
    x_input = gen_rd_x_input(num)
    y_state_label = UC_appro_teacher.ansatz_predict_with_trained_model(x_input,'./model_09994.ckpt')
    x_teach_train, x_teach_test, y_teach_train, y_teach_test = train_test_split(x_input, y_state_label, \
                                                                                test_size=0.2)
    hams_features_train = gen_transformed_hams_from_label(y_teach_train)
    ms_train_dataset = train_data = ds.NumpySlicesDataset( {"image": x_teach_train, "label": hams_features_train},shuffle=False)
    
    return ms_train_dataset, x_teach_train, x_teach_test, y_teach_train, y_teach_test



if __name__ == "__main__":

    test_x = np.load('./test_x.npy',allow_pickle=True)
    train_x = np.load('./train_x.npy',allow_pickle=True)
    train_y = np.load('./train_y.npy',allow_pickle=True)

    # This is teacher UC 
    UC_appro_teacher = UCirApproximator(5)
    pred_y = UC_appro_teacher.ansatz_predict_with_trained_model(train_x,'./model_09994.ckpt')
    acc = eval_acc(pred_y, train_y)
    print(f"Teacher Acc on training data: {acc} ")


    # this is the student UC
    TRAIN_EPOCH = 4
    TRAIN_BATCH = 16
    
    circuit_depths = [1,2,3,4]
    cir_types = [0,1,2,3,4,5,6,7,8,9,10,11]
    
    acc_type = {}
    def teach_a_class_of_students():
        
        for dep in circuit_depths:
            for ctype in cir_types:
                
                print('dep: {0}; type {1}'.format(dep,ctype))
                key_str = str(dep) + '_' + str(ctype)
                UC_appro_student = UCirApproximator(dep, ctype, UC_appro_teacher)
                
                # train from "textbook"
                X_test_independent, y_test_independent = UC_appro_student.train('./train_x.npy', './train_y.npy',False)

                # train from teacher 
                for _ in range(5):
                    UC_appro_student.train_as_student(bs=16,epochs=2)
                UC_appro_student.export_trained_parameters()
                UC_appro_student.checkpoint_name
                
                pred_y = UC_appro_student.ansatz_predict_with_trained_model(train_x,UC_appro_student.checkpoint_name)
                acc_train = eval_acc(pred_y, train_y)
                print(f"Acc on textbook training: {acc_train}")

                pred_y_test = UC_appro_student.ansatz_predict_with_trained_model(X_test_independent,'./model.ckpt')
                acc_test = eval_acc(pred_y_test, y_test_independent)
                print(f"Acc on textbook independent tests: {acc_test}")
                
                acc_type[key_str] = acc_train
                
        with open('result.json','w') as f:
            json.dump(acc_type,f)
    
teach_a_class_of_students()
