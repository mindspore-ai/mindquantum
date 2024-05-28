# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 20:12:52 2024
Z2 symmetry BEL test, random sampling
@author: jonze
"""
import math
import numpy as np
import random
import pickle
import copy
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, X, RX, RY, RZ, ZZ, CNOT, BarrierGate
import matplotlib.pyplot as plt
import matplotlib

np.random.seed(1)
#load sets from the folder

with open('training_set.pickle', 'rb') as f:
    training_pool = pickle.load(f)
with open('val_set.pickle', 'rb') as f:
    data_val = pickle.load(f)
with open('test_set.pickle', 'rb') as f:
    data_test = pickle.load(f)

#BEL-Z Encoder
def build_encoder():
    encoder = Circuit()
    encoder +=  RX(f'alpha{0}').on(0)
    encoder +=  RX(f'alpha{1}').on(1)
    encoder +=  RY(f'alpha{0}').on(0)
    encoder +=  RY(f'alpha{1}').on(1)
    encoder = encoder.no_grad()
    return encoder.as_encoder()

#BEL-Z Ansatz 
def build_circuit():
    circ = Circuit()
    circ += build_encoder().as_encoder()
    ansatz = Circuit()
    ansatz += RX(f'theta{0}').on(0)
    ansatz += RX(f'theta{1}').on(1)
    ansatz += CNOT.on(1,0)
    ansatz.as_ansatz()
    ansatz += BarrierGate()
    circ += ansatz
    
    ansatz = Circuit()
    ansatz += RX(f'theta{2}').on(0)
    ansatz += RX(f'theta{3}').on(1)
    ansatz += CNOT.on(1,0)
    ansatz.as_ansatz()
    ansatz += BarrierGate()
    circ += ansatz
    
    ansatz = Circuit()
    ansatz += RX(f'theta{4}').on(0)
    ansatz += RX(f'theta{5}').on(1)
    ansatz += CNOT.on(1,0)
    ansatz.as_ansatz()
    ansatz += BarrierGate()
    circ += ansatz
    
    ansatz = Circuit()
    ansatz += RX(f'theta{6}').on(0)
    ansatz += RX(f'theta{7}').on(1)
    ansatz += CNOT.on(1,0)
    ansatz.as_ansatz()
    ansatz += BarrierGate()
    circ += ansatz
    
    ansatz = Circuit()
    ansatz += RX(f'theta{8}').on(0)
    ansatz += RX(f'theta{9}').on(1)
    ansatz += CNOT.on(1,0)
    ansatz.as_ansatz()
    ansatz += BarrierGate()
    circ += ansatz
    
    ansatz = Circuit()
    ansatz += RX(f'theta{10}').on(0)
    ansatz += RX(f'theta{11}').on(1)
    ansatz += CNOT.on(1,0)
    ansatz.as_ansatz()
    ansatz += BarrierGate()
    circ += ansatz
    


    
    return circ

circ = build_circuit()
from mindquantum.core.operators import QubitOperator
from mindquantum.core.operators import Hamiltonian
ham = [Hamiltonian(0.5*QubitOperator('Z0')+0.5*QubitOperator('Z1'))]

import mindspore as ms
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator
from mindspore import nn, Tensor
from mindspore.nn import Adam, TrainOneStepCell, LossBase

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(2)
sim = Simulator('mqvector', circ.n_qubits)
grad_ops = sim.get_expectation_with_grad(ham, circ, parallel_worker=16)

def Param_ini():
    return ms.Tensor(np.random.uniform(-np.pi, np.pi, len(circ.ansatz_params_name)).astype(np.float32))

def generate_dataset(data):
    #we only look on the Z0, Z0<0 class -1, Z0 >0 class 1
    x, y = [], []
    for i, j in data:
        x.append(i)
        if j == 1:
            y.append([1])
        else:
            y.append([0])
    return np.array(x).astype(np.float32)*np.pi/2 , np.array(y).astype(np.float32)

import mindspore.dataset as ds
import mindspore as ms
import mindspore.ops as ops
from mindspore.ops import functional as F


class LossNet(ms.nn.Cell):

    def __init__(self, qnet):
        super(LossNet, self).__init__()
        self.qnet = qnet

    def construct(self, x, y):
        y_pred = self.qnet(x)
        # angle = F.arccos(y_pred)/2
        # pred_0 = (F.cos(angle))**2
        # pred_1 = (F.sin(angle))**2
        # pred_0 = (F.cos(angle))**2
        # pred_1 = (F.sin(angle))**2
        pred_0 = (y_pred+1)/2
        entropy = -(y*F.log(pred_0) + (1-y)*F.log(1-pred_0))
        # print(entropy)
        y = ops.mean(entropy)    
        #y = F.square(y_pred-y)
        y = F.mean(y)
        return y

def Sgn(x):
    for i in range(len(x)):
        if x[i] >= 0:
            x[i] = 1
        else:
            x[i] = -1
    return x


NUM_ENSEMBLE = 40
N_label = 20
epochs = 100

#verification and test set are not changed, we dont need to load it over and over :)
val_x, val_y = generate_dataset(data_val)
val_x_ms, val_y_ms = ms.Tensor(val_x), ms.Tensor(val_y)
test_x, test_y = generate_dataset(data_test)
test_x_ms, test_y_ms = ms.Tensor(test_x), ms.Tensor(test_y)


best_performance = np.zeros([NUM_ENSEMBLE,N_label])
for i in range(NUM_ENSEMBLE):
    #in each ensemble we should randomize a list of samples to be picked from the pool
    index_to_be_queried = np.random.randint(0,len(training_pool),N_label)
    training_set = []
    for j in range(N_label):
        training_set.append(training_pool[index_to_be_queried[j]])
        train_x, train_y = generate_dataset(training_set) #the difference between run.py and test.py
        train_x_ms, train_y_ms = ms.Tensor(train_x), ms.Tensor(train_y)
        batch_size = len(training_set)
        train_loader = ds.NumpySlicesDataset({
            'x': train_x,
            'y': train_y,
        }, shuffle=True).batch(batch_size)
        val_loader = ds.NumpySlicesDataset({
            'x': val_x,
            'y': val_y
        }).batch(batch_size)
        test_loader = ds.NumpySlicesDataset({
            'x': test_x,
            'y': test_y
        }).batch(batch_size)
        
        #now we train the QNN
        p0 = Param_ini() # initialize the parameter
        QuantumNet = MQLayer(grad_ops, weight=p0)
        loss_net = LossNet(QuantumNet)
        opti = ms.nn.Adam(loss_net.trainable_params(), learning_rate=0.1)
        train_net = ms.nn.TrainOneStepCell(loss_net, opti)
        #the QNN is initialized for each new size of training set
        corr = []
        acc_list = []
        loss_list = []
        #now we start training, we check the performance on the verification set, and record the best model for testing
        best = 0
        test_best = 0
        for epoch in range(epochs):
            for batch, (data,label) in enumerate(train_loader.create_tuple_iterator()):
                loss = train_net(data, label).asnumpy()
            pred_y = np.array(QuantumNet(val_x_ms).asnumpy())
            
            pred_y = Sgn(pred_y)
            corr = np.mean(pred_y == ((val_y-0.5)*2))
            if corr >= best:
                pred_test_y = np.array(QuantumNet(test_x_ms).asnumpy())
                pred_test_y = Sgn(pred_test_y)
                test_best = np.mean(pred_test_y == ((test_y-0.5)*2))
                best = corr
                
                ###PROBLEM: it doesnt record the best QuantumNet but QuantumNet at the end, overfitting.
                param_opt = copy.copy(QuantumNet.weight)
                best_num = epoch
                #print('the best quantum net is in epoch',epoch)
                
            acc_list.append(corr)
            loss_list.append(loss)
            #print(pred_test_y)
            #print(f"epoch: {epoch} loss: {loss:>7f} on verification corr: {corr:>7f} [{batch:>3d}/{batch_size:>3d}]")
        print('After all, the test set correct rate is ',test_best,'from the epoch',best_num,'with',j+1,'samples labeled in the',i+1,'ensemble')
        print('The queries sample is',index_to_be_queried[j])
        best_performance[i,j] = test_best
            

        
with open('BEL-Z_RS.pickle', 'wb') as f:
    pickle.dump(best_performance, f)
       




































