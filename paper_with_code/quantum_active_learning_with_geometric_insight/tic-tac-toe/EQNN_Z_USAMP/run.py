# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:05:11 2024
tictactoe, USAMP, entropy
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

#the very specefic QNN we take
L = 2
P = 5
network = 'tcemoidcemoidcemoidcemoidcemoidtcemoidcemoidcemoidcemoidcemoid'

#network = 'tcemoidcemoid'


# encoder circuit
def build_encoder():
    encoder = Circuit()
    encoder += RX(f'alpha{0}').on(0)
    encoder += RX(f'alpha{1}').on(1)
    encoder += RX(f'alpha{2}').on(2)
    encoder += RX(f'alpha{3}').on(3)
    encoder += RX(f'alpha{4}').on(4)
    encoder += RX(f'alpha{5}').on(5)
    encoder += RX(f'alpha{6}').on(6)
    encoder += RX(f'alpha{7}').on(7)
    encoder += RX(f'alpha{8}').on(8)
    encoder = encoder.no_grad()
    return encoder.as_encoder()


#write the QNN ansatz
def build_circuit(network):
    '''
    read the network config, including data encoding layer and permutation of CEMOID
    return the observation of three observables for classification
    '''
    count = 0
    circ = Circuit()
    for i in range(len(network)):
        if network[i] == 't':
            circ += build_encoder().as_encoder()
        elif network[i] == 'c':
            #corner single qubit
            ansatz = Circuit()
            ansatz += RX(f'theta{count}').on(0)
            ansatz += RX(f'theta{count}').on(2)
            ansatz += RX(f'theta{count}').on(6)
            ansatz += RX(f'theta{count}').on(8)
            count += 1
            ansatz += RY(f'theta{count}').on(0)
            ansatz += RY(f'theta{count}').on(2)
            ansatz += RY(f'theta{count}').on(6)
            ansatz += RY(f'theta{count}').on(8)
            count += 1
            ansatz.as_ansatz()
            circ += ansatz
        elif network[i] == 'e':
            #edge single qubit
            ansatz = Circuit()
            ansatz += RX(f'theta{count}').on(1)
            ansatz += RX(f'theta{count}').on(3)
            ansatz += RX(f'theta{count}').on(5)
            ansatz += RX(f'theta{count}').on(7)
            count += 1
            ansatz += RY(f'theta{count}').on(1)
            ansatz += RY(f'theta{count}').on(3)
            ansatz += RY(f'theta{count}').on(5)
            ansatz += RY(f'theta{count}').on(7)
            count += 1
            ansatz.as_ansatz()
            circ += ansatz
        elif network[i] == 'm':
            #middle single qubit
            ansatz = Circuit()
            ansatz += RX(f'theta{count}').on(4)
            count += 1
            ansatz += RY(f'theta{count}').on(4)
            count += 1
            ansatz.as_ansatz()
            circ += ansatz
        elif network[i] == 'o':
            #corner to neibouring edge
            ansatz = Circuit()
            ansatz += RY(f'theta{count}').on(1, 0)
            ansatz += RY(f'theta{count}').on(1, 2)
            ansatz += RY(f'theta{count}').on(5, 2)
            ansatz += RY(f'theta{count}').on(5, 8)
            ansatz += RY(f'theta{count}').on(7, 6)
            ansatz += RY(f'theta{count}').on(7, 8)
            ansatz += RY(f'theta{count}').on(3, 0)
            ansatz += RY(f'theta{count}').on(3, 6)
            count += 1
            ansatz.as_ansatz()
            circ += ansatz
        elif network[i] == 'i':
            #edges to middle
            ansatz = Circuit()
            ansatz += RY(f'theta{count}').on(4, 1)
            ansatz += RY(f'theta{count}').on(4, 3)
            ansatz += RY(f'theta{count}').on(4, 5)
            ansatz += RY(f'theta{count}').on(4, 7)
            count += 1
            ansatz.as_ansatz()
            circ += ansatz
        elif network[i] == 'd':
            #middle to corner
            ansatz = Circuit()
            ansatz += RY(f'theta{count}').on(0, 4)
            ansatz += RY(f'theta{count}').on(2, 4)
            ansatz += RY(f'theta{count}').on(6, 4)
            ansatz += RY(f'theta{count}').on(8, 4)
            count += 1
            ansatz.as_ansatz()
            circ += ansatz
        circ += BarrierGate()
    return circ


circ = build_circuit(network)

from mindquantum.core.operators import QubitOperator
from mindquantum.core.operators import Hamiltonian

ham_corner = Hamiltonian(0.25 * QubitOperator('Z0') +
                         0.25 * QubitOperator('Z2') +
                         0.25 * QubitOperator('Z6') +
                         0.25 * QubitOperator('Z8'))
ham_center = Hamiltonian(QubitOperator('Z4'))
ham_edge = Hamiltonian(0.25 * QubitOperator('Z1') +
                       0.25 * QubitOperator('Z3') +
                       0.25 * QubitOperator('Z5') + 0.25 * QubitOperator('Z7'))
hams = [ham_corner, ham_center, ham_edge]

import mindspore as ms
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator
from mindspore import nn, Tensor
from mindspore.nn import Adam, TrainOneStepCell, LossBase

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(2)
sim = Simulator('mqvector', circ.n_qubits)
grad_ops = sim.get_expectation_with_grad(hams, circ, parallel_worker=16)

def Param_ini():
    return ms.Tensor(np.random.uniform(-np.pi, np.pi, len(circ.ansatz_params_name)).astype(np.float32))

#generate the dataset to be fed to the QNN as rotation angles
def generate_dataset(data):
    x, y = [], []
    for i, j in data:
        x.append(i)
        if j == 1:
            y.append([1,-1, -1])
        elif j == 0:
            y.append([-1, 1, -1])
        elif j == -1:
            y.append([-1, -1, 1])
    return np.array(x).astype(np.float32) * 2 * np.pi / 3, np.array(y).astype(
        np.float32)

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
        y = F.square(y_pred - y)
        y = ops.sum(y, 1)
        y = ops.mean(y, 0)
        return y




NUM_ENSEMBLE = 40
N_label = 20
epochs = 200

#verification and test set are not changed, we dont need to load it over and over :)
val_x, val_y = generate_dataset(data_val)
val_x_ms, val_y_ms = ms.Tensor(val_x), ms.Tensor(val_y)
test_x, test_y = generate_dataset(data_test)
test_x_ms, test_y_ms = ms.Tensor(test_x), ms.Tensor(test_y)


best_performance = np.zeros([NUM_ENSEMBLE,N_label])


#we first Tensor the training pool
pool_x, pool_y = generate_dataset(training_pool)
pool_x_ms = ms.Tensor(pool_x)
#before training we have to write the function for evaluating the uncertainty on the training_pool
def USamp(param_opt,pool_x,query_status):
    QuantumNet = MQLayer(grad_ops, weight=param_opt)
    predict_pool = np.array(QuantumNet(pool_x_ms).asnumpy())    
    max_entropy = 0
    index = 0
    for i in range(len(pool_x)):
        y0 = predict_pool[i][0]
        y1 = predict_pool[i][1]
        y2 = predict_pool[i][2]
        p0 = np.exp(y0)/(np.exp(y0)+np.exp(y1)+np.exp(y2))
        p1 = np.exp(y1)/(np.exp(y0)+np.exp(y1)+np.exp(y2))
        p2 = np.exp(y2)/(np.exp(y0)+np.exp(y1)+np.exp(y2))
        entropy = -(p0*np.log(p0) + p1*np.log(p1) +p2*np.log(p2))
        if entropy >= max_entropy and query_status[i] == 0:
            max_entropy = entropy
            index = i
    print('sample',index,'is queried, its max entropy is', max_entropy)
    return index




for i in range(NUM_ENSEMBLE):
    #in each ensemble, we should initialize the query status
    query_status = np.zeros(len(pool_x))
    training_set = []
    for j in range(N_label):
        #we have to evaluate the uncertainty based on the model we trained on the last training set
        #if nothing was trained, we have to initalize the model by evaluating its initial parameters
        if j == 0:
            #nothing is trained
            p0 = Param_ini()
            index = USamp(p0, pool_x, query_status)
            query_status[index] = 1
        else:
            #a model was trained on the last training set
            param_opt = copy.copy(QuantumNet.weight) #last model
            index = USamp(param_opt, pool_x, query_status) #USamp for index
            query_status[index] = 1
        #now we should enlarge the training set
        training_set.append(training_pool[index])
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
        
        #now we have to initialize the model for training
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
            pred_y = -(np.argmax(pred_y,axis=1)-1)

            corr = np.mean(pred_y == -(np.argmax(val_y,axis=1)-1))
            if corr >= best:
                pred_test_y = np.array(QuantumNet(test_x_ms).asnumpy())
                pred_test_y = -(np.argmax(pred_test_y,axis=1)-1)
                test_best = np.mean(pred_test_y == -(np.argmax(test_y,axis=1)-1))
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
        print('The queries sample is',index)
        best_performance[i,j] = test_best

with open('EQNN-Z_entropy.pickle', 'wb') as f:
    pickle.dump(best_performance, f)
    
















































