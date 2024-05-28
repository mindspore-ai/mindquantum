# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 21:14:55 2024
test the BEL-Z
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
#use mindspore ds to batch and shuffle the data
train_x, train_y = generate_dataset(training_pool)
train_x_ms, train_y_ms = ms.Tensor(train_x), ms.Tensor(train_y)
val_x, val_y = generate_dataset(data_val)
val_x_ms, val_y_ms = ms.Tensor(val_x), ms.Tensor(val_y)
test_x, test_y = generate_dataset(data_test)
test_x_ms, test_y_ms = ms.Tensor(test_x), ms.Tensor(test_y)
batch_size = 300
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

p0 = Param_ini()
QuantumNet = MQLayer(grad_ops, weight=p0)
loss_net = LossNet(QuantumNet)
opti = ms.nn.Adam(loss_net.trainable_params(), learning_rate=0.1)
train_net = ms.nn.TrainOneStepCell(loss_net, opti)
epochs = 100
corr = []
acc_list = []
loss_list = []

def Sgn(x):
    for i in range(len(x)):
        if x[i] >= 0:
            x[i] = 1
        else:
            x[i] = -1
    return x


def TrainEQNN(QuantumNet,corr,acc_list,loss_list):
    best = 0
    test_best = 0
    for epoch in range(epochs):
        for batch, (data,label) in enumerate(train_loader.create_tuple_iterator()):
            loss = train_net(data, label).asnumpy()
        pred_y = np.array(QuantumNet(val_x_ms).asnumpy())
        
        pred_y = Sgn(pred_y)
        #print(pred_y)
        #print(val_y)
        corr = np.mean(pred_y == ((val_y-0.5)*2))
 
        #print(corr)
        
        if corr >= best:
            pred_test_y = np.array(QuantumNet(test_x_ms).asnumpy())
            pred_test_y = Sgn(pred_test_y)
            test_best = np.mean(pred_test_y == ((test_y-0.5)*2))
            best = corr
            
            ###PROBLEM: it doesnt record the best QuantumNet but QuantumNet at the end, overfitting.
            param_opt = copy.copy(QuantumNet.weight)
            best_num = epoch
            print('the best quantum net is in epoch',epoch)
            
        acc_list.append(corr)
        loss_list.append(loss)
        #print(pred_test_y)
        print(f"epoch: {epoch} loss: {loss:>7f} on verification corr: {corr:>7f} [{batch:>3d}/{batch_size:>3d}]")
    print('After all, the test set correct rate is ',test_best)
    
    return acc_list,loss_list,p0,param_opt,best_num,test_best

N_ENSEMBLE = 40
test_acc_list = np.zeros(N_ENSEMBLE)
epoch_list = np.zeros(N_ENSEMBLE)
for i in range(N_ENSEMBLE):
    p0 = Param_ini()
    QuantumNet = MQLayer(grad_ops, weight=p0)
    loss_net = LossNet(QuantumNet)
    opti = ms.nn.Adam(loss_net.trainable_params(), learning_rate=0.1)
    train_net = ms.nn.TrainOneStepCell(loss_net, opti)
    epochs = 100
    corr = []
    acc_list = []
    loss_list = []
    acc_list,loss_list,p0,param_opt,best_num,test_best = TrainEQNN(QuantumNet,corr,acc_list,loss_list)
    test_acc_list[i] = test_best
    epoch_list[i] = best_num

print(test_acc_list)
print(epoch_list)
print('average acc = ',np.mean(test_acc_list),'standard_deviation = ',np.std(test_acc_list))
print('average epoch = ',np.mean(epoch_list),'standard_deviation = ',np.std(epoch_list))

#visualize the decision boundary

# xx, yy = np.meshgrid(np.linspace(1,-1,101),np.linspace(-1,1,101))
# xx = xx.flatten()
# yy = yy.flatten()
# space_xy = np.zeros([10201,2])
# for i in range(10201):
#     space_xy[i,0] = xx[i]
#     space_xy[i,1] = yy[i]
# space_xy_ms = ms.Tensor(space_xy)
# predict_space = np.array(QuantumNet(space_xy_ms).asnumpy())

# predict_space = (predict_space + 1)/2
# predict_space = predict_space.reshape([101,101])

# cm_pt = matplotlib.colors.ListedColormap(["blue", "red"])
# fig,ax=plt.subplots(1, 1, figsize=(4,4), sharey=False, sharex=False)
# plt.imshow(predict_space,cmap='bwr',vmin=0,vmax=1,extent=([-1,1,-1,1]))
# plt.colorbar()

# #double check the acc_rate
# pred_test_y = np.array(QuantumNet(test_x_ms).asnumpy())
# pred_test_y = Sgn(pred_test_y)
# correct_list = (pred_test_y == ((test_y-0.5)*2))

# x0_test_list = np.zeros(len(correct_list))
# x1_test_list = np.zeros(len(correct_list))
# for i in range(len(correct_list)):
#     x0_test_list[i] = test_x[i][0] / (np.pi/2)
#     x1_test_list[i] = test_x[i][1] / (np.pi/2)
    
# cm_pt = matplotlib.colors.ListedColormap(["blue", "red"])

# #fig,ax=plt.subplots(1, 1, figsize=(4,4), sharey=False, sharex=False)
# #plt.scatter(x0_test_list,x1_test_list,c=correct_list,cmap=cm_pt)
# #plt.scatter(x0_test_list,x1_test_list,c=pred_test_y,cmap=cm_pt,marker='X')

# #plt.scatter(x0_test_list,x1_test_list,c='k',marker='.')
# plt.scatter(x0_test_list,x1_test_list,c='k',marker='o',label='True',s=0.5)
# plt.scatter([x0_test_list[i] for i in range(len(correct_list)) if not correct_list[i]],[x1_test_list[i] for i in range(len(correct_list)) if not correct_list[i]],c='k',marker='X',s=3)
# #fig.savefig('boundry.pdf',dpi=800,bbox_inches='tight',format='pdf')






    
