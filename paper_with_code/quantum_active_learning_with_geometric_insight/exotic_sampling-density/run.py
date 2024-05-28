# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:05:11 2024
Z2 symmetry equivalent QNN test, Fidelity
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

from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(1)
#load sets from the folder

with open('training_set.pickle', 'rb') as f:
    training_pool = pickle.load(f)
with open('F.pickle', 'rb') as f:
    Fid = pickle.load(f)

#EQNN-Z Encoder
def build_encoder():
    encoder = Circuit()
    encoder +=  RX(f'alpha{0}').on(0)
    encoder +=  RY(f'alpha{1}').on(1)
    encoder +=  RY(f'alpha{0}').on(0)
    encoder +=  RX(f'alpha{1}').on(1)
    #for qubit 0, XY, for qubit 1, YX, the SWAP symmetry is broken
    encoder = encoder.no_grad()
    return encoder.as_encoder()

#hea-Z Ansatz (single layer)
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



epochs = 17




#we first Tensor the training pool
pool_x, pool_y = generate_dataset(training_pool)
pool_x_ms = ms.Tensor(pool_x)
#before training we have to write the function for evaluating the uncertainty on the training_pool
def USamp(param_opt,pool_x,query_status):
    QuantumNet = MQLayer(grad_ops, weight=param_opt)
    predict_pool = np.array(QuantumNet(pool_x_ms).asnumpy()) # it's within [-1,1], we should find the point on the decision boundary
    min_uncertainty = 1
    index = 0
    for i in range(len(pool_x)):
        if abs(predict_pool[i][0]) <= min_uncertainty and query_status[i] == 0:
            min_uncertainty = abs(predict_pool[i][0])
            index = i
    print('sample',index,'is queried, its uncertainty is', min_uncertainty)
    return index




def PlotDecisionBoundary(p,num,dash):
    QuantumNet = MQLayer(grad_ops, weight=p)
    xx, yy = np.meshgrid(np.linspace(-0.2,0.2,101),np.linspace(0.2,-0.2,101))
    xx = xx.flatten()
    yy = yy.flatten()
    space_xy = np.zeros([10201,2])
    for i in range(10201):
        space_xy[i,0] = xx[i]
        space_xy[i,1] = yy[i]
    space_xy = space_xy * np.pi /2 #encoding
    space_xy_ms = ms.Tensor(space_xy)
    predict_space = np.array(QuantumNet(space_xy_ms).asnumpy())

    predict_space = (predict_space + 1)/2
    predict_space = predict_space.reshape([101,101])
    if dash == 1:
        plt.contour(np.linspace(-0.2,0.2,101),np.linspace(0.2,-0.2,101),predict_space,[0.5],colors='k',linewidths=num*0.1,linestyles='--')
    else:
        plt.contour(np.linspace(-0.2,0.2,101),np.linspace(0.2,-0.2,101),predict_space,[0.5],colors='k',linewidths=num*0.1)
    return 0
    
query_status = np.zeros(len(pool_x))  
print(query_status) 
training_set = []        
p0 = Param_ini()
index = np.random.randint(0,4)
print(index)
query_status[index] = 1
training_set.append(training_pool[index])
index = np.random.randint(0,4)+4
print(index)
query_status[index] = 1
training_set.append(training_pool[index])
train_x, train_y = generate_dataset(training_set)
train_x_ms, train_y_ms = ms.Tensor(train_x), ms.Tensor(train_y)

QuantumNet = MQLayer(grad_ops, weight=p0)
loss_net = LossNet(QuantumNet)
opti = ms.nn.Adam(loss_net.trainable_params(), learning_rate=0.099)
train_net = ms.nn.TrainOneStepCell(loss_net, opti)
batch_size=2
train_loader = ds.NumpySlicesDataset({
            'x': train_x,
            'y': train_y,
        }, shuffle=True).batch(batch_size)
for epoch in range(epochs):
    for batch, (data,label) in enumerate(train_loader.create_tuple_iterator()):
        loss = train_net(data, label).asnumpy()
    pred_y = np.array(QuantumNet(train_x_ms).asnumpy())
    pred_y = Sgn(pred_y)
    print('loss=',loss)

param_opt = copy.copy(QuantumNet.weight)
QuantumNet = MQLayer(grad_ops, weight=param_opt)
xx, yy = np.meshgrid(np.linspace(-0.2,0.2,101),np.linspace(0.2,-0.2,101))
xx = xx.flatten()
yy = yy.flatten()
space_xy = np.zeros([10201,2])
for i in range(10201):
    space_xy[i,0] = xx[i]
    space_xy[i,1] = yy[i]
space_xy = space_xy * np.pi /2 #encoding
space_xy_ms = ms.Tensor(space_xy)
predict_space = np.array(QuantumNet(space_xy_ms).asnumpy())

predict_space = (predict_space + 1)/2
predict_space = predict_space.reshape([101,101])


fig,ax=plt.subplots(1, 1, figsize=(4,2), sharey=False, sharex=False)
darkred='#8B0000'
darkblue='#00008B'
cm_pt = LinearSegmentedColormap.from_list('mycmap', [darkblue,'white',darkred])
plt.imshow(predict_space,cmap=cm_pt,vmin=predict_space.min(),vmax=predict_space.max()+0.035,extent=([-0.2,0.2,-0.1,0.1]))
#it's a simple trick to manually align the decision boundary and colormap white..
#im a shitty coder
plt.colorbar()
PlotDecisionBoundary(param_opt,num=10,dash=0)


#plot all samples from the pool
for i in range(4):
    plt.scatter(training_pool[i][0][0],training_pool[i][0][1],color=darkred,marker='.',s=10)
for i in range(4):
    plt.scatter(training_pool[i+4][0][0],training_pool[i+4][0][1],color=darkblue,marker='.',s=10)

plt.scatter(training_pool[8][0][0],training_pool[8][0][1],color=darkred,marker='.',s=10)
#PlotDecisionBoundary(param_opt, 20, dash=1)




def USamp(param_opt,pool_x,query_status):
    QuantumNet = MQLayer(grad_ops, weight=param_opt)
    predict_pool = np.array(QuantumNet(pool_x_ms).asnumpy()) # it's within [-1,1], we should find the point on the decision boundary
    min_uncertainty = 1
    index = 0
    for i in range(len(pool_x)):
        if abs(predict_pool[i][0]) <= min_uncertainty and query_status[i] == 0:
            min_uncertainty = abs(predict_pool[i][0])
            index = i
        print(predict_pool[i][0])
    print('sample',index,'is queried, its uncertainty is', min_uncertainty)
    return index


#let's write a new function for sampling USAMP+F
def FSamp(param_opt,pool_x,query_status,Fid):
    #we have to calculate the average fidelity of the i-th sample in the pool on other unlabeled samples
    F_mean = np.zeros(len(Fid))
    for i in range(len(Fid)):
        count = 0
        F_total = 0
        for j in range(len(Fid)):
            if query_status[i] != 1 and query_status[j] != 1: #not sampled
                count += 1
                F_total += Fid[i,j]
        if count == 0:
            count = 1
        F_mean[i] = F_total/count
    #the larger F_mean is, the more close the sample is to the other unlabeled sample in the pool
    QuantumNet = MQLayer(grad_ops, weight=param_opt)
    predict_pool = np.array(QuantumNet(pool_x_ms).asnumpy()) # it's within [-1,1], we should find the point on the decision boundary
    min_uncertainty = 100
    index = 0
    for i in range(len(pool_x)):
        temp = abs(predict_pool[i][0])+1*(1-F_mean[i])
        print(temp)
        if  temp <= min_uncertainty and query_status[i] == 0:

            min_uncertainty = temp
            index = i
    print('sample',index,'is queried, its uncertainty*(1-F_mean) is', min_uncertainty)
    return index

index_usamp = USamp(param_opt,pool_x,query_status)

plt.scatter(training_pool[2][0][0],training_pool[2][0][1],color=darkred,marker='x',s=10)
plt.scatter(training_pool[4][0][0],training_pool[4][0][1],color=darkblue,marker='x',s=10)

index_F = FSamp(param_opt,pool_x,query_status,Fid)
plt.scatter(training_pool[index_F][0][0],training_pool[index_F][0][1],edgecolors='k',facecolors='none',marker='o',s=20)
plt.scatter(training_pool[index_usamp][0][0],training_pool[index_usamp][0][1],edgecolors='k',facecolors='none',marker='o',s=20)
plt.text(training_pool[index_F][0][0]-0.09,training_pool[index_F][0][1]-0.015,'FSAMP')
plt.text(training_pool[index_usamp][0][0]+0.02,training_pool[index_usamp][0][1]-0.03,'USAMP')


plt.xlim([-0.2,0.2])
plt.ylim([-0.1,0.1])
plt.legend(fontsize=10,ncol=1,frameon=False)
plt.xticks([-0.2,-0.1,0,0.1,0.2],fontsize=15)
plt.yticks([-0.1,0,0.1],fontsize=15)
plt.xlabel('$x_0$',fontsize=15)
plt.ylabel('$x_1$',fontsize=15)
fig.savefig('exotic.pdf',dpi=800,bbox_inches='tight',format='pdf')




























