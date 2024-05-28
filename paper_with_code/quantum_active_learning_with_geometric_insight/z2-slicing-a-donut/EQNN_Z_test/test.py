# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 21:14:55 2024
Z2 symmetry equivalent QNN test
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

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'sans'
mpl.rcParams['mathtext.it'] = 'sans:italic'
mpl.rcParams['mathtext.default'] = 'it'

np.random.seed(1)
#load sets from the folder

with open('training_set.pickle', 'rb') as f:
    training_pool = pickle.load(f)
with open('val_set.pickle', 'rb') as f:
    data_val = pickle.load(f)
with open('test_set.pickle', 'rb') as f:
    data_test = pickle.load(f)

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

#EQNN-Z Ansatz (single layer)
def build_circuit():
    circ = Circuit()
    circ += build_encoder().as_encoder()
    ansatz = Circuit()
    ansatz += H.on(0)
    ansatz += H.on(1)
    ansatz += CNOT.on(1,0)
    ansatz += RZ(f'theta{0}').on(1)
    ansatz += CNOT.on(1,0)
    ansatz += H.on(0)
    ansatz += H.on(1)    
    ansatz += RZ(f'theta{1}').on(0)
    ansatz += RZ(f'theta{2}').on(1)
    ansatz.as_ansatz()
    ansatz += BarrierGate()
    circ += ansatz
    
    circ += build_encoder().as_encoder()
    ansatz = Circuit()
    ansatz += H.on(0)
    ansatz += H.on(1)
    ansatz += CNOT.on(1,0)
    ansatz += RZ(f'theta{3}').on(1)
    ansatz += CNOT.on(1,0)
    ansatz += H.on(0)
    ansatz += H.on(1)    
    ansatz += RZ(f'theta{4}').on(0)
    ansatz += RZ(f'theta{5}').on(1)
    ansatz.as_ansatz()
    ansatz += BarrierGate()
    circ += ansatz
    
    circ += build_encoder().as_encoder()
    ansatz = Circuit()
    ansatz += H.on(0)
    ansatz += H.on(1)
    ansatz += CNOT.on(1,0)
    ansatz += RZ(f'theta{6}').on(1)
    ansatz += CNOT.on(1,0)
    ansatz += H.on(0)
    ansatz += H.on(1)    
    ansatz += RZ(f'theta{7}').on(0)
    ansatz += RZ(f'theta{8}').on(1)
    ansatz.as_ansatz()
    ansatz += BarrierGate()
    circ += ansatz
    
    # circ += build_encoder().as_encoder()
    # ansatz = Circuit()
    # ansatz += H.on(0)
    # ansatz += H.on(1)
    # ansatz += CNOT.on(1,0)
    # ansatz += RZ(f'theta{9}').on(1)
    # ansatz += CNOT.on(1,0)
    # ansatz += H.on(0)
    # ansatz += H.on(1)    
    # ansatz += RZ(f'theta{10}').on(0)
    # ansatz += RZ(f'theta{11}').on(1)
    # ansatz.as_ansatz()
    # ansatz += BarrierGate()
    # circ += ansatz
    
    # circ += build_encoder().as_encoder()
    # ansatz = Circuit()
    # ansatz += H.on(0)
    # ansatz += H.on(1)
    # ansatz += CNOT.on(1,0)
    # ansatz += RZ(f'theta{12}').on(1)
    # ansatz += CNOT.on(1,0)
    # ansatz += H.on(0)
    # ansatz += H.on(1)    
    # ansatz += RZ(f'theta{13}').on(0)
    # ansatz += RZ(f'theta{14}').on(1)
    # ansatz.as_ansatz()
    # ansatz += BarrierGate()
    # circ += ansatz
    
    # circ += build_encoder().as_encoder()
    # ansatz = Circuit()
    # ansatz += H.on(0)
    # ansatz += H.on(1)
    # ansatz += CNOT.on(1,0)
    # ansatz += RZ(f'theta{15}').on(1)
    # ansatz += CNOT.on(1,0)
    # ansatz += H.on(0)
    # ansatz += H.on(1)    
    # ansatz += RZ(f'theta{16}').on(0)
    # ansatz += RZ(f'theta{17}').on(1)
    # ansatz.as_ansatz()
    # ansatz += BarrierGate()
    # circ += ansatz

    
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


def PlotDecisionBoundary(p,num,dash):
    QuantumNet = MQLayer(grad_ops, weight=p)
    xx, yy = np.meshgrid(np.linspace(1,-1,101),np.linspace(-1,1,101))
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
        plt.contour(np.linspace(1,-1,101),np.linspace(-1,1,101),predict_space,[0.5],colors='k',linewidths=num*0.1,linestyles='--')
    else:
        plt.contour(np.linspace(1,-1,101),np.linspace(-1,1,101),predict_space,[0.5],colors='k',linewidths=num*0.1)
    return 0

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
    
    return acc_list,loss_list,p0,param_opt,best_num

acc_list,loss_list,p0,param_opt,best_num = TrainEQNN(QuantumNet,corr,acc_list,loss_list)
QuantumNet = MQLayer(grad_ops, weight=param_opt)




def PlotDecisionBoundary(p,num,dash):
    QuantumNet = MQLayer(grad_ops, weight=p)
    xx, yy = np.meshgrid(np.linspace(-1,1,101),np.linspace(1,-1,101))
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
        plt.contour(np.linspace(-1,1,101),np.linspace(1,-1,101),predict_space,[0.5],colors='k',linewidths=num*0.1,linestyles='--')
    else:
        plt.contour(np.linspace(-1,1,101),np.linspace(1,-1,101),predict_space,[0.5],colors='k',linewidths=num*0.1)
    return 0

#visualize the decision boundary

xx, yy = np.meshgrid(np.linspace(-1,1,101),np.linspace(1,-1,101))
xx = xx.flatten()
yy = yy.flatten()
space_xy = np.zeros([10201,2])
for i in range(10201):
    space_xy[i,0] = xx[i]
    space_xy[i,1] = yy[i]
space_xy = space_xy * np.pi /2
space_xy_ms = ms.Tensor(space_xy)
predict_space = np.array(QuantumNet(space_xy_ms).asnumpy())

predict_space = (predict_space + 1)/2
predict_space = predict_space.reshape([101,101])


darkred='#8B0000'
darkblue='#00008B'
cm_pt = LinearSegmentedColormap.from_list('mycmap', [darkblue,'white',darkred])
fig,ax=plt.subplots(1, 1, figsize=(4,4), sharey=False, sharex=False)
plt.imshow(predict_space,cmap=cm_pt,vmin=0,vmax=1,extent=([-1,1,-1,1]))
plt.colorbar()

#plot samples in test sets
pred_test_y = QuantumNet(test_x_ms).asnumpy()
for i in range(len(pred_test_y)):
    if pred_test_y[i][0] < 0:
        y_pred = 0
    else:
        y_pred = 1
    if y_pred == test_y[i][0]:
        plt.scatter(test_x[i][0]/(np.pi/2),test_x[i][1]/(np.pi/2),marker='.',color='k')
    else:
        plt.scatter(test_x[i][0]/(np.pi/2),test_x[i][1]/(np.pi/2),marker='x',color='k')


PlotDecisionBoundary(param_opt, 20, dash=1)



plt.xlim([-1,1])
plt.ylim([-1,1])
plt.legend(fontsize=10,ncol=1,frameon=False)
plt.xticks([-1,-0.5,0,0.5,1],fontsize=15)
plt.yticks([-1,-0.5,0,0.5,1],fontsize=15)
plt.xlabel('$x_0$',fontsize=15)
plt.ylabel('$x_1$',fontsize=15)
fig.savefig('sleqnnz.pdf',dpi=800,bbox_inches='tight',format='pdf')

#fig.savefig('boundry.pdf',dpi=800,bbox_inches='tight',format='pdf')






    
