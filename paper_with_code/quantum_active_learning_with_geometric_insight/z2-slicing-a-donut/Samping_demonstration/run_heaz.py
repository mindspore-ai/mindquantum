# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:05:11 2024
Z2 symmetry nonequivalent QNN test, USAMP
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
ms.set_seed(4)
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


NUM_ENSEMBLE = 1
N_label = 6
epochs = 100

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

def PlotQueried(index,j):
    plt.text(training_pool[index][0][0],training_pool[index][0][1],s='%s'%(j+1))
    plt.scatter(training_pool[index][0][0],training_pool[index][0][1],marker='x',color='k')
    return 0
    



for i in range(NUM_ENSEMBLE):
    #in each ensemble, we should initialize the query status
    query_status = np.zeros(len(pool_x))
    training_set = []
    
    fig,ax=plt.subplots(1, 1, figsize=(4,4), sharey=False, sharex=False)
    for j in range(N_label):
        #we have to evaluate the uncertainty based on the model we trained on the last training set
        #if nothing was trained, we have to initalize the model by evaluating its initial parameters
        if j == 0:
            #nothing is trained
            p0 = Param_ini()
            index = USamp(p0, pool_x, query_status)
            print(index)
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
        if j == 0:
            PlotDecisionBoundary(p0,j+1,1)
        if j !=0:
            p0 = Param_ini()
            PlotDecisionBoundary(param_opt,j+1,1)# initialize the parameter
        PlotQueried(index,j)
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
        print('The queries sample is',index)
        best_performance[i,j] = test_best
        

# with open('EQNN-Z_USAMP.pickle', 'wb') as f:
#     pickle.dump(best_performance, f)

QuantumNet = MQLayer(grad_ops, weight=param_opt)
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

darkred='#8B0000'
darkblue='#00008B'
cm_pt = LinearSegmentedColormap.from_list('mycmap', [darkblue,'white',darkred])
plt.imshow(predict_space,cmap=cm_pt,vmin=0,vmax=1,extent=([-1,1,-1,1]))
plt.colorbar()
#PlotDecisionBoundary(param_opt,num=10,dash=0)


#plot all samples from the pool
for i in range(len(training_pool)):
    plt.scatter(training_pool[i][0][0],training_pool[i][0][1],color='k',marker='.',s=1)

PlotDecisionBoundary(param_opt, 20, dash=1)

plt.xlim([-1,1])
plt.ylim([-1,1])
plt.legend(fontsize=10,ncol=1,frameon=False)
plt.xticks([-1,-0.5,0,0.5,1],fontsize=15)
plt.yticks([-1,-0.5,0,0.5,1],fontsize=15)
plt.xlabel('$x_0$',fontsize=15)
plt.ylabel('$x_1$',fontsize=15)
fig.savefig('hea_usamp.pdf',dpi=800,bbox_inches='tight',format='pdf')


















































