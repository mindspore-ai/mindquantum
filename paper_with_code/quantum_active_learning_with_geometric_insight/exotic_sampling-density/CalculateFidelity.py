# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:33:08 2024

@author: jonze
"""
import math
import numpy as np
import random
import pickle
import copy
import matplotlib.pyplot as plt
import matplotlib

np.random.seed(1)

with open('training_set.pickle', 'rb') as f:
    training_pool = pickle.load(f)
    
def Rx(theta):
    U = np.zeros([2,2],dtype=complex)
    U[0,0] = np.cos(theta/2)
    U[1,1] = np.cos(theta/2)
    U[0,1] = -1j * np.sin(theta/2)
    U[1,0] = -1j * np.sin(theta/2)
    return U

def Ry(theta):
    U = np.zeros([2,2],dtype=complex)
    U[0,0] = np.cos(theta/2)
    U[1,1] = np.cos(theta/2)
    U[0,1] = -np.sin(theta/2)
    U[1,0] = np.sin(theta/2)
    return U

def EncodeClassicalData(initial_state,x0,x1):
    U = np.kron(Rx(x0*np.pi/2),Ry(x1*np.pi/2))
    psi = np.dot(U,initial_state)
    U = np.kron(Ry(x0*np.pi/2),Rx(x1*np.pi/2))
    psi = np.dot(U,psi)
    return psi

def Fidelity(psi_a,psi_b):
    F = np.dot(psi_b.conjugate().transpose(),psi_a)
    return (abs(F[0][0]))**2

initial_state = np.kron(np.array([[1],[0]],dtype=complex), np.array([[1],[0]],dtype=complex))

F = np.zeros([len(training_pool),len(training_pool)])
for i in range(len(training_pool)):
    for j in range(len(training_pool)):
        i0 = training_pool[i][0][0]
        i1 = training_pool[i][0][1]
        j0 = training_pool[j][0][0]
        j1 = training_pool[j][0][1]
        psi_i = EncodeClassicalData(initial_state, i0, i1)
        psi_j = EncodeClassicalData(initial_state, j0, j1)
        F[i,j] = Fidelity(psi_i, psi_j)

with open('F.pickle', 'wb') as f:
    pickle.dump(F, f)        