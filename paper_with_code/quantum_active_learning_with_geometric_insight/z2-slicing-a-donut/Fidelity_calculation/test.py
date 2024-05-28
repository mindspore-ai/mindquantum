# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:20:42 2024

@author: jonze
"""
import numpy as np

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
psi_a = EncodeClassicalData(initial_state, -0.8, 0.6)
psi_b = EncodeClassicalData(initial_state, 0.8, -0.6)
F = Fidelity(psi_a, psi_b)
print(F)
