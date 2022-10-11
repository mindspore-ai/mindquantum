# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 13:00:24 2022

@author: Jerryhts
"""

import numpy as np

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.array([[1, 0], [0, 1]])
'''
def RX(t):
    return np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]])
def RY(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])
def RZ(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])'''

def f(x):
    if x == 'X':
        return X
    elif x == 'Y':
        return Y
    elif x == 'Z':
        return Z
    else:
        return I
    
tot = 2
num = 2
    
def QubitOperator(string, para):
    al = ['I' for i in range(tot)]
    tem = string.split(' ')
    for i in tem:
        num = eval(i[1:])
        al[num] = i[0]
    ini = f(al[0])
    for i in range(1, tot):
        ini = np.kron(f(al[i]), ini)
    return para * ini

J = 1
ham = np.complex128(np.zeros((2 ** tot, 2 ** tot)))
for i in range(num - 1):
    ham += QubitOperator('X%d X%d' % (i, i + 1), J)
    ham += QubitOperator('Y%d Y%d' % (i, i + 1), J)
    ham += QubitOperator('Z%d Z%d' % (i, i + 1), J)
    
w, v = np.linalg.eig(ham)
res = np.sort(np.real(w))
print(res)