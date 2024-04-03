# -*- coding: utf-8 -*-
"""
Check the training set and generate the `real_test_y.npy`.

@NoEvaa
<Has a Bug>
"""
import numpy as np
from encoder_circuit import generate_encoder
from utils import param2dict

train_x = np.load('train_x.npy', allow_pickle=True)
train_y = np.load('train_y.npy', allow_pickle=True)

encoder, epn = generate_encoder()

A = []
B = []
for i in range(8):
    pp = param2dict(epn, train_x[i])
    state = encoder.get_qs(pr=pp)
    A.append(state)
    B.append(train_y[i])
A = np.array(A)
B = np.array(B)
X = np.dot(np.linalg.inv(A), B)
#print(X)

C = np.zeros((8, 8), dtype='complex128')
for i in range(8):
    for j in range(8):
        if abs(X[i][j]) > 1e-5:
            C[i][j] = X[i][j]
print(np.round(C, 3))

'''
#check train dataset
for i in range(800):
    pp = param2dict(epn, train_x[i])
    state = encoder.get_qs(pr=pp)
    if not np.allclose(np.dot(state, C), train_y[i]):
        print(i)
'''

'''
#generate real test_y
test_x = np.load('test_x.npy', allow_pickle=True)
real_test_y = []
for i in range(test_x.shape[0]):
    pp = param2dict(epn, train_x[i])
    state = encoder.get_qs(pr=pp)
    real_test_y.append(np.dot(state, C))
real_test_y = np.array(real_test_y)
np.save('real_test_y.npy', real_test_y)
'''
