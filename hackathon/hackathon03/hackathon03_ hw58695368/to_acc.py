# -*- coding: utf-8 -*-
import numpy as np
from encoder_circuit import generate_encoder
from ansatz_circuit import generate_ansatz
from utils import param2dict, calcu_acc

test_x = np.load('train_x.npy', allow_pickle=True)
real_test_y = np.load('train_y.npy', allow_pickle=True)

encoder, epn = generate_encoder()
ansatz, apn = generate_ansatz()
nw = len(apn)
circ = encoder + ansatz

weights = [0.39908293,  0.68672866, -0.16200331, -0.18892704, -0.9356558,
           -0.07870941, -0.27550808, -0.735237, -0.32230943] # Acc: 0.6701757226302132
test_y = []
app = param2dict(apn, weights)
for i in range(test_x.shape[0]):
    pp = dict(param2dict(epn, test_x[i]), **app)
    test_y.append(circ.get_qs(pr=pp))
test_y = np.array(test_y)
acc = calcu_acc(test_y, real_test_y)
print(f"Acc: {acc}")
