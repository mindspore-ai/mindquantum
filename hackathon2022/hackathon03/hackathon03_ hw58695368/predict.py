# -*- coding: utf-8 -*-
"""
Generate the `test_y.npy`.

@NE
"""
import numpy as np
from encoder_circuit import generate_encoder
from utils import param2dict

def J49U26A00(): # Acc: 0.6
    from ansatz_circuit import generate_ansatz
    weights = np.load('weights_J49U26A00.npy', allow_pickle=True)
    return generate_ansatz(), weights

# weights_J49U26B01.npy   Acc: 0.999967935851851
# weights_J49U26B02.npy   Acc: 0.9999999311257113
# weights_J49U26B03.npy   Acc: 0.9999991570897098
def J49U26B(file): 
    from ansatz_circuit_qsd import generate_ansatz
    weights = np.load(file, allow_pickle=True)
    return generate_ansatz(), weights


test_x = np.load('test_x.npy', allow_pickle=True)
encoder, epn = generate_encoder()
(ansatz, apn), weights = J49U26B('weights_J49U26B02.npy')
nw = len(apn)
circ = encoder + ansatz

test_y = []
app = param2dict(apn, weights)
for i in range(test_x.shape[0]):
    pp = dict(param2dict(epn, test_x[i]), **app)
    test_y.append(circ.get_qs(pr=pp))
test_y = np.array(test_y)
np.save('test_y.npy', test_y)
