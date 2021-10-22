# -*- coding: utf-8 -*-
import numpy as np
import mindquantum as mq
from mindquantum.core import X, Y, Z, H, RX, RY, RZ

print('Gate name:', X)
X.matrix()

print('Gate name:', Y)
Y.matrix()

print('Gate name:', Z)
Z.matrix()

print('Gate name:', H)
H.matrix()

cnot = X.on(0, 1)
print(cnot)

rx = RX('theta')
print('Gate name:', rx)
rx.matrix({'theta': 0})

ry = RY('theta')
print('Gate name:', ry)
ry.matrix({'theta': np.pi / 2})

rz = RZ('theta')
print('Gate name:', rz)
np.round(rz.matrix({'theta': np.pi}))

from mindquantum.core import Circuit

encoder = Circuit()
encoder += H.on(0)
encoder += X.on(1, 0)
encoder += RY('theta').on(2)

print(encoder)
encoder.summary()
