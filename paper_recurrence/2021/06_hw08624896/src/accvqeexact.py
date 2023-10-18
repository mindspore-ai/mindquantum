# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 13:00:24 2022

@author: Jerryhts
"""
import numpy as np
from mindquantum.core.operators import QubitOperator

tot = 2
num = 2

J = 1
ham = np.complex128(np.zeros((2 ** tot, 2 ** tot)))
ham = QubitOperator()
for i in range(num - 1):
    ham += QubitOperator(f"X{i} X{i + 1}", J)
    ham += QubitOperator(f"Y{i} Y{i + 1}", J)
    ham += QubitOperator(f"Z{i} Z{i + 1}", J)

m = ham.matrix().toarray()
w, v = np.linalg.eig(m)
res = np.sort(np.real(w))
print(res)
