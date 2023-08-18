import numpy as np
from mindquantum.core import Circuit                 # 导入Circuit模块，用于搭建量子线路
from mindquantum.algorithm.library import amplitude_encoder
from mindquantum.core import H, X, RZ,RX,CNOT  ,UN,RY              # 导入量子门H, X, RZ
#定义编码线路
def encoder(n):
    encoder = Circuit()
    encoder += UN(H,8)
    for i in range(n):
        encoder += RZ(f'alpha{i}').on(i)
    for i in range(n-1):
        encoder += CNOT.on(i+1,i)
        encoder +=RZ(f'alpha{i+8}').on(i+1)
    return encoder


#定义构造ansatz
def U(a, i,k):
    ansatz = Circuit()
    ansatz += RZ(f'theta{a}_1_{i}_{k}').on(a)
    ansatz += RX(-(np.pi / 2)).on(a)
    ansatz += RZ(f'theta{a}_2_{i}_{k}').on(a)
    ansatz += RX(np.pi / 2).on(a)
    ansatz += RZ(f'theta{a}_3_{i}_{k}').on(a)
    return ansatz

def conveolution_filter_ansatz_9(a, b, i, k):
    ansatz = Circuit()
    ansatz += U(a, i,k)
    ansatz += U(b, i,k)
    ansatz += CNOT(b, a)
    ansatz += RY(f'theta{a}_{i}_{k}_mid1').on(a)
    ansatz += RZ(f'theta{b}_{i}_{k}_mid2').on(b)
    ansatz += CNOT(a, b)
    ansatz += RY(f'theta{a}_{i}_{k}_mid3').on(a) #参数名识别标识_add
    ansatz += CNOT(b, a)
    ansatz += U(a, i+1,k)
    ansatz += U(b, i+1,k)
    return ansatz

def ansatz():
    ansatz = Circuit()
    conveolution_ansatz1_1 = conveolution_filter_ansatz_9(0, 1, 1, 1)
    conveolution_ansatz1_2 = conveolution_filter_ansatz_9(2, 3, 1, 2)
    conveolution_ansatz1_3 = conveolution_filter_ansatz_9(4, 5, 1, 3)
    conveolution_ansatz1_4 = conveolution_filter_ansatz_9(6, 7, 1, 4)

    conveolution_ansatz2_1 = conveolution_filter_ansatz_9(1, 2, 2, 1)
    conveolution_ansatz2_2 = conveolution_filter_ansatz_9(3, 4, 2, 2)
    conveolution_ansatz2_3 = conveolution_filter_ansatz_9(5, 6, 2, 3)

    conveolution_ansatz3_1 = conveolution_filter_ansatz_9(2, 3, 3, 1)
    conveolution_ansatz3_2 = conveolution_filter_ansatz_9(4, 5, 3, 2)

    conveolution_ansatz4_1 = conveolution_filter_ansatz_9(3, 4, 4, 1)

    conveolution_ansatz1 = conveolution_ansatz1_1 + conveolution_ansatz1_2+conveolution_ansatz1_3+conveolution_ansatz1_4
    conveolution_ansatz2 = conveolution_ansatz2_1 + conveolution_ansatz2_2 + conveolution_ansatz2_3
    conveolution_ansatz3 = conveolution_ansatz3_1 + conveolution_ansatz3_2
    conveolution_ansatz4 = conveolution_ansatz4_1
    # conveolution_ansatz1.summary()
    # conveolution_ansatz2.summary()
    # ansatz += conveolution_ansatz1 + conveolution_ansatz2
    # ansatz.summary()
    # conveolution_ansatz3.summary()
    # conveolution_ansatz4.summary()
    ansatz +=  conveolution_ansatz1 + conveolution_ansatz2 + conveolution_ansatz3 + conveolution_ansatz4
    #参数中存在同名参数，应当为165个参数实际只有147
    return ansatz