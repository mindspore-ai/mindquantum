# -*- coding: utf-8 -*-
"""
Testing various quantum circuits.

测试样例集锦

ex.
from qcexlib import QCircuitLib_ex
encoder, ansatz = QCircuitLib_ex().qc_original # 原始线路
encoder, ansatz = QCircuitLib_ex().hackathon01 # 最终答案
"""

from qclib import QCircuitLib_pure
from mindquantum import *
import numpy as np

class QCircuitLib_ex(QCircuitLib_pure):
    def __init__(self):
        QCircuitLib_pure.__init__(self)

    @property
    def hackathon01(self): # Answer to hackathon01 -- 2022
        '''
        经检查, train.mpy 文件中的5000个训练数据：
            (1)经过去重处理会只剩182个样本
            (2)182个样本中有62(31+31)个样本是同时具有两个便签(True|False)的
            (3)在训练集上测试无随机性能的模型, Acc 最高只能达到 0.9184
            (详细数据见/src/checkdataset.py)
            
        该线路与对应参数能够幸运地在训练集上达到 Acc: 0.9062 精确度
        在去重的训练集上测试达到 Acc: 0.6813186813186813 精确度
        '''
        # IQP编码改(circular)   bits=8qbits*2
        encoder = self.IQP(8, np.pi/2)
        # EfficientSU2   ZY   8qbits   entanglements-full   重复频率3
        ansatz = self.qvc(8, [RZ, RY], mode='full',reps=3)
        # 最后再纠缠一下下(circular)
        ansatz += self.entanglements_circular(8)
        # 测量第一个和最后一个比特
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [0, 7]]
        return encoder, ansatz, ham

    """
    Acc0 为训练集 train.npy 测试结果
    理论上无随机情况下 max(Acc0) = 0.9184
    以下测试线路仅保留 case(Acc0 >= 0.8) 且只取其最高值

    Acc1为对训练集 train.npy 去重后的 test.pkl 测试结果
    理论上无随机情况下 max(Acc1) = 0.8296703296703297
    以下测试线路在 Acc0 相同情况下取 Acc1 最高值
    """

    @property
    def qc_original(self): # 题目提供的原始线路
        '''
        Acc0: 0.8082
        Acc1: 0.6208791208791209
        '''
        circ = Circuit()
        for i in range(8):
            circ += RY(f'p{i}').on(i)
        circ += self.entanglements_cu(8)
        encoder = add_prefix(circ, 'e1') + add_prefix(circ, 'e2')
        ansatz = add_prefix(circ, 'a1')
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [6, 7]]
        return encoder, ansatz, ham

    @property
    def qc_test1(self): # 测试1 
        '''
        Acc0: 0.8274
        Acc1: 0.6428571428571429
        '''
        encoder = self.encoder_mli(8, [RY], 2, mode=self.entanglements_cu(8))
        #EfficientSU2  YZ  8qbits  重复频率1  full
        ansatz = self.qvc(8, [RY, RZ])
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [6, 7]]
        return encoder, ansatz, ham

    @property
    def qc_test2(self): # 测试2 
        '''
        Acc0: 0.8272
        Acc1: 0.6373626373626373
        '''
        encoder = self.encoder_mli(8, [RY], 2, mode=self.entanglements_cu(8))
        #EfficientSU2  YX  8qbits  重复频率1  full
        ansatz = self.qvc(8, [RY, RX])
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [6, 7]]
        return encoder, ansatz, ham

    @property
    def qc_test3(self): # 测试3 
        '''
        Acc0: 0.893
        Acc1: 0.6648351648351648
        '''
        #IQP编码
        encoder = self.IQP(8, np.pi/2)
        #EfficientSU2  ZY  8qbits  重复频率1  full
        ansatz = self.qvc(8, [RZ, RY], mode='full')
        ansatz += self.entanglements_circular(8)
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [6, 7]]
        return encoder, ansatz, ham

    @property
    def qc_test4(self): # 测试4 
        '''
        Acc0: 0.8942
        Acc1: 0.6703296703296703
        '''
        # 
        encoder = self.IQP(8, np.pi/2)
        #encoder += UN(H, 8)
        #
        ansatz = self.qvc(8, [RX, RZ, RY], mode='full')
        ansatz += self.entanglements_circular(8)
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [6, 7]]
        return encoder, ansatz, ham

    @property
    def qc_test5(self): # 测试5 
        '''
        Acc0: 0.8064
        Acc1: 0.5879120879120879
        '''
        #RY  circular   
        encoder = self.encoder_mli(8, [RY], 2, mode='circular')
        #EfficientSU2  YX  8qbits  重复频率1  circular
        ansatz = self.qvc(8, [RY, RX], mode='circular')
        #
        ansatz += self.entanglements_circular(8)
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [6, 7]]
        return encoder, ansatz, ham

    @property
    def qc_test6(self): # 测试6 
        '''
        Acc0: 0.8598 
        Acc1: 0.6373626373626373
        '''
        #RY  circular   
        encoder = self.encoder_mli(8, [RY], 2, mode='circular')
        #EfficientSU2  ZY  8qbits  重复频率1  circular
        ansatz = self.qvc(8, [RZ, RY], mode='circular')
        #
        ansatz += self.entanglements_circular(8)
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [6, 7]]
        return encoder, ansatz, ham

    @property
    def qc_test7(self): # 测试7
        '''
        Acc0: 0.8638
        Acc1: 0.6923076923076923
        '''
        #RY  circular   
        encoder = self.encoder_mli(8, [RY], 2, mode='circular')
        #EfficientSU2  ZY  8qbits  重复频率1  full
        ansatz = self.qvc(8, [RZ, RY], mode='full')
        #
        ansatz += self.entanglements_circular(8)
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [6, 7]]
        return encoder, ansatz, ham

    @property
    def qc_test8(self): # 测试8
        '''
        Acc0: 0.8316
        Acc1: 0.6923076923076923
        '''
        #   
        encoder = self.encoder_bs()
        #
        ansatz = self.qvc(4, [RZ, RY], mode='circular', reps=3,le=True, re=True)
        #ansatz += self.entanglements_circular(4)
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [2, 3]]
        return encoder, ansatz, ham

    @property
    def qc_test9(self): # 测试9
        '''
        Acc0: 0.8984
        Acc1: 0.6758241758241759
        '''
        encoder = self.IQP(8, np.pi/2)
        ansatz = self.qvc(8, [RZ, RY], mode='full',reps=3)
        ansatz += self.entanglements_circular(8)
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [6, 7]]
        return encoder, ansatz, ham

    @property
    def qc_test10(self): # 测试10
        '''
        Acc0: 0.9062
        Acc1: 0.6813186813186813
        '''
        encoder = self.IQP(8, np.pi/2)
        ansatz = self.qvc(8, [RZ, RY], mode='full',reps=3)
        ansatz += self.entanglements_circular(8)
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [0, 7]]
        return encoder, ansatz, ham

    @property
    def qc_test11(self): # 测试11
        '''
        Acc0: 
        Acc1: 
        '''
        encoder = self.IQP(8, np.pi/2)
        ansatz = self.qvc(8, [RZ, RY], mode='full',reps=3)
        ham = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [0, 7]]
        return encoder, ansatz, ham
