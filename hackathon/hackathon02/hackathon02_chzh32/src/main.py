# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 21:23:31 2022

用于测试提交

开始采用均匀叠加态
自己修改后的 9b 卷积

@author: Waikikilick
"""

import os
import sys
import time
import numpy as np
import mindspore as ms
from mindquantum import *
import matplotlib.pyplot as plt
os.environ['OMP_NUM_THREADS'] = '1'
import mindspore.context as context
from openfermionpyscf import run_pyscf
from hiqfermion.drivers import MolecularData
from mindquantum.framework import MQAnsatzOnlyLayer
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

ms.set_seed(1)

class Plot:
    def plot(self, prefix, blen_range, energy, time):
        x = blen_range
        data_time = time
        data_en = energy

        figen, axen = plt.subplots()
        axen.plot(x, data_en)
        axen.set_title('Energy')
        axen.set_xlabel('Bond Length')
        axen.set_ylabel('Energy')
        figen.savefig('figure_energy.png')

        figtime, axtime = plt.subplots()
        axtime.plot(x, data_time)
        axtime.set_title('Time')
        axtime.set_xlabel('Bond Length')
        axtime.set_ylabel('Time')
        figtime.savefig('figure_time.png')

        plt.close()


# 计时模块，用于统计 训练 和 执行 所用时间
class Timer:
    def __init__(self, t0=0.0):
        self.start_time = time.time()
        self.t0 = t0

    def runtime(self):
        return time.time() - self.start_time + self.t0

# 将用时单位从 秒，改成更清晰的 时 分 秒 显示
def format_time(t):
    hh = t // 3600
    mm = (t - 3600 * hh) // 60
    ss = t % 60
    return hh, mm, ss

# VQE算法 主体部分
class VQEoptimizer:
    def __init__(self, molecule=None, file=None): # molecule：分子数据 file：中间数据打印地址
        self.timer = Timer() # 开始计时
        self.molecule = molecule
        self.backend = 'projectq' # 后端模拟器
        self.file = file

        if molecule != None:
            self.generate_circuit(molecule) # 根据分子数据，生成所有量子线路

        print('Initialize finished! Time: ', self.timer.runtime(), file=self.file)
        sys.stdout.flush() # 可以在 Linux 系统中动态打印，而不是留到最后一次性打印

    def generate_circuit(self, molecule=None, ansatz_layers=11):
        if molecule == None:
            molecule = self.molecule

        _ , _ , _ , \
        self.hamiltonian, \
        self.n_qubits, \
        self.n_electrons = generate_uccsd(molecule) 
        
        self.circuit = Circuit([H.on(i) for i in range(self.n_qubits)]) # 生成编码线路
        
        for ii in range(self.n_qubits):
            self.circuit  += RX('theta0').on(ii)
            
        for ii in range(self.n_qubits):
            self.circuit  += RY('theta1').on(ii)
        
        self.circuit += Ansatz(num_qubit=self.n_qubits, num_layer=ansatz_layers) # 整体量子线路 = 编码线路 + 拟设线路
        self.simulator = Simulator(self.backend, self.n_qubits)
        grad_ops = self.simulator.get_expectation_with_grad(Hamiltonian(self.hamiltonian), self.circuit)
        self.qnet = MQAnsatzOnlyLayer(grad_ops)
        
    def optimize(self):
        optimizer = ms.nn.Adam(self.qnet.trainable_params(), learning_rate=0.01)
        train_qnet = ms.nn.TrainOneStepCell(self.qnet, optimizer)
        train_qnet.set_train()
        
        initial_energy = self.qnet()
        # print('Initial energy: ', initial_energy.asnumpy())

        eps = 1.e-9
        energy_diff = eps * 1000
        energy_last = initial_energy.asnumpy() + energy_diff
        iter_idx = 0
        mini_en = 0
        
        while abs(energy_diff) > eps:
            energy_i = train_qnet().asnumpy()
            mini_en = min(mini_en, energy_i)
            # print('Energy Now:',energy_i)
            # if iter_idx % 10 == 0:
            #     print('Step and energy: ', iter_idx, float(energy_i))
            energy_diff = energy_last - energy_i
            # print('diff: ',energy_diff)
            energy_last = energy_i
            iter_idx += 1
        
        print('Optimization completed at step:', iter_idx - 1)
        print('Optimized energy:', mini_en)
        
        return mini_en
        
            
# -------------------------------------------------------------------------------------

def Ansatz(num_qubit=8, num_layer=4):
    ansatz = Circuit()
    ansatz += BarrierGate()
    
    for ii in range(num_layer) :
        for jj in range(0, num_qubit, 2):
            ansatz += conv_circ(prefix= '%s'%(ii) + '_0_' + '%s'%(int(jj/2)) , bit_up= jj, bit_down=jj+1)

        ansatz += BarrierGate()
        for jj in range(1, num_qubit-1, 2):
            ansatz += conv_circ(prefix='%s'%(ii) + '_1_' + '%s'%(int((jj-1)/2)), bit_up= jj, bit_down=jj+1)
        ansatz += conv_circ(prefix='%s'%(ii) + '_1_' + '%s'%(int((jj+1)/2)), bit_up=num_qubit-1, bit_down=0)
        
    return ansatz

def conv_circ(prefix='0', bit_up=0, bit_down=1):
    _circ = Circuit()
    _circ += X.on(bit_down,bit_up)
    _circ += RY('theta10').on(bit_up)
    _circ += RZ('theta11').on(bit_down)
    _circ += X.on(bit_up,bit_down)
    _circ += RY('theta20').on(bit_up)
    _circ += X.on(bit_down,bit_up)
    _circ += RX('theta30').on(bit_up)
    _circ += RX('theta31').on(bit_down)
    _circ += RY('theta40').on(bit_up)
    _circ += RY('theta41').on(bit_down)
    _circ = add_prefix(_circ, prefix)
    return _circ

# ------------------------------------------------------------------------------------
class Main:
    def __init__(self):
        super().__init__()
        self.work_dir = './src/'
        sys.stdout.flush()

    def run(self, prefix, molecular_file, geom_list): # prefix 前缀 'LiH' / 'CH4' /
        if prefix == 'LiH':
            
            molecule = MolecularData(filename=self.work_dir+molecular_file)
            molecule.load() # 从指定的 hdf5 文件导入 分子数据
    
            with open(self.work_dir+prefix+'.o', 'a') as f: 
                # 将程序运行中生成的各种需要展示数据，保存到 .o 文件中
                # a: 打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
                print('\n',file=f)
                print('Start case: ', prefix, file=f)
                vqe = VQEoptimizer(file=f) # file 参数和 print 的 file 参数，为了将中间数据打印并保存到 .o 文件。
                en_list, time_list = [], []
                
                # for i in range(len(geom_list)):
                    
                mol0 = MolecularData(geometry=geom_list[0],
                                    basis=molecule.basis,
                                    charge=molecule.charge,
                                    multiplicity=molecule.multiplicity)
                
                mol = run_pyscf(mol0, run_scf=0, run_ccsd=1, run_fci=1) # 运行得到 FCI 法，得到参考能量值
                
                print("FCI energy: %20.16f Ha" % (mol.fci_energy), file=f)
                
                vqe.generate_circuit(mol, ansatz_layers=20)
                
                t = vqe.timer.runtime()
                en = vqe.optimize()
                print('Our Energy: ', en, 'Time: %i hrs %i mints %.2f sec.' % format_time(t), file=f)
                print('Precision compared with FCI is:', np.abs(en-mol.fci_energy), file=f)
                en_list.append(en[0])
                time_list.append(t)
    
                print('Optimization completed. Time: %i hrs %i mints %.2f sec.' % format_time(vqe.timer.runtime()), file=f)
                
                for _ in range(len(geom_list)-1):
                    en_list.append(0.)
                    time_list.append(1.)
    
            if len(en_list) == len(geom_list) and len(time_list) == len(geom_list):
                return en_list, time_list 
            else:
                raise ValueError('data lengths are not correct!')
                
        else:
            return list([0. for _ in range(len(geom_list))]), list([1 for _ in range(len(geom_list))])



