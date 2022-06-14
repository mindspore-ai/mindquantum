# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 21:23:31 2022

1. 拟设线路为自定义线路，而不是自动生成的 UCCSD 拟设。
   可设置层结构、层数等。

2. 采用 scipy.optimize.minimize 方法对参数进行优化。
   可设置 优化方法、终止优化精度、最大优化步数等。


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
from scipy.optimize import minimize
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

# 将变量名 和 数值 映射起来，以将参数传入网络对应位置
def param2dict(keys, values):
    param_dict = {}
    for (key, value) in zip(keys, values):
        param_dict[key] = value
    return param_dict

# scipy.optimize 优化时的目标函数
def func(x, grad_ops, file, show_iter_val=False):

    f, g = grad_ops(x)
    if show_iter_val:
        print(np.squeeze(f), file=file) # 通过改变 file 参数使 print() 函数输出到特定的文件中
        sys.stdout.flush() # 动态展示
    return np.real(np.squeeze(f)), np.squeeze(g)

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

    def generate_circuit(self, molecule=None):
        if molecule == None:
            molecule = self.molecule
        self.circuit = Circuit([X.on(i) for i in range(molecule.n_electrons)]) # 生成编码线路

        ansatz_0, \
        self.init_amp, \
        self.params_name, \
        self.hamiltonian, \
        self.n_qubits, \
        self.n_electrons = generate_uccsd(molecule, -1)

        ansatz_1 = Circuit(UN(H, range(self.n_qubits))) + Circuit(UN(H, range(self.n_qubits)))
        ansatz_circuit = ansatz_0 + ansatz_1
        self.circuit += ansatz_circuit 
        self.simulator = Simulator(self.backend, self.n_qubits, seed=124)
        
        # ansatz_circuit, init_amp, params_name
        # init_amp： uccsd 线路的初始参数值；params_name：各初始参数的名字；hamiltonian：分子的哈密顿量；
        # n_qubits：模拟器所需比特数；n_electrons：分子的电子数
        # amp_th：整数 决定对 uccsd 的振幅滤波方式。0 表示只保留正值。负值 表示保留所有振幅。
        
    def optimize(self, operator=None, circuit=None, init_amp=[], method='bfgs', tol=0.01, maxstep=200, iter_info=False):
        if operator == None:
            operator = self.hamiltonian
        if circuit == None:
            circuit = self.circuit
        if np.array(init_amp).size == 0:
            init_amp = self.init_amp
            
        grad_ops = self.simulator.get_expectation_with_grad(Hamiltonian(operator), circuit)
        # minimize 调用一次就行，内部就优化就会自动完成的
        self.res = minimize(func, # 目标函数
                            init_amp, # 初始迭代值
                            args=(grad_ops, self.file, iter_info), # 要输入到目标函数中的参数
                            method=method, # 求解的算法，目前可选的有 ‘Nelder-Mead’‘Powell’‘CG’‘BFGS’‘Newton-CG’‘L-BFGS-B’‘TNC’‘COBYLA’‘SLSQP’‘dogleg’‘trust-ncg’ 
                            # 以及在 version 0.14.0，还能自定义算法 以上算法的解释和相关用法见 minimize 函数的官方说明文档，一般求极值多用 'SLSQP'算法
                            jac=True, # 目标函数的雅可比矩阵。可选项，仅适用于CG，BFGS，Newton-CG，L-BFGS-B，TNC，SLSQP，dogleg，trust-ncg。
                            # 如果jac是布尔值并且为True，则假定fun与目标函数一起返回梯度。如果为False，将以数字方式估计梯度。jac也可以返回目标的梯度。此时，它的参数必须与fun相同。
                            bounds=(0, 2*np.pi), # 可选项，变量的边界
                            tol = tol, # 终止精度
                            options={'maxiter':maxstep,'disp':True}
                            # maxiter: 最大迭代次数  disp: 是否显示过程信息
                            )

# ------------------------------------------------------------------------------------
class Main:
    def __init__(self):
        super().__init__()
        self.work_dir = './src/'

    def run(self, prefix, molecular_file, geom_list): # prefix 前缀 'LiH' / 'CH4' /
        molecule = MolecularData(filename=self.work_dir+molecular_file)
        molecule.load() # 从指定的 hdf5 文件导入 分子数据

        with open(self.work_dir+prefix+'.o', 'a') as f: 
            # 将程序运行中生成的各种需要展示数据，保存到 .o 文件中
            # a: 打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
            print('Start case: ', prefix, file=f)
            
            vqe = VQEoptimizer(file=f) # file 参数和 print 的 file 参数，为了将中间数据打印并保存到 .o 文件。
            en_list, time_list = [], []

            for i in range(len(geom_list)):
                mol0 = MolecularData(geometry=geom_list[i],
                                    basis=molecule.basis,
                                    charge=molecule.charge,
                                    multiplicity=molecule.multiplicity)
                
                mol = run_pyscf(mol0, run_scf=0, run_ccsd=1, run_fci=0) # 运行得到 FCI 法，得到参考能量值
                
                vqe.generate_circuit(mol)
                vqe.optimize(method='CG', tol=0.032, maxstep=23)
                # ‘Nelder-Mead’‘Powell’‘CG’‘BFGS’‘Newton-CG’‘L-BFGS-B’‘TNC’‘COBYLA’‘SLSQP’‘dogleg’‘trust-ncg’ 
                param_dict = param2dict(vqe.circuit.params_name, vqe.res.x)
                vqe.simulator.apply_circuit(vqe.circuit, param_dict) # 执行设计好的 网络 及 参数
                t = vqe.timer.runtime()
                en = vqe.simulator.get_expectation(Hamiltonian(vqe.hamiltonian)).real
                # print('Our Energy: ', en, 'Time: %i hrs %i mints %.2f sec.' % format_time(t), file=f)
                # print('Precision compared with FCI is:', np.abs(en-mol.fci_energy), file=f)
                sys.stdout.flush()
                en_list.append(en)
                time_list.append(t)

        if len(en_list) == len(geom_list) and len(time_list) == len(geom_list):
            return en_list, time_list 
        else:
            raise ValueError('data lengths are not correct!')
