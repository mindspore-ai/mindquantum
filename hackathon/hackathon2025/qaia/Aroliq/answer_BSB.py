'''
此样例代码展示了一个纯相位的情况——2比特相位编码（不优化振幅），用来展示如何调用mindquantum中的量子启发式模块QAIA
QAIA代码仓地址：https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/mindquantum.algorithm.qaia.html#module-mindquantum.algorithm.qaia
QAIA代码介绍文档：https://www.mindspore.cn/mindquantum/docs/zh-CN/master/case_library/quantum_annealing_inspired_algorithm.html
'''

import os
import copy
from typing import Tuple, Dict, Any

import torch
import numpy as np
from matplotlib import pyplot as plt
from mindquantum.algorithm.qaia import BSB

Variables = Tuple[int, int, bool]   # [theta, nq_alpha, is_opt_amp]
Phases = np.ndarray
Amplitudes = np.ndarray


# 基于 bSB/dSB 进行波束赋形
class BF():
    '''
    此为基于量子启发算法进行优化的核心部分。
    样例代码进给出了一种优化思路，显然有很多方法可以进一步改善波束赋形的结果，包括但不限于：
        1、合理设置SB算法中的超参数，比如演化步长dt、迭代步数n_iter、损失函数控制系数xi等；
        2、合理选取其他形式的损失函数和损失函数中的超参数，样例代码中给出了一种参考目标函数的形式；
        3、将量子启发算法与其他优化策略结合；
        4、采用模拟分叉（SB）方法之外的其他量子启发算法，比如LQA，模拟伊辛机等。
    其中主要变量分别为如下形式：
        x, y, x_bit: 2维向量, size: param['N'] * param['encode_qubit']
        amp, phase_angle: 1维向量, size: param['N']
        efield: 2维向量, size: param['N'] * (180 * param['n_angle'] + 1)
    '''

    # 初始化变量
    def __init__(self, param:Dict[str, Any]):
        self.param = copy.deepcopy(param)
        EF = self._generate_power_pattern()
        AF = self._generate_array_factor()
        self.efield = torch.tensor(EF[None, ...] * AF)
        # 初始化振幅
        self.amp = np.ones(self.param['N'])

    # 生成单元阵子的辐射电场强度（随角度变化的函数）
    def _generate_power_pattern(self):
        theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        x = 12 * ((theta - 90) / 90) ** 2
        E_dB = -1.0 * np.where(x < 30, x, 30)
        E_theta = 10 ** (E_dB / 10)
        EF = E_theta ** 0.5
        return EF

    # 生成阵因子A_n
    def _generate_array_factor(self):
        theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        phase_x = 1j * np.pi * np.cos(theta * np.pi / 180)
        AF = np.exp(phase_x[None, :] * np.arange(self.param['N'])[:, None])
        return AF

    # 获得theta角度对应的矩阵指标
    def _get_index(self, angle_value):
        index = round(angle_value * self.param['n_angle'])
        return index

    # 比特相位编码函数，可以参考文献：https://arxiv.org/pdf/2409.19938
    def encode(self, x_bit:torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x_bit: 编码前的比特串 (spin形式±1)
        Returns:
            phase: 编码后的相位 (复张量)
        NOTE: 此编码函数仅针对2比特编码的情况
        '''
        c0 = 0.5 + 0.5j
        c1 = 0.5 - 0.5j
        phase = c0 * x_bit[:, 0] + c1 * x_bit[:, 1]
        return phase

    # 相位纯相位优化求解流程
    def solve(self):
        # 优化相位角
        self.x_final = self.opt_phase_QUBO()
        phase = self.encode(self.x_final)
        self.phase_angle = np.angle(phase)

        print(f'phase angle: {self.phase_angle}')
        print(f'amp: {self.amp}')

    # 基于模拟分叉方法的纯相位优化函数（调用mindquantum QAIA模块）
    def opt_phase_QUBO(self):
        '''
        对于QUBO问题，也可直接使用mindquantum中QAIA模块中的相关功能；
        我们采用2比特编码相位编码并忽略振幅，展示通过QAIA中的BSB模块实现相位优化的思路，为了取得更好的优化效果需要进一步改进代码；
        在这里，我们将损失函数主瓣信号强度减去加权的旁瓣信号强度写成哈密顿矩阵J和表示相位的向量x的内积的形式，进而匹配QAIA算法中的相应输入形式（比如这里用的BSB模块）。
        '''

        c1 = 0.5 + 0.5j
        c2 = 0.5 - 0.5j
        # 针对32个相位，对每个相位采用2比特编码，进而用64个实数变量描述损失函数，并且构建对应的64*64的J矩阵
        factor_array = torch.cat((self.efield[:, self._get_index(self.param['theta_0'])] * c1, self.efield[:, self._get_index(self.param['theta_0'])] * c2), dim=0)
        J_enhance = torch.einsum('i, j -> ij ', factor_array.conj(), factor_array)
        J_suppress = 0.0
        for i in range(len(self.param['range_list_weight'])):
            num = 0
            a_0 = 0.0
            for j in range(round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])), round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][1])), 1):
                num += 1
                factor_array = torch.cat((self.efield[:, j] * c1, self.efield[:, j] * c2), dim=0)
                a_0 += torch.einsum('i, j -> ij', factor_array.conj(), factor_array)
            J_suppress += self.param['range_list_weight'][i] * a_0 / num
        # 由矩阵的构造可知J[i, j] = J*[j, i]，因此取实部不影响计算结果
        J = torch.real((self.param['weight'] * J_enhance - (1 - self.param['weight']) * J_suppress)).numpy()

        # 根据损失函数的形式完成J矩阵构建后调用mindquantum中QAIA的BSB模块进行优化
        solver = BSB(np.asarray(J, dtype=np.float64), batch_size=1)
        solver.update()
        x_bit = np.sign(solver.x.reshape(2 * self.param['N'], 1))
        # 将x_bit的形式统一为 self.param['N'] * self.param['encode_qubit'] 这一二维矩阵的形式
        return x_bit.reshape(self.param['N'], self.param['encode_qubit'], order='F')

    # 优化结果画图函数
    def plot(self, filename:str=None):
        # 计算画图相关数据
        theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        F = torch.einsum('i, ij -> j', torch.tensor(self.amp) * np.exp(1.0j * self.phase_angle), self.efield).numpy()
        FF = np.real(F.conj() * F)
        y = 10 * np.log10(FF / np.max(FF))
        # 画图
        os.makedirs('./out', exist_ok=True)
        fn = filename or f'theta={self.param["theta_0"]}.jpg'
        plt.figure()
        plt.plot(theta, y)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$lg|F(\theta)|^2 - lg|F(\theta)|^2_{max}$ (dB)')
        plt.title(f'Beamforming Outcome: theta={self.param["theta_0"]}')
        plt.savefig(f'./out/{fn}')
        plt.close()


def optimized(variables:Variables) -> Tuple[Phases, Amplitudes]:
    """
    相位振幅优化函数，请勿修改或删除函数名称，请勿修改函数的返回值，否则判分失败
    Args:
        variables 一个包含3个元素的列表；
        列表中的第1个元素代表波束成形方向theta_0，45 <= theta_0 <= 135
        列表中的第2个元素代表相位角度取多少比特离散值，可能取值为1,2,3,4
        列表中的第3个元素是控制振幅是否优化，取值为True 或 False
    Returns：
        phase_angle 相位角，32个元素的列表float
        amp 阵子振幅，32个元素的列表float
    本样例代码中采用torch微分库实现了偏导计算
    """

    # 在后续优化过程中使用的参数
    param = {
        'theta_0': variables[0],            # 波束成形方向，以90度为例
        'N': 32,                            # 天线阵子总数
        'n_angle': 10,                      # 1度中被细分的次数
        'encode_qubit': variables[1],       # 进行相位编码的比特个数
        # bSB参数界面
        'xi': 0.1,                          # 模拟分叉算法中调节损失函数的相对大小
        'dt': 0.3,                          # 演化步长
        'n_iter': 2000,                     # 迭代步数
        # 相位损失函数界面
        'weight': 0.01,                     # 调节损失函数中分子和分母的相对大小
        'range_list': [[-30, -6], [6, 30]], # 需要压制的旁瓣范围相对于主瓣波束成形方向的角度表示
        'range_list_weight': [1, 1],        # 每个压制的旁瓣范围各自的权重
        # 连续振幅优化
        'lr': 0.001,                        # 学习率
    }

    A = BF(param)
    A.solve()
    A.plot()

    return A.phase_angle, A.amp
