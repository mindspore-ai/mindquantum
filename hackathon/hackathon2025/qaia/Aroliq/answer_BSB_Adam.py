import os
import copy
from typing import Tuple, Dict, Any

import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

Variables = Tuple[int, int, bool]   # [theta, nq_alpha, is_opt_amp]
Phases = np.ndarray
Amplitudes = np.ndarray


# 基于 bSB/dSB + Adama 进行波束赋形
class BF():
    '''
    此为基于量子启发算法进行优化的核心部分。
    样例代码进给出了一种优化思路，显然有很多方法可以进一步改善波束赋形的结果，包括但不限于：
        1、合理设置SB算法中的超参数，比如演化步长dt、迭代步数n_iter、损失函数控制系数xi等；
        2、合理选取其他形式的损失函数和损失函数中的超参数，样例代码中给出了一种参考目标函数的形式；
        3、将量子启发算法与其他优化策略结合；
        4、采用模拟分叉（SB）方法之外的其他量子启发算法，比如LQA，模拟伊辛机等。
        5、选手可以重点改进样例代码中相位和振幅的优化流程函数、纯相位优化函数、纯振幅优化函数、相位比特编码函数这些函数。

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

    # 相位比特编码函数，可以参考文献：https://arxiv.org/pdf/2409.19938
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
        #N = x_bit.shape[0] # [N=32, D=2]
        #phase.reshape(N)
        return phase        # [N]

    # 相位和振幅的优化求解流程
    def solve(self):
        # 优化相位角
        self.x_final = self.opt_phase()
        phase = self.encode(self.x_final)
        phase_angle = np.angle(phase)

        # 优化振幅
        if self.param['opt_amp_or_not'] is True:
            amp = self.opt_amp(self.amp, phase_angle)
        else:
            amp = self.amp

        # 后处理：归一化振幅，并且将振幅按照量化比特数的要求进行离散化；将负数振幅转化为正数的形式，同时给相位角加上pi
        phase_angle = np.angle(np.exp(1.0j * np.where(amp < 0, phase_angle + np.pi, phase_angle)))
        # 确保最后相位角变化为0到2\pi
        phase_angle += np.pi
        # 将振幅归一化
        amp = np.abs(amp)
        amp /= np.max(amp)

        self.phase_angle = phase_angle
        self.amp = amp
        #print(f'phase angle: {self.phase_angle}')
        #print(f'amp: {self.amp}')

    # 基于模拟分叉方法的纯相位优化函数
    def opt_phase(self):
        # 优化相位的损失函数
        def cost_func(x_bit) -> torch.Tensor:
            '''
            Args:
                x_bit: SB算法中的变量x
            Returns:
                obj: 损失函数的数值
            '''

            phase = self.encode(x_bit)
            amp = torch.from_numpy(self.amp)    # 此时应该都是1

            main_lobe = torch.einsum('i, i -> ', phase * amp, self.efield[:, self._get_index(self.param['theta_0'])])
            loss_2 = 1.0 * self.param['weight'] * torch.real(torch.conj(main_lobe) * main_lobe)
            loss_1 = 0.0
            for i in range(len(self.param['range_list_weight'])):
                one_range = torch.einsum('i, ij -> j', phase * amp, self.efield[:, round(
                    self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])): round(
                    self._get_index(self.param['theta_0'] + self.param['range_list'][i][1]))])
                loss_1 += (1 - self.param['weight']) * self.param['range_list_weight'][i] * (
                    torch.real(torch.conj(one_range) * one_range)).mean()
            # 目标函数需要重点修改，用户可以自定义目标函数，如加权求和等
            obj = loss_1 / loss_2
            return obj

        # 初始化
        x = 0.01 * np.random.randn(self.param['N'], self.param['encode_qubit'])
        y = 0.01 * np.random.randn(self.param['N'], self.param['encode_qubit'])
        for iter in tqdm(range(self.param['n_iter'] + 1)):
            # 计算梯度
            x_torch = torch.tensor(x, requires_grad=True)
            x_sign = x_torch - (x_torch - torch.sign(x_torch)).detach()
            loss = cost_func(x_sign)
            loss.backward()
            x_grad = (x_torch.grad).clone().detach().numpy()
            x_grad /= np.linalg.norm(x_grad)
            # 参数更新
            y += (-(0.5 - iter / self.param['n_iter']) * x - self.param['xi'] * x_grad) * self.param['dt']
            x = x + y * self.param['dt']
            cond = np.abs(x) > 1
            x = np.where(cond, np.sign(x), x)
            y = np.where(cond, np.zeros_like(y), y)
        return np.sign(x)

    # 纯振幅优化函数
    def opt_amp(self, amp:Amplitudes, phase_angle:Phases):
        '''
        Args:
            amp: 优化前的振幅
            phase_angle: 固定的相位角
        Returns:
            amplitude: 优化过后得到的振幅
            loss: 优化过后损失函数的数值
        '''

        # 优化振幅的损失函数
        def cost_func_for_amp(amp:Amplitudes, phase_angle:Phases) -> torch.Tensor:
            '''
            Args:
                amp: 振幅
                phase_angle: 相位角
            Returns:
                obj: 损失函数的数值
            '''

            phase = torch.exp(1.0j * phase_angle)

            main_lobe = torch.einsum('i, i -> ', phase * amp, self.efield[:, self._get_index(self.param['theta_0'])])
            loss_2 = 1.0 * self.param['weight'] * torch.real(torch.conj(main_lobe) * main_lobe)
            loss_1 = 1.0
            for i in range(len(self.param['range_list_weight'])):
                one_range = torch.einsum('i, ij -> j', phase * amp, self.efield[:, round(
                    self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])): round(
                    self._get_index(self.param['theta_0'] + self.param['range_list'][i][1]))])
                loss_1 += (1 - self.param['weight']) * self.param['range_list_weight'][i] * (
                    torch.real(torch.conj(one_range) * one_range)).mean()
            # 目标函数需要重点修改，用户可以自定义目标函数，如加权求和等
            obj = loss_1 / loss_2
            return obj

        amplitude = torch.from_numpy(amp)
        amplitude.requires_grad = True
        optimizer = torch.optim.Adam([amplitude], lr=self.param['lr'])
        for iter in tqdm(range(1000)):
            optimizer.zero_grad()
            loss = cost_func_for_amp(amplitude, torch.tensor(phase_angle))
            loss.backward()
            optimizer.step()
        return amplitude.clone().detach().numpy()

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
    本样例代码中采用torch微分库实现了偏导计算。
    """

    # 在后续优化过程中使用的参数
    param = {
        'theta_0': variables[0],            # 波束成形方向，以90度为例
        'N': 32,                            # 天线阵子总数 (测试时取32)
        'n_angle': 10,                      # 1度中被细分的次数 (测试时取500)
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
        'opt_amp_or_not': variables[2],     # 根据输入控制是否优化振幅
        'lr': 0.001,                        # 学习率
    }

    A = BF(param)
    A.solve()
    A.plot()

    return A.phase_angle, A.amp
