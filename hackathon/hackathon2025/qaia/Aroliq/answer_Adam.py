import os
from typing import Tuple, Dict, Any

import torch
from torch import Tensor
from torch.optim import Adam
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from matplotlib import pyplot as plt
from moviepy import *

device = 'cpu'

Variables = Tuple[int, int, bool]   # [theta_0, n_bit_phase, is_opt_amp]
Phases = ndarray
Amplitudes = ndarray


# 经典基线: 基于 Adama 进行波束赋形
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

    def __init__(self, param:Dict[str, Any]):
        self.param = param
        self.efield = self.make_efield()
        self.amp, self.phi = self.make_init_amp_phi()

        self.DEBUG = False
        self.params = []
        self.frames = []
        self.opt_iter = 0

    def make_efield(self) -> Tensor:
        theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        x = 12 * ((theta - 90) / 90) ** 2
        E_dB = np.where(x < 30, x, 30)
        E_theta = 10 ** (-E_dB / 10)
        # 生成单元阵子的辐射电场强度（随角度变化的函数）: [D=1801]，每个天线阵子独立产生的一个辐射场(像个钟形曲线的概率分布列)
        EF = np.sqrt(E_theta)
        phase_x = 1j * np.pi * np.cos(np.deg2rad(theta))
        # 生成阵因子A_n : [N=32, D=1801], N天线阵子数，D各方位角度细分数
        AF = np.exp(phase_x[None, :] * np.arange(self.param['N'])[:, None])
        efield = EF[None, ...] * AF
        return torch.from_numpy(efield).to(device, torch.complex64)

    def make_init_amp_phi(self) -> Tuple[ndarray, ndarray]:
        N = self.param['N']
        # 优化时: 相角值域不加约束 (反正是个周期函数)
        # 优化后: 相角须正则化到值域 [0,2*pi]
        phi = 0.01 * np.random.uniform(low=-1, high=1, size=[N])
        if self.param['opt_amp_or_not'] is True:
            # 优化时: 振幅值域约束到 [-1,1]
            # 优化后: 振幅须归一化到值域 [0,1], 振幅加负号 等价于 相角加pi
            amp = 0.01 * np.random.uniform(low=0, high=1, size=[N]) + 1
        else:
            amp = np.ones([N], dtype=np.float32)
        return amp, phi

    # 值域规范化
    def normalize_amp_phi(self):
        phi, amp = self.phi, self.amp
        for i in range(len(phi)):
            if amp[i] < 0:
                amp[i] = -amp[i]
                phi[i] += np.pi
        n_bit_phase = self.param['encode_qubit']
        phi = np.angle(np.exp(1.0j * phi)) + np.pi  # [-pi, pi] => [0, 2*pi]
        phi = np.round(phi / (2 * np.pi) * (2 ** n_bit_phase)) / (2 ** n_bit_phase) * (2 * np.pi)   # 量化
        self.phi, self.amp = phi, amp

    @torch.no_grad
    def normalize_amp_phi_debug(self, phi:Tensor, amp:Tensor) -> Tuple[Tensor, Tensor]:
        phi = torch.angle(np.exp(1.0j * torch.where(amp < 0, phi + torch.pi, phi))) + torch.pi
        n_bit_phase = self.param['encode_qubit']
        phi_vq = (2 ** n_bit_phase) / (2 * torch.pi)
        phi = torch.round(phi * phi_vq) / phi_vq
        amp = torch.abs(amp)
        amp = amp / torch.max(amp)
        return phi, amp

    # 获得theta角度对应的矩阵指标
    def _get_index(self, angle_value:float) -> int:
        idx = round(angle_value * self.param['n_angle'])    # 角度数 * 每1°细分数
        nlen = self.efield.shape[-1]
        return max(0, min(nlen, idx))

    def loss_fn(self, phi:Tensor, amp:Tensor) -> Tensor:
        # 值域约束
        #amp = amp / torch.abs(amp).max()
        amp = amp / amp.max()
        # 渲染/调制
        phase = torch.exp(1.0j * phi)
        F = torch.einsum('i, ij -> j', amp * phase, self.efield)
        FF = torch.real(F.conj() * F)
        FF_max = torch.max(FF)
        #FF = 10 * torch.log10(FF / FF_max) # 转为dB度量

        # https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
        self.opt_iter += 1
        if self.DEBUG and (self.opt_iter + 1) % 1000 == 0:
            # params
            phi, amp = self.normalize_amp_phi_debug(phi.detach(), amp.detach())
            self.params.append((amp.numpy(), phi.numpy()))
            # frames
            FF_o = FF.detach()
            FF_n = 10 * torch.log10(FF_o / torch.max(FF_o))
            plt.clf()
            fig = plt.figure()
            ax1 = fig.add_subplot(211) ; ax1.plot(FF_o.cpu().numpy()) ; ax1.set_title('linear')
            ax2 = fig.add_subplot(212) ; ax2.plot(FF_n.cpu().numpy()) ; ax2.set_title('log-scale')
            fig.canvas.draw()
            fig.tight_layout()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
            self.frames.append(frame)
            plt.close()

        # 主瓣峰值强度
        θ = self.param['theta_0']
        y_main_peak = FF[self._get_index(θ)]
        # 主瓣谷值强度差 (保证主瓣被检测出!!)
        valley_l = self.param['side_lobe_inner'][0][1]
        valley_r = self.param['side_lobe_inner'][1][0]
        y_main_l = FF[self._get_index(θ + valley_l)]
        y_main_r = FF[self._get_index(θ + valley_r)]
        y_main_max_valley = torch.where(y_main_l > y_main_r, y_main_l, y_main_r)
        # dB_rng 必须超过3dB
        dB_rng = torch.log10(y_main_peak / FF_max) - torch.log10((y_main_max_valley + 1e-9) / FF_max)
        p0 = torch.where(dB_rng > 3, 0, 3 - dB_rng)
        # 内旁瓣强度
        y_side_inner = 0.0
        for left, right in self.param['side_lobe_inner']:
            L = self._get_index(θ + left)
            R = self._get_index(θ + right)
            y_side_inner += FF[L:R].max()
        y_side_inner /= len(self.param['side_lobe_inner'])
        # 外旁瓣强度
        y_side_outer = 0.0
        for left, right in self.param['side_lobe_outer']:
            L = self._get_index(θ + left)
            R = self._get_index(θ + right)
            y_side_outer += FF[L:R].max()
        y_side_outer /= len(self.param['side_lobe_outer'])
        # 损失函数
        # - 外旁瓣惩罚因子100，宽度120°
        # - 内旁瓣惩罚因子20，宽度60°
        w = self.param['weight']    # 平衡分子分母的量级
        p1 = (1 - w) * (y_side_inner + 1) * 10 * 2
        p2 = (1 - w) * (y_side_outer + 1)
        q =       w  *  y_main_peak
        if self.DEBUG and (self.opt_iter + 1) % 10000 == 0:
            print('>> p0:', p0.item())
            print('>> p1:', p1.item())
            print('>> p2:', p2.item())
            print('>> q:', q.item())
        # 乘积形式，要求分子尽量小，分母尽量大
        return (p0 + p1 + p2) / q

    def opt_phi_amp(self):
        phi = torch.from_numpy(self.phi).to(device).float()
        amp = torch.from_numpy(self.amp).to(device).float()
        if self.param['opt_amp_or_not'] is True:
            phi.requires_grad = True
            amp.requires_grad = True
            optim = Adam([phi, amp], lr=self.param['lr_phi'])
        else:
            phi.requires_grad = True
            amp.requires_grad = False
            optim = Adam([phi], lr=self.param['lr_phi'])

        for i in tqdm(range(self.param['n_iter_phi'])):
            optim.zero_grad()
            loss = self.loss_fn(phi, amp)
            loss.backward()
            optim.step()
            #if (i + 1) % 1000 == 0: print(f'>> [step {i + 1}] {loss.item():.5f}')

        self.phi = phi.detach().cpu().numpy()
        self.amp = amp.detach().cpu().numpy()
        self.normalize_amp_phi()

        if False:
            print('[opt_amp_phi]')
            print(f'  phi: {self.phi}')
            print(f'  amp: {self.amp}')

    def opt_amp(self):
        # 第一阶段优化保证phi已量化
        phi = torch.from_numpy(self.phi).to(device).float()
        phi.requires_grad = False
        amp = torch.from_numpy(self.amp).to(device).float()
        amp.requires_grad = True
        optim = Adam([amp], lr=self.param['lr_amp'] * 10)

        for i in tqdm(range(self.param['n_iter_amp'])):
            optim.zero_grad()
            loss = self.loss_fn(phi, amp)
            loss.backward()
            optim.step()
            #if (i + 1) % 1000 == 0: print(f'>> [step {i + 1}] {loss.item():.5f}')

        self.amp = amp.detach().cpu().numpy()
        self.normalize_amp_phi()

        if False:
            print('[opt_amp]')
            print(f'  amp: {self.amp}')

    def plot(self, filename:str=None):
        # 计算画图相关数据
        theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        # [N=32] @ [N=32, D=1801] => [D=1801]
        F = np.einsum('i, ij -> j', self.amp * np.exp(1.0j * self.phi), self.efield.cpu().numpy())
        FF = np.real(F.conj() * F)
        y = 10 * np.log10(FF / np.max(FF))
        # 画图
        os.makedirs('./out', exist_ok=True)
        θ = self.param["theta_0"]
        suffix = ''
        if self.param['opt_amp_or_not'] is False:
            suffix += '_noamp'
        fn = filename or f'method=Adam_θ={θ}{suffix}.jpg'
        save_fp = f'./out/{fn}'
        print(f'>> savefig: {save_fp}')
        plt.figure()
        plt.plot(theta, y)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$lg|F(\theta)|^2 - lg|F(\theta)|^2_{max}$ (dB)')
        plt.title(f'method=Adam_θ={θ}{suffix}')
        plt.savefig(save_fp)
        plt.close()

    def plot_debug(self):
        if not self.DEBUG: return

        os.makedirs('./tmp', exist_ok=True)
        θ = self.param["theta_0"]
        suffix = ''
        if self.param['opt_amp_or_not'] is False:
            suffix += '_noamp'

        if self.frames:
            fn = f'method=Adam_θ={θ}{suffix}.mp4'
            ImageSequenceClip(self.frames, fps=6).write_videofile(f'./tmp/{fn}')

        amp_history = np.stack([a for a, p in self.params], axis=1)
        phi_history = np.stack([p for a, p in self.params], axis=1)
        fn = f'method=Adam_θ={θ}{suffix}_p.jpg'
        plt.clf()
        plt.subplot(211)
        for amp in amp_history: plt.plot(amp)
        plt.title('amp')
        plt.subplot(212)
        for phi in phi_history: plt.plot(phi)
        plt.title('phi')
        plt.tight_layout()
        plt.savefig(f'./tmp/{fn}', dpi=400)
        plt.close()


''' ↓↓↓ score utils '''

def get_efield(n_angle:int=500, N:int=32):
    # Eq. 3~4
    theta = np.linspace(0, 180, 180 * n_angle + 1)  # [90001]
    x = 12 * ((theta - 90) / 90) ** 2               # [90001], 下凹双曲线
    E_dB = -1.0 * np.where(x < 30, x, 30)           # [90001], 上凸双曲线
    E_theta = 10 ** (E_dB / 10)                     # [90001], 钟形/山坡
    EF = E_theta ** 0.5                             # [90001], WHY??
    # Eq. 2
    phase_x = 1j * np.pi * np.cos(theta * np.pi / 180)
    AF = np.exp(phase_x[None, :] * np.arange(N)[:, None])   # [32, 90001]
    # Eq. 1
    efield = EF[None, ...] * AF                             # [32, 90001]
    return efield

def get_score(phase_angle:Phases, amplitude:Amplitudes, variables:Variables) -> float:
    """
    打分函数,请参赛选手不要修改此函数和其中调用的任何函数
    Args:
        phase_angle 相位角，32个元素的列表float，取值为0到2\pi的弧度制
        amplitude  阵子振幅，32个元素的列表float
        variables  参数变量列表，变量列表中包含3个元素
            第一个元素是 theta_0 信号方向，float数
            第二个元素是 相位量化比特数，取值为 1, 2, 3, 4
            第三个元素是 控制振幅是否优化，取值为True 或 False
    Returns：
        单个角度对应优化参数的分数
    """

    n_angle = 500
    N = 32
    theta_0 = variables[0]
    n_bit_phase = variables[1]
    opt_amp_or_not = variables[2]

    # 确保相位和振幅是按照赛题要求的取值
    if opt_amp_or_not is True:
        amplitude = amplitude / np.max(amplitude)
    else:
        amplitude = np.ones(N)
    phase_angle = np.angle(np.exp(1.0j * phase_angle)) + np.pi
    phase_angle = np.round(phase_angle / (2 * np.pi) * (2 ** n_bit_phase)) / (2 ** n_bit_phase) * (2 * np.pi)

    efield = get_efield(n_angle, N)
    theta_array = np.linspace(0, 180, 180 * n_angle + 1)
    amp_phase = []
    for i in range(N):
        amp_phase.append(amplitude[i] * np.exp(1.0j * phase_angle[i]))
    F = np.einsum('i, ij -> j', np.array(amp_phase), efield)
    FF = np.real(F.conj() * F)
    # 峰值移到0dB
    db_array = 10 * np.log10(FF / np.max(FF))

    # [0°,180°] 位置范围，把θ移到原点0
    x = theta_array - theta_0
    value_list = []
    for i in range(theta_array.shape[0]):
        # >30°外旁瓣区域
        if abs(x[i]) >= 30:
            # 须在峰值-15dB下，故+15后不应大于0
            value_list.append(db_array[i] + 15)
    # 外旁瓣惩罚
    a = max(np.max(value_list), 0)

    # 峰值位置
    target = np.max(db_array)
    for i in range(theta_array.shape[0]):
        if db_array[i] == target:
            max_index = i
            break

    # 主瓣两侧最近的压制到 -30dB 的位置
    theta_up = 180
    theta_down = 0
    # 主瓣两侧最近的局部最小值位置
    theta_min_up = 180
    theta_min_down = 0
    # 峰值位置与目标位置相差大于1°，直接没分
    if abs(theta_array[max_index] - theta_0) > 1:
        y = 0
        print(f'Incorrect beamforming direction: {theta_array[max_index]}, with target: {theta_0}')
        print(f'final score: {y}')
    else:
        for i in range(1, 10000):
            if db_array[i + max_index] <= -30:
                theta_up = theta_array[i + max_index]
                break

        for i in range(1, 10000):
            if db_array[-i + max_index] <= -30:
                theta_down = theta_array[-i + max_index]
                break

        for i in range(1, 10000):
            if (db_array[i + max_index] < db_array[i - 1 + max_index]) and (db_array[i + max_index] < db_array[i + 1 + max_index]):
                theta_min_up = theta_array[i + max_index]
                break

        for i in range(1, 10000):
            if (db_array[-i + max_index] < db_array[-i - 1 + max_index]) and (db_array[-i + max_index] < db_array[-i + 1 + max_index]):
                theta_min_down = theta_array[-i + max_index]
                break

        if theta_up == 180 or theta_down == 0:
            # 主瓣在这里意味着一个高出左右地平线 30dB 的尖峰
            y = 0
            print(f'Failed to identify expected mainlobe.')
            print(f'final score: {y}')
        elif theta_min_up < theta_up or theta_min_down > theta_down:
            # 主瓣须是单峰，这意味着左右极小值必在-30dB检测点之外，即 theta_min_down <= theta_down < peak < theta_up <= theta_min_up
            y = 0
            print('>> assert fail: theta_min_down <= theta_down < peak < theta_up <= theta_min_up')
            print(f'   {theta_min_down} <= {theta_down} < {theta_array[max_index]} < {theta_up} <= {theta_min_up}')
            print(f'The intensity of mainlobe did not decrease to -30 dB')
            print(f'final score: {y}')
        else:
            # 主瓣宽度
            W = theta_up - theta_down
            # 大于6°有惩罚
            b = max(W - 6, 0)

            value_list_2 = []
            for i in range(theta_array.shape[0]):
                # <30°内旁瓣区域 and >主瓣左右极小值外(的非主瓣)区域
                if abs(x[i]) <= 30 and (x[i] >= theta_min_up - theta_0 or x[i] <= theta_min_down - theta_0):
                    # 须在峰值-30dB下，故+30后不应大于0
                    value_list_2.append(db_array[i] + 30)
            # 外旁瓣惩罚
            c = np.max(value_list_2)

            y_sum = 1000 - 100 * a - 80 * b - 20 * c
            # 负分直接归为0分
            y = max(y_sum, 0)
            print(f'W = {W:.5f}, a = {a:.5f}, b = {b:.5f}, c = {c:.5f}; y_sum = {y_sum:.5f}, final score: y = {y:.5f}')
    return float(y)

''' ↑↑↑ score utils '''


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
        'N': 32,                            # 天线阵子总数 (测试时取32)
        'n_angle': 7,                       # 1度中被细分的次数 (测试时取500)
        'theta_0': variables[0],            # 波束成形方向，以90度为例
        'encode_qubit': variables[1],       # 进行相位编码的比特个数
        # 损失函数
        'weight': 0.01,                              # 调节损失函数中分子和分母的相对大小
        'side_lobe_inner': [[-30, -3], [3, 30]],     # 需要压制的内旁瓣范围相对于主瓣波束成形方向的角度表示
        'side_lobe_outer': [[-180, -30], [30, 180]], # 需要压制的外旁瓣范围相对于主瓣波束成形方向的角度表示
        # 优化器
        'opt_amp_or_not': variables[2],     # 根据输入控制是否优化振幅
        'n_iter_phi': 45000,                # Adam 需要很长的时间收敛
        'n_iter_amp': 15000,
        'lr_phi': 0.005,
        'lr_amp': 0.01,
    }

    A = BF(param)
    A.opt_phi_amp()     # joint optim phi & amp
    print('>> score 1:', get_score(A.phi, A.amp, variables))
    if param['opt_amp_or_not'] is True:
        A.opt_amp()         # fix phi, refine amp
        print('>> score 2:', get_score(A.phi, A.amp, variables))
    #A.plot()
    #A.plot_debug()
    return A.phi, A.amp
