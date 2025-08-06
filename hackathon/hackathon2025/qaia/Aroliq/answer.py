import os
from time import time
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt

if os.getenv('DEBUG', False):
  from moviepy import *

Variables = Tuple[int, int, bool]   # [theta, nq_alpha, is_opt_amp]
Phases = ndarray
Amplitudes = ndarray

BASE_PATH = Path(__file__).parent


class MLP(nn.Module):
  def __init__(self, d_in:int, d_hid:int):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(d_in, d_hid),
      nn.ReLU(inplace=True),
      nn.Linear(d_hid, 2, bias=False),
    )
  def forward(self, x:Tensor) -> Tensor:
    return self.mlp(x)


if 'load pretrained phase encoder':
    mlp3 = MLP(3, 4).eval()
    mlp3.load_state_dict(torch.load(str(BASE_PATH / 'mlp-3.pth')))
    mlp3.requires_grad_(False)
    mlp4 = MLP(4, 6).eval()
    mlp4.load_state_dict(torch.load(str(BASE_PATH / 'mlp-4.pth')))
    mlp4.requires_grad_(False)


# 我的实验田
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
        self.name = 'armit'
        self.param = param
        self.init_efield()
        self.init_amp_phi()

        self.DEBUG = False
        self.params = []
        self.frames = []
        self.opt_iter = 0
        self.opt_phase = 0

    def init_efield(self) -> Tensor:
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
        self.efield = torch.from_numpy(efield).to(torch.complex64)

    def init_amp_phi(self) -> Tuple[ndarray, ndarray]:
        N = self.param['N']
        phi = np.zeros([N], dtype=np.float32)
        amp = np.ones ([N], dtype=np.float32)
        self.amp, self.phi = amp, phi

    # 后处理：归一化振幅，并且将振幅按照量化比特数的要求进行离散化；将负数振幅转化为正数的形式，同时给相位角加上pi
    def norm_phi_amp(self):
        phi, amp = self.phi, self.amp
        # 确保最后相位角变化为(0,2*pi]
        phi = np.angle(np.exp(1.0j * np.where(amp < 0, phi + np.pi, phi))) + np.pi
        phi_vq = (2 ** self.param['encode_qubit']) / (2 * np.pi)
        phi = np.round(phi * phi_vq) / phi_vq
        # 将振幅归一化
        amp = np.abs(amp)
        amp /= np.max(amp)
        self.phi, self.amp = phi, amp

    def norm_phi_amp_debug(self, phi:Tensor, amp:Tensor) -> Tuple[Tensor, Tensor]:
        # 确保最后相位角变化为(0,2*pi]
        phi = torch.angle(torch.exp(1.0j * torch.where(amp < 0, phi + torch.pi, phi))) + torch.pi
        phi_vq = (2 ** self.param['encode_qubit']) / (2 * torch.pi)
        phi = torch.round(phi * phi_vq) / phi_vq
        # 将振幅归一化
        amp = torch.abs(amp)
        amp /= torch.max(amp)
        return phi, amp

    # 获得theta角度对应的矩阵指标
    def _get_index(self, phi:float) -> int:
        index = round(phi * self.param['n_angle'])
        nlen = self.efield.shape[-1]
        return max(0, min(nlen, index))

    # 相位比特编码函数，须可微分
    def encode(self, x_spin:Tensor) -> Tensor:
        nq = self.param['encode_qubit']
        if nq in [1, 2]:
            return self.encode_arXiv_2409_19938(x_spin)
        else:
            mlp = mlp3 if nq == 3 else mlp4
            out = mlp(x_spin)
            return out[..., 0] + 1j * out[..., 1]

    def encode_arXiv_2409_19938(self, x_spin:Tensor) -> Tensor:
        nq = self.param['encode_qubit']
        if nq == 1:
            '''
            bit spin phi phase
             0   +    0    1
             1   -    π   -1
            '''
            # Eq. 3
            phase = x_spin.to(torch.complex64).squeeze(-1)
        elif nq == 2:
            '''
            bit spin phi  phase
             00  ++   0     1
             01  +-   π/2   i
             11  --   π    -1
             10  -+  -π/2  -i
            '''
            # Eq. 4
            c1 = (1 + 1j) / 2
            c2 = (1 - 1j) / 2
            sp1 = x_spin[..., 0]
            sp2 = x_spin[..., 1]
            phase = c1 * sp1 + c2 * sp2
        elif nq == 3:
            '''
            bit  spin   phi
            000   +++    0
            001   ++-    π/4
            010   +-+    π/2    <- 跳变
            011   +--   3π/4
            111   ---    π
            110   --+  -3π/4
            101   -+-   -π/2    <- 跳变
            100   -+    -π/4
            '''
            # Eq. 5~6 & Fig. 4(b); 注意此时是一个不完全的格雷码
            A = np.sqrt(4 + 2 * np.sqrt(2)) / 4
            B = np.sqrt(4 - 2 * np.sqrt(2)) / 4
            c1 = A * np.exp(1j*(3/8)*np.pi)
            c2 = A * np.exp(1j*(-1/8)*np.pi)
            c3 = B * np.exp(1j*(-1/8)*np.pi)
            c4 = B * np.exp(1j*(-5/8)*np.pi)
            if not 'debug':
                # 编号顺序: 从x正半轴出发逆时针画半圆，然后从x正半轴出发顺时针画半圆
                nums = [bin(e)[2:].rjust(nq, '0') for e in range(2**nq)]
                bits = np.asarray([[int(i) for i in e] for e in nums])
                spin = 1 - 2 * bits
                phase = c1 * spin[:,0] + c2 * spin[:,1] + c3 * spin[:,2] + c4 * spin[:,0] * spin[:,1] * spin[:,2]
                phi = np.angle(phase)
            sp1 = x_spin[..., 0]
            sp2 = x_spin[..., 1]
            sp3 = x_spin[..., 2]
            phase = c1 * sp1 + c2 * sp2 + c3 * sp3 + c4 * sp1 * sp2 * sp3
        return phase

    def loss_fn(self, phase:Tensor, amp:Tensor) -> Tensor:
        # 渲染/调制
        amp = amp / amp.max()
        F = torch.einsum('bi, ij -> bj', amp * phase, self.efield)
        FF = torch.real(F.conj() * F)
        FF_max = torch.max(FF)
        #FF = 10 * torch.log10(FF / FF_max) # 转为dB度量

        # 主瓣峰值强度
        θ = self.param['theta_0']
        y_main_peak = FF[..., self._get_index(θ)]
        # 主瓣谷值强度差 (保证主瓣被检测出!!)
        if False:
            valley_l = self.param['side_lobe_inner'][0][1]
            valley_r = self.param['side_lobe_inner'][1][0]
            y_main_l = FF[..., self._get_index(θ + valley_l)]
            y_main_r = FF[..., self._get_index(θ + valley_r)]
            y_main_max_valley = torch.where(y_main_l > y_main_r, y_main_l, y_main_r)
            # dB_rng 必须超过3dB
            dB_rng = torch.log10(y_main_peak / FF_max) - torch.log10((y_main_max_valley + 1e-9) / FF_max)
            p0 = torch.where(dB_rng > 3, 0, 3 - dB_rng)
        # 内旁瓣强度
        y_side_inner = 0.0
        for left, right in self.param['side_lobe_inner']:
            L = self._get_index(θ + left)
            R = self._get_index(θ + right)
            y_side_inner += FF[..., L:R].max()
        y_side_inner /= len(self.param['side_lobe_inner'])
        # 外旁瓣强度
        y_side_outer = 0.0
        for left, right in self.param['side_lobe_outer']:
            L = self._get_index(θ + left)
            R = self._get_index(θ + right)
            y_side_outer += FF[..., L:R].max()
        y_side_outer /= len(self.param['side_lobe_outer'])
        # 损失函数
        # - 外旁瓣惩罚因子100，宽度120°
        # - 内旁瓣惩罚因子20，宽度60°
        w = self.param['weight']    # 平衡分子分母的量级
        p1 = (1 - w) * (y_side_inner + 1) * 10
        p2 = (1 - w) * (y_side_outer + 1)
        q =       w  *  y_main_peak
        if self.DEBUG and (self.opt_iter + 1) % 500 == 0:
            print('>> p0:', p0.mean().item())
            print('>> p1:', p1.mean().item())
            print('>> p2:', p2.mean().item())
            print('>> q:',  q .mean().item())
        # 乘积形式，要求分子尽量小，分母尽量大
        loss = (p1 + p2) / q

        # https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
        self.opt_iter += 1
        if self.DEBUG and (self.opt_iter + 1) % 100 == 0:
            # params
            best = loss.argmin()
            amp0 = amp[best].detach()
            phi0 = torch.angle(phase[best].detach())
            phi0, amp0 = self.norm_phi_amp_debug(phi0, amp0)
            self.params.append((amp0.numpy(), phi0.numpy()))
            # frames
            FF_o = FF[best].detach()
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

        return loss

    # 基于模拟分叉方法的纯相位优化函数
    def opt_phi(self):
        # 初始化
        B = self.param['B_phi']
        x = np.random.randn(B, self.param['N'], self.param['encode_qubit']).astype(np.float32) * 0.01
        y = np.random.randn(B, self.param['N'], self.param['encode_qubit']).astype(np.float32) * 0.01
        amp = torch.from_numpy(self.amp.astype(np.float32)).unsqueeze(0).expand(B, -1)
        for iter in range(self.param['n_iter']):
            # 计算梯度
            x_torch = torch.tensor(x, requires_grad=True)
            x_sign = x_torch - (x_torch - x_torch.sign()).detach()  # sign()的可微实现
            phase = self.encode(x_sign)
            loss = self.loss_fn(phase, amp).mean()  # mini-batch
            loss.backward()
            x_grad = x_torch.grad.detach()
            x_grad /= torch.linalg.norm(x_grad)
            x_grad = x_grad.numpy()

            # 参数更新 (bSB)
            # - 广义动量: 温度+梯度+步长
            T = iter / self.param['n_iter'] - 0.5
            y += (T * x - self.param['xi'] * x_grad) * self.param['dt']
            # - 广义坐标: 沿动量方向走一步
            x += y * self.param['dt']

            # 终止条件
            y = np.where(np.abs(x) >= 1, 0, y)
            x = np.clip(x, -1, +1)

        with torch.no_grad():
            phase = self.encode(torch.sign(torch.from_numpy(x)))
            loss = self.loss_fn(phase, amp)
            best = loss.argmin()
            if self.DEBUG:
                print('>> best loss phi:', loss[best].item())
            self.phi = np.angle(phase[best].detach().cpu().numpy())

        self.on_opt_hook()

    # 纯振幅优化函数
    def opt_amp(self):
        if self.param['opt_amp'] is False: return

        B = self.param['B_amp']
        phase = torch.from_numpy(np.exp(1j * self.phi).astype(np.complex64)).unsqueeze(0).expand(B, -1)
        phase.requires_grad = False
        amp = torch.from_numpy(self.amp.astype(np.float32)).unsqueeze(0).expand(B, -1)
        amp = amp + torch.rand_like(amp) * 0.01 - 0.005
        amp.requires_grad = True
        optim = Adam([amp], lr=self.param['lr_amp'])

        for iter in range(self.param['n_iter_amp']):
            optim.zero_grad()
            loss = self.loss_fn(phase, amp).mean()
            loss.backward()
            optim.step()

        with torch.no_grad():
            loss = self.loss_fn(phase, amp)
            best = loss.argmin()
            if self.DEBUG:
                print('>> best loss amp:', loss[best].item())
            self.amp = amp[best].detach().cpu().numpy()

        self.on_opt_hook()

    def on_opt_hook(self):
        if not self.DEBUG: return
        self.opt_phase += 1
        variables = [
            self.param['theta_0'],
            self.param['encode_qubit'],
            self.param['opt_amp'],
        ]
        print(f'>> score {self.opt_phase}:', get_score(self.phi, self.amp, variables))

    def plot(self):
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
        if self.param['opt_amp'] is False:
            suffix += '_noamp'
        fn = f'method={self.name}_θ={θ}{suffix}.jpg'
        save_fp = f'./out/{fn}'
        print(f'>> savefig: {save_fp}')
        plt.figure()
        plt.plot(theta, y)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$lg|F(\theta)|^2 - lg|F(\theta)|^2_{max}$ (dB)')
        plt.title(f'method={self.name}_θ={θ}{suffix}')
        plt.savefig(save_fp)
        plt.close()

    def plot_debug(self):
        if not self.DEBUG: return

        os.makedirs('./tmp', exist_ok=True)
        θ = self.param['theta_0']
        nq = self.param['encode_qubit']
        suffix = ''
        if self.param['opt_amp'] is False:
            suffix += '_noamp'

        if self.frames:
            fn = f'method={self.name}_θ={θ}_nq={nq}{suffix}.mp4'
            ImageSequenceClip(self.frames, fps=6).write_videofile(f'./tmp/{fn}')

        amp_history = np.stack([a for a, p in self.params], axis=1)
        phi_history = np.stack([p for a, p in self.params], axis=1)
        fn = f'method={self.name}_θ={θ}_nq={nq}{suffix}_p.jpg'
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

    n_angle = 100
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
    amp_phase = amplitude * np.exp(1.0j * phase_angle)
    F = np.einsum('i, ij -> j', amp_phase, efield)
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
        elif theta_min_up < theta_up or theta_min_down > theta_down:
            # 主瓣须是单峰，这意味着左右极小值必在-30dB检测点之外，即 theta_min_down <= theta_down < peak < theta_up <= theta_min_up
            y = 0
            print('>> assert fail: theta_min_down <= theta_down < peak < theta_up <= theta_min_up')
            print(f'   {theta_min_down} <= {theta_down} < {theta_array[max_index]} < {theta_up} <= {theta_min_up}')
            print(f'The intensity of mainlobe did not decrease to -30 dB')
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

    ts_start = time()
    TIME_LIMIT = 90
    ts_soft_limit = ts_start + TIME_LIMIT - 10
    ts_hard_limit = ts_start + TIME_LIMIT - 3

    param = {
        # 损失函数
        'N': 32,                            # 天线阵子总数 (测试时固定32)
        'n_angle': 9,                       # 1度中被细分的次数 (测试时固定500)
        'theta_0': variables[0],            # 波束成形方向，以90度为例
        'weight': 0.01,                     # 调节损失函数中分子和分母的相对大小
        'side_lobe_inner': [[-30, -3], [3, 30]],     # 需要压制的内旁瓣范围相对于主瓣波束成形方向的角度表示
        'side_lobe_outer': [[-180, -30], [30, 180]], # 需要压制的外旁瓣范围相对于主瓣波束成形方向的角度表示
        # 相位优化 (bSB)
        'encode_qubit': variables[1],       # 进行相位编码的比特个数 (相位量化数)
        'B_phi': 64,
        'n_iter': 1500,                     # 迭代步数
        'xi': 0.15,                         # 模拟分叉算法中调节损失函数的相对大小
        'dt': 0.20,                         # 演化步长
        # 振幅优化 (Adam)
        'opt_amp': variables[2],            # 根据输入控制是否优化振幅
        'B_amp': 8,
        'n_iter_amp': 1000,
        'lr_amp': 0.01,
    }

    best_phi = None
    best_amp = None
    best_score = 0
    A = BF(param)
    while time() < ts_soft_limit:
        A.init_amp_phi()
        A.opt_phi()
        if time() >= ts_hard_limit: break
        A.opt_amp()
        if time() >= ts_hard_limit: break
        A.norm_phi_amp()
        #A.plot()
        #A.plot_debug()
        try:
            score = get_score(A.phi, A.amp, variables)
        except:
            if time() >= ts_hard_limit: break
            else: continue
        #print('single round score:', score)
        if score > best_score:
            best_score = score
            best_phi = A.phi.copy()
            best_amp = A.amp.copy()
    #score = get_score(best_phi, best_amp, variables)
    #print('Local final score:', score)
    return best_phi, best_amp
