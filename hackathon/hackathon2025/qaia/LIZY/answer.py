'''
LIZY-2025-Mindspore-hackathon-wireless-决赛最终代码-answer.py
'''

import numpy as np
import torch
from matplotlib import pyplot as plt
import copy
import time

from sympy.abc import alpha
from tqdm import tqdm

def optimized(variables):
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
        'theta_0': variables[0],  # 波束成形方向，以90度为例
        'N': 32,  # 天线阵子总数
        'n_angle': 10,  # 1度中被细分的次数
        'encode_qubit': variables[1],  # 进行相位编码的比特个数，样例代码中固定为2，实际对应变量 variables[1]

        # bSB参数界面
        'xi': 0.2,  # 模拟分叉算法中调节损失函数的相对大小
        'dt': 0.8,  # 演化步长
        'n_iter': 2750,  # 迭代步数
        'batch_size': 65,

        # 相位损失函数界面
        'weight': 0.01,  # 调节损失函数中分子和分母的相对大小
        'range_list': [[-30, -6], [6, 30]], # 需要压制的旁瓣范围相对于主瓣波束成形方向的角度表示
        'range_list_weight': [1, 1], # 每个压制的旁瓣范围各自的权重

        # 连续振幅优化
        'opt_amp_or_not': variables[2],
        'lr': 0.005, # 学习率
    }

    A = BF(param)
    A.solve()
    A.plot()

    phase_angle = A.phase_angle
    amp = A.amp
    score = A.score

    return phase_angle, amp


# 计算角度制下的三角余弦函数
def cosd_f(x):
    return np.cos(x * np.pi / 180)

# 计算角度制下的三角正弦函数
def sind_f(x):
    return np.sin(x * np.pi / 180)

# 基于bSB或dSB算法进行波束赋形
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
    # ============================================================================================================================ #
    def __init__(self, param):
        self.param = copy.deepcopy(param)
        self.EF = self._generate_power_pattern() # 生成单元阵子的辐射电场强度 EF E(theta)
        self.AF = self._generate_array_factor() # 生成阵因子 AF A_n(theta)
        self.amp = np.ones(self.param['N']) # 幅度amp初始化为全1
        self.efield = torch.tensor(self.EF[None, ...] * self.AF) # 构造天线方向响应张量

    # 相位纯相位优化求解流程
    # ============================================================================================================================ #
    def solve(self):
        # QIA独立优化相位
        x_final_QIA, score_max_QIA = self.opt_phase_QUBO()
        phase_QIA = self.encode(x_final_QIA)
        phase_angle_QIA = np.angle(phase_QIA)

        # Torch独立优化相位与振幅
        x_final_Tor = self.opt_phase()
        phase_Tor = self.encode(x_final_Tor)
        phase_angle_Tor = np.angle(phase_Tor)
        amp_Tor, _ = self.opt_amp(self.amp, phase_angle_Tor)
        amp_Tor = np.array(amp_Tor.clone().detach().numpy())
        cond = amp_Tor < 0
        phase_angle_Tor = np.angle(np.exp(1.0j * np.where(cond, phase_angle_Tor + np.pi, phase_angle_Tor)))
        phase_angle_Tor = phase_angle_Tor + np.pi  # 确保最后相位角变化为0到2\pi
        amp_Tor = np.abs(amp_Tor) / np.max(np.abs(amp_Tor))  # 将振幅归一化
        score_max_Tor = get_score(phase_angle_Tor, amp_Tor, self.param['theta_0'], self.param['encode_qubit'])

        # QIA优化相位+Torch优化振幅
        amp_QIA_Tor, _ = self.opt_amp(self.amp, phase_angle_QIA)
        amp_QIA_Tor = np.array(amp_QIA_Tor.clone().detach().numpy())
        cond = amp_QIA_Tor < 0
        phase_angle_QIA_Tor = np.angle(np.exp(1.0j * np.where(cond, phase_angle_QIA + np.pi, phase_angle_QIA)))
        phase_angle_QIA_Tor = phase_angle_QIA_Tor + np.pi  # 确保最后相位角变化为0到2\pi
        amp_QIA_Tor = np.abs(amp_QIA_Tor) / np.max(np.abs(amp_QIA_Tor))  # 将振幅归一化
        score_max_QIA_Tor = get_score(phase_angle_QIA_Tor, amp_QIA_Tor, self.param['theta_0'],
                                      self.param['encode_qubit'])

        # score_max_QIA_Tor = 0
        # score_max_QIA = 0
        # score_max_Tor = 0

        scores = [score_max_QIA, score_max_Tor, score_max_QIA_Tor]
        phases = [phase_angle_QIA, phase_angle_Tor, phase_angle_QIA_Tor]
        amps = [np.ones(self.param['N']), amp_Tor, amp_QIA_Tor]  # self.amp 对应 QIA 的原始振幅

        max_idx = np.argmax(scores)
        # print('max_idx', max_idx)
        self.phase_angle = phases[max_idx]
        self.amp = amps[max_idx]
        self.score = scores[max_idx]

        print(scores)
        # print(f'phase angle: {self.phase_angle}')
        # print(f'amp: {self.amp}')

    # 基于模拟分叉方法的纯相位优化函数（调用mindquantum QAIA模块）
    # ============================================================================================================================ #
    def opt_phase_QUBO(self):
        from mindquantum.algorithm.qaia import BSB, DSB, CFC, SimCIM, SFC, ASB, LQA

        def test_score(x_total):
            score = []
            for i in range(x_total.shape[1]):
                x_bit_temp = x_total[:, i:i + 1]
                x_final_temp = x_bit_temp.reshape(self.param['N'], 2**(self.param['encode_qubit']-1), order='F')
                phase_temp = self.encode(x_final_temp)
                phase_angle_temp = np.angle(phase_temp)
                # amptitude = self.opt_amp(self.amp, phase_angle_temp)

                score.append(get_score(phase_angle_temp, np.ones(self.param['N']), self.param['theta_0'], self.param['encode_qubit']))
                # score.append(get_score(phase_angle_temp, amptitude, self.param['theta_0']))

            best_batch_idx = np.argmax(score)
            score_max = np.max(score)
            x_bit = np.sign(x_total[:, best_batch_idx:best_batch_idx + 1])  # 形状: (64, 1)
            return x_bit, score_max

        '''
            对于QUBO问题，也可直接使用mindquantum中QAIA模块中的相关功能；
            我们采用2比特编码相位编码并忽略振幅，展示通过QAIA中的BSB模块实现相位优化的思路，为了取得更好的优化效果需要进一步改进代码；
            在这里，我们将损失函数主瓣信号强度减去加权的旁瓣信号强度写成哈密顿矩阵J和表示相位的向量x的内积的形式，进而匹配QAIA算法中的相应输入形式（比如这里用的BSB模块）。 
        '''
        if self.param['encode_qubit'] == 2 or self.param['encode_qubit'] == 1:
            c1 = 0.5 + 0.5j
            c2 = 0.5 - 0.5j
            # 针对32个相位，对每个相位采用2比特编码，进而用64个实数变量描述损失函数，并且构建对应的64*64的J矩阵
            factor_array = torch.cat((self.efield[:, self._get_index(self.param['theta_0'])] * c1,self.efield[:, self._get_index(self.param['theta_0'])] * c2), dim=0)
            J_enhance = torch.einsum('i, j -> ij ', factor_array.conj(), factor_array)
            J_suppress = 0.0
            for i in range(len(self.param['range_list_weight'])):
                num = 0
                a_0 = 0.0
                for j in range(round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])),round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][1])), 1):
                    num += 1
                    factor_array = torch.cat((self.efield[:, j] * c1, self.efield[:, j] * c2), dim=0)
                    a_0 += torch.einsum('i, j -> ij', factor_array.conj(), factor_array)
                J_suppress += self.param['range_list_weight'][i] * a_0 / num
            J = torch.real((self.param['weight'] * J_enhance - (1 - self.param['weight']) * J_suppress)).numpy()  # 由矩阵的构造可知J[i, j] = J*[j, i]，因此取实部不影响计算结果

        else:
            c1 = 0.5 + 0j
            c2 = 0 + 0.5j
            c3 = 0.5 + 0j
            c4 = 0 + 0.5j
            # c1 = 0.25 + 0.6036j
            # c2 = 0.6036 - 0.25j
            # c3 = 0.25 - 0.1036j
            # c4 = - 0.1036 - 0.25j
            factor_array = torch.cat((
                self.efield[:, self._get_index(self.param['theta_0'])] * c1,
                self.efield[:, self._get_index(self.param['theta_0'])] * c2,
                self.efield[:, self._get_index(self.param['theta_0'])] * c3,
                self.efield[:, self._get_index(self.param['theta_0'])] * c4), dim=0)
            J_enhance = torch.einsum('i, j -> ij ', factor_array.conj(), factor_array)
            J_suppress = 0.0
            for i in range(len(self.param['range_list_weight'])):
                num = 0
                a_0 = 0.0
                for j in range(round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])),
                               round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][1])), 1):
                    num += 1
                    factor_array = torch.cat((
                        self.efield[:, j] * c1,
                        self.efield[:, j] * c2,
                        self.efield[:, j] * c3,
                        self.efield[:, j] * c4), dim=0)
                    a_0 += torch.einsum('i, j -> ij', factor_array.conj(), factor_array)
                J_suppress += self.param['range_list_weight'][i] * a_0 / num
            J = torch.real((self.param['weight'] * J_enhance - (
                        1 - self.param['weight']) * J_suppress)).numpy()  # 由矩阵的构造可知J[i, j] = J*[j, i]，因此取实部不影响计算结果

        x_bit_temp = []
        score_max_temp = []
        # 根据损失函数的形式完成J矩阵构建后调用mindquantum中QAIA的BSB模块进行优化
        batch_size = self.param['batch_size']

        solver = BSB(np.array(J, dtype="float64"), batch_size=batch_size, xi=0.587, dt=0.405)
        # solver = BSB(np.array(J, dtype="float64"), batch_size=batch_size)
        solver.update()
        x_total = np.sign(solver.x)
        x_total = np.unique(x_total.T, axis=0)
        x_total = x_total.T
        x_bit, score_max = test_score(x_total)
        x_bit_temp.append(x_bit), score_max_temp.append(score_max)

        solver = DSB(np.array(J, dtype="float64"), batch_size=batch_size, xi=0.716, dt=0.019)
        # solver = DSB(np.array(J, dtype="float64"), batch_size=batch_size)
        solver.update()
        x_total = np.sign(solver.x)
        x_total = np.unique(x_total.T, axis=0)
        x_total = x_total.T
        x_bit, score_max = test_score(x_total)
        x_bit_temp.append(x_bit), score_max_temp.append(score_max)

        solver = CFC(np.array(J, dtype="float64"), batch_size=batch_size, dt=0.013)
        # solver = CFC(np.array(J, dtype="float64"), batch_size=batch_size)
        solver.update()
        x_total = np.sign(solver.x)
        x_total = np.unique(x_total.T, axis=0)
        x_total = x_total.T
        x_bit, score_max = test_score(x_total)
        x_bit_temp.append(x_bit), score_max_temp.append(score_max)

        solver = SimCIM(np.array(J, dtype="float64"), batch_size=batch_size, dt=1.122, sigma=0.048, pt=7.432)
        # solver = SimCIM(np.array(J, dtype="float64"), batch_size=batch_size)
        solver.update()
        x_total = np.sign(solver.x)
        x_total = np.unique(x_total.T, axis=0)
        x_total = x_total.T
        x_bit, score_max = test_score(x_total)
        x_bit_temp.append(x_bit), score_max_temp.append(score_max)

        solver = SFC(np.array(J, dtype="float64"), batch_size=batch_size, dt=0.186, k=0.122)
        # solver = SFC(np.array(J, dtype="float64"), batch_size=batch_size)
        solver.update()
        x_total = np.sign(solver.x)
        x_total = np.unique(x_total.T, axis=0)
        x_total = x_total.T
        x_bit, score_max = test_score(x_total)
        x_bit_temp.append(x_bit), score_max_temp.append(score_max)

        solver = ASB(np.array(J, dtype="float64"), batch_size=batch_size, xi=1.709, dt=0.129)
        # solver = ASB(np.array(J, dtype="float64"), batch_size=batch_size)
        solver.update()
        x_total = np.sign(solver.x)
        x_total = np.unique(x_total.T, axis=0)
        x_total = x_total.T
        x_bit, score_max = test_score(x_total)
        x_bit_temp.append(x_bit), score_max_temp.append(score_max)

        solver = LQA(np.array(J, dtype="float64"), batch_size=batch_size, dt=0.018, gamma=0.487)
        # solver = LQA(np.array(J, dtype="float64"), batch_size=batch_size)
        solver.update()
        x_total = np.sign(solver.x)
        x_total = np.unique(x_total.T, axis=0)
        x_total = x_total.T
        x_bit, score_max = test_score(x_total)
        x_bit_temp.append(x_bit), score_max_temp.append(score_max)

        best_batch_idx = np.argmax(score_max_temp)
        score_max = np.max(score_max_temp)
        x_bit = np.sign(x_bit_temp[best_batch_idx])  # 形状: (64, 1)

        return x_bit.reshape(self.param['N'], 2**(self.param['encode_qubit']-1), order='F'), score_max  # 将x_bit的形式统一为 self.param['N'] * self.param['encode_qubit'] 这一二维矩阵的形式

    # 基于模拟分叉方法的纯相位优化函数
    # ============================================================================================================================ #
    def opt_phase(self):
        # 优化相位的损失函数
        def cost_func(x_bit):
            '''
            Args:
                x_bit: SB算法中的变量x
            Returns:
                obj: 损失函数的数值
            '''
            phase = self.encode(x_bit)
            amp = torch.tensor(self.amp.copy())

            main_lobe = torch.einsum('i, i -> ', phase * amp,
                                     self.efield[:, self._get_index(self.param['theta_0'])])
            loss_2 = 1.0 * self.param['weight'] * torch.real(torch.conj(main_lobe) * main_lobe)
            loss_1 = 0.0
            for i in range(len(self.param['range_list_weight'])):
                one_range = torch.einsum('i, ij -> j', phase * amp, self.efield[:, round(
                    self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])): round(
                    self._get_index(self.param['theta_0'] + self.param['range_list'][i][1]))])
                loss_1 += (1 - self.param['weight']) * self.param['range_list_weight'][i] * (
                    torch.real(torch.conj(one_range) * one_range)).mean()
            obj = loss_1 / loss_2  # 目标函数需要重点修改，用户可以自定义目标函数，如加权求和等
            return obj

            # 初始化

        x = 0.01 * (np.random.randn(self.param['N'], 2**(self.param['encode_qubit']-1)))
        y = 0.01 * (np.random.randn(self.param['N'], 2**(self.param['encode_qubit']-1)))

        for iter in tqdm(range(self.param['n_iter'] + 1)):
            x_torch = torch.tensor(x)
            x_torch.requires_grad = True

            # 计算梯度与更新参数
            x_sign = x_torch - (x_torch - torch.sign(x_torch)).detach()  # dSB
            loss = cost_func(x_sign)
            loss.backward()
            x_grad = (x_torch.grad).clone().detach().numpy()
            y += (-(0.5 - iter / self.param['n_iter']) * x - self.param['xi'] * x_grad / np.linalg.norm(x_grad)) * \
                 self.param['dt']
            x = x + y * self.param['dt']
            cond = np.abs(x) > 1
            x = np.where(cond, np.sign(x), x)
            y = np.where(cond, np.zeros_like(y), y)

        return np.sign(x)

        # 纯振幅优化函数
        # ============================================================================================================================ #
    def opt_amp(self, amp, phase_angle):
        '''
        Args:
            amp: 优化前的振幅
            phase_angle: 固定的相位角
        Returns:
            amplitude: 优化过后得到的振幅
            loss: 优化过后损失函数的数值
        '''

        # 优化振幅的损失函数
        def cost_func_for_amp(amp, phase_angle):
            '''
            Args:
                amp: 振幅
                phase_angle: 相位角
            Returns:
                obj: 损失函数的数值
            '''
            phase = torch.exp(1.0j * phase_angle)

            main_lobe = torch.einsum('i, i -> ', phase * amp,
                                     self.efield[:, self._get_index(self.param['theta_0'])])
            loss_2 = 1.0 * self.param['weight'] * torch.real(torch.conj(main_lobe) * main_lobe)
            loss_1 = 1.0
            for i in range(len(self.param['range_list_weight'])):
                one_range = torch.einsum('i, ij -> j', phase * amp, self.efield[:, round(
                    self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])): round(
                    self._get_index(self.param['theta_0'] + self.param['range_list'][i][1]))])
                loss_1 += (1 - self.param['weight']) * self.param['range_list_weight'][i] * (
                    torch.real(torch.conj(one_range) * one_range)).mean()

            obj = loss_1 / loss_2
            return obj

        amplitude = torch.tensor(amp.copy())
        amplitude.requires_grad = True
        optimizer = torch.optim.Adam([amplitude], lr=self.param['lr'])

        for iter in tqdm(range(1000)):
            optimizer.zero_grad()
            loss = cost_func_for_amp(amplitude, torch.tensor(phase_angle))
            loss.backward()
            optimizer.step()
        return amplitude, loss


    # 辅助函数
    # ============================================================================================================================ #
    # 比特相位编码函数，可以参考文献：https://arxiv.org/pdf/2409.19938
    def encode(self, x_bit):
        '''
        Args:
            x_bit: 编码前的比特串
        Returns:
            phase: 编码后的相位

        此编码函数仅针对2比特编码的情况
        '''
        if self.param['encode_qubit'] == 2 or self.param['encode_qubit'] == 1:
            c1 = 0.5 + 0.5j
            c2 = 0.5 - 0.5j
            N = x_bit.shape[0]
            phase = c1 * x_bit[:, 0] + c2 * x_bit[:, 1]

        else:
            c1 = 0.5 + 0j
            c2 = 0 + 0.5j
            c3 = 0.5 + 0j
            c4 = 0 + 0.5j
            # c1 = 0.25 + 0.6036j
            # c2 = 0.6036 - 0.25j
            # c3 = 0.25 - 0.1036j
            # c4 = - 0.1036 - 0.25j
            N = x_bit.shape[0]
            # phase = (c1 * x_bit[:, 0] +
            #          c2 * x_bit[:, 1] +
            #          c3 * x_bit[:, 2] +
            #          c4 * x_bit[:, 0] * x_bit[:, 1] * x_bit[:, 2] * x_bit[:, 3])
            phase = (c1 * x_bit[:, 0] +
                     c2 * x_bit[:, 1] +
                     c3 * x_bit[:, 2] +
                     c4 * x_bit[:, 3])
        phase.reshape(N)
        return phase

    def plot(self):
        import matplotlib.pyplot as plt

        # 设置中文字体显示
        plt.rcParams["font.family"] = ["SimHei"]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.rcParams['figure.dpi'] = 300
        plt.figure(figsize=(8, 6))

        # 计算画图相关数据
        self.theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        F = torch.einsum('i, ij -> j', torch.tensor(self.amp) * np.exp(1.0j * self.phase_angle), self.efield).numpy()
        self.FF = np.real(F.conj() * F)
        self.y = 10 * np.log10(self.FF / np.max(self.FF))

        # 主瓣方向 θ₀ 及其 ±30°
        theta_0 = self.param['theta_0']
        theta_inner_left = theta_0 - 30
        theta_inner_right = theta_0 + 30

        # 主瓣 ±30° 范围内，低于 -30dB 的区域
        mask_inner = (self.theta >= theta_inner_left) & (self.theta <= theta_inner_right) & (self.y <= -30)
        plt.fill_between(self.theta, self.y, -30, where=mask_inner, color='red', alpha=0.2, label='成形区间低于$-30dB$')

        # 主瓣 ±30° 范围外，低于 -15dB 的区域
        mask_outer = ((self.theta < theta_inner_left) | (self.theta > theta_inner_right)) & (self.y <= -15)
        plt.fill_between(self.theta, self.y, -15, where=mask_outer, color='blue', alpha=0.2, label='非成形区间低于$-15dB$')

        # 目标 ±1° 范围内
        plt.axvspan(theta_0 - 1, theta_0 + 1, color='green', alpha=0.2, label=r'实际角度允许区间 $\theta_0 \pm 1$°')

        # 画图
        plt.plot(self.theta, self.y, color='blue', alpha=0.5, label='波束图')

        # 标注目标方向 θ₀
        plt.axvline(theta_0, color='green', linestyle='--', linewidth=1.5, label=r'目标角度 $\theta_0$', alpha=0.7)
        plt.text(theta_0 + 1, 0, r'$\theta_0=$' + str(theta_0) + '°', color='green', fontsize=12, alpha=0.7)

        # 辅助线
        plt.axvline(theta_inner_left, color='gray', linestyle='--')
        plt.axvline(theta_inner_right, color='gray', linestyle='--')

        # 设置中文显示和标签
        plt.xlabel(r'方向角 $\theta$ $($度$)$', fontsize=12)
        plt.ylabel(r'归一化信号强度 $(dB)$', fontsize=12)
        # plt.title(r'目标角度 $\theta_0$='+str(theta_0)+'°下波束赋形结果', fontsize=14)
        plt.ylim([-50, 5])  # 限制y轴显示范围
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True)

        # 保存图像
        plt.savefig(str(theta_0) + '_beamforming_marked_' + str(self.param['encode_qubit']) + 'bits.jpg')
        plt.show()

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
        phase_x = 1j * np.pi * cosd_f(theta)
        AF = np.exp(phase_x[None, :] * np.arange(self.param['N'])[:, None])
        return AF

    # 获得theta角度对应的矩阵指标
    def _get_index(self, angle_value):
        index = round(angle_value * self.param['n_angle'])
        return index


def get_score(phase_angle, amplitude, theta, encode_qubit):
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
    theta_0 = theta
    n_bit_phase = encode_qubit
    opt_amp_or_not = True

    # 确保相位和振幅是按照赛题要求的取值
    if opt_amp_or_not is True:
        amplitude = amplitude / np.max(amplitude)
    else:
        amplitude = np.ones(N)
    phase_angle = np.angle(np.exp(1.0j * phase_angle)) + np.pi
    phase_angle = np.round(phase_angle / (2 * np.pi) * (2 ** n_bit_phase)) / (2 ** n_bit_phase) * (2 * np.pi)

    def cosd_f(x):
        return np.cos(x * np.pi / 180)

    def get_efield(n_angle, N, theta_0):
        theta = np.linspace(0, 180, 180 * n_angle + 1)
        x = 12 * ((theta - 90) / 90) ** 2
        E_dB = -1.0 * np.where(x < 30, x, 30)
        E_theta = 10 ** (E_dB / 10)
        EF = E_theta ** 0.5

        phase_x = 1j * np.pi * cosd_f(theta)
        AF = np.exp(phase_x[None, :] * np.arange(N)[:, None])

        efield = EF[None, ...] * AF

        return efield

    efield = get_efield(n_angle, N, theta_0)
    theta_array = np.linspace(0, 180, 180 * n_angle + 1)
    amp_phase = []
    for i in range(N):
        amp_phase.append(amplitude[i] * np.exp(1.0j * phase_angle[i]))
    F = np.einsum('i, ij -> j', np.array(amp_phase), efield)
    FF = np.real(F.conj() * F)
    db_array = 10 * np.log10(FF / np.max(FF))

    x = theta_array - theta_0
    value_list = []
    for i in range(theta_array.shape[0]):
        if abs(x[i]) >= 30:
            value_list.append(db_array[i] + 15)
    a = max(np.max(value_list), 0)

    target = np.max(db_array)
    for i in range(theta_array.shape[0]):
        if db_array[i] == target:
            max_index = i
            break

    theta_up = 180
    theta_down = 0
    theta_min_up = 180
    theta_min_down = 0
    if abs(theta_array[max_index] - theta_0) > 1:
        y = 0

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
            if (db_array[i + max_index] < db_array[i - 1 + max_index]) and (
                    db_array[i + max_index] < db_array[i + 1 + max_index]):
                theta_min_up = theta_array[i + max_index]
                break

        for i in range(1, 10000):
            if (db_array[-i + max_index] < db_array[-i - 1 + max_index]) and (
                    db_array[-i + max_index] < db_array[-i + 1 + max_index]):
                theta_min_down = theta_array[-i + max_index]
                break

        if theta_up == 180 or theta_down == 0:
            y = 0

        elif theta_min_up < theta_up or theta_min_down > theta_down:
            y = 0

        else:
            W = theta_up - theta_down
            b = max(W - 6, 0)

            value_list_2 = []
            for i in range(theta_array.shape[0]):
                if abs(x[i]) <= 30 and (x[i] >= theta_min_up - theta_0 or x[i] <= theta_min_down - theta_0):
                    value_list_2.append(db_array[i] + 30)
            c = np.max(value_list_2)

            # 负分直接归为0分
            y = max(1000 - 100 * a - 80 * b - 20 * c, 0)
    return y


if __name__ == '__main__':
    variable_list = [[50.0, 2, True], [60.0, 3, True], [70.0, 3, True], [80.0, 3, True], [90.0, 3, True],
                     [100.0, 3, True], [110.0, 3, True], [120.0, 3, True], [130.0, 3, True]]
    # variable_list = [[50, 3, True]]
    for variable in variable_list:
        phase_angle, amp = optimized(variable)