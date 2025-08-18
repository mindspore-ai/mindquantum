 ## 关于相位编码，可以参考文献：https://arxiv.org/pdf/2409.19938 ##
import numpy as np
import torch
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm


from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Binary,Integer
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize



import os
import time
import concurrent.futures
import multiprocessing

from pymoo.core.population import Population

from pymoo.algorithms.soo.nonconvex.ga import GA
from mindquantum.algorithm.qaia import BSB
import random


# 设置全局种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# from answer import optimized
def optimized(variables):

    # 在后续优化过程中使用的参数
    param = {
        'theta_0': variables[0],  # 波束成形方向，以90度为例
        'N': 32,  # 天线阵子总数
        'n_angle': 40,  # 1度中被细分的次数
        'encode_qubit': variables[1],  # 进行相位编码的比特个数，样例代码中固定为2，实际对应变量 variables[1]

        # bSB参数界面
        'xi': 0.1,  # 模拟分叉算法中调节损失函数的相对大小
        'dt': 0.1,  # 演化步长
        'n_iter': 2000,  # 迭代步数

        # 相位损失函数界面 
        'weight': 0.3,  # 调节损失函数中分子和分母的相对大小
        'range_list': [[-30, -6], [6, 30]], # 需要压制的旁瓣范围相对于主瓣波束成形方向的角度表示
        'range_list_weight': [1, 1], # 每个压制的旁瓣范围各自的权重

        # 连续振幅优化
        'opt_amp_or_not': variables[2], # 根据输入控制是否优化振幅
        'lr': 0.001, # 学习率
    }

    A = QUBO_GA(param)
    A.solve()
    A.plot()

    phase_angle = A.phase_angle  # 相位角
    amp = A.amp  # 振幅保持不变
    print(f'pppppppphase angle: {phase_angle}')
    print(f'aaaaaaaaaamp: {amp}')
    return phase_angle, amp

"""
下面为样例代码
"""

# 计算角度制下的三角余弦函数
def cosd_f(x):
    return np.cos(x * np.pi / 180)
# 计算角度制下的三角正弦函数
def sind_f(x):
    return np.sin(x * np.pi / 180)

class BF():
        def __init__(self, param):
            self.param = copy.deepcopy(param)
            self.EF = self._generate_power_pattern()
            self.AF = self._generate_array_factor()
            self.amp = np.ones(self.param['N'])
            self.efield = torch.tensor(self.EF[None, ...] * self.AF)
            np.random.seed(42)
            torch.manual_seed(42)
            random.seed(42)
        def solve(self):
            self.x_final = self.opt_phase_QUBO()
            self.phase = self.encode(self.x_final)
            self.phase_angle = np.angle(self.phase)
            # print(f'phase angle: {self.phase_angle}')
            # print(f'amp: {self.amp}')

        def opt_phase_QUBO(self):
            c1 = 0.5 + 0.5j
            c2 = 0.5 - 0.5j
            factor_array = torch.cat((self.efield[:, self._get_index(self.param['theta_0'])] * c1, 
                                    self.efield[:, self._get_index(self.param['theta_0'])] * c2), dim=0)
            J_enhance = torch.einsum('i, j -> ij', factor_array.conj(), factor_array).real

            J_suppress = 0.0
            for i, (start, end) in enumerate(self.param['range_list']):
                num = 0
                a_0 = 0.0
                max_energy = torch.zeros((64, 64))
                for j in range(round(self._get_index(self.param['theta_0'] + start)), 
                            round(self._get_index(self.param['theta_0'] + end)), 1):
                    num += 1
                    factor_array = torch.cat((self.efield[:, j] * c1, self.efield[:, j] * c2), dim=0)
                    a_0 += torch.einsum('i, j -> ij', factor_array.conj(), factor_array)
                    
                    current_energy = torch.einsum('i, j -> ij', factor_array.conj(), factor_array)
                    norm_a = torch.norm(max_energy, p='fro')
                    norm_b = torch.norm(current_energy, p='fro')
                    max_energy = max_energy if norm_a > norm_b else current_energy
                # J_suppress = max_energy
                J_suppress += self.param['range_list_weight'][i] * a_0 / num

            J = torch.real((self.param['weight'] * J_enhance - (1 - self.param['weight']) * J_suppress)).numpy()
            solver = BSB(np.array(J, dtype="float64"), batch_size= 5, n_iter=self.param['n_iter'], 
                        xi=self.param['xi'], backend='cpu-float32')
            solver.update()
            
            array = solver.x
            min_energy_index = np.argmin(solver.calc_energy())
            max_cut_index = np.argmax(solver.calc_cut())
            solution_max_cut = array[:, max_cut_index].reshape(-1, 1)
            x_bit = np.sign(solution_max_cut.reshape(2 * self.param['N'], 1))
            return x_bit.reshape(self.param['N'], 2, order='F')

        def encode(self, x_bit):
            c0 = 0.5 + 0.5j
            c1 = 0.5 - 0.5j
            N = x_bit.shape[0]
            phase = c0 * x_bit[:, 0] + c1 * x_bit[:, 1]
            phase.reshape(N)
            return phase
        def _generate_power_pattern(self):
            theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
            x = 12 * ((theta - 90) / 90) ** 2
            E_dB = -1.0 * np.where(x < 30, x, 30)
            E_theta = 10 ** (E_dB / 10)
            return E_theta ** 0.5
        def _generate_array_factor(self):
            theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
            phase_x = 1j * np.pi * cosd_f(theta)
            return np.exp(phase_x[None, :] * np.arange(self.param['N'])[:, None])
        def _get_index(self, angle_value):
            return round(angle_value * self.param['n_angle'])
def get_efield1(n_angle, N, theta_0):
    theta = np.linspace(0, 180, 180 * n_angle + 1)
    x = 12 * ((theta - 90) / 90) ** 2
    E_dB = -1.0 * np.where(x < 30, x, 30)
    E_theta = 10 ** (E_dB / 10)
    EF = E_theta ** 0.5

    phase_x = 1j * np.pi * cosd_f(theta)
    AF = np.exp(phase_x[None, :] * np.arange(N)[:, None])

    efield = EF[None, ...] * AF

    return efield
def optimized_with_params(param):
    A = BF(param)
    A.solve()
    return A.phase_angle, A.amp

# def optimized_with_params(param):
#     pass 
#     return None, None

class GA_OPT(ElementwiseProblem):
    def __init__(self,variables):
        self.variables = variables
                        # 计算全局数据
        self.n_angle = 500
        self.N = 32
        self.theta_0 = variables[0]  # 可替换为其他 theta_0（如 81）
        self.efield = get_efield1(self.n_angle, self.N, self.theta_0)
        self.theta_array = np.linspace(0, 180, 180 * self.n_angle + 1)
        # 定义离散步长
        xi_min, xi_max, xi_step = 0.03, 0.10, 0.01
        weight_min, weight_max, weight_step = 0.03, 0.10, 0.01
        
        # 计算离散值数量
        xi_n = int((xi_max - xi_min) / xi_step) + 1
        weight_n = int((weight_max - weight_min) / weight_step) + 1
        
        # 定义整数变量
        vars = {
            "xi": Integer(bounds=(0, xi_n - 1)),
            "weight": Integer(bounds=(0, weight_n - 1)),
        }
        super().__init__(vars=vars, n_obj=1)
        # 预计算离散值列表
        self.xi_values = [xi_min + i * xi_step for i in range(xi_n)]
        self.weight_values = [weight_min + i * weight_step for i in range(weight_n)]
        self.param_cache = {}  # key: (xi_idx, weight_idx) -> value: (phase_angle, amp)

    def _evaluate(self, x, out, *args, **kwargs):
        xi_idx = x["xi"]
        weight_idx = x["weight"]
        xi = self.xi_values[x["xi"]]
        weight = self.weight_values[x["weight"]]
        left1 = -55
        right1 = -25
        left2 = 25
        right2 = 55
        param = {
            'theta_0': self.variables[0],
            'N': 32,
            'n_angle': 100,
            'encode_qubit': self.variables[1],
            'xi': xi,
            'dt': 0.3,
            'n_iter': 300,
            'weight': weight,
            'range_list': [[left1, right1], [left2, right2]],
            'range_list_weight': [1, 1],
        }
        encode_qubit = self.variables[1]

        phase_angle, amp = optimized_with_params(param)
        # print(f"xi={xi:.4f}, weight={weight:.4f}, range_list=[[ {left1:.2f}, {right1:.2f} ], [ {left2:.2f}, {right2:.2f} ]]")
        # 保存到缓存
        self.param_cache[(xi_idx, weight_idx)] = (phase_angle, amp)
        # 2. 归一化到 [0, 2π]
        phase_angle = np.angle(np.exp(1j * phase_angle)) + np.pi
        # 3. 统一相位离散化（与 get_score 一致）
        phase_angle = np.round(phase_angle / (2 * np.pi) * (2 ** encode_qubit)) / (2 **  encode_qubit) * (2 * np.pi)

        amp_phase = []
        N=32
        for i in range(N):
            amp_phase.append(amp[i] * np.exp(1.0j * phase_angle[i]))
        F = np.einsum('i, ij -> j', np.array(amp_phase), self.efield)
        FF = np.real(F.conj() * F)
        db_array = 10 * np.log10(FF / np.max(FF))
        
        x_theta = self.theta_array - self.theta_0
        mask = np.abs(x_theta) >= 30
        selected_values = db_array[mask] + 15
        a = max(np.max(selected_values) if selected_values.size > 0 else 0, 0)
        
        # 定位主瓣方向
        target = np.max(db_array)
        for i in range(self.theta_array.shape[0]):
            if db_array[i] == target:
                max_index = i
                break
            
        theta_up = self.theta_array[-1]
        theta_down = self.theta_array[0]
        right_mask = db_array[max_index + 1:] <= -30
        if np.any(right_mask):
            theta_up = self.theta_array[max_index + 1 + np.argmax(right_mask)]
        left_mask = np.flip(db_array[:max_index]) <= -30
        if np.any(left_mask):
            theta_down = self.theta_array[max_index - 1 - np.argmax(left_mask)]
        W = theta_up - theta_down
        b = max(W - 6, 0)
        
        # 计算 theta_min_up 和 theta_min_down
        right_section = db_array[max_index + 1:-1]
        if right_section.size > 1:
            is_min_right = (right_section < np.roll(right_section, 1)) & (right_section < np.roll(right_section, -1))
            if np.any(is_min_right):
                theta_min_up = self.theta_array[max_index + 1 + np.argmax(is_min_right)]
            else:
                theta_min_up = self.theta_array[-1]
        else:
            theta_min_up = self.theta_array[-1]
        
        left_section = db_array[1:max_index][::-1]
        if left_section.size > 1:
            is_min_left = (left_section < np.roll(left_section, 1)) & (left_section < np.roll(left_section, -1))
            if np.any(is_min_left):
                theta_min_down = self.theta_array[max_index - 1 - np.argmax(is_min_left)]
            else:
                theta_min_down = self.theta_array[0]
        else:
            theta_min_down = self.theta_array[0]
        
        # 计算 c
        mask1 = np.abs(x_theta) <= 30
        mask2 = (x_theta >= (theta_min_up - self.theta_0)) | (x_theta <= (theta_min_down - self.theta_0))
        combined_mask = mask1 & mask2
        selected_db = np.full_like(db_array, -np.inf)
        selected_db[combined_mask] = db_array[combined_mask] + 30
        c = np.max(selected_db)
        if not np.isfinite(c):
            c = 0.0
        
        # 加入主瓣方向惩罚项
        direction_penalty = abs(self.theta_array[max_index] - self.theta_0)
        obj = 1000 - (100 * a + 80 * b + 20 * c) # - 5 * direction_penalty
        print(f"xi={xi:.4f}, weight={weight:.4f}，a: {a}, b: {b}, c: {c},  max direction: {self.theta_array[max_index]}，obj: {obj}")
    
        out["F"] = [-obj]

    # ============================================================================================================================ #
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
            

class QUBO_GA():
    # 初始化变量
    # ============================================================================================================================ #
    def __init__(self, param):
        self.param = copy.deepcopy(param)
        self.EF = self._generate_power_pattern()
        self.AF = self._generate_array_factor()
        self.amp = np.ones(self.param['N'])
        self.efield = torch.tensor(self.EF[None, ...] * self.AF)

    # 相位和振幅的优化求解流程
    # ============================================================================================================================ #
    def solve(self):
        if self.param['opt_amp_or_not'] is True:
             ##### 第一次相位优化  优化相位角
            self.x_final = self.opt_phase()
            # print(f'x_final11111: {self.x_final}')
            self.phase = self.encode(self.x_final)
            self.phase_angle = np.angle(self.phase)

            ##### 第一次振幅优化
            self.amp, _ = self.opt_amp(self.amp, self.phase_angle)
             # amp2, _ = self.opt_amp(self.amp, self.phase_angle)
            # print(f'self.amp类型: {type(self.amp)}, {self.amp.shape}')
            # print(f'self.phase_angle类型: {type(self.phase_angle)}, {self.phase_angle.shape}')
            
            ######################            结合振幅和相位进行微调优化
            self.amp,x_bit_solution,phase_N  = self.joint_optimization(self.amp, self.x_final)
            self.phase_angle = self.encode1(x_bit_solution,phase_N)

            # print(f'self.amp类型: {type(self.amp)}, {self.amp.shape}')
            # print(f'self.phase_angle类型: {type(self.phase_angle)}, {self.phase_angle.shape}')
            
            # 训练后的相位补偿
            theta_0_deg = self.param['theta_0']
            theta_0_rad = np.deg2rad(theta_0_deg)
            N = 32
            n = np.arange(N)
            delta_phase = -np.pi * np.cos(theta_0_rad) * n  # 补偿项
            phase_angle_compensated = self.phase_angle + delta_phase  # 添加补偿
            # 归一化相位到 [0, 2π]
            self.phase_angle  = np.angle(np.exp(1j * phase_angle_compensated)) + np.pi
            
        else :
            variables = [self.param['theta_0'], self.param['encode_qubit'], self.param['opt_amp_or_not']]
            # print(f"vvvvvvvvvvvariables: {variables}")
            self.phase_angle, self.amp, best_params = self.optimize_hyperparameters(variables)
            print(f"Optimized phase_angle: {self.phase_angle}")
            print(f"Best hyperparameters: xi={best_params['xi']}, weight={best_params['weight']},amp = {self.amp}")
            
            
# ============    self.param['opt_amp_or_not'] is True:     ================================================================================================================ #
    def encode1(self,x_bit,phase_N):
        bit_count = x_bit.shape[1]  # 获取比特数
        if bit_count == 2:
            bin_weights = [2, 1]
        elif bit_count == 3:
            bin_weights = [4, 2, 1]
        elif bit_count == 4:
            bin_weights = [8, 4, 2, 1]
        elif bit_count == 1:
            bin_weights = [1]
        else:
            raise ValueError("Unsupported bit count. Supported: 2, 3, 4")
        binary_value = np.sum(x_bit * bin_weights, axis=1)
        phase = (2 * np.pi / phase_N) * binary_value - np.pi  # 相位范围 -π 到 π
        return phase
    def joint_optimization(self, amp, x_bit):
        def encode2(x_bit,theta_0,encode_qubit, phase_N):
            bit_count = x_bit.shape[1]  # 获取比特数
            if bit_count == 2:
                bin_weights = [2, 1]
            elif bit_count == 3:
                bin_weights = [4, 2, 1]
            elif bit_count == 4:
                bin_weights = [8, 4, 2, 1]
            elif bit_count == 1:
                bin_weights = [1]
            else:
                raise ValueError("Unsupported bit count. Supported: 2, 3, 4")
            binary_value = np.sum(x_bit * bin_weights, axis=1)
            phase = (2 * np.pi / phase_N) * binary_value - np.pi  # 相位范围 -π 到 π
            # 1. 添加相位补偿项
            n = np.arange(32)
            delta_phase = -np.pi * np.cos(np.deg2rad(theta_0)) * n  # 补偿项
            compensated_phase = phase + delta_phase
            # 2. 归一化到 [0, 2π]
            compensated_phase = np.angle(np.exp(1j * compensated_phase)) + np.pi
            
            # 3. 统一相位离散化（与 get_score 一致）
            phase = np.round(compensated_phase / (2 * np.pi) * (2 ** encode_qubit)) / (2 **  encode_qubit) * (2 * np.pi)
    
            
            return phase

        def get_efield1(n_angle, N, theta_0):
            theta = np.linspace(0, 180, 180 * n_angle + 1)
            x = 12 * ((theta - 90) / 90) ** 2
            E_dB = -1.0 * np.where(x < 30, x, 30)
            E_theta = 10 ** (E_dB / 10)
            EF = E_theta ** 0.5

            phase_x = 1j * np.pi * cosd_f(theta)
            AF = np.exp(phase_x[None, :] * np.arange(N)[:, None])

            efield = EF[None, ...] * AF

            return efield
        
        # 定义优化问题
        class MyProblem(ElementwiseProblem):
            def __init__(self,theta_0, encode_qubit):
                # 计算全局数据
                self.n_angle = 500
                self.N = 32
                self.theta_0 = theta_0  # 可替换为其他 theta_0（如 81）
                self.efield = get_efield1(self.n_angle, self.N, self.theta_0)
                self.theta_array = np.linspace(0, 180, 180 * self.n_angle + 1)
                
                # 定义变量
                vars = {}
                for i in range(32):
                    vars[f"amp_{i}"] = Real(bounds=(0, 1))
                for i in range(32):
                    for j in range(encode_qubit):
                        vars[f"x_bit_{i}_{j}"] = Binary()
                vars[f"phase_N"] = Integer(bounds=(2**encode_qubit, 50))
                super().__init__(
                    vars=vars,
                    n_obj=1,
                    n_constr=1
                )

            def _evaluate(self, x, out, *args, **kwargs):
                # 提取变量
                amp = np.array([x[f"amp_{i}"] for i in range(32)])
                amp = amp/np.max(amp)
                x_bit = np.array([[x[f"x_bit_{i}_{j}"] for j in range(encode_qubit)] for i in range(32)])
                
                phase_N = x[f"phase_N"]
                # 强制 x_bit 为二元（0 或 1）
                x_bit = np.round(x_bit).astype(int)
                # print(f"x_bit: \n{x_bit}")
                # 计算目标函数
                phase = encode2(x_bit,self.theta_0 ,encode_qubit, phase_N)
                # phase_angle = phase
          
                phase_angle = phase
                # # 计算方向图
                # amp_phase = amp * np.exp(1j * phase_angle)
                # F = amp_phase @ self.efield
                # FF = np.real(np.conj(F) * F)
                # max_FF = np.max(FF)
                # db_array = 10 * np.log10(FF / max_FF + 1e-10)
                amp_phase = []
                N=32
                for i in range(N):
                    amp_phase.append(amp[i] * np.exp(1.0j * phase_angle[i]))
                    
                F = np.einsum('i, ij -> j', np.array(amp_phase), self.efield)
                FF = np.real(F.conj() * F)
                db_array = 10 * np.log10(FF / np.max(FF))
                
                x_theta = self.theta_array - self.theta_0
                mask = np.abs(x_theta) >= 30
                selected_values = db_array[mask] + 15
                a = max(np.max(selected_values) if selected_values.size > 0 else 0, 0)
                
                # 定位主瓣方向
                target = np.max(db_array)
                for i in range(self.theta_array.shape[0]):
                    if db_array[i] == target:
                        max_index = i
                        break
                    
                    
                theta_up = self.theta_array[-1]
                theta_down = self.theta_array[0]
                right_mask = db_array[max_index + 1:] <= -30
                if np.any(right_mask):
                    theta_up = self.theta_array[max_index + 1 + np.argmax(right_mask)]
                left_mask = np.flip(db_array[:max_index]) <= -30
                if np.any(left_mask):
                    theta_down = self.theta_array[max_index - 1 - np.argmax(left_mask)]
                W = theta_up - theta_down
                b = max(W - 6, 0)
                
                # 计算 theta_min_up 和 theta_min_down
                right_section = db_array[max_index + 1:-1]
                if right_section.size > 1:
                    is_min_right = (right_section < np.roll(right_section, 1)) & (right_section < np.roll(right_section, -1))
                    if np.any(is_min_right):
                        theta_min_up = self.theta_array[max_index + 1 + np.argmax(is_min_right)]
                    else:
                        theta_min_up = self.theta_array[-1]
                else:
                    theta_min_up = self.theta_array[-1]
                
                left_section = db_array[1:max_index][::-1]
                if left_section.size > 1:
                    is_min_left = (left_section < np.roll(left_section, 1)) & (left_section < np.roll(left_section, -1))
                    if np.any(is_min_left):
                        theta_min_down = self.theta_array[max_index - 1 - np.argmax(is_min_left)]
                    else:
                        theta_min_down = self.theta_array[0]
                else:
                    theta_min_down = self.theta_array[0]
                
                # 计算 c
                mask1 = np.abs(x_theta) <= 30
                mask2 = (x_theta >= (theta_min_up - self.theta_0)) | (x_theta <= (theta_min_down - self.theta_0))
                combined_mask = mask1 & mask2
                selected_db = np.full_like(db_array, -np.inf)
                selected_db[combined_mask] = db_array[combined_mask] + 30
                c = np.max(selected_db)
                if not np.isfinite(c):
                    c = 0.0
                
                # 加入主瓣方向惩罚项
                direction_penalty = abs(self.theta_array[max_index] - self.theta_0)
                obj = 1000 - (100 * a + 80 * b + 20 * c) # - 5 * direction_penalty
                # print(f"a: {a}, b: {b}, c: {c}, obj: {obj}, max direction: {self.theta_array[max_index]}")
                
                # 计算约束
                constr = abs(self.theta_array[max_index] - self.theta_0) - 1
                out["F"] = [-obj]  # 最小化问题，返回负值
                out["G"] = [constr]  # 约束 <= 0

        def build_initial_solution(amp, x_bit, phase_N, encode_qubit,problem):
                solution = {}

                # 添加 amp_i
                for i in range(32):
                    solution[f"amp_{i}"] = amp[i]

                # 添加 x_bit_i_j
                for i in range(32):
                    for j in range(encode_qubit):
                        solution[f"x_bit_{i}_{j}"] = int(x_bit[i][j])  # 确保是整数 0/1

                # 添加 phase_N
                solution[f"phase_N"] = int(phase_N)  # 确保是整数

                return solution


        # 设置参数
        theta_0 = self.param['theta_0']
        encode_qubit = self.param['encode_qubit']
        problem = MyProblem(theta_0, encode_qubit) 
        
        phase_N = 2**encode_qubit
        
        solution_dict = build_initial_solution(amp, x_bit, phase_N, encode_qubit, problem)
        population = Population.new("X", [solution_dict])

        algorithm = MixedVariableGA(
            pop_size=10,
            seed=1,
            termination=('n_gen', 100),
            initialization = population
        )

        # 求解
        res = minimize(problem, algorithm, verbose=True)

        # 输出结果
        solution = res.X
        objective_value = -res.F[0]

        # 提取并分开显示解
        amp_solution = np.array([solution[f"amp_{i}"] for i in range(32)])
        x_bit_solution = np.array([[solution[f"x_bit_{i}_{j}"] for j in range(encode_qubit)] for i in range(32)])
        phase_N = solution[f"phase_N"]
        # 将布尔值转换为 +1 和 -1
        x_bit_solution = np.where(x_bit_solution, 1, 0)
        amp_solution = amp_solution/np.max(amp_solution)
        print(f"RETURN_amp 解: {amp_solution}")
        print(f"x_bit 解: \n{x_bit_solution}")
        print(f"目标值: {objective_value}")

        return amp_solution,x_bit_solution,phase_N
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

            main_lobe = torch.einsum('i, i -> ', phase * amp, self.efield[:, self._get_index(self.param['theta_0'])])
            loss_2 = 1.0 * self.param['weight'] * torch.real(torch.conj(main_lobe) * main_lobe)
            
            loss_1 = 0.0
            for i in range(len(self.param['range_list_weight'])):
                one_range = torch.einsum('i, ij -> j', phase * amp, self.efield[:, round(
                    self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])): round(
                    self._get_index(self.param['theta_0'] + self.param['range_list'][i][1]))])
                loss_1 += (1 - self.param['weight']) * self.param['range_list_weight'][i] * (
                    torch.real(torch.conj(one_range) * one_range)).mean()
            obj = loss_1 / (loss_2 + 1e-8) # 目标函数需要重点修改，用户可以自定义目标函数，如加权求和等
            return obj 
    
        # 初始化
        x = 0.01 * (np.random.randn(self.param['N'], self.param['encode_qubit']))
        # print(f'x_init: {x}')
        y = 0.01 * (np.random.randn(self.param['N'], self.param['encode_qubit']))
        print(f' 第一次相位优化 ')
        for iter in tqdm(range(self.param['n_iter'] + 1)):
            x_torch = torch.tensor(x)
            x_torch.requires_grad = True

            # 计算梯度与更新参数
            x_sign = x_torch - (x_torch - torch.sign(x_torch)).detach() #dSB离散化
            loss = cost_func(x_sign)
            loss.backward()
            x_grad = (x_torch.grad).clone().detach().numpy()
            y += (-(0.5 - iter / self.param['n_iter']) * x - self.param['xi'] * x_grad / np.linalg.norm(x_grad)) * self.param['dt']
            x = x + y * self.param['dt']
            cond = np.abs(x) > 1
            x = np.where(cond, np.sign(x), x)
            y = np.where(cond, np.zeros_like(y), y)

        return np.sign(x)
            # ==========================================                          ================================================================================== #
    def encode(self, x_bit):
        '''
        Args: 
            x_bit: 编码前的比特串
        Returns:
            phase: 编码后的相位

        此编码函数仅针对2比特编码的情况
        '''
        c0 = 0.5 + 0.5j
        c1 = 0.5 - 0.5j
        N = x_bit.shape[0]
        phase = c0 * x_bit[:, 0] + c1 * x_bit[:, 1]
        phase.reshape(N)
        return phase
    
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

            main_lobe = torch.einsum('i, i -> ', phase * amp, self.efield[:, self._get_index(self.param['theta_0'])])
            loss_2 = 1.0 * self.param['weight'] * torch.real(torch.conj(main_lobe) * main_lobe)
            loss_1 = 1.0
            for i in range(len(self.param['range_list_weight'])):
                one_range = torch.einsum('i, ij -> j', phase * amp, self.efield[:, round(
                    self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])): round(
                    self._get_index(self.param['theta_0'] + self.param['range_list'][i][1]))])
                loss_1 += (1 - self.param['weight']) * self.param['range_list_weight'][i] * (
                    torch.real(torch.conj(one_range) * one_range)).mean()

            obj = loss_1 / (loss_2 + 1e-8)
            return obj

        amplitude = torch.tensor(amp.copy())
        amplitude.requires_grad = True
        optimizer = torch.optim.Adam([amplitude], lr=self.param['lr'])
        print(f' 第一次振幅优化 ')
        for iter in tqdm(range(1000)):
            optimizer.zero_grad()
            loss = cost_func_for_amp(amplitude, torch.tensor(phase_angle))
            loss.backward()
            optimizer.step()
        return amplitude, loss

# ===========  self.param['opt_amp_or_not'] is False:  ================================================================================================================ #
    def optimize_hyperparameters(self,variables):

        try:
            problem = GA_OPT(variables)  # 添加 try/except 前先测试
        except Exception as e:
            print(f"Error when instantiating GA_OPT: {e}")
            raise
        algorithm = MixedVariableGA(
                pop_size= 6,
                seed=1,
                termination=('n_gen', 7)
            )
        res = minimize(problem, algorithm, verbose=True)
        best_params = res.X
        # 将整数索引转换为真实值
        xi_real = problem.xi_values[best_params["xi"]]
        weight_real = problem.weight_values[best_params["weight"]]
        
        objective_value = -res.F[0]
        print(f"目标值: {objective_value}")

        ### 取出最后最好的结果
    # 从缓存中提取 phase_angle 和 amp
        phase_angle, amp = problem.param_cache.get((best_params["xi"], best_params["weight"]), (None, None))

        if phase_angle is None:
            raise ValueError("最优解对应的 phase_angle 和 amp 未缓存")
        return phase_angle, amp,  {"xi": xi_real, "weight": weight_real}
    
    
    

    # ============================================================================================================================ #
    # 优化结果画图函数
    def plot(self):
        # 计算画图相关数据
        self.theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        F = torch.einsum('i, ij -> j', torch.tensor(self.amp) * np.exp(1.0j * self.phase_angle), self.efield).numpy()
        self.FF = np.real(F.conj() * F)
        self.y = 10 * np.log10(self.FF / np.max(self.FF))
        # 画图
        plt.figure()
        plt.plot(self.theta, self.y)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$lg|F(\theta)|^2 - lg|F(\theta)|^2_{max}$' + ' (dB)')
        plt.title('Beamforming Outcome')
        plt.savefig(str(self.param['theta_0']) + '_beamforming_0513_answer_run.jpg')
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



