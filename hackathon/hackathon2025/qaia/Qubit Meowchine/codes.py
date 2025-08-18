import numpy as np
import torch
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm
from itertools import product

def optimized(variables):
    # 构造参数字典，包含波束中心角度、阵元数、扫描分辨率、编码比特数等
    param = {
        "theta_0": variables[0],
        "N": 32,
        "n_angle": 10,
        "encode_qubit": 2,
        "weight": 0.05,
        "range_list": [
            [-30, -3],
            [3, 30],
        ],
        "range_list_weight": [1, 1],
        "opt_amp_or_not": variables[2],
        "lr": 0.0001,
        "iter": 5000,
    }
    # 初始化 BF 对象并执行求解、绘图
    A = BF(param)
    A.solve()
    A.plot()
    phase_angle = A.phase_angle
    amp = A.amp

    return phase_angle, amp


# 以度为单位的余弦函数
def cosd_f(x):
    return np.cos(x * np.pi / 180)


# 以度为单位的正弦函数
def sind_f(x):
    return np.sin(x * np.pi / 180)


class BF:
    def __init__(self, param):
        # 深拷贝参数，设置幅度和相位优化的初始范围
        self.param = copy.deepcopy(param)
        # 重新调整旁瓣压制范围
        theta_0 = self.param["theta_0"]
        range1_start = 0
        range1_end = max(0, theta_0 - 3)
        range2_start = min(180, theta_0 + 3)
        range2_end = 180
        self.param["range_list"] = [
            [range1_start - theta_0, range1_end - theta_0],
            [range2_start - theta_0, range2_end - theta_0],
        ]
        self.param["range_list_weight"] = [1, 1]
        # 生成天线单元方向图和阵列因子
        self.EF = self._generate_power_pattern()
        self.AF = self._generate_array_factor()
        # 初始 amplitudes 全部设为 1
        self.amp = np.ones(self.param["N"])
        # 将方向图与阵列因子相乘，得到场分布张量
        self.efield = torch.tensor(self.EF[None, ...] * self.AF)

    def solve(self):
        # 求解最优相位编码，解码并计算相位角
        self.x_final = self.opt_phase_QUBO()
        self.phase = self.encode(self.x_final)
        self.phase_angle = np.angle(self.phase)
        # 若启用振幅优化，则调用 opt_amp
        if self.param["opt_amp_or_not"] is True:
            self.amp, _ = self.opt_amp(self.amp, self.phase_angle)
            self.amp = np.array(self.amp.clone().detach().numpy())

        # 归一化振幅并打印结果
        self.amp = np.abs(self.amp) / np.max(np.abs(self.amp))
        print(f"phase angle: {self.phase_angle}")
        print(f"amp: {self.amp}")

    def opt_phase_QUBO(self):
        # 导入 QUBO 求解器 LQA
        from mindquantum.algorithm.qaia import LQA

        # 复权系数 c1, c2 用于主瓣增强与旁瓣抑制
        c1 = 0.5 + 0.5j
        c2 = 0.5 - 0.5j
        factor_array = torch.cat(
            (
                self.efield[:, self._get_index(self.param["theta_0"])] * c1,
                self.efield[:, self._get_index(self.param["theta_0"])] * c2,
            ),
            dim=0,
        )
        # 计算主瓣增强矩阵 J_enhance
        J_enhance = torch.einsum("i, j -> ij", factor_array.conj(), factor_array)
        # 计算各范围内的抑制矩阵 J_suppress
        J_suppress = 0.0
        for i in range(len(self.param["range_list_weight"])):
            num = 0
            a_0 = 0.0
            start_idx = round(
                self._get_index(self.param["theta_0"] + self.param["range_list"][i][0])
            )
            end_idx = round(
                self._get_index(self.param["theta_0"] + self.param["range_list"][i][1])
            )
            for j in range(start_idx, end_idx):
                num += 1
                factor_array = torch.cat(
                    (self.efield[:, j] * c1, self.efield[:, j] * c2), dim=0
                )
                a_0 += torch.einsum("i, j -> ij", factor_array.conj(), factor_array)
            J_suppress += self.param["range_list_weight"][i] * a_0 / num

        # 组合主瓣增强与旁瓣抑制，形成总 QUBO 矩阵 J
        J = torch.real(
            (self.param["weight"] * J_enhance - (1 - self.param["weight"]) * J_suppress)
        ).numpy()

        # 用 LQA 求解 QUBO 问题
        batch_size = 1
        solver = LQA(
            np.array(J, dtype="float64"),
            batch_size=batch_size,
        )

        solver.update()

        # 获取最佳比特解，并按行列重构为 (N, encode_qubit)
        if batch_size == 1:
            x_best = solver.x.reshape(2 * self.param["N"], 1)
        else:
            energies = solver.calc_energy()
            best_idx = np.argmin(energies)
            x_best = solver.x[:, best_idx].reshape(2 * self.param["N"], 1)

        return x_best.reshape(self.param["N"], self.param["encode_qubit"], order="F")

    def opt_amp(self, amp, phase_angle):
        # 定义振幅优化的目标函数
        def cost_func_for_amp(amp, phase_angle):
            phase = torch.exp(1.0j * phase_angle)
            # 计算主瓣能量
            main_lobe = torch.einsum(
                "i, i -> ",
                phase * amp,
                self.efield[:, self._get_index(self.param["theta_0"])],
            )
            loss_2 = (
                1.0
                * self.param["weight"]
                * torch.real(torch.conj(main_lobe) * main_lobe)
            )
            loss_1 = 1.0
            # 计算各抑制区间的平均能量
            for i in range(len(self.param["range_list_weight"])):
                start_idx = round(
                    self._get_index(
                        self.param["theta_0"] + self.param["range_list"][i][0]
                    )
                )
                end_idx = round(
                    self._get_index(
                        self.param["theta_0"] + self.param["range_list"][i][1]
                    )
                )
                one_range = torch.einsum(
                    "i, ij -> j",
                    phase * amp,
                    self.efield[:, start_idx:end_idx],
                )
                loss_1 += (
                    (1 - self.param["weight"])
                    * self.param["range_list_weight"][i]
                    * (torch.real(torch.conj(one_range) * one_range)).mean()
                )
            # 构造综合目标函数
            obj = loss_1**3 / (loss_2 + 1e-8)
            return obj

        # 将 numpy 振幅转换为可微张量并构建优化器
        amplitude = torch.tensor(amp.copy())
        amplitude.requires_grad = True
        optimizer = torch.optim.Adam([amplitude], lr=self.param["lr"])

        # 迭代优化
        for iter in tqdm(range(self.param["iter"])):
            optimizer.zero_grad()
            loss = cost_func_for_amp(amplitude, torch.tensor(phase_angle))
            loss.backward()
            optimizer.step()
        return amplitude, loss

    def encode(self, x_bit):
        # 将 QUBO 解码为复数相位值，根据 encode_qubit 选择不同映射
        is_torch = isinstance(x_bit, torch.Tensor)
        complex_dtype = torch.complex64 if is_torch else np.complex64
        Nb = self.param["encode_qubit"]
        if Nb == 1:
            # 单比特映射
            c0 = (
                torch.tensor(1.0 + 0.0j, dtype=complex_dtype)
                if is_torch
                else np.array(1.0 + 0.0j, dtype=complex_dtype)
            )
            phase = c0 * x_bit[:, 0]

        elif Nb == 2:
            # 2 比特 QPSK 映射
            c0 = (
                torch.tensor(0.5 + 0.5j, dtype=complex_dtype)
                if is_torch
                else np.array(0.5 + 0.5j, dtype=complex_dtype)
            )
            c1 = (
                torch.tensor(0.5 - 0.5j, dtype=complex_dtype)
                if is_torch
                else np.array(0.5 - 0.5j, dtype=complex_dtype)
            )
            phase = c0 * x_bit[:, 0] + c1 * x_bit[:, 1]

        elif Nb == 3:
            # 3 比特映射（八相位）
            c0 = (
                torch.tensor(0.2500 + 0.6036j, dtype=complex_dtype)
                if is_torch
                else np.array(0.2500 + 0.6036j, dtype=complex_dtype)
            )
            c1 = (
                torch.tensor(0.6036 - 0.2500j, dtype=complex_dtype)
                if is_torch
                else np.array(0.6036 - 0.2500j, dtype=complex_dtype)
            )
            c2 = (
                torch.tensor(0.2500 - 0.1036j, dtype=complex_dtype)
                if is_torch
                else np.array(0.2500 - 0.1036j, dtype=complex_dtype)
            )
            c3 = (
                torch.tensor(-0.1036 - 0.2500j, dtype=complex_dtype)
                if is_torch
                else np.array(-0.1036 - 0.2500j, dtype=complex_dtype)
            )
            s1, s2, s3 = x_bit[:, 0], x_bit[:, 1], x_bit[:, 2]
            phase = c0 * s1 + c1 * s2 + c2 * s3 + c3 * (s1 * s2 * s3)

        elif Nb == 4:
            # 4 比特复杂映射，系数从 coeffs 列表读取
            coeffs = [
                0.000000 + 0.000000j,
                -0.125000 - 0.628417j,
                0.000000 + 0.000000j,
                -0.000000 - 0.000000j,
                -0.000000 - 0.000000j,
                0.628417 - 0.125000j,
                0.260299 - 0.051777j,
                0.125000 - 0.024864j,
                -0.000000 - 0.000000j,
                0.000000 - 0.000000j,
                -0.000000 - 0.000000j,
                0.051777 + 0.260299j,
                0.024864 + 0.125000j,
                0.010299 + 0.051777j,
                -0.000000 - 0.000000j,
                -0.051777 + 0.010299j,
            ]
            c4 = (
                torch.tensor(coeffs, dtype=complex_dtype)
                if is_torch
                else np.array(coeffs, dtype=complex_dtype)
            )
            s = x_bit
            # 构造所有组合特征向量 feats
            feats = [
                torch.ones_like(s[:, 0], dtype=complex_dtype)
                if is_torch
                else np.ones_like(s[:, 0], dtype=complex_dtype)
            ]
            for i in range(4):
                feats.append(
                    s[:, i].to(complex_dtype)
                    if is_torch
                    else s[:, i].astype(complex_dtype)
                )
            for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
                feats.append(
                    (s[:, i] * s[:, j]).to(complex_dtype)
                    if is_torch
                    else (s[:, i] * s[:, j]).astype(complex_dtype)
                )
            for i, j, k in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
                feats.append(
                    (s[:, i] * s[:, j] * s[:, k]).to(complex_dtype)
                    if is_torch
                    else (s[:, i] * s[:, j] * s[:, k]).astype(complex_dtype)
                )
            feats.append(
                (s[:, 0] * s[:, 1] * s[:, 2] * s[:, 3]).to(complex_dtype)
                if is_torch
                else (s[:, 0] * s[:, 1] * s[:, 2] * s[:, 3]).astype(complex_dtype)
            )
            F = torch.stack(feats, dim=1) if is_torch else np.stack(feats, axis=1)
            phase = F.matmul(c4) if is_torch else F.dot(c4)

        else:
            raise ValueError(f"Unsupported encode_qubit={Nb}")
        return phase

    def plot(self):
        # 计算扫描角度和阵列方向图
        self.theta = np.linspace(0, 180, 180 * self.param["n_angle"] + 1)
        F = torch.einsum(
            "i, ij -> j",
            torch.tensor(self.amp) * np.exp(1.0j * self.phase_angle),
            self.efield,
        ).numpy()
        self.FF = np.real(F.conj() * F)
        self.y = 10 * np.log10(self.FF / np.max(self.FF))

        # 绘制并保存波束形成结果
        plt.figure()
        plt.plot(self.theta, self.y)
        plt.xlabel(r"$\theta$ (°)")
        plt.ylabel(r"$10\log_{10}|F(\theta)|^2 - 10\log_{10}|F(\theta)|^2_{\max}$ (dB)")
        plt.title("Beamforming Outcome")
        plt.savefig(str(self.param["theta_0"]) + "_beamforming.jpg")

    def _generate_power_pattern(self):
        # 计算单元方向图：剪切后 -30dB 外设定为 -30dB
        theta = np.linspace(0, 180, 180 * self.param["n_angle"] + 1)
        x = 12 * ((theta - 90) / 90) ** 2
        E_dB = -1.0 * np.where(x < 30, x, 30)
        E_theta = 10 ** (E_dB / 10)
        EF = E_theta**0.5
        return EF

    def _generate_array_factor(self):
        # 计算阵列因子 AF = exp(jπn cosθ)
        theta = np.linspace(0, 180, 180 * self.param["n_angle"] + 1)
        phase_x = 1j * np.pi * cosd_f(theta)
        AF = np.exp(phase_x[None, :] * np.arange(self.param["N"])[:, None])
        return AF

    def _get_index(self, angle_value):
        # 将角度映射到数组索引
        index = round(angle_value * self.param["n_angle"])
        return index


if __name__ == "__main__":
    # 测试入口：传入 [theta_0, encode_qubit, opt_amp_flag]
    result = optimized([112, 2, False])
