import numpy as np
import torch
from matplotlib import pyplot as plt
import copy
from mindquantum.algorithm.qaia import *

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
    """
    # 在后续优化过程中使用的参数
    param = {
        'theta_0': variables[0],  # 波束成形方向，以90度为例
        'N': 32,  # 天线阵子总数
        'n_angle': 10,  # 1度中被细分的次数
        'encode_qubit_phase': variables[1],  # 进行相位编码的比特个数，对应变量 variables[1]
        'encode_qubit_amp': 4,  # 进行振幅编码的比特个数

        # qaia参数界面
        'xi': 0.1,  # 模拟分叉算法中调节损失函数的相对大小
        'dt': 0.3,  # 演化步长
        'batch_size_phase': 30,  # 相位批处理大小，重要参数，考虑时间条件时，可在程序中增加时间测量动态调节该参数，使之最大化
        'batch_size_amp': 2,  # 振幅批处理大小
        'qaia': [CAC, CFC, BSB, DSB, SimCIM, LQA, NMFA], # 彩虹量子启发算法的算法集合，[CAC, CFC, BSB, DSB, SimCIM, LQA, NMFA, SFC]等可选

        # torch参数界面
        'n_iter_phase': 50,  # 优化相位迭代步数
        'n_iter_amp': 10,  # 优化振幅的迭代步数

        # 相位损失函数界面 
        'weight': 0.5,  # 调节损失函数中分子和分母的相对大小
        'range_list': [[-30, -6], [6, 30]], # 需要压制的旁瓣范围相对于主瓣波束成形方向的角度表示
        'range_list_weight': [1, 1], # 每个压制的旁瓣范围各自的权重

        # 连续振幅优化
        'opt_amp_or_not': variables[2], # 根据输入控制是否优化振幅
        'lr': 0.1, # 学习率
    }
    A = BF(param)
    A.solve()
    #A.plot()
    phase_angle = A.phase_angle
    amp = A.amp
    return phase_angle, amp

# 计算角度制下的三角余弦函数
def cosd_f(x):
    return np.cos(x * np.pi / 180)

# 计算角度制下的三角正弦函数
def sind_f(x):
    return np.sin(x * np.pi / 180)

class BF():
    # 初始化
    # ============================================================================================================================ #
    def __init__(self, param):
        self.param = copy.deepcopy(param)
        self.EF = self._generate_power_pattern()
        self.AF = self._generate_array_factor()
        self.amp = np.ones(self.param['N'])
        self.efield = torch.tensor(self.EF[None, ...] * self.AF)

        self.N = self.param['N']  #天线数
        self.theta_0 = self.param['theta_0']  # 波束成形方向
        self.opt_amp_or_not = self.param['opt_amp_or_not']  # 是否优化振幅
        self.encode_qubit_phase = self.param['encode_qubit_phase']  # 相位编码比特数
        self.c_phase = self._get_c_phase()  # 相位编码系数
        if self.opt_amp_or_not:
            self.encode_qubit_amp = self.param['encode_qubit_amp']  # 振幅编码比特数
            self.c_amp = self._get_c_amp()  # 振幅编码系数
        print(f'用例: {[self.theta_0, self.encode_qubit_phase, self.opt_amp_or_not]}')

    # 求解主流程
    # ============================================================================================================================ #
    def solve(self):
        # 量子启发式部分
        print('量子启发式部分计算中...')
        x_phase_angle_candidate = self.opt_phase_Qrainbow()  # 优化相位角（彩虹量子启发式算法）
        phase_angle_candidate = self.encode_phase(x_phase_angle_candidate)  # 编码相位角
        phase_angle_candidate = self.refine_phase(phase_angle_candidate, is_hard=True) # 候选解精炼
        if self.opt_amp_or_not:
            amp_candidate = self.opt_amp_Qrainbow(phase_angle_candidate)  # 优化振幅(一阶段：彩虹量子启发式算法)
            phase_angle_candidate, amp_candidate = self.opt_amp_Cgradient(phase_angle_candidate, amp_candidate)  # 优化振幅(二阶段：经典梯度算法)
        else:
            amp_candidate = np.ones((phase_angle_candidate.shape[0], self.N))

        # 经典梯度部分
        print('经典梯度部分计算中...')
        phase_angle_candidate_C = self.opt_phase_Cgradient()  # 优化相位角（经典梯度下降算法）
        amp_candidate_C = np.ones((phase_angle_candidate_C.shape[0], self.N))
        if self.opt_amp_or_not:            
            phase_angle_candidate_C, amp_candidate_C = self.opt_amp_Cgradient(phase_angle_candidate_C, amp_candidate_C)  # 优化振幅（经典梯度算法）

        # 候选解合并与评估
        print('候选解合并与评估中...')
        phase_angle_candidate = np.concatenate((phase_angle_candidate, phase_angle_candidate_C), axis=0)
        amp_candidate = np.concatenate((amp_candidate, amp_candidate_C), axis=0)    
        self.candidate_eval(phase_angle_candidate, amp_candidate)
     
    # 彩虹量子启发式算法（优化相位角）
    # ============================================================================================================================ #
    def opt_phase_Qrainbow(self):
        theta0_idx = self._get_index(self.theta_0)
        weight = self.param['weight']
        batch_size = self.param['batch_size_phase']
        mainlobe = self.build_mainlobe(self.c_phase, theta0_idx) # 构建增强主瓣的J矩阵部分
        sidelobe = self.build_sidelobe(self.c_phase) # 构建抑制旁瓣的J矩阵部分
        J = weight * mainlobe - (1 - weight) * sidelobe # 目标函数
        all_batch_size = len(self.param['qaia'])*batch_size
        all_x_bit = np.array([])
        for qaia in self.param['qaia']:
            try:
                solver = qaia(np.array(J, dtype="float64"), batch_size=batch_size)
                solver.update()
                x_bit = np.sign(solver.x.reshape(-1, batch_size)) 
                if all_x_bit.size == 0:
                    all_x_bit = x_bit
                else:
                    all_x_bit = np.concatenate([all_x_bit, x_bit], axis=1) 
            except:
                all_batch_size = all_batch_size - batch_size
                print(f'{qaia.__name__} failed')
        return all_x_bit.reshape(self.N, -1, all_batch_size, order='F').transpose((2, 0, 1))  # 形状为 (batch_size, N, encode_qubit_phase)

    # 彩虹量子启发式算法（优化幅度）
    # ============================================================================================================================ #    
    def opt_amp_Qrainbow(self, phase_angle_candidate):
        amp_candidate = np.array([])
        theta0_idx = self._get_index(self.theta_0)
        batch_size = self.param['batch_size_amp']
        factor1 = np.array(self.efield[:, theta0_idx])  # factor1[n] 表示 exp(jπncosθ)
        for phase_angle in phase_angle_candidate:
            factor2 = np.exp(1.0j * phase_angle)  # factor2[n] 表示 exp(jφn)
            factor = factor1 * factor2  # factor[n] 表示除了振幅之外的An(θ)的部分
            amp_term = np.outer(factor, self.c_amp).flatten(order='F')
            J_h = (np.outer(amp_term, amp_term.conj())).real
            h = np.sum(J_h[-self.N:], axis=0)[:-self.N]  #分离出h，也就是把外积矩阵的最后N行相加（再截断最后N个元素），得到h
            J = J_h[:-self.N, :-self.N]  #分离出J
            all_batch_size = len(self.param['qaia'])*batch_size
            all_x_bit = np.array([])
            for qaia in self.param['qaia']:
                try:
                    solver = qaia(J=np.array(J, dtype="float64"), h=np.array(h, dtype="float64"), batch_size=batch_size)
                    solver.update()
                    x_bit = np.sign(solver.x.reshape(batch_size, -1))
                    if all_x_bit.size == 0:
                        all_x_bit = x_bit
                    else:
                        all_x_bit = np.concatenate([all_x_bit, x_bit], axis=1)
                except:
                    all_batch_size = all_batch_size - batch_size
                    print(f'{qaia.__name__} failed')
            all_x_bit = all_x_bit.reshape(self.N, -1, all_batch_size, order='F').transpose((2, 0, 1))  # 形状为 (batch_size, N, encode_qubit_amp)
            x_amp_candidate_local = np.unique(all_x_bit, axis=0)  #当前相位对应的候选振幅
            amp_candidate_local = self.encode_amp(x_amp_candidate_local)
            amp_candidate_local = np.append(amp_candidate_local, np.ones(self.N)).reshape(-1, self.N)  # 添加一个全1的振幅作为候选解
            score_candidate_amp_local = np.array([])
            for amp_local in amp_candidate_local:
                score_candidate_amp_local = np.append(score_candidate_amp_local, score(phase_angle, amp_local, [self.theta_0, self.encode_qubit_phase, self.opt_amp_or_not]))
            index = np.argmax(score_candidate_amp_local)
            amp_candidate = np.append(amp_candidate, amp_candidate_local[index])
        amp_candidate = amp_candidate.reshape(-1, self.N)
        return amp_candidate

    # 经典梯度算法（优化相位角）
    # ============================================================================================================================ #
    def opt_phase_Cgradient(self):
        # 优化相位的损失函数
        def cost_func(x_bit):
            '''
            Args: 
                x_bit: SB算法中的变量x
            Returns:
                obj: 损失函数的数值
            '''
            coeffs = self._get_coefficients(self.encode_qubit_phase)
            phase = torch.zeros(self.N, dtype=torch.complex128)            
            for n in range(self.N):
                phase[n] = sum(c * x_bit[n, i] for i, c in enumerate(coeffs))
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
            obj = loss_1 / loss_2 # 目标函数需要重点修改，用户可以自定义目标函数，如加权求和等
            return obj 
    
        # 初始化
        x = 0.01 * (np.random.randn(self.N, 2**(self.encode_qubit_phase-1)))
        y = 0.01 * (np.random.randn(self.N, 2**(self.encode_qubit_phase-1)))
        for iter in range(self.param['n_iter_phase'] + 1):
            x_torch = torch.tensor(x)
            x_torch.requires_grad = True
            # 计算梯度与更新参数
            x_sign = x_torch - (x_torch - torch.sign(x_torch)).detach() #dSB
            loss = cost_func(x_sign)
            loss.backward()
            x_grad = (x_torch.grad).clone().detach().numpy()
            y += (-(0.5 - iter / self.param['n_iter_phase']) * x - self.param['xi'] * x_grad / np.linalg.norm(x_grad)) * self.param['dt']
            x = x + y * self.param['dt']
            cond = np.abs(x) > 1
            x = np.where(cond, np.sign(x), x)
            y = np.where(cond, np.zeros_like(y), y)
        x_bit = np.sign(x)  # 最终的x_bit
        coeffs = self._get_coefficients(self.encode_qubit_phase)
        phase = torch.zeros(self.N, dtype=torch.complex128)            
        for n in range(self.N):
            phase[n] = sum(c * x_bit[n, i] for i, c in enumerate(coeffs))
        phase_angle = np.angle(phase)
        phase_angle_candidate = phase_angle[np.newaxis, :]  # 经典梯度算法只输出一个batch_size的结果
        return phase_angle_candidate

    # 经典梯度算法（优化振幅）
    # ============================================================================================================================ #
    def opt_amp_Cgradient(self, phase_angle_candidate, amp_candidate):
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
            main_lobe = torch.einsum('i, i -> ', phase * amp, self.efield[:, self._get_index(self.theta_0)])
            loss_2 = 1.0 * self.param['weight'] * torch.real(torch.conj(main_lobe) * main_lobe)
            loss_1 = 1.0
            for i in range(len(self.param['range_list_weight'])):
                one_range = torch.einsum('i, ij -> j', phase * amp, self.efield[:, round(
                    self._get_index(self.theta_0 + self.param['range_list'][i][0])): round(
                    self._get_index(self.theta_0 + self.param['range_list'][i][1]))])
                loss_1 += (1 - self.param['weight']) * self.param['range_list_weight'][i] * (
                    torch.real(torch.conj(one_range) * one_range)).mean()
            obj = loss_1 / loss_2
            return obj
        
        i = 0
        for phase_angle, amp in zip(phase_angle_candidate, amp_candidate):
            amplitude = torch.tensor(amp.copy())
            amplitude.requires_grad = True
            optimizer = torch.optim.Adam([amplitude], lr=self.param['lr'])

            for iter in range(self.param['n_iter_amp']):
                optimizer.zero_grad()
                loss = cost_func_for_amp(amplitude, torch.tensor(phase_angle))
                loss.backward()
                optimizer.step()
            amp_tmp = amplitude.clone().detach().numpy()
            cond = amp_tmp < 0 # 后处理：归一化振幅，并且将振幅按照量化比特数的要求进行离散化；将负数振幅转化为正数的形式，同时给相位角加上pi
            phase_angle_tmp = np.angle(np.exp(1.0j * np.where(cond, phase_angle + np.pi, phase_angle)))  # 确保最后相位角变化为0到2\pi
            amp_tmp = np.abs(amp_tmp) / np.max(np.abs(amp_tmp)) # 将振幅归一化
            if score(phase_angle_tmp, amp_tmp, [self.theta_0, self.encode_qubit_phase, self.opt_amp_or_not]) > score(phase_angle, amp, [self.theta_0, self.encode_qubit_phase, self.opt_amp_or_not]):
                phase_angle_candidate[i] = phase_angle_tmp.copy()
                amp_candidate[i] = amp_tmp.copy()
            i += 1
        return phase_angle_candidate, amp_candidate
    
    # 候选解合并与评估
    # ============================================================================================================================ #
    def candidate_eval(self, phase_angle_candidate, amp_candidate):
        score_candidate = np.array([])  # 初始化候选解的分数列表
        for phase_angle, amp in zip(phase_angle_candidate, amp_candidate):
            score_candidate = np.append(score_candidate, score(phase_angle, amp, [self.theta_0, self.encode_qubit_phase, self.opt_amp_or_not]))
        index = np.argmax(score_candidate)
        self.amp = amp_candidate[index]
        self.phase_angle = phase_angle_candidate[index]
        # 打印结果
        print(f'候选得分: ')
        print(f'{score_candidate}')
        print(f'最终得分: {np.max(score_candidate)}')
        print('')

    # 编码相关函数
    # ============================================================================================================================ #
    # 主瓣的J矩阵构建（基于双外积法）
    def build_mainlobe(self, c, theta0_idx):
        factor = np.array(self.efield[:, theta0_idx])  # factor[i] 表示 exp(jπncosθ)
        mainlobe_term = np.outer(factor, c).flatten(order='F')
        J_mainlobe = (np.outer(mainlobe_term, mainlobe_term.conj())).real  # 主瓣信号强度（模平方）
        return J_mainlobe

    # 旁瓣的J矩阵构建（基于双外积法）
    def build_sidelobe(self, c):
        J_sidelobe = 0.0
        for i in range(len(self.param['range_list_weight'])):
            num = 0
            a_0 = 0.0
            for j in range(round(self._get_index(self.theta_0 + self.param['range_list'][i][0])), round(self._get_index(self.theta_0 + self.param['range_list'][i][1])), 1):
                num += 1
                factor= np.array(self.efield[:, j])
                sidelobe_term = np.outer(factor, c).flatten(order='F')
                a_0 += (np.outer(sidelobe_term, sidelobe_term.conj())).real
            J_sidelobe += self.param['range_list_weight'][i] * a_0 / num
        return J_sidelobe    

    # 相位编码
    def encode_phase(self, x_bit):
        phase = np.einsum('ijk,k->ij', x_bit, self.c_phase)  
        phase_angle = np.angle(phase)
        return phase_angle
    
    # 振幅编码
    def encode_amp(self, x_bit):
        amp = np.einsum('ijk,k->ij', (1-x_bit), self.c_amp[:self.encode_qubit_amp])  
        return amp

    # 获得相位编码的系数，外部程序预生成，这里直接使用
    def _get_c_phase(self):
        if self.encode_qubit_phase == 1:
            return np.array([1.0+0j])
        elif self.encode_qubit_phase == 2:
            return np.array([0.5+0.5j, 0.5-0.5j])
        elif self.encode_qubit_phase == 3:
            return np.array([0.25+0.60355339j, 0.60355339-0.25j, -0.10355339-0.25j, 0.25-0.10355339j])
        # elif self.encode_qubit_phase == 4:  # 灰奇组合编码，可选备用
        #     return np.array([0.125+0.62841744j, 0.62841744-0.125j, -0.0517767-0.26029903j, -0.01029903-0.0517767j,
        #                      0.26029903-0.0517767j, -0.0517767+0.01029903j, -0.02486405-0.125j, 0.125-0.02486405j])
        elif self.encode_qubit_phase == 4:  # 另一种编码方式，计算效率更高
            angles = [np.pi/8, 3*np.pi/8, 5*np.pi/8, 7*np.pi/8]
            return [np.exp(1j * ang)/2 for ang in angles]
        else:
            raise ValueError("Unsupported encode_qubit")
    
    # 获得振幅编码的系数
    def _get_c_amp(self):
        arr = 1 / (2 ** np.arange(2, self.encode_qubit_amp + 2))
        arr = np.append(arr, -arr.sum())  # 偏置项
        return arr

    # 基于层次聚类的候选解精炼
    # ============================================================================================================================ #
    def refine_phase(self, phase_angle_candidate, min_num_candidates = 20, retention_ratio = 0.2, is_hard = False):  # min_num_candidates最少候选解个数，retention_ratio精炼度，is_hard=True表示只作绝对去重
        """
        基于层次聚类的候选解精炼
        通过计算相位角候选解的相似度，减少冗余解的数量
        """
        # 1. 预处理：去除完全相同的候选解
        phase_angle_candidate = np.unique(phase_angle_candidate, axis=0)
        if is_hard == True:
            return phase_angle_candidate
        num_candidates = phase_angle_candidate.shape[0]
        # 2. 如果候选解数量较少，直接返回
        if num_candidates <= min_num_candidates:
            return phase_angle_candidate
        # 3. 计算相位角距离矩阵（考虑2π周期）
        D = np.zeros((num_candidates, num_candidates))
        for i in range(num_candidates):
            for j in range(i+1, num_candidates):
                diff = np.abs(phase_angle_candidate[i] - phase_angle_candidate[j])  # 计算两个候选解的相位角差值（考虑2π周期性）
                circular_diff = np.minimum(diff, 2*np.pi - diff)
                D[i, j] = np.mean(circular_diff) # 计算平均相位角距离
                D[j, i] = D[i, j]
        # 4. 层次聚类（自底向上）
        clusters = [{i} for i in range(num_candidates)]  # 初始化：每个候选解是一个簇
        cluster_dist = np.full((num_candidates, num_candidates), np.inf)  # 簇间距离矩阵（初始为候选解间的距离）
        for i in range(num_candidates):
            for j in range(i+1, num_candidates):
                cluster_dist[i, j] = D[i, j]
                cluster_dist[j, i] = D[i, j]
        target_clusters = max(min_num_candidates, int(num_candidates * retention_ratio))  # 目标簇数量
        while len(clusters) > target_clusters:  # 当簇数量大于目标数量时合并
            min_dist = np.inf  
            merge_i, merge_j = -1, -1
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    if cluster_dist[i, j] < min_dist:
                        min_dist = cluster_dist[i, j]  # 找到距离最小的两个簇
                        merge_i, merge_j = i, j
            if min_dist == np.inf:
                break  # 没有可合并的簇
            new_cluster = clusters[merge_i] | clusters[merge_j]  # 合并两个簇
            clusters.pop(max(merge_i, merge_j))
            clusters.pop(min(merge_i, merge_j))
            clusters.append(new_cluster)  # 更新簇列表
            n = len(clusters)
            new_dist = np.full((n, n), np.inf)  # 更新簇间距离矩阵          
            for i in range(n-1):
                for j in range(i+1, n-1):
                    new_dist[i, j] = cluster_dist[i, j]  # 复制未变动的距离
            new_idx = n-1
            for idx in range(n-1):
                dist_sum = 0
                count = 0
                for a in clusters[new_idx]:
                    for b in clusters[idx]:
                        dist_sum += D[a, b]
                        count += 1
                if count > 0:
                    avg_dist = dist_sum / count
                    new_dist[new_idx, idx] = avg_dist
                    new_dist[idx, new_idx] = avg_dist
            cluster_dist = new_dist  # 计算新簇与其他簇的距离（平均链接法），计算新簇与当前簇所有元素对的距离
        # 5. 从每个簇中选择代表解（中心解）
        representatives = []
        for cluster in clusters:
            cluster_list = list(cluster)
            dist_sums = np.zeros(len(cluster_list))
            for i, cand_i in enumerate(cluster_list):
                for j, cand_j in enumerate(cluster_list):
                    if i != j:
                        dist_sums[i] += D[cand_i, cand_j]  # 计算簇内每个解与其他解的距离和
            rep_idx = cluster_list[np.argmin(dist_sums)]  # 选择距离和最小的解作为代表解
            representatives.append(phase_angle_candidate[rep_idx])
        refined_phase_angle_candidate = np.array(representatives)
        return refined_phase_angle_candidate

    # 辅助函数
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
        plt.savefig(str(self.param['theta_0']) + '_beamforming.jpg')

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

    # 获得相位编码的系数（经典梯度算法使用）
    def _get_coefficients(self, encode_qubit):
        if encode_qubit == 1:
            return [1.0]
        elif encode_qubit == 2:
            return [(1 + 1j)/2, (1 - 1j)/2]
        elif encode_qubit == 3:
            return [0.25+0.60355339j, 0.60355339-0.25j, -0.10355339-0.25j, 0.25-0.10355339j]                         
        elif encode_qubit == 4:
            return [0.125+0.62841744j, 0.62841744-0.125j, -0.0517767-0.26029903j, -0.01029903-0.0517767j,
                    0.26029903-0.0517767j, -0.0517767+0.01029903j, -0.02486405-0.125j, 0.125-0.02486405j]
        else:
            raise ValueError("Unsupported encode_qubit")
        
# 得分计算函数
def score(phase_angle, amplitude, variables):
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
    
    n_angle = 20  # 修改以提高运算效率
    range_n_angle = range(1, 20 * n_angle)
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
    max_index = np.argmax(db_array)  # 修改以提高运算效率
    theta_up = 180
    theta_down = 0
    theta_min_up = 180
    theta_min_down = 0
    if abs(theta_array[max_index] - theta_0) > 1:
        y = 0
    else:
        for i in range_n_angle:
            if db_array[i + max_index] <= -30:
                theta_up = theta_array[i + max_index]
                break
        for i in range_n_angle:
            if db_array[-i + max_index] <= -30:
                theta_down = theta_array[-i + max_index]
                break
        for i in range_n_angle:
            if (db_array[i + max_index] < db_array[i - 1 + max_index]) and (
                    db_array[i + max_index] < db_array[i + 1 + max_index]):
                theta_min_up = theta_array[i + max_index]
                break
        for i in range_n_angle:
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

