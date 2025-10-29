import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
from collections import defaultdict
import pandas as pd
import itertools
import warnings
import random
import math
import json
warnings.filterwarnings("ignore")

class HUBOSolver:
    def plot_energy_history(self, title="Energy Optimization", dataset_idx=None):
        """绘制能量函数曲线图"""
        plt.figure(figsize=(12, 6))
        # 绘制能量曲线
        for i, run_energy in enumerate(self.energy_history):
            plt.plot(run_energy, label=f"Run {i+1}", linewidth=2)
            # 添加最终能量标记
            final_energy = run_energy[-1]
            plt.scatter(len(run_energy)-1, final_energy, s=100, zorder=5, color='red')
            plt.text(len(run_energy)-1, final_energy, 
                    f"{final_energy:.4f}", fontsize=10, ha='right', va='bottom')
        plt.title(f"{self.method}", fontsize=14)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Energy", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        # 添加统计信息
        stats_text = [
            f"Method: {self.method}",
            f"Best Energy: {max(self.best_energy_history):.4f}",
            f"Avg Final Energy: {np.mean([run[-1] for run in self.energy_history]):.4f}",
            f"Runtime: {self.runtime:.2f} seconds"
        ]
        plt.figtext(0.70, 0.20, "\n".join(stats_text), fontsize=12, 
                   ha="left", bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        plt.tight_layout()
        # 保存图像
        if not os.path.exists("./four/result_three_end1"):
            os.makedirs("./four/result_three_end1")
        if dataset_idx is not None:
            filename = f"./four/result_three_end1/energy_history_{self.method}_ds{dataset_idx}.png"
        else:
            filename = f"./four/result_three_end1/energy_history_{self.method}.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved energy history plot to {filename}")
        #plt.show()
        
    def __init__(self, terms_dict, n_vars, 
                 xi=0.3, dt=0.15, n_iter=1000, method='dSB', device='cpu', runs=5):
        """
        初始化HUBO求解器
        
        参数:
        terms_dict: 字典，键为项阶数(1-8)，值为(系数, [变量索引列表])的列表
        n_vars: 变量个数
        xi: 学习率
        dt: 时间步长
        n_iter: 迭代次数
        method: 求解方法 ('bSB', 'dSB', 'CAC', 'CFC')
        device: 计算设备 ('cpu' or 'cuda')
        runs: 运行次数 (基础功能用5次，调参功能用1次)
        """
        self.terms_dict = terms_dict
        self.n_vars = n_vars
        self.xi = xi
        self.dt = dt
        self.n_iter = n_iter
        self.method = method
        self.device = device
        self.runs = runs
        
        # 初始化变量
        self.x = 0.01 * torch.randn(n_vars, device=device)
        self.y = 0.01 * torch.randn(n_vars, device=device)
        
        # 固定参数
        self.Delta = 0.5
        
        # 用于记录历史
        self.cut_history = []  # 每次迭代的HUBO值
        self.energy_history = []      # 每次迭代的能量值
        self.best_energy_history = [] # 每次迭代的最佳能量值
        self.runtime = 0              # 总运行时间
        self.x_history = []           # 变量演化历史
        self.p_history = []           # p参数历史
        
        # 预计算能量计算的矩阵表示
        self._precompute_energy_matrix()

    def _precompute_energy_matrix(self):
        """预计算能量函数的矩阵表示，用于加速计算"""
        self.term_matrices = []
        self.term_coeffs = []
        
        # 遍历所有阶数的项
        for order, terms in self.terms_dict.items():
            for coef, vars_list in terms:
                # 创建稀疏矩阵表示
                indices = torch.tensor(vars_list, dtype=torch.long, device=self.device)
                # 存储系数和索引
                self.term_coeffs.append(coef)
                self.term_matrices.append(indices)
        
        # 转换为张量
        self.term_coeffs = torch.tensor(self.term_coeffs, device=self.device)

    def compute_energy(self, x):
        """计算HUBO能量函数 - 矩阵优化版本"""
        energy = torch.tensor(0.0, device=self.device)
        
        # 批量计算所有项
        for i, indices in enumerate(self.term_matrices):
            # 获取对应的变量值
            vars_values = x[indices]
            # 计算乘积
            prod = torch.prod(vars_values)
            # 累加能量
            energy += self.term_coeffs[i] * prod
        
        return energy

    def compute_energy_batch(self, x_batch):
        """批量计算能量函数 - 用于并行评估多个参数"""
        # x_batch: [batch_size, n_vars]
        batch_size = x_batch.shape[0]
        energies = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            energies[i] = self.compute_energy(x_batch[i])
            
        return energies

    def solve(self):
        best_energy = float('-inf')  # 初始化为负无穷（最大化）
        best_x = None
        self.energy_history = []
        self.best_energy_history = []
        self.x_history = []
        self.p_history = []
        self.cut_history=[]
        
        start_time = time.time()
        
        # 多次运行取最佳解，这样如果不是优化参数，可视化算法时间对比的时候，改成多轮运行
        for run in range(self.runs):
            # 重新初始化变量
            self.x = 0.01 * torch.randn(self.n_vars, device=self.device)
            self.y = 0.01 * torch.randn(self.n_vars, device=self.device)

            run_hubo=[]  # 本次运行的HUBO值历史
            run_energy = []          # 本次运行的能量值历史
            run_best_energy = float('-inf')   # 本次运行的最佳能量值（最大化）
            
            # 使用进度条
            for iter in tqdm(range(self.n_iter + 1), desc=f"Run {run+1}"):
                p_t = iter / self.n_iter  # 时间参数
                self.p_history.append(p_t)  # 记录p值历史
                
                # 创建需要梯度的变量
                x_torch = self.x.clone().detach().requires_grad_(True)
                
                # 计算能量
                energy = self.compute_energy(x_torch)
                
                # 记录当前值
                current_energy = energy.item()
                run_energy.append(current_energy)
                run_hubo.append(current_energy)
                
                # 更新本次运行的最佳能量值（最大化）
                if current_energy > run_best_energy:
                    run_best_energy = current_energy
                
                # 计算梯度
                energy.backward()
                grad_E = x_torch.grad.clone()
                x_torch.grad.zero_()
                
                # 记录变量演化历史（仅最后一次运行）
                if run == self.runs - 1:
                    self.x_history.append(self.x.detach().cpu().numpy().copy())
                
                # 根据公式更新变量
                dx = self.y  # dx_i/dt = y_i
                
                if self.method == 'bSB':
                    dy = -(self.Delta - p_t) * self.x + self.xi * grad_E
                
                elif self.method == 'dSB':
                    with torch.no_grad():
                        sign_x = torch.sign(self.x)
                    dy = -(self.Delta - p_t) * sign_x + self.xi * grad_E
                
                # 实现CAC算法（相干模拟耦合）
                elif self.method == 'CAC':
                    with torch.no_grad():
                        sign_x = torch.sign(self.x)
                    # CAC方程：dy/dt = -Γy - [Δ - p(t)]sign_x + ξ∇E
                    gamma = 0.1  # 阻尼系数
                    dy = -gamma * self.y - (self.Delta - p_t) * sign_x + self.xi * grad_E
                
                # 实现CFC算法（相干反馈控制）
                elif self.method == 'CFC':
                    with torch.no_grad():
                        sign_x = torch.sign(self.x)
                        global_avg = torch.mean(self.x)  # 全局平均值
                    
                    # CFC方程：dy/dt = -Γy - [Δ - p(t)]sign_x + ξ∇E + β(x - <x>)
                    gamma = 0.1  # 阻尼系数
                    beta = 0.05  # 反馈增益
                    dy = -gamma * self.y - (self.Delta - p_t) * sign_x + self.xi * grad_E + beta * (self.x - global_avg)
                
                # 更新位置和动量
                new_x = self.x + dx * self.dt
                new_y = self.y + dy * self.dt
                
                # 边界处理：当 |x_i| >= 1
                over_bound = torch.abs(new_x) >= 1.0
                new_x[over_bound] = torch.sign(new_x[over_bound])
                new_y[over_bound] = 0.0
                
                # 更新变量
                self.x = new_x.detach()
                self.y = new_y.detach()
            
            # 记录本次运行的历史
            self.energy_history.append(run_energy)
            self.best_energy_history.append(run_best_energy)
            self.cut_history.append(run_hubo)
            # 记录当前运行的最佳解（最大化）
            with torch.no_grad():
                current_x = torch.sign(self.x)
                current_energy = self.compute_energy(current_x).item()
                
                if current_energy > best_energy:
                    best_energy = current_energy
                    best_x = current_x
        
        self.runtime = time.time() - start_time
        return best_x, best_energy
    
    def plot_bifurcation(self, dataset_idx=None):
        """绘制变量分叉图"""
        if not self.x_history:
            print("No x history recorded for bifurcation plot.")
            return
        
        # 转换为numpy数组
        x_hist = np.array(self.x_history)
        total_iters = np.arange(len(x_hist))

        plt.figure(figsize=(10, 6))

        # 绘制每个变量的演化
        for var_idx in range(self.n_vars):
            plt.plot(total_iters, x_hist[:, var_idx], alpha=0.6, linewidth=1.5)

        plt.title(f"{self.method} - Variable Bifurcation", fontsize=14)
        plt.xlabel("Total Iteration", fontsize=12)
        plt.ylabel("Variable Value", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存图像
        if dataset_idx is not None:
            filename = f"./four/result_three_end1/bifurcation_{self.method}_ds{dataset_idx}.png"
        else:
            filename = f"./four/result_three_end1/bifurcation_{self.method}.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved bifurcation plot to {filename}")
        #plt.show()

    def plot_final_distribution(self, dataset_idx=None):
        """绘制HUBO值分布直方图"""
        if not self.cut_history:
            print("No HUBO history recorded for final distribution plot.")
            return

        # 展开所有run的所有iteration的HUBO值
        all_HUBO_values = np.array([hubo for run in self.cut_history for hubo in run])

        plt.figure(figsize=(10, 6))
        plt.hist(all_HUBO_values, bins=20, alpha=0.7, color='blue', edgecolor='black')

        plt.title(f"{self.method} - HUBO Value Distribution (All Iterations)", fontsize=14)
        plt.xlabel("HUBO Value (All Iterations)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 添加统计信息
        stats_text = [
            f"Mean: {np.mean(all_HUBO_values):.4f}",
            f"Std Dev: {np.std(all_HUBO_values):.4f}",
            f"Min: {np.min(all_HUBO_values):.4f}",
            f"Max: {np.max(all_HUBO_values):.4f}"
        ]

        plt.figtext(0.75, 0.75, "\n".join(stats_text), fontsize=12, 
                ha="left", bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

        plt.tight_layout()

        # 保存图像
        if dataset_idx is not None:
            filename = f"./four/result_three_end1/distribution_{self.method}_ds{dataset_idx}.png"
        else:
            filename = f"./four/result_three_end1/distribution_{self.method}.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved distribution plot to {filename}") 

def generate_hubo_terms(n_vars, max_order=3, density=0.2, coef_range=(-10, 10)):
    """
    生成随机HUBO项
    
    参数:
    n_vars: 变量个数
    max_order: 最高阶数 (1-3)
    density: 项密度 (0-1)
    coef_range: 系数范围
    
    返回:
    terms_dict: 字典，键为阶数，值为(系数, [变量索引])列表
    """
    terms_dict = defaultdict(list)

    # 生成不同阶数的项
    for order in range(1, max_order + 1):
        # 计算可能的组合数
        total_combinations = math.comb(n_vars, order)
        # 根据密度计算要生成的项数
        num_terms = max(1, int(total_combinations * density))

        # 生成指定数量的唯一项
        generated = set()
        while len(generated) < num_terms:
            # 随机选择order个不重复的变量
            vars = tuple(sorted(np.random.choice(n_vars, size=order, replace=False)))
            if vars not in generated:
                generated.add(vars)
                # 生成随机系数
                coef = np.random.randint(coef_range[0], coef_range[1])
                terms_dict[order].append((coef, vars))

    return dict(terms_dict)

def square_hubo_terms(terms_dict):
    """
    将HUBO项平方，得到新的HUBO项（最高6阶）
    参数:
    terms_dict: 原始HUBO项字典
    返回:
    squared_terms_dict: 平方后的HUBO项字典（最高6阶）
    """
    squared_terms_dict = defaultdict(list)
        
    # 获取所有项
    all_terms = []
    for order, terms in terms_dict.items():
        for coef, vars in terms:
            all_terms.append((coef, vars))
    
    n = len(all_terms)
    
    # 计算所有项对的乘积
    for i in range(n):
        for j in range(i, n):
            coef_i, vars_i = all_terms[i]
            coef_j, vars_j = all_terms[j]
            
            # 计算新系数
            new_coef = coef_i * coef_j
            
            # 如果i != j，需要乘以2（因为交叉项对称）
            if i != j:
                new_coef *= 2
            
            # 计算新变量的集合（并集）
            all_vars = set(vars_i) | set(vars_j)
            # 转换为有序元组
            new_vars = tuple(sorted(all_vars))
            
            # 添加新项
            order = len(new_vars)
            if order > 0:  # 忽略零阶项（常数项）
                squared_terms_dict[order].append((new_coef, new_vars))
        
    # 合并同类项
    final_terms_dict = defaultdict(list)
    for order, terms in squared_terms_dict.items():
        term_dict = {}
        for coef, vars in terms:
            if vars in term_dict:
                term_dict[vars] += coef
            else:
                term_dict[vars] = coef
        
        for vars, coef in term_dict.items():
            final_terms_dict[order].append((coef, vars))
    
    return dict(final_terms_dict)

def combine_hubo_terms(terms_dict1, terms_dict2, lambda_val=1.0):
    """
    合并两个HUBO项字典，并乘以相应的系数
    
    参数:
    terms_dict1: 第一个HUBO项字典
    terms_dict2: 第二个HUBO项字典
    lambda_val: 第二个字典的缩放系数
    
    返回:
    合并后的HUBO项字典
    """
    combined_dict = defaultdict(list)
    
    # 添加第一个字典的所有项
    for order, terms in terms_dict1.items():
        for coef, vars in terms:
            combined_dict[order].append((coef, vars))
    
    # 添加第二个字典的所有项（乘以-lambda_val）
    for order, terms in terms_dict2.items():
        for coef, vars in terms:
            combined_dict[order].append((-lambda_val * coef, vars))
    
    # 合并同类项
    final_dict = defaultdict(list)
    for order, terms in combined_dict.items():
        term_dict = {}
        for coef, vars in terms:
            # 将变量元组作为键
            if vars in term_dict:
                term_dict[vars] += coef
            else:
                term_dict[vars] = coef
        
        for vars, coef in term_dict.items():
            # 只添加非零项
            if abs(coef) > 1e-6:
                final_dict[order].append((coef, vars))
    
    return dict(final_dict)

def hubo_terms_to_string(terms_dict):
    """将HUBO项转换为可读字符串"""
    terms_str = []
    for order, terms in terms_dict.items():
        for coef, vars in terms:
            if order == 1:
                terms_str.append(f"{coef:.3f} * x{vars[0]}")
            else:
                var_str = "".join([f"x{i}" for i in vars])
                terms_str.append(f"{coef:.3f} * {var_str}")
    return " + ".join(terms_str)

def run_algorithm_comparison(n_datasets=100, n_vars=8, density=0.15, runs=5):
    """
    运行四种算法的比较实验
    
    参数:
    n_datasets: 数据集数量
    n_vars: 变量个数
    density: 项密度
    runs: 运行次数 (基础功能用5次，调参功能用1次)
    """
    # 创建输出目录
    if not os.path.exists("./four/result_three_end1"):
        os.makedirs("./four/result_three_end1")
    
    # 存储结果
    results = []
    
    # 进度条
    pbar = tqdm(total=n_datasets, desc="Running y4 HUBO algorithm comparison")
    
    for dataset_idx in range(n_datasets):
        # 生成两个随机3阶HUBO项（分子和分母）
        num_terms = generate_hubo_terms(n_vars, max_order=3, density=density)
        den_terms = generate_hubo_terms(n_vars, max_order=3, density=density)
        
        # 将分子和分母项平方
        num_squared = square_hubo_terms(num_terms)
        den_squared = square_hubo_terms(den_terms)
        
        # 随机生成lambda值（0到1之间）
        lambda_val = random.uniform(0.5, 2.0)
        
        # 构造y4 = A² - λB²
        y4_terms = combine_hubo_terms(num_squared, den_squared, lambda_val)
        
        # 固定参数
        xi = 0.05
        dt = 0.1
        n_iter = 100
        
        # 运行四种算法
        for method in ['bSB', 'dSB', 'CAC', 'CFC']:
            # 创建求解器
            solver = HUBOSolver(
                terms_dict=y4_terms,
                n_vars=n_vars,
                xi=xi,
                dt=dt,
                n_iter=n_iter,
                method=method,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                runs=runs
            )
            
            # 求解并获取最终解和能量值
            solution, best_energy = solver.solve()
            
            # 记录结果
            results.append({
                'dataset': dataset_idx,
                'method': method,
                'energy': best_energy,
                'runtime': solver.runtime,
                'n_vars': n_vars,
                'density': density,
                'lambda': lambda_val
            })
            
            # 保存图像（每个算法前25个数据集）
            if dataset_idx < 25:
                # 获取表达式
                num_expr = hubo_terms_to_string(num_terms)
                den_expr = hubo_terms_to_string(den_terms)
                
                # 创建标题
                title = f"{method} Optimization - y4 = ({num_expr})² - {lambda_val:.3f}*({den_expr})²"
                
                # 绘制能量历史
                solver.plot_energy_history(title=title, dataset_idx=dataset_idx)
                
                # 绘制分叉图
                solver.plot_bifurcation(dataset_idx=dataset_idx)

                #绘制直方图
                solver.plot_final_distribution(dataset_idx=dataset_idx)
        
        pbar.update(1)
    
    pbar.close()
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存结果
    results_df.to_csv("./four/result_three_end1/y4_algorithm_comparison_results.csv", index=False)
    print("Saved algorithm comparison results to ./four/result_three_end1/y4_algorithm_comparison_results.csv")
    
    return results_df

def estimate_lambda_range(num_terms, den_terms, n_vars, sample_size=100, device='cpu', save_path=None):
    """
    采样法估算lambda的合理区间（Dinkelbach方法）：
    lambda ∈ [min(f(x)/g(x)), max(f(x)/g(x))]
    
    参数：
        num_terms: 分子HUBO项字典
        den_terms: 分母HUBO项字典
        n_vars: 变量个数
        sample_size: 采样数量
        device: 计算设备
        save_path: 可选，保存区间到json
    返回：
        (min_lambda, max_lambda)
    """
    # 随机采样二进制向量
    x_samples = torch.randint(0, 2, (sample_size, n_vars), device=device, dtype=torch.float32)
    # 构造分子和分母的solver
    num_solver = HUBOSolver(num_terms, n_vars, device=device)
    den_solver = HUBOSolver(den_terms, n_vars, device=device)
    # 计算能量
    f_vals = num_solver.compute_energy_batch(x_samples)
    g_vals = den_solver.compute_energy_batch(x_samples)
    # 过滤g(x)为0的样本
    mask = (g_vals.abs() > 1e-8)
    ratios = f_vals[mask] / g_vals[mask]
    if ratios.numel() == 0:
        raise ValueError("采样中分母全为0，无法估算lambda区间")
    min_lambda = ratios.min().item()
    max_lambda = ratios.max().item()
    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump({'min_lambda': min_lambda, 'max_lambda': max_lambda}, f)
    return min_lambda, max_lambda

# 基础功能主函数
def run_basic_functionality():
    """运行基础功能"""
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    # 运行算法比较实验
    n_datasets = 5
    n_vars = 10  # 减少变量数以降低计算复杂度
    density = 0.10  # 降低密度以控制问题规模
    
    print(f"Running y4 HUBO algorithm comparison on {n_datasets} datasets")
    print(f"Variables: {n_vars}, Density: {density}")
    
    # 基础功能使用5次运行
    results_df = run_algorithm_comparison(n_datasets, n_vars, density, runs=5)
    
    return results_df

if __name__ == "__main__":
    # 运行基础功能
    results_df = run_basic_functionality()