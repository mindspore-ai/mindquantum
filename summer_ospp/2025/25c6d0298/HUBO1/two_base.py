import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time
from collections import defaultdict
import pandas as pd
import itertools
import warnings
import random
import math
warnings.filterwarnings("ignore")

class HUBOSolver:
    def __init__(self, terms_dict, n_vars, 
                 xi=0.3, dt=0.15, n_iter=500, method='dSB', device='cpu'):
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
        """
        self.terms_dict = terms_dict
        self.n_vars = n_vars
        self.xi = xi
        self.dt = dt
        self.n_iter = n_iter
        self.method = method
        self.device = device
        
        # 初始化变量
        self.x = 0.01 * torch.randn(n_vars, device=device)
        self.y = 0.01 * torch.randn(n_vars, device=device)
        
        # 固定参数
        self.Delta = 0.5
        
        # 用于记录历史
        self.cut_history = []          # 每次迭代的HUBO值
        self.energy_history = []      # 每次迭代的能量值
        self.best_energy_history = [] # 每次迭代的最佳能量值
        self.runtime = 0              # 总运行时间
        self.x_history = []           # 变量演化历史
        self.p_history = []           # p参数历史

    def compute_energy(self, x):
        """计算HUBO能量函数"""
        energy = torch.tensor(0.0, device=self.device)
        
        # 遍历所有阶数的项
        for order, terms in self.terms_dict.items():
            for coef, vars_list in terms:
                # 计算该项的乘积
                prod = torch.tensor(1.0, device=self.device)
                for var_idx in vars_list:
                    prod *= x[var_idx]
                energy += coef * prod
        
        return energy

    def solve(self):
        best_energy = float('-inf')  # 初始化为负无穷（最大化）
        best_x = None
        self.cut_history = []
        self.energy_history = []
        self.best_energy_history = []
        self.x_history = []
        self.p_history = []
        
        start_time = time.time()
        
        # 多次运行取最佳解
        for run in range(5):
            # 重新初始化变量
            self.x = 0.01 * torch.randn(self.n_vars, device=self.device)
            self.y = 0.01 * torch.randn(self.n_vars, device=self.device)

            run_hubo=[]  # 本次运行的HUBO值历史
            run_energy = []          # 本次运行的能量值历史
            run_best_energy = float('-inf')   # 本次运行的最佳能量值（最大化）
            
            # 使用进度条
            for iter in tqdm(range(self.n_iter + 1), desc=f"Run {run+1}/5"):
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
                
                # 只记录第5次运行的变量演化历史
                if run == 4:
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
    
    def plot_energy_history(self, title="Energy Optimization", save_dir="two"):
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
        
        plt.title(f"{self.method} - {title}", fontsize=14)
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
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, f"energy_history_{self.method}.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved energy history plot to {filename}")
        #plt.show()
    
    def plot_bifurcation(self, save_dir="two"):
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

        # # 添加分叉点标记（通常在最后一次迭代位置）
        # plt.axvline(x=len(x_hist)-1, color='r', linestyle='--', alpha=0.7)
        # plt.text(len(x_hist)-1 + 2, 0.8, "Bifurcation Point", rotation=90, fontsize=10, color='r')

        plt.tight_layout()

        # 保存图像
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, f"bifurcation_{self.method}.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved bifurcation plot to {filename}")
        #plt.show()
    def plot_final_distribution(self, save_dir="two"):
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
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, f"distribution_{self.method}.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved distribution plot to {filename}")
        

def generate_hubo_terms(n_vars, max_order=8, density=0.1, coef_range=(-10, 10)):
    """
    生成随机HUBO项
    参数:
    n_vars: 变量个数
    max_order: 最高阶数 (1-8)
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
                coef = np.random.uniform(coef_range[0], coef_range[1])
                terms_dict[order].append((coef, vars))
    
    return dict(terms_dict)

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

def run_algorithm_comparison(n_datasets=100, n_vars=10, max_order=8, density=0.1):
    """
    运行四种算法的比较实验
    
    参数:
    n_datasets: 数据集数量
    n_vars: 变量个数
    max_order: 最高阶数
    density: 项密度
    """
    # 创建输出目录
    if not os.path.exists("two"):
        os.makedirs("two")
    
    # 存储结果
    resultstwo = []
    
    # 进度条
    pbar = tqdm(total=n_datasets, desc="Running HUBO algorithm comparison")
    
    for dataset_idx in range(n_datasets):
        # 生成随机HUBO项
        terms_dict = generate_hubo_terms(n_vars, max_order, density)

        # 创建每个数据集的输出目录
        dataset_dir = os.path.join("two_end/resultstwo", f"dataset_{dataset_idx}")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # 固定参数
        xi = 0.05
        dt = 0.1
        n_iter = 500

        # 运行四种算法
        for method in ['bSB', 'dSB', 'CAC', 'CFC']:
            # 创建求解器
            solver = HUBOSolver(
                terms_dict=terms_dict,
                n_vars=n_vars,
                xi=xi,
                dt=dt,
                n_iter=n_iter,
                method=method,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            # 求解并获取最终解和能量值
            solution, best_energy = solver.solve()

            # 记录结果
            resultstwo.append({
                'dataset': dataset_idx,
                'method': method,
                'energy': best_energy,
                'runtime': solver.runtime,
                'n_vars': n_vars,
                'max_order': max_order,
                'density': density
            })

            # 保存图像到每个数据集的文件夹
            solver.plot_energy_history(title=f"{method} Optimization - Dataset {dataset_idx}", save_dir=dataset_dir)
            solver.plot_bifurcation(save_dir=dataset_dir)
            solver.plot_final_distribution(save_dir=dataset_dir)

        pbar.update(1)
    
    pbar.close()
    
    # 转换为DataFrame
    resultstwo_df = pd.DataFrame(resultstwo)
    
    # # 保存结果
    # resultstwo_df.to_csv("resultstwo/hubo_algorithm_comparison_resultstwo.csv", index=False)
    # print("Saved algorithm comparison resultstwo to resultstwo/hubo_algorithm_comparison_resultstwo.csv")
    
    return resultstwo_df

def plot_algorithm_comparison(resultstwo_df):
    """绘制算法比较结果"""
    # 计算每种方法的平均能量值和运行时间
    summary = resultstwo_df.groupby('method').agg(
        avg_energy=pd.NamedAgg(column='energy', aggfunc='mean'),
        std_energy=pd.NamedAgg(column='energy', aggfunc='std'),
        min_energy=pd.NamedAgg(column='energy', aggfunc='min'),
        max_energy=pd.NamedAgg(column='energy', aggfunc='max'),
        avg_runtime=pd.NamedAgg(column='runtime', aggfunc='mean')
    ).reset_index()
    
    # 打印结果
    print("\nAlgorithm Performance Summary:")
    print(summary)
    
    # 绘制能量值比较
    plt.figure(figsize=(12, 8))
    
    # 箱线图比较能量值
    plt.subplot(2, 1, 1)
    plt.boxplot([resultstwo_df[resultstwo_df['method'] == m]['energy'] for m in ['bSB', 'dSB', 'CAC', 'CFC']])
    plt.xticks([1, 2, 3, 4], ['bSB', 'dSB', 'CAC', 'CFC'])
    plt.title("HUBO Energy Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Energy Value")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 运行时间比较
    plt.subplot(2, 1, 2)
    runtime_data = [resultstwo_df[resultstwo_df['method'] == m]['runtime'] for m in ['bSB', 'dSB', 'CAC', 'CFC']]
    plt.bar(['bSB', 'dSB', 'CAC', 'CFC'], [np.mean(d) for d in runtime_data], 
            yerr=[np.std(d) for d in runtime_data], capsize=5)
    plt.title("Runtime Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图像
    filename = "./two/result_two_end1/hubo_algorithm_comparison_summary.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved algorithm comparison summary to {filename}")
   # #plt.show()

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    # 运行算法比较实验
    n_datasets = 5
    n_vars = 10
    max_order = 8
    density = 0.15
    
    print(f"Running HUBO algorithm comparison on {n_datasets} datasets")
    print(f"Variables: {n_vars}, Max Order: {max_order}, Density: {density}")
    
    resultstwo_df = run_algorithm_comparison(n_datasets, n_vars, max_order, density)
    
    # 绘制算法比较结果
    print("\nPlotting algorithm comparison resultstwo...")
    plot_algorithm_comparison(resultstwo_df)
    
    # 打印最终结果
    print("\nFinal resultstwo Summary:")
    print(resultstwo_df.groupby('method')['energy'].describe())
    
   