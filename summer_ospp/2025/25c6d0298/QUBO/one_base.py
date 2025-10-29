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
import networkx as nx
warnings.filterwarnings("ignore")

class QUBOSolver:
    def __init__(self, J_matrix, n_vars, 
                 xi=0.3, dt=0.15, n_iter=1000, method='dSB', device='cpu'):
        self.J = J_matrix
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
        self.cut_history = []         # 每次迭代的切割值
        self.energy_history = []      # 每次迭代的能量值
        self.best_cut_history = []    # 每次迭代的最佳切割值
        self.runtime = 0              # 总运行时间
        self.x_history = []           # 变量演化历史
        self.p_history = []           # p参数历史

    def compute_qubo_value(self, x):
        """计算QUBO值: y1 = Σ J_ij x_i x_j"""
        # 使用矩阵乘法高效计算
        x_matrix = x.unsqueeze(0)  # 转换为行向量
        return torch.mm(torch.mm(x_matrix, self.J), x_matrix.T).squeeze()

    def compute_energy(self, x):
        """计算能量函数：QUBO值"""
        return self.compute_qubo_value(x)

    def solve(self):
        best_qubo = float('-inf')  # 初始化为负无穷（最大化）
        best_x = None
        self.cut_history = []
        self.energy_history = []
        self.best_cut_history = []
        self.x_history = []
        self.p_history = []
        
        start_time = time.time()
        
        # 多次运行取最佳解
        for run in range(5):
            # 重新初始化变量
            self.x = 0.01 * torch.randn(self.n_vars, device=self.device)
            self.y = 0.01 * torch.randn(self.n_vars, device=self.device)
            
            run_qubo = []            # 本次运行的QUBO值历史
            run_energy = []          # 本次运行的能量值历史
            run_best_qubo = float('-inf')   # 本次运行的最佳QUBO值（最大化）
            
            # 使用进度条
            for iter in tqdm(range(self.n_iter + 1), desc=f"Run {run+1}"):
                p_t = iter / self.n_iter  # 时间参数
                self.p_history.append(p_t)  # 记录p值历史
                
                # 创建需要梯度的变量
                x_torch = self.x.clone().detach().requires_grad_(True)
                
                # 计算QUBO值
                if self.method in ['dSB', 'CAC', 'CFC']:
                    # 使用符号函数的直通估计器
                    with torch.no_grad():
                        sign_x = torch.sign(x_torch)
                    x_ste = sign_x - x_torch.detach() + x_torch
                    qubo_value = self.compute_qubo_value(x_ste)
                else:  # bSB
                    # 直接使用连续值
                    qubo_value = self.compute_qubo_value(x_torch)
                
                # 计算能量（最大化能量）
                energy = self.compute_energy(x_torch)
                
                # 记录当前值
                current_qubo = qubo_value.item()
                run_qubo.append(current_qubo)
                run_energy.append(energy.item())
                
                # 更新本次运行的最佳QUBO值（最大化）
                if current_qubo > run_best_qubo:
                    run_best_qubo = current_qubo
                
                # 计算梯度
                energy.backward()
                grad_E = x_torch.grad.clone()
                x_torch.grad.zero_()
                
                # 记录变量演化历史（仅最后一次运行）
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
            self.cut_history.append(run_qubo)
            self.energy_history.append(run_energy)
            self.best_cut_history.append(run_best_qubo)
            
            # 记录当前运行的最佳解（最大化）
            with torch.no_grad():
                current_x = torch.sign(self.x)
                current_qubo = self.compute_qubo_value(current_x).item()
                
                if current_qubo > best_qubo:
                    best_qubo = current_qubo
                    best_x = current_x
        
        self.runtime = time.time() - start_time
        return best_x, best_qubo
    
    def plot_energy_history(self, title="Energy Optimization", save_dir="resultsone"):
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
        
        plt.title(f"{self.method} - Energy Optimization", fontsize=14)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Energy", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # 添加统计信息
        stats_text = [
            f"Method: {self.method}",
            f"Best QUBO Value: {max(self.best_cut_history):.4f}",
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
       # plt.show()
    
    def plot_bifurcation(self, save_dir="resultsone"):
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
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, f"bifurcation_{self.method}.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved bifurcation plot to {filename}")
        #plt.show()
    
    def plot_final_distribution(self, save_dir="resultsone"):
        """绘制QUBO值分布直方图"""
        if not self.cut_history:
            print("No QUBO history recorded for final distribution plot.")
            return

        # 展开所有run的所有iteration的QUBO值
        all_qubo_values = np.array([qubo for run in self.cut_history for qubo in run])

        plt.figure(figsize=(10, 6))
        plt.hist(all_qubo_values, bins=20, alpha=0.7, color='blue', edgecolor='black')

        plt.title(f"{self.method} - QUBO Value Distribution (All Iterations)", fontsize=14)
        plt.xlabel("QUBO Value (All Iterations)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 添加统计信息
        stats_text = [
            f"Mean: {np.mean(all_qubo_values):.4f}",
            f"Std Dev: {np.std(all_qubo_values):.4f}",
            f"Min: {np.min(all_qubo_values):.4f}",
            f"Max: {np.max(all_qubo_values):.4f}"
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
       # plt.show()

def generate_qubo_matrix(n_vars, density=0.3):
    """生成随机QUBO矩阵J"""
    J = torch.zeros((n_vars, n_vars))
    
    # 随机生成非对角元素
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if random.random() < density:
                # 随机权重，范围在[-10, 10]
                weight = random.uniform(-10, 10)
                J[i, j] = weight
                J[j, i] = weight  # 保持对称
    
    return J

def run_algorithm_comparison(n_datasets=100, n_vars=10, density=0.3):
    """运行四种算法的比较实验"""
    # 创建输出目录/
    if not os.path.exists("./one/resultsone_end1"):
        os.makedirs("./one/resultsone_end1")
    
    # 存储结果
    resultsone = defaultdict(list)
    
    # 进度条
    pbar = tqdm(total=n_datasets, desc="Running algorithm comparison")
    
    for dataset_idx in range(n_datasets):
        # 生成随机QUBO矩阵
        J_matrix = generate_qubo_matrix(n_vars, density)

        # 固定参数
        xi = 0.05
        dt = 0.1
        n_iter = 1000

        # 为每个数据集创建单独的文件夹
        dataset_dir = os.path.join("./one/resultsone_end1", f"dataset_{dataset_idx}")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # 运行四种算法
        for method in ['bSB', 'dSB', 'CAC', 'CFC']:
            # 创建求解器
            solver = QUBOSolver(
                J_matrix=J_matrix,
                n_vars=n_vars,
                xi=xi,
                dt=dt,
                n_iter=n_iter,
                method=method,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            # 求解并获取最终解和QUBO值
            solution, best_qubo = solver.solve()

            # 记录结果
            resultsone['dataset'].append(dataset_idx)
            resultsone['method'].append(method)
            resultsone['qubo_value'].append(best_qubo)
            resultsone['runtime'].append(solver.runtime)

            # 保存图像（每个数据集都保存）
            solver.plot_energy_history(title=f"{method} Optimization - Dataset {dataset_idx}", save_dir=dataset_dir)
            solver.plot_bifurcation(save_dir=dataset_dir)
            solver.plot_final_distribution(save_dir=dataset_dir)

        pbar.update(1)
    
    pbar.close()
    
    # 转换为DataFrame
    resultsone_df = pd.DataFrame(resultsone)
    
    return resultsone_df

def plot_algorithm_comparison(resultsone_df):
    """绘制算法比较结果"""
    # 计算每种方法的平均QUBO值和运行时间
    summary = resultsone_df.groupby('method').agg(
        avg_qubo=pd.NamedAgg(column='qubo_value', aggfunc='mean'),
        std_qubo=pd.NamedAgg(column='qubo_value', aggfunc='std'),
        min_qubo=pd.NamedAgg(column='qubo_value', aggfunc='min'),
        max_qubo=pd.NamedAgg(column='qubo_value', aggfunc='max'),
        avg_runtime=pd.NamedAgg(column='runtime', aggfunc='mean')
    ).reset_index()
    
    # 打印结果
    print("\nAlgorithm Performance Summary:")
    print(summary)
    
    # 绘制QUBO值比较
    plt.figure(figsize=(12, 6))
    
    # 箱线图比较QUBO值
    plt.subplot(1, 2, 1)
    plt.boxplot([resultsone_df[resultsone_df['method'] == m]['qubo_value'] for m in ['bSB', 'dSB', 'CAC', 'CFC']])
    plt.xticks([1, 2, 3, 4], ['bSB', 'dSB', 'CAC', 'CFC'])
    plt.title("QUBO Value Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("QUBO Value")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 运行时间比较
    plt.subplot(1, 2, 2)
    runtime_data = [resultsone_df[resultsone_df['method'] == m]['runtime'] for m in ['bSB', 'dSB', 'CAC', 'CFC']]
    plt.bar(['bSB', 'dSB', 'CAC', 'CFC'], [np.mean(d) for d in runtime_data], 
            yerr=[np.std(d) for d in runtime_data], capsize=5)
    plt.title("Runtime Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图像
    filename = "./one/resultsone_end1/algorithm_comparison_summary.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved algorithm comparison summary to {filename}")
    #plt.show()

if __name__ == "__main__":
    # 运行算法比较实验
    n_datasets = 5
    n_vars = 20
    density = 0.3
    
    print(f"Running algorithm comparison on {n_datasets} datasets with {n_vars} variables...")
    resultsone_df = run_algorithm_comparison(n_datasets, n_vars, density)
    
    # 绘制算法比较结果
    print("\nPlotting algorithm comparison resultsone...")
    plot_algorithm_comparison(resultsone_df)