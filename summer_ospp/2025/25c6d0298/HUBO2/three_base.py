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
import json
from json import JSONEncoder
warnings.filterwarnings("ignore")

class NumpyEncoder(JSONEncoder):
    """自定义JSON编码器，用于处理numpy数据类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class HUBOSolver:
    def __init__(self, terms_dict, n_vars, 
                 xi=0.3, dt=0.15, n_iter=1000, method='dSB', device='cpu'):
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
        self.cut_history = []  # 每次迭代的cut值
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
        self.cut_history = []  # 重置cut历史
        self.energy_history = []
        self.best_energy_history = []
        self.x_history = []
        self.p_history = []
        
        start_time = time.time()
        
        # 多次运行取最佳解
        for run in range(5):  # 固定为5次运行
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
    
    def plot_energy_history(self, dataset_idx, method, title="Energy Optimization"):
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
   
        plt.title(f"{title}\nDataset {dataset_idx} - {method}", fontsize=14)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Energy", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # 添加统计信息
        stats_text = [
            f"Method: {method}",
            f"Dataset: {dataset_idx}",
            f"Best Energy: {max(self.best_energy_history):.4f}",
            f"Avg Final Energy: {np.mean([run[-1] for run in self.energy_history]):.4f}",
            f"Runtime: {self.runtime:.2f} seconds"
        ]
        
        plt.figtext(0.70, 0.20, "\n".join(stats_text), fontsize=12, 
                   ha="left", bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        
        # 保存图像
        if not os.path.exists(f"./three/results_three_end1/dataset_{dataset_idx}"):
            os.makedirs(f"./three/results_three_end1/dataset_{dataset_idx}")
        filename = f"./three/results_three_end1/dataset_{dataset_idx}/energy_history_{method}.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved energy history plot to {filename}")
        plt.close()
    
    def plot_bifurcation(self, dataset_idx, method):
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

        plt.title(f"Dataset {dataset_idx} - {method} - Variable Bifurcation", fontsize=14)
        plt.xlabel("Total Iteration", fontsize=12)
        plt.ylabel("Variable Value", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        
        # 保存图像
        if not os.path.exists(f"./three/results_three_end1/dataset_{dataset_idx}"):
            os.makedirs(f"./three/results_three_end1/dataset_{dataset_idx}")
        filename = f"./three/results_three_end1/dataset_{dataset_idx}/bifurcation_{method}.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved bifurcation plot to {filename}")
        plt.close()
        
    def plot_final_distribution(self, dataset_idx, method):
        """绘制HUBO值分布直方图"""
        if not self.cut_history:
            print("No HUBO history recorded for final distribution plot.")
            return

        # 展开所有run的所有iteration的HUBO值
        all_HUBO_values = np.array([hubo for run in self.cut_history for hubo in run])

        plt.figure(figsize=(10, 6))
        plt.hist(all_HUBO_values, bins=20, alpha=0.7, color='blue', edgecolor='black')

        plt.title(f"Dataset {dataset_idx} - {method} - HUBO Value Distribution (All Iterations)", fontsize=14)
        plt.xlabel("HUBO Value (All Iterations)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 添加统计信息
        stats_text = [
            f"Dataset: {dataset_idx}",
            f"Method: {method}",
            f"Mean: {np.mean(all_HUBO_values):.4f}",
            f"Std Dev: {np.std(all_HUBO_values):.4f}",
            f"Min: {np.min(all_HUBO_values):.4f}",
            f"Max: {np.max(all_HUBO_values):.4f}"
        ]

        plt.figtext(0.75, 0.75, "\n".join(stats_text), fontsize=12, 
                   ha="left", bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

        plt.tight_layout()

        # 保存图像
        if not os.path.exists(f"./three/result_three_end1/dataset_{dataset_idx}"):
            os.makedirs(f"./three/result_three_end1/dataset_{dataset_idx}")
        filename = f"./three/result_three_end1/dataset_{dataset_idx}/distribution_{method}.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved distribution plot to {filename}")
        plt.close()

def convert_to_python_types(data):
    """将数据转换为Python内置类型，以便JSON序列化"""
    if isinstance(data, dict):
        return {k: convert_to_python_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_python_types(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_python_types(item) for item in data)
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

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
                coef = np.random.uniform(coef_range[0], coef_range[1])
                # 转换为Python内置类型
                coef = float(coef)
                vars = tuple(int(v) for v in vars)
                terms_dict[order].append((coef, vars))
    
    return dict(terms_dict)

def square_hubo_terms(terms_dict):
    """
    将3阶HUBO项平方，得到新的HUBO项（最高6阶）
    参数:terms_dict: 原始3阶HUBO项字典
    返回:squared_terms_dict: 平方后的HUBO项字典（最高6阶）
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

def run_algorithm_comparison(n_datasets=5, n_vars=8, density=0.15):
    """
    运行四种算法的比较实验
    
    参数:
    n_datasets: 数据集数量
    n_vars: 变量个数
    density: 项密度
    """
    # 创建输出目录
    if not os.path.exists("./three/result_three_end1"):
        os.makedirs("./three/result_three_end1")
    
    # 存储结果
    results = []
    
    # 进度条
    pbar = tqdm(total=n_datasets, desc="Running y3 HUBO algorithm comparison")
    
    for dataset_idx in range(n_datasets):
        # 为每个数据集创建子目录
        dataset_dir = f"./three/result_three_end1/dataset_{dataset_idx}"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        # 生成随机3阶HUBO项
        original_terms = generate_hubo_terms(n_vars, max_order=3, density=density)
        
        # 将HUBO项平方
        squared_terms = square_hubo_terms(original_terms)
        
        # 保存HUBO问题
        with open(os.path.join(dataset_dir, 'hubo_problem.json'), 'w') as f:
            json.dump({
                'original_terms': convert_to_python_types(original_terms),
                'squared_terms': convert_to_python_types(squared_terms)
            }, f, indent=4, cls=NumpyEncoder)
        
        # 固定参数
        xi = 0.05
        dt = 0.1
        n_iter = 1000
        
        # 运行四种算法
        for method in ['bSB', 'dSB', 'CAC', 'CFC']:
            # 创建求解器
            solver = HUBOSolver(
                terms_dict=squared_terms,
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
            results.append({
                'dataset': dataset_idx,
                'method': method,
                'energy': float(best_energy),  # 转换为Python float
                'runtime': float(solver.runtime),  # 转换为Python float
                'n_vars': n_vars,
                'density': density
            })
            
            # 获取原始和平方后的表达式
            original_expr = hubo_terms_to_string(original_terms)
            squared_expr = hubo_terms_to_string(squared_terms)
            
            # 创建标题
            title = f"{method} Optimization - y3 = ({original_expr})²"
            
            # 绘制能量历史
            solver.plot_energy_history(dataset_idx, method, title=title)
            
            # 绘制分叉图
            solver.plot_bifurcation(dataset_idx, method)

            # 绘制HUBO值分布直方图
            solver.plot_final_distribution(dataset_idx, method)
        
        pbar.update(1)
    
    pbar.close()
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 绘制算法比较结果
    plot_algorithm_comparison(results_df)
    
    return results_df

def plot_algorithm_comparison(results_df):
    """绘制算法比较结果"""
    # 计算每种方法的平均能量值和运行时间
    summary = results_df.groupby('method').agg(
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
    plt.boxplot([results_df[results_df['method'] == m]['energy'] for m in ['bSB', 'dSB', 'CAC', 'CFC']])
    plt.xticks([1, 2, 3, 4], ['bSB', 'dSB', 'CAC', 'CFC'])
    plt.title("y3 HUBO Energy Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Energy Value")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 运行时间比较
    plt.subplot(2, 1, 2)
    runtime_data = [results_df[results_df['method'] == m]['runtime'] for m in ['bSB', 'dSB', 'CAC', 'CFC']]
    plt.bar(['bSB', 'dSB', 'CAC', 'CFC'], [np.mean(d) for d in runtime_data], 
            yerr=[np.std(d) for d in runtime_data], capsize=5)
    plt.title("Runtime Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图像
    filename = "./three/results_three_end1/algorithm_comparison_summary.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved algorithm comparison summary to {filename}")
    plt.show()

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    # 运行算法比较实验
    n_datasets = 5
    n_vars = 8
    density = 0.15
    
    print(f"Running y3 HUBO algorithm comparison on {n_datasets} datasets")
    print(f"Variables: {n_vars}, Density: {density}")
    
    results_df = run_algorithm_comparison(n_datasets, n_vars, density)
    
    # 打印最终结果
    print("\nFinal Results Summary:")
    print(results_df.groupby('method')['energy'].describe())
    
    # 保存结果到CSV文件
    results_df.to_csv("./three/result_three_end1/all_results.csv", index=False)
    print("Results saved to ./three/result_three_end1/all_results.csv")