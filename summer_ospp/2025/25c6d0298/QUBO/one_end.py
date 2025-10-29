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
import networkx as nx
warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use('Agg')
import json
from json import JSONEncoder
from skopt import gp_minimize
from scipy.optimize import dual_annealing, minimize
import cma
from functools import partial
from hyperopt import fmin, tpe, hp, Trials
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import math

# ===== 量子遗传算法实现 =====
class QuantumGeneticAlgorithm:
    def __init__(self, obj_func, bounds, n_calls, pop_size=10, n_qubits=8):
        self.obj_func = obj_func
        self.bounds = bounds
        self.n_calls = n_calls
        self.pop_size = pop_size
        self.n_qubits = n_qubits
        self.n_params = len(bounds)
        self.best_fitness = float('inf')
        self.best_solution = None
        self.fitness_history = []
        
        # 初始化量子种群
        self.quantum_pop = np.ones((pop_size, self.n_params, n_qubits, 2)) * (1/np.sqrt(2))
        
    def measure(self, quantum_pop):
        """测量量子个体，得到经典个体"""
        pop_size, n_params, n_qubits, _ = quantum_pop.shape
        classical_pop = np.zeros((pop_size, n_params))
        
        for i in range(pop_size):
            for j in range(n_params):
                # 对每个量子比特进行测量
                bits = []
                for k in range(n_qubits):
                    alpha = quantum_pop[i, j, k, 0]
                    prob_0 = alpha**2
                    if random.random() < prob_0:
                        bits.append(0)
                    else:
                        bits.append(1)
                
                # 将二进制转换为十进制
                decimal = 0
                for idx, bit in enumerate(bits):
                    decimal += bit * (2 ** (n_qubits - idx - 1))
                
                # 映射到参数空间
                min_val, max_val = self.bounds[j]
                classical_pop[i, j] = min_val + decimal * (max_val - min_val) / (2**n_qubits - 1)
        
        return classical_pop
    
    def evaluate(self, classical_pop):
        """评估经典种群"""
        fitness = []
        for i in range(classical_pop.shape[0]):
            fitness.append(self.obj_func(classical_pop[i]))
        return np.array(fitness)
    
    def update_quantum_pop(self, quantum_pop, classical_pop, fitness, rotation_angle=0.01*np.pi):
        """更新量子种群"""
        pop_size, n_params, n_qubits, _ = quantum_pop.shape
        best_idx = np.argmin(fitness)
        
        for i in range(pop_size):
            for j in range(n_params):
                for k in range(n_qubits):
                    # 计算旋转角度和方向
                    alpha, beta = quantum_pop[i, j, k]
                    best_alpha, _ = quantum_pop[best_idx, j, k]
                    
                    # 确定旋转方向
                    if alpha * beta > 0:
                        delta_theta = rotation_angle
                    else:
                        delta_theta = -rotation_angle
                    
                    # 应用量子旋转门
                    new_alpha = alpha * np.cos(delta_theta) - beta * np.sin(delta_theta)
                    new_beta = alpha * np.sin(delta_theta) + beta * np.cos(delta_theta)
                    
                    # 归一化
                    norm = np.sqrt(new_alpha**2 + new_beta**2)
                    quantum_pop[i, j, k, 0] = new_alpha / norm
                    quantum_pop[i, j, k, 1] = new_beta / norm
        
        return quantum_pop
    
    def mutate(self, quantum_pop, mutation_rate=0.01):
        """量子变异操作"""
        pop_size, n_params, n_qubits, _ = quantum_pop.shape
        
        for i in range(pop_size):
            for j in range(n_params):
                for k in range(n_qubits):
                    if random.random() < mutation_rate:
                        # 交换α和β
                        quantum_pop[i, j, k, 0], quantum_pop[i, j, k, 1] = \
                            quantum_pop[i, j, k, 1], quantum_pop[i, j, k, 0]
        
        return quantum_pop
    
    def optimize(self):
        """执行优化"""
        n_iterations = max(1, self.n_calls // self.pop_size)
        
        for iteration in range(n_iterations):
            # 测量量子种群得到经典种群
            classical_pop = self.measure(self.quantum_pop)
            
            # 评估经典种群
            fitness = self.evaluate(classical_pop)
            
            # 更新最佳解
            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]
            
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = classical_pop[current_best_idx]
            
            self.fitness_history.append(self.best_fitness)
            
            # 更新量子种群
            self.quantum_pop = self.update_quantum_pop(self.quantum_pop, classical_pop, fitness)
            
            # 量子变异
            self.quantum_pop = self.mutate(self.quantum_pop)
        
        return self.best_solution, self.best_fitness

class QUBOSolverBatch:
    def __init__(self, J_batch, n_vars, batch_size,
                 xi=0.3, dt=0.15, n_iter=1000, method='dSB', device='cpu'):
        """
        J_batch: [batch_size, n_vars, n_vars]，每个batch一个QUBO矩阵
        """
        self.J_batch = J_batch.to(device)
        self.n_vars = n_vars
        self.batch_size = batch_size
        self.xi = xi
        self.dt = dt
        self.n_iter = n_iter
        self.method = method
        self.device = device
        self.Delta = 0.5
        # 初始化变量 shape: [batch_size, n_vars]
        self.x = 0.01 * torch.randn(batch_size, n_vars, device=device)
        self.y = 0.01 * torch.randn(batch_size, n_vars, device=device)
        # 历史
        self.cut_history = []
        self.energy_history = []
        self.best_cut_history = []
        self.runtime = 0
        self.x_history = []
        self.p_history = []

    def compute_qubo_value_batch(self, x):
        """
        x: [batch_size, n_vars]
        J_batch: [batch_size, n_vars, n_vars]
        return: [batch_size]
        """
        x = x.unsqueeze(1)  # [batch, 1, n_vars]
        xT = x.transpose(1,2)  # [batch, n_vars, 1]
        # x @ J @ x^T for each batch
        return torch.bmm(torch.bmm(x, self.J_batch), xT).squeeze()

    def compute_energy_batch(self, x):
        return self.compute_qubo_value_batch(x)

    def solve(self):
        best_qubo = torch.full((self.batch_size,), float('-inf'), device=self.device)
        best_x = torch.zeros_like(self.x)
        self.cut_history = []
        self.energy_history = []
        self.best_cut_history = []
        self.x_history = []
        self.p_history = []
        start_time = time.time()
        for run in range(1):
            self.x = 0.01 * torch.randn(self.batch_size, self.n_vars, device=self.device)
            self.y = 0.01 * torch.randn(self.batch_size, self.n_vars, device=self.device)
            run_qubo = []
            run_energy = []
            run_best_qubo = torch.full((self.batch_size,), float('-inf'), device=self.device)
            for iter in tqdm(range(self.n_iter + 1), desc=f"Batch Run {run+1}"):
                p_t = iter / self.n_iter
                self.p_history.append(p_t)
                x_torch = self.x.clone().detach().requires_grad_(True)
                if self.method in ['dSB', 'CAC', 'CFC']:
                    with torch.no_grad():
                        sign_x = torch.sign(x_torch)
                    x_ste = sign_x - x_torch.detach() + x_torch
                    qubo_value = self.compute_qubo_value_batch(x_ste)
                else:
                    qubo_value = self.compute_qubo_value_batch(x_torch)
                energy = self.compute_energy_batch(x_torch)
                run_qubo.append(qubo_value.detach().cpu().numpy())
                run_energy.append(energy.detach().cpu().numpy())
                # 更新本次运行的最佳QUBO值
                run_best_qubo = torch.max(run_best_qubo, qubo_value)
                # 计算梯度
                energy.sum().backward()
                grad_E = x_torch.grad.clone()
                x_torch.grad.zero_()
                dx = self.y
                if self.method == 'bSB':
                    dy = -(self.Delta - p_t) * self.x + self.xi * grad_E
                elif self.method == 'dSB':
                    with torch.no_grad():
                        sign_x = torch.sign(self.x)
                    dy = -(self.Delta - p_t) * sign_x + self.xi * grad_E
                elif self.method == 'CAC':
                    with torch.no_grad():
                        sign_x = torch.sign(self.x)
                    gamma = 0.1
                    dy = -gamma * self.y - (self.Delta - p_t) * sign_x + self.xi * grad_E
                elif self.method == 'CFC':
                    with torch.no_grad():
                        sign_x = torch.sign(self.x)
                        global_avg = torch.mean(self.x, dim=1, keepdim=True)
                    gamma = 0.1
                    beta = 0.05
                    dy = -gamma * self.y - (self.Delta - p_t) * sign_x + self.xi * grad_E + beta * (self.x - global_avg)
                new_x = self.x + dx * self.dt
                new_y = self.y + dy * self.dt
                over_bound = torch.abs(new_x) >= 1.0
                new_x[over_bound] = torch.sign(new_x[over_bound])
                new_y[over_bound] = 0.0
                self.x = new_x.detach()
                self.y = new_y.detach()
            self.cut_history.append(run_qubo)
            self.energy_history.append(run_energy)
            self.best_cut_history.append(run_best_qubo.detach().cpu().numpy())
            with torch.no_grad():
                current_x = torch.sign(self.x)
                current_qubo = self.compute_qubo_value_batch(current_x)
                mask = current_qubo > best_qubo
                best_qubo[mask] = current_qubo[mask]
                best_x[mask] = current_x[mask]
        self.runtime = time.time() - start_time
        return best_x, best_qubo

# ===== 封装优化器函数 =====
class HyperparameterOptimizer:
    def __init__(self, n_vars, n_iter, method_name, density, J_batch, n_calls=10):
        self.n_vars = n_vars
        self.n_iter = n_iter
        self.method_name = method_name
        self.density = density
        self.J_batch = J_batch
        self.n_calls = n_calls
        self.batch_size = J_batch.shape[0] if J_batch is not None else 1
        
    def objective(self, params):
        xi, dt = params
        try:
            if self.J_batch is not None:
                solver = QUBOSolverBatch(self.J_batch, self.n_vars, self.batch_size, 
                                       xi=xi, dt=dt, n_iter=self.n_iter, 
                                       method=self.method_name, device='cpu')
                _, best_qubo = solver.solve()
                # 返回所有batch的平均最优能量
                return -float(torch.mean(best_qubo).item())
            else:
                raise ValueError('J_batch must be provided for batch optimization')
        except Exception as e:
            print(f"Error during solver execution: {e}")
            return 1000
    
    def hyperopt_objective(self, params):
        xi = params['xi']
        dt = params['dt']
        return self.objective((xi, dt))
    
    def optimize_bayes(self, space):
        """贝叶斯优化"""
        res = gp_minimize(self.objective, space, n_calls=self.n_calls, random_state=42)
        best_xi, best_dt = res.x
        best_score = -res.fun
        return best_xi, best_dt, best_score, res
    
    def optimize_sa(self, bounds):
        """模拟退火"""
        res = dual_annealing(self.objective, bounds, maxiter=self.n_calls//2, seed=42)
        best_xi, best_dt = res.x
        best_score = -res.fun
        return best_xi, best_dt, best_score, res
    
    def optimize_cma_es(self, x0):
        """CMA-ES优化"""
        try:
            es = cma.CMAEvolutionStrategy(x0, 0.1, {
                'bounds': [[0.01, 0.01], [0.5, 0.3]],
                'maxfevals': self.n_calls,
                'verbose': -1
            })
            es.optimize(self.objective)
            best_xi, best_dt = es.result.xbest
            best_score = -es.result.fbest
            return best_xi, best_dt, best_score, es
        except Exception as e:
            print(f"CMA-ES bounds error: {e}")
            es = cma.CMAEvolutionStrategy(x0, 0.1, {
                'maxfevals': self.n_calls,
                'verbose': -1
            })
            es.optimize(self.objective)
            best_xi, best_dt = es.result.xbest
            best_score = -es.result.fbest
            return best_xi, best_dt, best_score, es
    
    def optimize_nelder_mead(self, x0, bounds):
        """Nelder-Mead优化"""
        res = minimize(self.objective, x0, method='Nelder-Mead', bounds=bounds, 
                      options={'maxfev': self.n_calls})
        best_xi, best_dt = res.x
        best_score = -res.fun
        return best_xi, best_dt, best_score, res
    
    def optimize_hyperopt(self):
        """Hyperopt优化"""
        hyperopt_space = {
            'xi': hp.uniform('xi', 0.01, 0.5),
            'dt': hp.uniform('dt', 0.01, 0.3)
        }
        trials = Trials()
        best_hyperopt = fmin(
            fn=self.hyperopt_objective,
            space=hyperopt_space,
            algo=tpe.suggest,
            max_evals=self.n_calls,
            trials=trials,
            rstate=np.random.default_rng(42)
        )
        best_trial = trials.best_trial
        best_xi = best_trial['misc']['vals']['xi'][0]
        best_dt = best_trial['misc']['vals']['dt'][0]
        best_score = -best_trial['result']['loss']
        return best_xi, best_dt, best_score, trials
    
    def optimize_qga(self, bounds):
        """量子遗传算法优化"""
        qga = QuantumGeneticAlgorithm(self.objective, bounds, self.n_calls, pop_size=10, n_qubits=8)
        best_params_qga, best_score_qga = qga.optimize()
        best_xi, best_dt = best_params_qga
        best_score = -best_score_qga
        return best_xi, best_dt, best_score, qga

# ===== 实验管理器 =====
class QUBOExperimentManager:
    def __init__(self, result_dir='./QUBO/results'):
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)
        
    def generate_qubo_batch(self, batch_size, n_vars, density):
        """生成QUBO矩阵批次"""
        J_list = []
        for _ in range(batch_size):
            J = np.random.randn(n_vars, n_vars)
            mask = np.random.rand(n_vars, n_vars) < density
            J = J * mask
            J = np.triu(J)
            J = J + J.T - np.diag(np.diag(J))
            J_list.append(J)
        J_batch = np.stack(J_list, axis=0)
        return torch.tensor(J_batch, dtype=torch.float32)
    
    def convert_to_python_types(self, data):
        """将数据转换为Python原生类型"""
        if isinstance(data, dict):
            return {k: self.convert_to_python_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_to_python_types(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self.convert_to_python_types(item) for item in data)
        elif isinstance(data, np.generic):
            return data.item()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    
    def run_single_experiment(self, n_iter, method, opt_method, datasets, n_calls=10):
        """运行单个实验配置"""
        print(f"运行 {method}-{opt_method} (迭代={n_iter})...")
        
        best_scores = []
        best_params = []
        total_times = []
        last_10_avg_scores = []  # 最后10次平均分数
        
        # 在多个数据集上运行以获得统计结果
        for dataset_idx, J_batch in enumerate(datasets):
            try:
                # 创建优化器
                optimizer = HyperparameterOptimizer(
                    n_vars=J_batch.shape[1], 
                    n_iter=n_iter, 
                    method_name=method, 
                    density=0.1, 
                    J_batch=J_batch,
                    n_calls=n_calls
                )
                
                # 定义搜索空间
                space = [(0.01, 0.5), (0.01, 0.3)]
                bounds = [(0.01, 0.5), (0.01, 0.3)]
                x0 = [0.1, 0.1]
                
                # 记录端到端开始时间
                start_time = time.time()
                
                # 执行优化
                if opt_method == 'Bayes':
                    best_xi, best_dt, best_score, _ = optimizer.optimize_bayes(space)
                elif opt_method == 'SA':
                    best_xi, best_dt, best_score, _ = optimizer.optimize_sa(bounds)
                elif opt_method == 'CMA-ES':
                    best_xi, best_dt, best_score, _ = optimizer.optimize_cma_es(x0)
                elif opt_method == 'Nelder-Mead':
                    best_xi, best_dt, best_score, _ = optimizer.optimize_nelder_mead(x0, bounds)
                elif opt_method == 'Hyperopt':
                    best_xi, best_dt, best_score, _ = optimizer.optimize_hyperopt()
                elif opt_method == 'QGA':
                    best_xi, best_dt, best_score, _ = optimizer.optimize_qga(bounds)
                
                # 记录端到端时间
                total_time = time.time() - start_time
                
                # 使用最佳参数重新运行求解器以获取最后10次迭代的平均
                solver = QUBOSolverBatch(J_batch, J_batch.shape[1], J_batch.shape[0],
                                       xi=best_xi, dt=best_dt, n_iter=n_iter, 
                                       method=method, device='cpu')
                _, best_qubo = solver.solve()
                
                # 计算最后10次迭代的平均（如果迭代次数足够）
                if len(solver.energy_history) > 0 and len(solver.energy_history[0]) >= 10:
                    last_10_avg = np.mean(solver.energy_history[0][-10:])
                else:
                    last_10_avg = float(torch.mean(best_qubo).item())
                
                best_scores.append(best_score)
                best_params.append((best_xi, best_dt))
                total_times.append(total_time)
                last_10_avg_scores.append(last_10_avg)
                
            except Exception as e:
                print(f"数据集 {dataset_idx+1} 上 {opt_method} 失败: {e}")
                best_scores.append(0)
                best_params.append((0.1, 0.1))
                total_times.append(float('inf'))
                last_10_avg_scores.append(0)
        
        # 返回最佳结果（而不是平均结果）
        if best_scores:
            best_idx = np.argmax(best_scores)  # 找到最佳分数的索引
            return {
                'best_score': best_scores[best_idx],
                'best_params': best_params[best_idx],
                'total_time': total_times[best_idx],
                'last_10_avg': last_10_avg_scores[best_idx],
                'all_scores': best_scores,
                'all_times': total_times
            }
        else:
            return {
                'best_score': 0,
                'best_params': (0.1, 0.1),
                'total_time': float('inf'),
                'last_10_avg': 0,
                'all_scores': [],
                'all_times': []
            }
    
    def run_comprehensive_experiment(self, methods, opt_methods, n_iter_values, 
                                   n_vars=10, density=0.1, n_calls=10, 
                                   batch_size=4, n_datasets=50):
        """运行综合实验"""
        print("=== 超参数比较实验 ===")
        print(f"算法: {methods}")
        print(f"优化方法: {opt_methods}")
        print(f"迭代次数: {n_iter_values}")
        print(f"数据集数量: {n_datasets}")
        print(f"变量数量: {n_vars}")
        
        all_results = {}
        performance_matrix = {}
        
        # 为每个迭代次数运行实验
        for n_iter in n_iter_values:
            print(f"\n{'='*60}")
            print(f"运行迭代次数: {n_iter}")
            print(f"{'='*60}")
            
            iteration_results = {}
            performance_matrix[n_iter] = {}
            
            # 生成多个数据集
            datasets = []
            for dataset_idx in range(n_datasets):
                J_batch = self.generate_qubo_batch(batch_size, n_vars, density)
                datasets.append(J_batch)
            
            for method in methods:
                print(f"\n--- 算法: {method} ---")
                method_results = {
                    'params': {}, 
                    'scores': {}, 
                    'times': {}, 
                    'last_10_avg': {},
                    'energy_histories': {}
                }
                method_performance = {}
                
                # 为每个优化方法运行
                for opt_method in opt_methods:
                    result = self.run_single_experiment(n_iter, method, opt_method, datasets, n_calls)
                    
                    method_results['params'][opt_method] = result['best_params']
                    method_results['scores'][opt_method] = result['best_score']
                    method_results['times'][opt_method] = result['total_time']
                    method_results['last_10_avg'][opt_method] = result['last_10_avg']
                    
                    method_performance[opt_method] = {
                        'score': result['best_score'], 
                        'time': result['total_time'],
                        'last_10_avg': result['last_10_avg']
                    }
                    
                    print(f"{opt_method}: 最佳分数={result['best_score']:.4f}, "
                          f"端到端时间={result['total_time']:.2f}s, "
                          f"最后10次平均={result['last_10_avg']:.4f}")
                
                # 记录能量历史（使用第一个数据集和最佳参数）
                print("记录能量历史...")
                for opt_name in opt_methods:
                    if opt_name in method_results['params']:
                        xi, dt = method_results['params'][opt_name]
                        try:
                            solver = QUBOSolverBatch(datasets[0], n_vars, batch_size, 
                                                   xi=xi, dt=dt, n_iter=n_iter, method=method, device='cpu')
                            _, _ = solver.solve()
                            energy_history = [float(np.mean(e)) for e in solver.energy_history[0]]
                            method_results['energy_histories'][opt_name] = energy_history
                        except Exception as e:
                            print(f"记录 {opt_name} 能量历史时出错: {e}")
                
                iteration_results[method] = self.convert_to_python_types(method_results)
                performance_matrix[n_iter][method] = method_performance
            
            all_results[n_iter] = iteration_results
        
        return all_results, performance_matrix
    
    def plot_results(self, all_results, performance_matrix, n_iter_values, methods, opt_methods):
        """绘制结果图表"""
        print("\n=== 创建综合性能分析 ===")
        
        # 1. 为每个迭代次数绘制能量历史图
        print("绘制能量历史图...")
        for n_iter in n_iter_values:
            if n_iter in all_results:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.flatten()
                
                for idx, method in enumerate(methods):
                    if idx >= len(axes):
                        break
                    ax = axes[idx]
                    if method in all_results[n_iter]:
                        for opt_name, history in all_results[n_iter][method]['energy_histories'].items():
                            xi, dt = all_results[n_iter][method]['params'][opt_name]
                            ax.plot(history, label=f'{opt_name} (xi={xi:.3f}, dt={dt:.3f})', 
                                linewidth=2, alpha=0.8)
                        ax.set_xlabel('Iteration', fontsize=12)
                        ax.set_ylabel('Energy', fontsize=12)
                        ax.set_title(f'{method} Method (n_iter={n_iter})', fontsize=14)
                        ax.legend(fontsize=10)
                        ax.grid(True, linestyle='--', alpha=0.7)
                        ax.tick_params(axis='both', which='major', labelsize=10)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.result_dir, f'energy_history_niter_{n_iter}.png'), 
                        dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. 重新组织热力图数据 - 横轴为Iteration，纵轴为调优算法
        print("创建新的热力图布局...")
        
        # 准备数据矩阵
        n_iters = len(n_iter_values)
        n_opts = len(opt_methods)
        
        # 创建四个数据矩阵
        best_score_matrix = np.zeros((n_opts, n_iters))
        last_10_matrix = np.zeros((n_opts, n_iters)) 
        time_matrix = np.zeros((n_opts, n_iters))
        
        # 启发算法性能矩阵
        method_performance_matrix = np.zeros((len(methods), n_iters))
        
        # 填充数据
        for i, opt_method in enumerate(opt_methods):
            for j, n_iter in enumerate(n_iter_values):
                opt_scores = []
                opt_last_10 = []
                opt_times = []
                
                for method in methods:
                    if (method in performance_matrix[n_iter] and 
                        opt_method in performance_matrix[n_iter][method]):
                        opt_scores.append(performance_matrix[n_iter][method][opt_method]['score'])
                        opt_last_10.append(performance_matrix[n_iter][method][opt_method]['last_10_avg'])
                        opt_times.append(performance_matrix[n_iter][method][opt_method]['time'])
                
                best_score_matrix[i, j] = np.mean(opt_scores) if opt_scores else 0
                last_10_matrix[i, j] = np.mean(opt_last_10) if opt_last_10 else 0
                time_matrix[i, j] = np.mean(opt_times) if opt_times else 0
        
        # 填充启发算法性能矩阵
        for i, method in enumerate(methods):
            for j, n_iter in enumerate(n_iter_values):
                method_scores = []
                for opt_method in opt_methods:
                    if (method in performance_matrix[n_iter] and 
                        opt_method in performance_matrix[n_iter][method]):
                        method_scores.append(performance_matrix[n_iter][method][opt_method]['score'])
                method_performance_matrix[i, j] = np.mean(method_scores) if method_scores else 0
        
        # 3. 绘制新的热力图布局 - 修复字体居中问题
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 设置全局字体大小
        plt.rcParams.update({'font.size': 14})
        
        # 计算正确的文本位置 - 使用imshow的坐标系统
        # 对于n_opts行n_iters列的热力图，每个单元格的中心位置
        x_centers = np.arange(n_iters) + 0.5
        y_centers = np.arange(n_opts) + 0.5
        
        x_centers_methods = np.arange(n_iters) + 0.5
        y_centers_methods = np.arange(len(methods)) + 0.5
        
        # 图1: 调优算法最佳能量 vs Iteration
        im1 = axes[0, 0].imshow(best_score_matrix, cmap='viridis', aspect='auto')
        axes[0, 0].set_xlabel('Iteration', fontsize=16, fontweight='bold')
        axes[0, 0].set_ylabel('Optimization Algorithm', fontsize=16, fontweight='bold')
        axes[0, 0].set_title('Best Energy by Optimization Algorithm vs Iteration', fontsize=18, fontweight='bold')
        
        # 设置正确的刻度位置和标签
        axes[0, 0].set_xticks(np.arange(n_iters))
        axes[0, 0].set_xticklabels([str(x) for x in n_iter_values], fontsize=14)
        axes[0, 0].set_yticks(np.arange(n_opts))
        axes[0, 0].set_yticklabels(opt_methods, fontsize=14)
        
        cbar1 = plt.colorbar(im1, ax=axes[0, 0])
        cbar1.set_label('Best Energy', fontsize=14)
        cbar1.ax.tick_params(labelsize=12)
        
        # 添加数值标注 - 使用正确的坐标系统
        for i in range(best_score_matrix.shape[0]):
            for j in range(best_score_matrix.shape[1]):
                text_color = 'white' if best_score_matrix[i, j] > np.max(best_score_matrix) * 0.5 else 'black'
                axes[0, 0].text(j, i, f"{best_score_matrix[i, j]:.2f}",
                                ha='center', va='center', color=text_color, 
                                fontsize=16, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.3))
        
        # 图2: 调优算法最后10次平均能量 vs Iteration
        im2 = axes[0, 1].imshow(last_10_matrix, cmap='plasma', aspect='auto')
        axes[0, 1].set_xlabel('Iteration', fontsize=16, fontweight='bold')
        axes[0, 1].set_ylabel('Optimization Algorithm', fontsize=16, fontweight='bold')
        axes[0, 1].set_title('Last 10 Iterations Average Energy by Optimization Algorithm vs Iteration', 
                            fontsize=18, fontweight='bold')
        
        axes[0, 1].set_xticks(np.arange(n_iters))
        axes[0, 1].set_xticklabels([str(x) for x in n_iter_values], fontsize=14)
        axes[0, 1].set_yticks(np.arange(n_opts))
        axes[0, 1].set_yticklabels(opt_methods, fontsize=14)
        
        cbar2 = plt.colorbar(im2, ax=axes[0, 1])
        cbar2.set_label('Last 10 Iterations Average Energy', fontsize=14)
        cbar2.ax.tick_params(labelsize=12)
        
        # 添加数值标注
        for i in range(last_10_matrix.shape[0]):
            for j in range(last_10_matrix.shape[1]):
                text_color = 'white' if last_10_matrix[i, j] > np.max(last_10_matrix) * 0.5 else 'black'
                axes[0, 1].text(j, i, f"{last_10_matrix[i, j]:.2f}",
                                ha='center', va='center', color=text_color, 
                                fontsize=16, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.3))
        
        # 图3: 启发算法性能 vs Iteration
        im3 = axes[1, 0].imshow(method_performance_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_xlabel('Iteration', fontsize=16, fontweight='bold')
        axes[1, 0].set_ylabel('Heuristic Algorithm', fontsize=16, fontweight='bold')
        axes[1, 0].set_title('Best Energy by Heuristic Algorithm vs Iteration', 
                            fontsize=18, fontweight='bold')
        
        axes[1, 0].set_xticks(np.arange(n_iters))
        axes[1, 0].set_xticklabels([str(x) for x in n_iter_values], fontsize=14)
        axes[1, 0].set_yticks(np.arange(len(methods)))
        axes[1, 0].set_yticklabels(methods, fontsize=14)
        
        cbar3 = plt.colorbar(im3, ax=axes[1, 0])
        cbar3.set_label('Best Energy', fontsize=14)
        cbar3.ax.tick_params(labelsize=12)
        
        # 添加数值标注
        for i in range(method_performance_matrix.shape[0]):
            for j in range(method_performance_matrix.shape[1]):
                text_color = 'white' if method_performance_matrix[i, j] > np.max(method_performance_matrix) * 0.5 else 'black'
                axes[1, 0].text(j, i, f"{method_performance_matrix[i, j]:.2f}",
                                ha='center', va='center', color=text_color, 
                                fontsize=16, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.3))
        
        # 图4: 调优算法时间 vs Iteration
        im4 = axes[1, 1].imshow(time_matrix, cmap='RdYlBu_r', aspect='auto')
        axes[1, 1].set_xlabel('Iteration', fontsize=16, fontweight='bold')
        axes[1, 1].set_ylabel('Optimization Algorithm', fontsize=16, fontweight='bold')
        axes[1, 1].set_title('Total Time by Optimization Algorithm vs Iteration', 
                            fontsize=18, fontweight='bold')
        
        axes[1, 1].set_xticks(np.arange(n_iters))
        axes[1, 1].set_xticklabels([str(x) for x in n_iter_values], fontsize=14)
        axes[1, 1].set_yticks(np.arange(n_opts))
        axes[1, 1].set_yticklabels(opt_methods, fontsize=14)
        
        cbar4 = plt.colorbar(im4, ax=axes[1, 1])
        cbar4.set_label('Total Time (seconds)', fontsize=14)
        cbar4.ax.tick_params(labelsize=12)
        
        # 添加数值标注
        for i in range(time_matrix.shape[0]):
            for j in range(time_matrix.shape[1]):
                text_color = 'white' if time_matrix[i, j] > np.max(time_matrix) * 0.5 else 'black'
                axes[1, 1].text(j, i, f"{time_matrix[i, j]:.1f}",
                                ha='center', va='center', color=text_color, 
                                fontsize=16, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'reorganized_performance_heatmaps.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 性能对比图（增大字体）
        print("创建性能对比图...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 设置子图字体大小
        for ax in axes.flat:
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)
            ax.title.set_size(16)

        # 1. 各调参算法在不同迭代次数下的最佳分数
        for idx, opt_method in enumerate(opt_methods):
            scores_by_iter = []
            for n_iter in n_iter_values:
                method_scores = [performance_matrix[n_iter][method][opt_method]['score']
                                for method in methods if opt_method in performance_matrix[n_iter][method]]
                best_score = np.max(method_scores) if method_scores else 0
                scores_by_iter.append(best_score)
            axes[0,0].plot(n_iter_values, scores_by_iter, marker='o', label=opt_method, linewidth=2, markersize=8)
        axes[0,0].set_xlabel('Iterations', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('Best Score', fontsize=14, fontweight='bold')
        axes[0,0].set_title('Optimizer Best Score vs Iterations', fontsize=16, fontweight='bold')
        axes[0,0].legend(fontsize=12)
        axes[0,0].grid(True, alpha=0.3)

        # 2. 各调参算法在不同迭代次数下的最后10次平均分数
        for idx, opt_method in enumerate(opt_methods):
            last_10_by_iter = []
            for n_iter in n_iter_values:
                method_last_10 = [performance_matrix[n_iter][method][opt_method]['last_10_avg']
                                for method in methods if opt_method in performance_matrix[n_iter][method]]
                avg_last_10 = np.mean(method_last_10) if method_last_10 else 0
                last_10_by_iter.append(avg_last_10)
            axes[0,1].plot(n_iter_values, last_10_by_iter, marker='s', label=opt_method, linewidth=2, markersize=8)
        axes[0,1].set_xlabel('Iterations', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('Last 10 Iterations Average', fontsize=14, fontweight='bold')
        axes[0,1].set_title('Last 10 Iterations Average vs Iterations', fontsize=16, fontweight='bold')
        axes[0,1].legend(fontsize=12)
        axes[0,1].grid(True, alpha=0.3)

        # 3. 各调参算法在不同迭代次数下的总时间
        for idx, opt_method in enumerate(opt_methods):
            times_by_iter = []
            for n_iter in n_iter_values:
                method_times = [performance_matrix[n_iter][method][opt_method]['time']
                                for method in methods if opt_method in performance_matrix[n_iter][method]]
                avg_time = np.mean(method_times) if method_times else 0
                times_by_iter.append(avg_time)
            axes[1,0].plot(n_iter_values, times_by_iter, marker='^', label=opt_method, linewidth=2, markersize=8)
        axes[1,0].set_xlabel('Iterations', fontsize=14, fontweight='bold')
        axes[1,0].set_ylabel('Total Time (s)', fontsize=14, fontweight='bold')
        axes[1,0].set_title('Optimizer Total Time vs Iterations', fontsize=16, fontweight='bold')
        axes[1,0].legend(fontsize=12)
        axes[1,0].grid(True, alpha=0.3)

        # 4. 最佳分数与最后10次平均的相关性
        all_best_scores = []
        all_last_10_avgs = []
        for n_iter in n_iter_values:
            for method in methods:
                for opt_method in opt_methods:
                    if (method in performance_matrix[n_iter] and 
                        opt_method in performance_matrix[n_iter][method]):
                        all_best_scores.append(performance_matrix[n_iter][method][opt_method]['score'])
                        all_last_10_avgs.append(performance_matrix[n_iter][method][opt_method]['last_10_avg'])
        
        scatter = axes[1,1].scatter(all_best_scores, all_last_10_avgs, alpha=0.6, s=60)
        axes[1,1].set_xlabel('Best Score', fontsize=14, fontweight='bold')
        axes[1,1].set_ylabel('Last 10 Iterations Average', fontsize=14, fontweight='bold')
        axes[1,1].set_title('Correlation: Best Score vs Last 10 Average', fontsize=16, fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(all_best_scores) > 1:
            z = np.polyfit(all_best_scores, all_last_10_avgs, 1)
            p = np.poly1d(z)
            axes[1,1].plot(all_best_scores, p(all_best_scores), "r--", alpha=0.8, linewidth=2)

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_results(self, all_results, performance_matrix, n_iter_values, methods, opt_methods):
        """保存结果到文件"""
        print("保存详细结果...")
        
        # 保存详细结果
        with open(os.path.join(self.result_dir, 'hyperparameter_comparison_results.json'), 'w') as f:
            json.dump(all_results, f, indent=4, cls=NumpyEncoder)
        
        # 创建性能总结表
        summary_data = []
        for n_iter in n_iter_values:
            for method in methods:
                for opt_method in opt_methods:
                    if (method in performance_matrix[n_iter] and 
                        opt_method in performance_matrix[n_iter][method]):
                        score = performance_matrix[n_iter][method][opt_method]['score']
                        time_val = performance_matrix[n_iter][method][opt_method]['time']
                        last_10_avg = performance_matrix[n_iter][method][opt_method]['last_10_avg']
                        summary_data.append({
                            'Iterations': n_iter,
                            'Method': method,
                            'Optimizer': opt_method,
                            'Best_Score': score,
                            'Last_10_Avg': last_10_avg,
                            'Total_Time': time_val
                        })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.result_dir, 'performance_summary.csv'), index=False)
        
        # 打印最佳性能组合
        print(f"\n=== 最佳性能组合 ===")
        best_combinations = summary_df.nlargest(10, 'Best_Score')
        for idx, row in best_combinations.iterrows():
            print(f"第{idx+1}名: {row['Method']}-{row['Optimizer']} "
                f"(迭代={row['Iterations']}, 最佳分数={row['Best_Score']:.4f}, "
                f"最后10次平均={row['Last_10_Avg']:.4f}, 总时间={row['Total_Time']:.2f}s)")
        
        return summary_df

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# ===== 主程序 =====
if __name__ == "__main__":
    # 实验配置
    methods = ['bSB', 'dSB', 'CAC', 'CFC']
    opt_methods = ['Bayes', 'CMA-ES', 'Nelder-Mead', 'Hyperopt', 'QGA']
    n_iter_values = [10, 100, 300, 500, 800]
    n_vars = 20
    density = 0.10
    n_calls = 10
    batch_size = 4
    n_datasets = 50
    
    # 创建实验管理器
    result_dir = './QUBO/results_one_end3_update'
    experiment_manager = QUBOExperimentManager(result_dir)
    
    # 运行综合实验
    all_results, performance_matrix = experiment_manager.run_comprehensive_experiment(
        methods=methods,
        opt_methods=opt_methods,
        n_iter_values=n_iter_values,
        n_vars=n_vars,
        density=density,
        n_calls=n_calls,
        batch_size=batch_size,
        n_datasets=n_datasets
    )
    
    # 绘制结果
    experiment_manager.plot_results(all_results, performance_matrix, n_iter_values, methods, opt_methods)
    
    # 保存结果
    summary_df = experiment_manager.save_results(all_results, performance_matrix, n_iter_values, methods, opt_methods)
    
    print(f"\n=== 实验完成 ===")
    print(f"结果保存在: {result_dir}")
    print(f"生成了以下文件:")
    print(f"- hyperparameter_comparison_results.json: 详细结果")
    print(f"- performance_summary.csv: 性能总结表")
    print(f"- reorganized_performance_heatmaps.png: 重新组织的性能热力图")
    print(f"- performance_comparison.png: 性能对比图")
    print(f"- energy_history_niter_*.png: 各迭代次数的能量历史图")