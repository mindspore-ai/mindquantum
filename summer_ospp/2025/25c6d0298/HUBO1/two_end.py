import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from skopt import gp_minimize
from scipy.optimize import dual_annealing, minimize
import cma
import time
from functools import partial
import json
from json import JSONEncoder
from hyperopt import fmin, tpe, hp, Trials
from tqdm import tqdm
import warnings
import random
import math
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
mpl.use('Agg')


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# ===== HUBO问题生成器 =====
class HUBOProblemGenerator:
    def __init__(self, n_vars=300, max_terms=10, max_order=3):
        self.n_vars = n_vars
        self.max_terms = max_terms
        self.max_order = max_order
    
    def generate_hubo_terms(self):
        """生成HUBO项，从300个变量中随机选择少量项"""
        terms_dict = {}
        
        # 生成不同阶数的项
        for order in range(1, self.max_order + 1):
            # 计算该阶数下的项数（确保总项数不超过max_terms）
            remaining_terms = self.max_terms - sum(len(terms) for terms in terms_dict.values())
            if remaining_terms <= 0:
                break
                
            # 该阶数的项数不超过剩余项数
            n_terms = min(remaining_terms, max(1, self.max_terms // self.max_order))
            
            terms = []
            for _ in range(n_terms):
                # 随机生成系数 (-10到10之间)
                coef = np.random.randint(-10, 10)
                # 从300个变量中随机选择order个变量
                vars_list = random.sample(range(self.n_vars), order)
                terms.append((coef, vars_list))
            
            if terms:
                terms_dict[order] = terms
        
        return terms_dict
    
    def generate_multiple_problems(self, n_problems):
        """生成多个HUBO问题"""
        problems = []
        for _ in range(n_problems):
            problems.append(self.generate_hubo_terms())
        return problems


# ===== 超参数优化器封装 =====
class HyperparameterOptimizer:
    def __init__(self, terms_dict, n_vars, n_iter, method_name, n_calls=10):
        self.terms_dict = terms_dict
        self.n_vars = n_vars
        self.n_iter = n_iter
        self.method_name = method_name
        self.n_calls = n_calls
        
    def objective(self, params):
        """目标函数：评估超参数性能"""
        xi, dt = params
        try:
            term_orders, term_coeffs, term_indices = terms_dict_to_tensor(self.terms_dict, self.n_vars)
            solver = HUBOSolverBatch(term_orders, term_coeffs, term_indices, self.n_vars, 1, 
                                   xi=xi, dt=dt, n_iter=self.n_iter, method=self.method_name, device='cpu')
            _, best_energy = solver.solve()
            
            # 计算最后10次迭代的平均能量
            if len(solver.energy_history) > 0 and len(solver.energy_history[0]) >= 10:
                last_10_avg = np.mean(solver.energy_history[0][-10:])
            else:
                last_10_avg = float(best_energy[0].item())
                
            return float(best_energy[0].item()), last_10_avg
        except Exception as e:
            print(f"求解器执行错误: {e}")
            return 1000, 1000
    
    def hyperopt_objective(self, params):
        """Hyperopt的目标函数"""
        xi = params['xi']
        dt = params['dt']
        best_energy, last_10_avg = self.objective((xi, dt))
        return best_energy
    
    def optimize_bayes(self, space):
        """贝叶斯优化"""
        def obj_func(params):
            return self.objective(params)[0]
            
        res = gp_minimize(obj_func, space, n_calls=self.n_calls, random_state=42)
        best_xi, best_dt = res.x
        best_score = res.fun
        return best_xi, best_dt, best_score, res
    
    def optimize_sa(self, bounds):
        """模拟退火优化"""
        def obj_func(params):
            return self.objective(params)[0]
            
        res = dual_annealing(obj_func, bounds, maxiter=self.n_calls, seed=42)
        best_xi, best_dt = res.x
        best_score = res.fun
        return best_xi, best_dt, best_score, res
    
    def optimize_cma_es(self, x0):
        """CMA-ES优化"""
        def obj_func(params):
            return self.objective(params)[0]
            
        try:
            es = cma.CMAEvolutionStrategy(x0, 0.1, {
                'bounds': [[0.01, 0.01], [0.5, 0.3]],
                'maxfevals': self.n_calls,
                'verbose': -1
            })
            es.optimize(obj_func)
            best_xi, best_dt = es.result.xbest
            best_score = es.result.fbest
            return best_xi, best_dt, best_score, es
        except Exception as e:
            print(f"CMA-ES边界设置失败: {e}")
            es = cma.CMAEvolutionStrategy(x0, 0.1, {
                'maxfevals': self.n_calls,
                'verbose': -1
            })
            es.optimize(obj_func)
            best_xi, best_dt = es.result.xbest
            best_score = es.result.fbest
            return best_xi, best_dt, best_score, es
    
    def optimize_nelder_mead(self, x0, bounds):
        """Nelder-Mead优化"""
        def obj_func(params):
            return self.objective(params)[0]
            
        res = minimize(obj_func, x0, method='Nelder-Mead', bounds=bounds, 
                      options={'maxfev': self.n_calls})
        best_xi, best_dt = res.x
        best_score = res.fun
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
        best_score = best_trial['result']['loss']
        return best_xi, best_dt, best_score, trials
    
    def optimize_qga(self, bounds):
        """量子遗传算法优化"""
        def obj_func(params):
            return self.objective(params)[0]
            
        qga = QuantumGeneticAlgorithm(obj_func, bounds, self.n_calls, pop_size=10, n_qubits=8)
        best_params_qga, best_score_qga = qga.optimize()
        best_xi, best_dt = best_params_qga
        best_score = best_score_qga
        return best_xi, best_dt, best_score, qga


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


# ===== HUBO求解器 =====
class HUBOSolverBatch:
    def __init__(self, term_orders, term_coeffs, term_indices, n_vars, batch_size,
                 xi=0.3, dt=0.15, n_iter=1000, method='dSB', device='cpu'):
        self.term_orders = term_orders.to(device)
        self.term_coeffs = term_coeffs.to(device)
        self.term_indices = term_indices.to(device)
        self.n_vars = n_vars
        self.batch_size = batch_size
        self.xi = xi
        self.dt = dt
        self.n_iter = n_iter
        self.method = method
        self.device = device
        self.Delta = 0.5
        self.x = 0.01 * torch.randn(batch_size, n_vars, device=device)
        self.y = 0.01 * torch.randn(batch_size, n_vars, device=device)
        self.cut_history = []
        self.energy_history = []
        self.best_energy_history = []
        self.runtime = 0
        self.x_history = []
        self.p_history = []

    def compute_energy_batch(self, x):
        n_terms = self.term_coeffs.shape[0]
        max_order = self.term_indices.shape[1]
        batch_size = x.shape[0]
        idx = self.term_indices.unsqueeze(0).expand(batch_size, -1, -1)
        idx_safe = idx.clone()
        idx_safe[idx_safe == -1] = 0  # 将-1替换为0，防止越界
        x_gathered = torch.gather(x.unsqueeze(1).expand(-1, n_terms, -1), 2, idx_safe)
        mask = (idx == -1)
        x_gathered[mask] = 1.0
        prod = torch.prod(x_gathered, dim=2)
        energy = torch.matmul(prod, self.term_coeffs)
        return energy

    def solve(self):
        best_energy = torch.full((self.batch_size,), float('-inf'), device=self.device)
        best_x = torch.zeros_like(self.x)
        self.cut_history = []
        self.energy_history = []
        self.best_energy_history = []
        self.x_history = []
        self.p_history = []
        start_time = time.time()
        for run in range(1):
            self.x = 0.01 * torch.randn(self.batch_size, self.n_vars, device=self.device)
            self.y = 0.01 * torch.randn(self.batch_size, self.n_vars, device=self.device)
            run_energy = []
            run_best_energy = torch.full((self.batch_size,), float('-inf'), device=self.device)
            for iter in tqdm(range(self.n_iter + 1), desc=f"Batch Run {run+1}"):
                p_t = iter / self.n_iter
                self.p_history.append(p_t)
                x_torch = self.x.clone().detach().requires_grad_(True)
                energy = self.compute_energy_batch(x_torch)
                run_energy.append(energy.detach().cpu().numpy())
                run_best_energy = torch.max(run_best_energy, energy)
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
            self.energy_history.append(run_energy)
            self.best_energy_history.append(run_best_energy.detach().cpu().numpy())
            with torch.no_grad():
                current_x = torch.sign(self.x)
                current_energy = self.compute_energy_batch(current_x)
                mask = current_energy > best_energy
                best_energy[mask] = current_energy[mask]
                best_x[mask] = current_x[mask]
        self.runtime = time.time() - start_time
        return best_x, best_energy


def terms_dict_to_tensor(terms_dict, n_vars):
    """将HUBO项字典转换为张量格式"""
    all_terms = []
    for order, terms in terms_dict.items():
        for coef, vars_list in terms:
            all_terms.append((order, coef, vars_list))
    n_terms = len(all_terms)
    max_order = max([len(t[2]) for t in all_terms]) if all_terms else 1
    term_orders = torch.tensor([len(t[2]) for t in all_terms], dtype=torch.long)
    term_coeffs = torch.tensor([t[1] for t in all_terms], dtype=torch.float32)
    term_indices = -torch.ones((n_terms, max_order), dtype=torch.long)
    for i, (_, _, vars_list) in enumerate(all_terms):
        term_indices[i, :len(vars_list)] = torch.tensor(vars_list, dtype=torch.long)
    return term_orders, term_coeffs, term_indices


# ===== 实验管理器 =====
class HUBOExperimentManager:
    def __init__(self, result_dir='./HUBO1/result_two_end3_update'):
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)
        
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
    
    def run_single_experiment(self, n_iter, method, opt_method, hubo_problems, n_vars, n_calls=10):
        """运行单个实验配置"""
        print(f"运行 {method}-{opt_method} (迭代={n_iter})...")
        
        best_scores = []
        best_params = []
        total_times = []
        last_10_avg_scores = []  # 最后10次平均分数
        
        # 在多个HUBO问题上运行以获得统计结果
        for problem_idx, terms_dict in enumerate(hubo_problems):
            try:
                # 创建优化器
                optimizer = HyperparameterOptimizer(
                    terms_dict=terms_dict,
                    n_vars=n_vars,
                    n_iter=n_iter,
                    method_name=method,
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
                term_orders, term_coeffs, term_indices = terms_dict_to_tensor(terms_dict, n_vars)
                solver = HUBOSolverBatch(term_orders, term_coeffs, term_indices, n_vars, 1,
                                       xi=best_xi, dt=best_dt, n_iter=n_iter, method=method, device='cpu')
                _, best_qubo = solver.solve()
                
                # 计算最后10次迭代的平均（如果迭代次数足够）
                if len(solver.energy_history) > 0 and len(solver.energy_history[0]) >= 10:
                    last_10_avg = np.mean(solver.energy_history[0][-10:])
                else:
                    last_10_avg = float(best_qubo[0].item())
                
                best_scores.append(best_score)
                best_params.append((best_xi, best_dt))
                total_times.append(total_time)
                last_10_avg_scores.append(last_10_avg)
                
            except Exception as e:
                print(f"HUBO问题 {problem_idx+1} 上 {opt_method} 失败: {e}")
                best_scores.append(0)
                best_params.append((0.1, 0.1))
                total_times.append(float('inf'))
                last_10_avg_scores.append(0)
        
        # 返回最佳结果（而不是平均结果）
        if best_scores:
            best_idx = np.argmin(best_scores)  # 找到最佳分数的索引（最小化问题）
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
                                   n_vars=300, max_terms=10, max_order=3, 
                                   n_calls=10, n_problems=50):
        """运行综合实验"""
        print("=== HUBO超参数比较实验 ===")
        print(f"算法: {methods}")
        print(f"优化方法: {opt_methods}")
        print(f"迭代次数: {n_iter_values}")
        print(f"HUBO问题数量: {n_problems}")
        print(f"变量数量: {n_vars}")
        print(f"最大项数: {max_terms}")
        print(f"最大阶数: {max_order}")
        
        all_results = {}
        performance_matrix = {}
        
        # 生成HUBO问题
        problem_generator = HUBOProblemGenerator(n_vars=n_vars, max_terms=max_terms, max_order=max_order)
        hubo_problems = problem_generator.generate_multiple_problems(n_problems)
        
        # 为每个迭代次数运行实验
        for n_iter in n_iter_values:
            print(f"\n{'='*60}")
            print(f"运行迭代次数: {n_iter}")
            print(f"{'='*60}")
            
            iteration_results = {}
            performance_matrix[n_iter] = {}
            
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
                    result = self.run_single_experiment(n_iter, method, opt_method, hubo_problems, n_vars, n_calls)
                    
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
                
                # 记录能量历史（使用第一个HUBO问题和最佳参数）
                print("记录能量历史...")
                for opt_name in opt_methods:
                    if opt_name in method_results['params']:
                        xi, dt = method_results['params'][opt_name]
                        try:
                            term_orders, term_coeffs, term_indices = terms_dict_to_tensor(hubo_problems[0], n_vars)
                            solver = HUBOSolverBatch(term_orders, term_coeffs, term_indices, n_vars, 1, 
                                                   xi=xi, dt=dt, n_iter=n_iter, method=method, device='cpu')
                            _, _ = solver.solve()
                            energy_history = [float(e[0]) for e in solver.energy_history[0]]
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
                        ax.set_xlabel('Iteration')
                        ax.set_ylabel('Energy')
                        ax.set_title(f'{method} Method (n_iter={n_iter})')
                        ax.legend(fontsize=8)
                        ax.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.result_dir, f'energy_history_niter_{n_iter}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. 性能热力图
        print("创建性能热力图...")
        score_matrix = np.zeros((len(n_iter_values), len(methods) * len(opt_methods)))
        time_matrix = np.zeros((len(n_iter_values), len(methods) * len(opt_methods)))
        last_10_matrix = np.zeros((len(n_iter_values), len(methods) * len(opt_methods)))
        
        method_opt_labels = []
        for method in methods:
            for opt_method in opt_methods:
                method_opt_labels.append(f'{method}-{opt_method}')
        
        for i, n_iter in enumerate(n_iter_values):
            for j, method in enumerate(methods):
                for k, opt_method in enumerate(opt_methods):
                    col_idx = j * len(opt_methods) + k
                    if (method in performance_matrix[n_iter] and 
                        opt_method in performance_matrix[n_iter][method]):
                        score_matrix[i, col_idx] = performance_matrix[n_iter][method][opt_method]['score']
                        time_matrix[i, col_idx] = performance_matrix[n_iter][method][opt_method]['time']
                        last_10_matrix[i, col_idx] = performance_matrix[n_iter][method][opt_method]['last_10_avg']
        
        # 绘制分数热力图
        plt.figure(figsize=(20, 12))
        plt.subplot(1, 3, 1)
        im1 = plt.imshow(score_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(im1, label='Best Score')
        plt.xlabel('Method-Optimizer Combination')
        plt.ylabel('Iterations')
        plt.title('Best Score Heatmap')
        plt.xticks(range(len(method_opt_labels)), method_opt_labels, rotation=45, ha='right')
        plt.yticks(range(len(n_iter_values)), n_iter_values)
        
        # 绘制时间热力图
        plt.subplot(1, 3, 2)
        im2 = plt.imshow(time_matrix, cmap='plasma', aspect='auto')
        plt.colorbar(im2, label='Total Time (s)')
        plt.xlabel('Method-Optimizer Combination')
        plt.ylabel('Iterations')
        plt.title('Total Time Heatmap')
        plt.xticks(range(len(method_opt_labels)), method_opt_labels, rotation=45, ha='right')
        plt.yticks(range(len(n_iter_values)), n_iter_values)
        
        # 绘制最后10次平均热力图
        plt.subplot(1, 3, 3)
        im3 = plt.imshow(last_10_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar(im3, label='Last 10 Iterations Average')
        plt.xlabel('Method-Optimizer Combination')
        plt.ylabel('Iterations')
        plt.title('Last 10 Iterations Average Heatmap')
        plt.xticks(range(len(method_opt_labels)), method_opt_labels, rotation=45, ha='right')
        plt.yticks(range(len(n_iter_values)), n_iter_values)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'performance_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 性能对比图
        print("创建性能对比图...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 各调参算法在不同迭代次数下的最佳分数
        for idx, opt_method in enumerate(opt_methods):
            scores_by_iter = []
            for n_iter in n_iter_values:
                # 取所有method的最佳分数
                method_scores = [performance_matrix[n_iter][method][opt_method]['score']
                                for method in methods if opt_method in performance_matrix[n_iter][method]]
                best_score = np.min(method_scores) if method_scores else 0  # 最小化问题
                scores_by_iter.append(best_score)
            axes[0,0].plot(n_iter_values, scores_by_iter, marker='o', label=opt_method, linewidth=2)
        axes[0,0].set_xlabel('Iterations')
        axes[0,0].set_ylabel('Best Score')
        axes[0,0].set_title('Optimizer Best Score vs Iterations')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # 2. 各调参算法在不同迭代次数下的最后10次平均分数
        for idx, opt_method in enumerate(opt_methods):
            last_10_by_iter = []
            for n_iter in n_iter_values:
                method_last_10 = [performance_matrix[n_iter][method][opt_method]['last_10_avg']
                                for method in methods if opt_method in performance_matrix[n_iter][method]]
                avg_last_10 = np.mean(method_last_10) if method_last_10 else 0
                last_10_by_iter.append(avg_last_10)
            axes[0,1].plot(n_iter_values, last_10_by_iter, marker='s', label=opt_method, linewidth=2)
        axes[0,1].set_xlabel('Iterations')
        axes[0,1].set_ylabel('Last 10 Iterations Average')
        axes[0,1].set_title('Last 10 Iterations Average vs Iterations')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # 3. 各调参算法在不同迭代次数下的总时间
        for idx, opt_method in enumerate(opt_methods):
            times_by_iter = []
            for n_iter in n_iter_values:
                method_times = [performance_matrix[n_iter][method][opt_method]['time']
                                for method in methods if opt_method in performance_matrix[n_iter][method]]
                avg_time = np.mean(method_times) if method_times else 0
                times_by_iter.append(avg_time)
            axes[1,0].plot(n_iter_values, times_by_iter, marker='^', label=opt_method, linewidth=2)
        axes[1,0].set_xlabel('Iterations')
        axes[1,0].set_ylabel('Total Time (s)')
        axes[1,0].set_title('Optimizer Total Time vs Iterations')
        axes[1,0].legend()
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
        
        axes[1,1].scatter(all_best_scores, all_last_10_avgs, alpha=0.6)
        axes[1,1].set_xlabel('Best Score')
        axes[1,1].set_ylabel('Last 10 Iterations Average')
        axes[1,1].set_title('Correlation: Best Score vs Last 10 Average')
        axes[1,1].grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(all_best_scores) > 1:
            z = np.polyfit(all_best_scores, all_last_10_avgs, 1)
            p = np.poly1d(z)
            axes[1,1].plot(all_best_scores, p(all_best_scores), "r--", alpha=0.8)

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
                            'Best_Score': abs(score),
                            'Last_10_Avg': abs(last_10_avg),
                            'Total_Time': time_val
                        })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.result_dir, 'performance_summary.csv'), index=False)
        
        # 打印最佳性能组合
        print(f"\n=== 最佳性能组合 ===")
        best_combinations = summary_df.nsmallest(10, 'Best_Score')  # 最小化问题
        for idx, row in best_combinations.iterrows():
            print(f"第{idx+1}名: {row['Method']}-{row['Optimizer']} "
                  f"(迭代={row['Iterations']}, 最佳分数={row['Best_Score']:.4f}, "
                  f"最后10次平均={row['Last_10_Avg']:.4f}, 总时间={row['Total_Time']:.2f}s)")
        
        return summary_df


# ===== 可视化函数 =====
def create_simple_visualizations(success_df, result_dir):
    """创建简化的可视化图表"""
    print("\nCreating simplified visualization charts...")
    
    # 设置图形样式
    plt.style.use('default')
    
    # 1. Energy vs Iterations
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Energy-Iteration curve for all optimizers
    plt.subplot(2, 2, 1)
    for opt_method in success_df['Optimizer'].unique():
        method_data = success_df[success_df['Optimizer'] == opt_method]
        iter_means = method_data.groupby('Iterations')['Best_Score'].mean()
        plt.plot(iter_means.index, iter_means.values, label=opt_method, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Iterations')
    plt.ylabel('Average Best Energy')
    plt.title('Optimizer: Energy vs Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Energy-Iteration curve for all HUBO methods
    plt.subplot(2, 2, 2)
    for method in success_df['Method'].unique():
        method_data = success_df[success_df['Method'] == method]
        iter_means = method_data.groupby('Iterations')['Best_Score'].mean()
        plt.plot(iter_means.index, iter_means.values, label=method, marker='s', linewidth=2, markersize=6)
    plt.xlabel('Iterations')
    plt.ylabel('Average Best Energy')
    plt.title('HUBO Method: Energy vs Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Time vs Iterations
    plt.subplot(2, 2, 3)
    for opt_method in success_df['Optimizer'].unique():
        method_data = success_df[success_df['Optimizer'] == opt_method]
        iter_means = method_data.groupby('Iterations')['Total_Time'].mean()
        plt.plot(iter_means.index, iter_means.values, label=opt_method, marker='^', linewidth=2, markersize=6)
    plt.xlabel('Iterations')
    plt.ylabel('Average Time (s)')
    plt.title('Optimizer: Time vs Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Energy vs Time scatter plot
    plt.subplot(2, 2, 4)
    for opt_method in success_df['Optimizer'].unique():
        method_data = success_df[success_df['Optimizer'] == opt_method]
        plt.scatter(method_data['Total_Time'], method_data['Best_Score'], 
                   label=opt_method, alpha=0.6, s=50)
    plt.xlabel('Time (s)')
    plt.ylabel('Best Energy')
    plt.title('Energy vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'iteration_analysis_simple.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 热力图分析
    create_heatmap_visualizations(success_df, result_dir)
    
    print(f"Simplified visualization charts saved to {result_dir}/")


def create_heatmap_visualizations(success_df, result_dir):
    """创建热力图可视化"""
    print("Creating heatmap analysis...")
    
    # 创建热力图
    plt.figure(figsize=(16, 10))
    
    # Subplot 1: Optimizer vs Iterations (Energy Heatmap)
    plt.subplot(2, 2, 1)
    pivot_energy = success_df.pivot_table(values='Best_Score', index='Optimizer', columns='Iterations', aggfunc='mean')
    sns.heatmap(pivot_energy, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'Average Energy'})
    plt.title('Optimizer vs Iterations (Energy Heatmap)')
    plt.xlabel('Iterations')
    plt.ylabel('Optimizer')
    
    # Subplot 2: HUBO Method vs Iterations (Energy Heatmap)
    plt.subplot(2, 2, 2)
    pivot_energy_hubo = success_df.pivot_table(values='Best_Score', index='Method', columns='Iterations', aggfunc='mean')
    sns.heatmap(pivot_energy_hubo, annot=True, fmt='.3f', cmap='plasma', cbar_kws={'label': 'Average Energy'})
    plt.title('HUBO Method vs Iterations (Energy Heatmap)')
    plt.xlabel('Iterations')
    plt.ylabel('HUBO Method')
    
    # Subplot 3: Optimizer vs Iterations (Time Heatmap)
    plt.subplot(2, 2, 3)
    pivot_time = success_df.pivot_table(values='Total_Time', index='Optimizer', columns='Iterations', aggfunc='mean')
    sns.heatmap(pivot_time, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'Average Time (s)'})
    plt.title('Optimizer vs Iterations (Time Heatmap)')
    plt.xlabel('Iterations')
    plt.ylabel('Optimizer')
    
    # Subplot 4: Efficiency Heatmap (Energy/Time)
    plt.subplot(2, 2, 4)
    pivot_efficiency = success_df.pivot_table(values='Best_Score', index='Optimizer', columns='Iterations', aggfunc='mean')
    pivot_time_norm = success_df.pivot_table(values='Total_Time', index='Optimizer', columns='Iterations', aggfunc='mean')
    efficiency = pivot_efficiency / pivot_time_norm  # Energy/Time as efficiency
    sns.heatmap(efficiency, annot=True, fmt='.3f', cmap='RdYlBu_r', cbar_kws={'label': 'Efficiency (Energy/Time)'})
    plt.title('Efficiency Heatmap')
    plt.xlabel('Iterations')
    plt.ylabel('Optimizer')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'heatmap_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ===== 主程序 =====
if __name__ == "__main__":
    # 实验配置
    methods = ['bSB', 'dSB', 'CAC', 'CFC']
    opt_methods = ['Bayes', 'CMA-ES', 'Nelder-Mead', 'Hyperopt', 'QGA']#可选优化方法，有Bayes, SA, CMA-ES, Nelder-Mead, Hyperopt, QGA
    n_iter_values = [10, 100, 300,500,800]  # 不同的迭代次数
    n_vars = 300  # 更大的体系
    max_terms = 10  # 只选取少量项
    max_order = 3
    n_calls = 10
    n_problems = 50  # 生成多个HUBO问题进行测试
    
    # 创建实验管理器
    result_dir = './HUBO1/result_two_end3_update'
    experiment_manager = HUBOExperimentManager(result_dir)
    
    # 运行综合实验
    all_results, performance_matrix = experiment_manager.run_comprehensive_experiment(
        methods=methods,
        opt_methods=opt_methods,
        n_iter_values=n_iter_values,
        n_vars=n_vars,
        max_terms=max_terms,
        max_order=max_order,
        n_calls=n_calls,
        n_problems=n_problems
    )
    
    # 绘制结果
    experiment_manager.plot_results(all_results, performance_matrix, n_iter_values, methods, opt_methods)
    
    # 保存结果
    summary_df = experiment_manager.save_results(all_results, performance_matrix, n_iter_values, methods, opt_methods)
    
    # 创建热力图可视化
    print("\n=== 创建热力图可视化 ===")
    create_simple_visualizations(summary_df, result_dir)
    
    print(f"\n=== 实验完成 ===")
    print(f"结果保存在: {result_dir}")
    print(f"生成了以下文件:")
    print(f"- hyperparameter_comparison_results.json: 详细结果")
    print(f"- performance_summary.csv: 性能总结表")
    print(f"- performance_heatmaps.png: 性能热力图")
    print(f"- performance_comparison.png: 性能对比图")
    print(f"- energy_history_niter_*.png: 各迭代次数的能量历史图")
    print(f"- iteration_analysis_simple.png: 简化分析图表")
    print(f"- heatmap_analysis.png: 热力图分析")