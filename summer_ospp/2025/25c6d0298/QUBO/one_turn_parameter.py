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
    def __init__(self, obj_func, bounds, n_calls, pop_size=12, n_qubits=8):
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



def objective(params, n_vars, n_iter, method_name, density, J_batch=None):
    xi, dt = params
    try:
        if J_batch is not None:
            batch_size = J_batch.shape[0]
            solver = QUBOSolverBatch(J_batch, n_vars, batch_size, xi=xi, dt=dt, n_iter=n_iter, method=method_name, device='cpu')
            _, best_qubo = solver.solve()
            # 返回所有batch的平均最优能量
            return -float(torch.mean(best_qubo).item())
        else:
            raise ValueError('J_batch must be provided for batch optimization')
    except Exception as e:
        print(f"Error during solver execution: {e}")
        return 1000

def batch_objective(params_list, n_vars, n_iter, method_name, density, J_batch=None):
    results = []
    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), len(params_list))) as executor:
        futures = {
            executor.submit(objective, params, n_vars, n_iter, method_name, density, J_batch): i
            for i, params in enumerate(params_list)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                print(f"Task failed: {e}")
                results.append((idx, 1000))
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]

def hyperopt_objective(params, n_vars, n_iter, method_name, density, J_batch=None):
    xi = params['xi']
    dt = params['dt']
    return objective((xi, dt), n_vars, n_iter, method_name, density, J_batch)

def convert_to_python_types(data):
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

def create_cma_es(x0, n_calls):
    try:
        es = cma.CMAEvolutionStrategy(x0, 0.1, {
            'bounds': [[0.01, 0.01], [0.5, 0.3]],
            'maxfevals': n_calls,
            'verbose': -1
        })
        return es
    except Exception as e:
        print(f"CMA-ES bounds error: {e}")
        return cma.CMAEvolutionStrategy(x0, 0.1, {
            'maxfevals': n_calls,
            'verbose': -1
        })

def generate_qubo_batch(batch_size, n_vars, density):
    # 生成batch_size个QUBO矩阵
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

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

if __name__ == "__main__":
    methods = ['bSB', 'dSB', 'CAC', 'CFC']
    opt_methods = ['Bayes', 'SA', 'CMA-ES', 'Nelder-Mead', 'Hyperopt', 'QGA']  # 添加QGA
    n_vars = 12
    n_iter = 300
    density = 0.3
    n_calls = 10
    batch_size = 4  # 并行batch数
    result_dir = './one/resultsone_end2'
    os.makedirs(result_dir, exist_ok=True)
    all_results = {}
    J_batch = generate_qubo_batch(batch_size, n_vars, density)
    for method in methods:
        print(f"\n=== 优化方法: {method} ===")
        method_dir = os.path.join(result_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        obj_func = partial(objective, n_vars=n_vars, n_iter=n_iter, method_name=method, density=density, J_batch=J_batch)
        batch_obj_func = partial(batch_objective, n_vars=n_vars, n_iter=n_iter, method_name=method, density=density, J_batch=J_batch)
        hyperopt_obj = partial(hyperopt_objective, n_vars=n_vars, n_iter=n_iter, method_name=method, density=density, J_batch=J_batch)
        space = [(0.01, 0.5), (0.01, 0.3)]
        bounds = [(0.01, 0.5), (0.01, 0.3)]
        x0 = [0.1, 0.1]
        method_results = {'params': {}, 'scores': {}, 'times': {}, 'energy_histories': {}}
        print("运行贝叶斯优化...")
        start_time = time.time()
        try:
            res_bayes = gp_minimize(obj_func, space, n_calls=n_calls, random_state=42)
            bayes_time = time.time() - start_time
            best_xi_bayes, best_dt_bayes = res_bayes.x
            best_score_bayes = -res_bayes.fun
            print(f"Bayesian Optimization: xi={best_xi_bayes:.4f}, dt={best_dt_bayes:.4f}, "
                  f"最优值={best_score_bayes:.4f}, 耗时={bayes_time:.2f}s")
            method_results['params']['Bayes'] = (best_xi_bayes, best_dt_bayes)
            method_results['scores']['Bayes'] = best_score_bayes
            method_results['times']['Bayes'] = bayes_time
        except Exception as e:
            print(f"贝叶斯优化失败: {e}")
        print("运行模拟退火...")
        start_time = time.time()
        try:
            res_sa = dual_annealing(obj_func, bounds, maxiter=n_calls, seed=42)
            sa_time = time.time() - start_time
            best_xi_sa, best_dt_sa = res_sa.x
            best_score_sa = -res_sa.fun
            print(f"Simulated Annealing: xi={best_xi_sa:.4f}, dt={best_dt_sa:.4f}, "
                  f"最优值={best_score_sa:.4f}, 耗时={sa_time:.2f}s")
            method_results['params']['SA'] = (best_xi_sa, best_dt_sa)
            method_results['scores']['SA'] = best_score_sa
            method_results['times']['SA'] = sa_time
        except Exception as e:
            print(f"模拟退火失败: {e}")
        print("运行CMA-ES...")
        start_time = time.time()
        try:
            es = create_cma_es(x0, n_calls)
            es.optimize(obj_func)
            cma_time = time.time() - start_time
            best_xi_cma, best_dt_cma = es.result.xbest
            best_score_cma = -es.result.fbest
            print(f"CMA-ES: xi={best_xi_cma:.4f}, dt={best_dt_cma:.4f}, "
                  f"最优值={best_score_cma:.4f}, 耗时={cma_time:.2f}s")
            method_results['params']['CMA-ES'] = (best_xi_cma, best_dt_cma)
            method_results['scores']['CMA-ES'] = best_score_cma
            method_results['times']['CMA-ES'] = cma_time
        except Exception as e:
            print(f"CMA-ES失败: {e}")
        print("运行Nelder-Mead...")
        start_time = time.time()
        try:
            res_nm = minimize(obj_func, x0, method='Nelder-Mead', bounds=bounds, options={'maxfev': n_calls})
            nm_time = time.time() - start_time
            best_xi_nm, best_dt_nm = res_nm.x
            best_score_nm = -res_nm.fun
            print(f"Nelder-Mead: xi={best_xi_nm:.4f}, dt={best_dt_nm:.4f}, "
                  f"最优值={best_score_nm:.4f}, 耗时={nm_time:.2f}s")
            method_results['params']['Nelder-Mead'] = (best_xi_nm, best_dt_nm)
            method_results['scores']['Nelder-Mead'] = best_score_nm
            method_results['times']['Nelder-Mead'] = nm_time
        except Exception as e:
            print(f"Nelder-Mead失败: {e}")
        print("运行Hyperopt...")
        start_time = time.time()
        try:
            hyperopt_space = {
                'xi': hp.uniform('xi', 0.01, 0.5),
                'dt': hp.uniform('dt', 0.01, 0.3)
            }
            trials = Trials()
            best_hyperopt = fmin(
                fn=hyperopt_obj,
                space=hyperopt_space,
                algo=tpe.suggest,
                max_evals=n_calls,
                trials=trials,
                rstate=np.random.default_rng(42)
            )
            hyperopt_time = time.time() - start_time
            best_trial = trials.best_trial
            best_xi_hyperopt = best_trial['misc']['vals']['xi'][0]
            best_dt_hyperopt = best_trial['misc']['vals']['dt'][0]
            best_score_hyperopt = best_trial['result']['loss']
            print(f"Hyperopt: xi={best_xi_hyperopt:.4f}, dt={best_dt_hyperopt:.4f}, "
                  f"最优值={best_score_hyperopt:.4f}, 耗时={hyperopt_time:.2f}s")
            method_results['params']['Hyperopt'] = (best_xi_hyperopt, best_dt_hyperopt)
            method_results['scores']['Hyperopt'] = best_score_hyperopt
            method_results['times']['Hyperopt'] = hyperopt_time
            with open(os.path.join(method_dir, f'hyperopt_trials_{method}.json'), 'w') as f:
                json.dump(trials.vals, f, indent=4)
        except Exception as e:
            print(f"Hyperopt失败: {e}")
        print("运行量子遗传算法(QGA)...")
        start_time = time.time()
        try:
            # 量子遗传算法
            qga = QuantumGeneticAlgorithm(obj_func, bounds, n_calls, pop_size=10, n_qubits=8)
            best_params_qga, best_score_qga = qga.optimize()
            qga_time = time.time() - start_time
            best_xi_qga, best_dt_qga = best_params_qga
            # 取负号，保证score为正，与其他优化器一致
            best_score_qga = -best_score_qga
            print(f"Quantum Genetic Algorithm: xi={best_xi_qga:.4f}, dt={best_dt_qga:.4f}, "
                  f"最优值={best_score_qga:.4f}, 耗时={qga_time:.2f}s")
            method_results['params']['QGA'] = (float(best_xi_qga), float(best_dt_qga))
            method_results['scores']['QGA'] = float(best_score_qga)
            method_results['times']['QGA'] = qga_time
        except Exception as e:
            print(f"量子遗传算法失败: {e}")
            import traceback
            traceback.print_exc()
        print("\n运行最优参数并记录能量历史...")
        for opt_name in opt_methods:
            if opt_name in method_results['params']:
                xi, dt = method_results['params'][opt_name]
                try:
                    solver = QUBOSolverBatch(J_batch, n_vars, batch_size, xi=xi, dt=dt, n_iter=n_iter, method=method, device='cpu')
                    _, _ = solver.solve()
                    energy_history = [float(np.mean(e)) for e in solver.energy_history[0]]
                    method_results['energy_histories'][opt_name] = energy_history
                except Exception as e:
                    print(f"使用{opt_name}最优参数运行时出错: {e}")
        all_results[method] = convert_to_python_types(method_results)
        print(f"绘制 {method} 方法的能量历史...")
        plt.figure(figsize=(12, 8))
        for opt_name, history in method_results['energy_histories'].items():
            xi, dt = method_results['params'][opt_name]
            plt.plot(history, label=f'{opt_name} (xi={xi:.3f}, dt={dt:.3f})', linewidth=2, alpha=0.8)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Energy', fontsize=12)
        plt.title(f'{method} Method: Energy History Comparison (n_vars={n_vars})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(method_dir, f'{method}_energy_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    with open(os.path.join(result_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4, cls=NumpyEncoder)
    print("\n所有优化完成！结果已保存到", result_dir)