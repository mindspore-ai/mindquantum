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


def generate_hubo_terms(n_vars, max_order=8, density=0.15):
    terms_dict = {}
    for order in range(1, max_order+1):
        n_terms = int(density * n_vars ** order)
        terms = []
        for _ in range(n_terms):
            coef = np.random.randint(-10,10)
            vars_list = random.sample(range(n_vars), order)
            terms.append((coef, vars_list))
        if terms:
            terms_dict[order] = terms
    return terms_dict


def objective(params, terms_dict, n_vars, n_iter, method_name):
    xi, dt = params
    term_orders, term_coeffs, term_indices = terms_dict_to_tensor(terms_dict, n_vars)
    solver = HUBOSolverBatch(term_orders, term_coeffs, term_indices, n_vars, 1, xi=xi, dt=dt, n_iter=n_iter, method=method_name, device='cpu')
    _, best_energy = solver.solve()
    # 返回正的能量值
    return float(best_energy[0].item())


def hyperopt_objective(params, terms_dict, n_vars, n_iter, method_name):
    xi = params['xi']
    dt = params['dt']
    return objective((xi, dt), terms_dict, n_vars, n_iter, method_name)


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
        print(f"新版CMA-ES边界设置失败，尝试旧版设置: {e}")
        try:
            es = cma.CMAEvolutionStrategy(x0, 0.1, {
                'bounds': [0.01, 0.5, 0.01, 0.3],
                'maxfevals': n_calls,
                'verbose': -1
            })
            return es
        except Exception as e2:
            print(f"旧版CMA-ES边界设置也失败: {e2}")
            print("尝试无边界CMA-ES")
            return cma.CMAEvolutionStrategy(x0, 0.1, {
                'maxfevals': n_calls,
                'verbose': -1
            })


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


# ===== 来自 two_tensor.py 的实现 =====
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
    all_terms = []
    for order, terms in terms_dict.items():
        for coef, vars_list in terms:
            all_terms.append((order, coef, vars_list))
    n_terms = len(all_terms)
    max_order = max([len(t[2]) for t in all_terms])
    term_orders = torch.tensor([len(t[2]) for t in all_terms], dtype=torch.long)
    term_coeffs = torch.tensor([t[1] for t in all_terms], dtype=torch.float32)
    term_indices = -torch.ones((n_terms, max_order), dtype=torch.long)
    for i, (_, _, vars_list) in enumerate(all_terms):
        term_indices[i, :len(vars_list)] = torch.tensor(vars_list, dtype=torch.long)
    return term_orders, term_coeffs, term_indices


if __name__ == "__main__":
    methods = ['bSB', 'dSB', 'CAC', 'CFC']
    opt_methods = ['Bayes', 'SA', 'CMA-ES', 'Nelder-Mead', 'Hyperopt', 'QGA']
    n_vars = 10
    n_iter = 500
    max_order = 3
    density = 0.15
    n_calls = 10
    result_dir = './two/result_two_end2'
    os.makedirs(result_dir, exist_ok=True)
    all_results = {}
    terms_dict = generate_hubo_terms(n_vars, max_order=max_order, density=density)
    with open(os.path.join(result_dir, 'hubo_problem.json'), 'w') as f:
        json.dump(convert_to_python_types(terms_dict), f, indent=4, cls=NumpyEncoder)
    
    for method in methods:
        print(f"\n=== 优化方法: {method} ===")
        method_dir = os.path.join(result_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        obj_func = partial(objective, terms_dict=terms_dict, n_vars=n_vars, n_iter=n_iter, method_name=method)
        hyperopt_obj = partial(hyperopt_objective, terms_dict=terms_dict, n_vars=n_vars, n_iter=n_iter, method_name=method)
        space = [(0.01, 0.5), (0.01, 0.3)]
        bounds = [(0.01, 0.5), (0.01, 0.3)]
        x0 = [0.1, 0.1]
        method_results = {'params': {}, 'scores': {}, 'times': {}, 'energy_histories': {}}
        
        # 定义所有优化方法的执行函数
        opt_functions = {
            'Bayes': lambda: gp_minimize(obj_func, space, n_calls=n_calls, random_state=42),
            'SA': lambda: dual_annealing(obj_func, bounds, maxiter=n_calls, seed=42),
            'CMA-ES': lambda: create_cma_es(x0, n_calls).optimize(obj_func),
            'Nelder-Mead': lambda: minimize(obj_func, x0, method='Nelder-Mead', bounds=bounds, options={'maxfev': n_calls}),
            'Hyperopt': lambda: fmin(
                fn=hyperopt_obj,
                space={'xi': hp.uniform('xi', 0.01, 0.5), 'dt': hp.uniform('dt', 0.01, 0.3)},
                algo=tpe.suggest,
                max_evals=n_calls,
                trials=Trials(),
                rstate=np.random.default_rng(42)
            ),
            'QGA': lambda: QuantumGeneticAlgorithm(obj_func, bounds, n_calls).optimize()
        }
        
        # 执行所有优化方法
        for opt_name in opt_methods:
            print(f"运行{opt_name}...")
            start_time = time.time()
            try:
                if opt_name == 'Bayes':
                    res = opt_functions[opt_name]()
                    best_xi, best_dt = res.x
                    best_score = res.fun
                    method_results['params'][opt_name] = (float(best_xi), float(best_dt))
                    method_results['scores'][opt_name] = float(best_score)
                    method_results['times'][opt_name] = time.time() - start_time
                    print(f"{opt_name}: xi={best_xi:.4f}, dt={best_dt:.4f}, 最佳值={best_score:.4f}, 耗时={method_results['times'][opt_name]:.2f}s")
                
                elif opt_name == 'SA':
                    res = opt_functions[opt_name]()
                    best_xi, best_dt = res.x
                    best_score = res.fun
                    method_results['params'][opt_name] = (float(best_xi), float(best_dt))
                    method_results['scores'][opt_name] = float(best_score)
                    method_results['times'][opt_name] = time.time() - start_time
                    print(f"{opt_name}: xi={best_xi:.4f}, dt={best_dt:.4f}, 最佳值={best_score:.4f}, 耗时={method_results['times'][opt_name]:.2f}s")
                
                elif opt_name == 'CMA-ES':
                    es = opt_functions[opt_name]()
                    best_xi, best_dt = es.result.xbest
                    best_score = es.result.fbest
                    method_results['params'][opt_name] = (float(best_xi), float(best_dt))
                    method_results['scores'][opt_name] = float(best_score)
                    method_results['times'][opt_name] = time.time() - start_time
                    print(f"{opt_name}: xi={best_xi:.4f}, dt={best_dt:.4f}, 最佳值={best_score:.4f}, 耗时={method_results['times'][opt_name]:.2f}s")
                
                elif opt_name == 'Nelder-Mead':
                    res = opt_functions[opt_name]()
                    best_xi, best_dt = res.x
                    best_score = res.fun
                    method_results['params'][opt_name] = (float(best_xi), float(best_dt))
                    method_results['scores'][opt_name] = float(best_score)
                    method_results['times'][opt_name] = time.time() - start_time
                    print(f"{opt_name}: xi={best_xi:.4f}, dt={best_dt:.4f}, 最佳值={best_score:.4f}, 耗时={method_results['times'][opt_name]:.2f}s")
                
                elif opt_name == 'Hyperopt':
                    trials = Trials()
                    best_hyperopt = fmin(
                        fn=hyperopt_obj,
                        space={'xi': hp.uniform('xi', 0.01, 0.5), 'dt': hp.uniform('dt', 0.01, 0.3)},
                        algo=tpe.suggest,
                        max_evals=n_calls,
                        trials=trials,
                        rstate=np.random.default_rng(42)
                    )
                    best_trial = trials.best_trial
                    best_xi = best_trial['misc']['vals']['xi'][0]
                    best_dt = best_trial['misc']['vals']['dt'][0]
                    best_score = best_trial['result']['loss']
                    method_results['params'][opt_name] = (float(best_xi), float(best_dt))
                    method_results['scores'][opt_name] = float(best_score)
                    method_results['times'][opt_name] = time.time() - start_time
                    print(f"{opt_name}: xi={best_xi:.4f}, dt={best_dt:.4f}, 最佳值={best_score:.4f}, 耗时={method_results['times'][opt_name]:.2f}s")
                
                elif opt_name == 'QGA':
                    best_params, best_score = opt_functions[opt_name]()
                    best_xi, best_dt = best_params
                    method_results['params'][opt_name] = (float(best_xi), float(best_dt))
                    method_results['scores'][opt_name] = float(best_score)
                    method_results['times'][opt_name] = time.time() - start_time
                    print(f"{opt_name}: xi={best_xi:.4f}, dt={best_dt:.4f}, 最佳值={best_score:.4f}, 耗时={method_results['times'][opt_name]:.2f}s")
                
            except Exception as e:
                print(f"{opt_name}失败: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n运行最优参数并记录能量历史...")
        for opt_name in opt_methods:
            if opt_name in method_results['params']:
                xi, dt = method_results['params'][opt_name]
                try:
                    term_orders, term_coeffs, term_indices = terms_dict_to_tensor(terms_dict, n_vars)
                    solver = HUBOSolverBatch(term_orders, term_coeffs, term_indices, n_vars, 1, xi=xi, dt=dt, n_iter=n_iter, method=method, device='cpu')
                    _, _ = solver.solve()
                    energy_history = [float(e[0]) for e in solver.energy_history[0]]
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