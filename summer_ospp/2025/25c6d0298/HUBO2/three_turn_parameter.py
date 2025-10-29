import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from skopt import gp_minimize
from scipy.optimize import dual_annealing, minimize
import cma
import time
from functools import partial
import json
import matplotlib as mpl
from json import JSONEncoder
from hyperopt import fmin, tpe, hp, Trials
import random
import warnings
from three_base import generate_hubo_terms, square_hubo_terms
from tqdm import tqdm
import math

mpl.use('Agg')

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
    
    def optimize(self, pbar=None):
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
            
            # 更新进度条
            if pbar is not None:
                pbar.update(self.pop_size)
                pbar.set_postfix({
                    'xi': f'{self.best_solution[0]:.3f}',
                    'dt': f'{self.best_solution[1]:.3f}',
                    'energy': f'{self.best_fitness:.4f}'
                })
        
        return self.best_solution, self.best_fitness

class HUBOSolverBatch:
    def __init__(self, term_orders, term_coeffs, term_indices, n_vars, batch_size, 
                 xi=0.3, dt=0.15, n_iter=1000, method='dSB', device='cpu'):
        """
        批处理版本的HUBO求解器，用于参数优化
        
        参数:
        term_orders: 项阶数张量
        term_coeffs: 项系数张量
        term_indices: 项索引张量
        n_vars: 变量个数
        batch_size: 批处理大小
        xi: 学习率
        dt: 时间步长
        n_iter: 迭代次数
        method: 求解方法 ('bSB', 'dSB', 'CAC', 'CFC')
        device: 计算设备 ('cpu' or 'cuda')
        """
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
        
        # 初始化变量
        self.x = 0.01 * torch.randn(batch_size, n_vars, device=device)
        self.y = 0.01 * torch.randn(batch_size, n_vars, device=device)
        
        # 固定参数
        self.Delta = 0.5
        
        # 用于记录历史
        self.energy_history = []  # 每次迭代的能量值

    def compute_energy(self, x):
        """计算HUBO能量函数（批处理版本）"""
        energy = torch.zeros(self.batch_size, device=self.device)
        
        # 遍历所有项
        for i in range(len(self.term_orders)):
            order = self.term_orders[i]
            coef = self.term_coeffs[i]
            indices = self.term_indices[i, :order]
            
            # 计算该项的乘积
            prod = torch.ones(self.batch_size, device=self.device)
            for j in range(order):
                var_idx = indices[j]
                prod *= x[:, var_idx]
            
            energy += coef * prod
        
        return energy

    def solve(self):
        """求解HUBO问题（批处理版本）"""
        self.energy_history = []
        
        for iter in range(self.n_iter):
            p_t = iter / self.n_iter  # 时间参数
            
            # 创建需要梯度的变量
            x_torch = self.x.clone().detach().requires_grad_(True)
            
            # 计算能量
            energy = self.compute_energy(x_torch)
            
            # 记录当前能量值
            self.energy_history.append(energy.detach().cpu().numpy())
            
            # 计算梯度
            energy.backward(torch.ones_like(energy))
            grad_E = x_torch.grad.clone()
            x_torch.grad.zero_()
            
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
                    global_avg = torch.mean(self.x, dim=1, keepdim=True)  # 每批的平均值
                
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
        
        # 获取最终解和能量值
        with torch.no_grad():
            final_x = torch.sign(self.x)
            final_energy = self.compute_energy(final_x)
        
        return final_x, final_energy

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def terms_dict_to_tensor(terms_dict, n_vars):
    """将terms_dict转换为张量表示"""
    # 收集所有项
    all_terms = []
    for order, terms in terms_dict.items():
        for coef, vars_list in terms:
            all_terms.append((order, coef, vars_list))
    
    # 找到最大阶数
    max_order = max([order for order, _, _ in all_terms])
    
    # 创建张量
    n_terms = len(all_terms)
    term_orders = torch.zeros(n_terms, dtype=torch.long)
    term_coeffs = torch.zeros(n_terms)
    term_indices = torch.zeros(n_terms, max_order, dtype=torch.long)
    
    for i, (order, coef, vars_list) in enumerate(all_terms):
        term_orders[i] = order
        term_coeffs[i] = coef
        for j, var_idx in enumerate(vars_list):
            term_indices[i, j] = var_idx
    
    return term_orders, term_coeffs, term_indices

def objective(params, terms_dict, n_vars, n_iter, method_name, pbar=None):
    """目标函数，返回单个值"""
    xi, dt = params
    term_orders, term_coeffs, term_indices = terms_dict_to_tensor(terms_dict, n_vars)
    solver = HUBOSolverBatch(term_orders, term_coeffs, term_indices, n_vars, 1, 
                            xi=xi, dt=dt, n_iter=n_iter, method=method_name, device='cpu')
    _, best_energy = solver.solve()
    
    # 更新进度条
    if pbar is not None:
        pbar.update(1)
        pbar.set_postfix({
            'xi': f'{xi:.3f}',
            'dt': f'{dt:.3f}',
            'energy': f'{-best_energy[0].item():.4f}'
        })
    
    return -float(best_energy[0].item())

def hyperopt_objective(params, terms_dict, n_vars, n_iter, method_name, pbar=None):
    xi = params['xi']
    dt = params['dt']
    result = objective((xi, dt), terms_dict, n_vars, n_iter, method_name, pbar)
    
    # 更新进度条
    if pbar is not None:
        pbar.update(1)
        pbar.set_postfix({
            'xi': f'{xi:.3f}',
            'dt': f'{dt:.3f}',
            'energy': f'{result:.4f}'
        })
    
    return result

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

def optimize_bayes_for_method(obj_func, space, method, n_calls, n_initial_points, pbar=None):
    """专门为CFC和dSB方法优化的贝叶斯优化函数"""
    print(f"运行贝叶斯优化 for {method}...")
    start_time = time.time()
    
    try:
        # 根据不同方法选择不同的acquisition函数
        if method == 'CFC':
            acq_func = "EI"
            
        elif method == 'dSB':
            acq_func = "PI"
            
        elif method == 'bSB':
            acq_func = "EI"
            
        else:
            acq_func = "EI"
        
        # 运行贝叶斯优化
        res_bayes = gp_minimize(
            obj_func,
            space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            random_state=42
        )
        
        best_params = res_bayes.x
        best_score = -res_bayes.fun
        opt_time = time.time() - start_time
        
        print(f"Bayes: xi={best_params[0]:.4f}, dt={best_params[1]:.4f}, "
              f"最佳值={best_score:.4f}, 耗时={opt_time:.2f}s")
        
        return {
            'params': best_params,
            'score': best_score,
            'time': opt_time
        }
        
    except Exception as e:
        print(f"贝叶斯优化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def optimize_with_method(opt_name, obj_func, hyperopt_obj, space, bounds, x0, n_calls, method_dir, method, pbar=None):
    """通用优化函数，用于不同的优化方法"""
    method_results = {}
    print(f"运行{opt_name}优化...")
    start_time = time.time()
    
    try:
        if opt_name == 'Bayes':
            # 使用专门的贝叶斯优化函数
            n_initial_points = min(10, n_calls // 2)
            result = optimize_bayes_for_method(obj_func, space, method, n_calls, n_initial_points, pbar)
            if result is None:
                return None
                
            method_results['params'] = result['params']
            method_results['score'] = result['score']
            method_results['time'] = result['time']
            
        elif opt_name == 'SA':
            # 为SA创建进度条
            sa_pbar = tqdm(total=n_calls, desc=f"{method} - SA Progress", leave=False)
            
            def sa_callback(x, f, context):
                if context == 0:  # 只在每次函数评估后更新
                    sa_pbar.update(1)
                    sa_pbar.set_postfix({
                        'xi': f'{x[0]:.3f}',
                        'dt': f'{x[1]:.3f}',
                        'energy': f'{-f:.4f}'
                    })
            
            res_sa = dual_annealing(obj_func, bounds, maxiter=n_calls, seed=42, callback=sa_callback)
            sa_pbar.close()
            
            best_params = res_sa.x
            best_score = -res_sa.fun
            opt_time = time.time() - start_time
            
            method_results['params'] = best_params
            method_results['score'] = best_score
            method_results['time'] = opt_time
            
        elif opt_name == 'CMA-ES':
            # 为CMA-ES创建进度条
            cma_pbar = tqdm(total=n_calls, desc=f"{method} - CMA-ES Progress", leave=False)
            
            es = create_cma_es(x0, n_calls)
            
            # 自定义CMA-ES的优化函数，添加进度条
            while not es.stop() and es.result.evaluations < n_calls:
                solutions = es.ask()
                fitnesses = [obj_func(x) for x in solutions]
                es.tell(solutions, fitnesses)
                
                # 更新进度条
                cma_pbar.update(len(solutions))
                cma_pbar.set_postfix({
                    'xi': f'{es.result.xbest[0]:.3f}',
                    'dt': f'{es.result.xbest[1]:.3f}',
                    'energy': f'{-es.result.fbest:.4f}'
                })
            
            cma_pbar.close()
            
            best_params = es.result.xbest
            best_score = -es.result.fbest
            opt_time = time.time() - start_time
            
            method_results['params'] = best_params
            method_results['score'] = best_score
            method_results['time'] = opt_time
            
        elif opt_name == 'Nelder-Mead':
            # 为Nelder-Mead创建进度条
            nm_pbar = tqdm(total=n_calls, desc=f"{method} - Nelder-Mead Progress", leave=False)
            
            def nm_callback(x):
                nm_pbar.update(1)
                nm_pbar.set_postfix({
                    'xi': f'{x[0]:.3f}',
                    'dt': f'{x[1]:.3f}',
                    'energy': f'{obj_func(x):.4f}'
                })
            
            res_nm = minimize(obj_func, x0, method='Nelder-Mead', bounds=bounds, 
                             options={'maxfev': n_calls}, callback=nm_callback)
            nm_pbar.close()
            
            best_params = res_nm.x
            best_score = -res_nm.fun
            opt_time = time.time() - start_time
            
            method_results['params'] = best_params
            method_results['score'] = best_score
            method_results['time'] = opt_time
            
        elif opt_name == 'Hyperopt':
            # 为Hyperopt创建进度条
            hyperopt_pbar = tqdm(total=n_calls, desc=f"{method} - Hyperopt Progress", leave=False)
            
            hyperopt_space = {
                'xi': hp.uniform('xi', 0.01, 0.5),
                'dt': hp.uniform('dt', 0.01, 0.3)
            }
            
            # 包装目标函数以包含进度条
            def hyperopt_obj_with_pbar(params):
                result = hyperopt_obj(params)
                hyperopt_pbar.update(1)
                hyperopt_pbar.set_postfix({
                    'xi': f'{params["xi"]:.3f}',
                    'dt': f'{params["dt"]:.3f}',
                    'energy': f'{result:.4f}'
                })
                return result
            
            trials = Trials()
            best_hyperopt = fmin(
                fn=hyperopt_obj_with_pbar,
                space=hyperopt_space,
                algo=tpe.suggest,
                max_evals=n_calls,
                trials=trials,
                rstate=np.random.default_rng(42)
            )
            hyperopt_pbar.close()
            
            best_trial = trials.best_trial
            best_params = [
                best_trial['misc']['vals']['xi'][0],
                best_trial['misc']['vals']['dt'][0]
            ]
            best_score = best_trial['result']['loss']
            opt_time = time.time() - start_time
            
            method_results['params'] = best_params
            method_results['score'] = best_score
            method_results['time'] = opt_time
            
            # 保存Hyperopt trials
            with open(os.path.join(method_dir, f'hyperopt_trials_{method}.json'), 'w') as f:
                json.dump(trials.vals, f, indent=4)
        
        elif opt_name == 'QGA':
            # 为QGA创建进度条
            qga_pbar = tqdm(total=n_calls, desc=f"{method} - QGA Progress", leave=False)
            
            # 量子遗传算法
            qga = QuantumGeneticAlgorithm(obj_func, bounds, n_calls, pop_size=10, n_qubits=8)
            best_params_qga, best_score_qga = qga.optimize(pbar=qga_pbar)
            qga_pbar.close()
            
            best_params = best_params_qga
            best_score = best_score_qga
            opt_time = time.time() - start_time
            
            method_results['params'] = best_params
            method_results['score'] = best_score
            method_results['time'] = opt_time
        
        print(f"{opt_name}: xi={method_results['params'][0]:.4f}, dt={method_results['params'][1]:.4f}, "
              f"最佳值={method_results['score']:.4f}, 耗时={method_results['time']:.2f}s")
        
    except Exception as e:
        print(f"{opt_name}失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return method_results

if __name__ == "__main__":
    methods = ['bSB', 'dSB', 'CAC', 'CFC']
    opt_methods = ['Bayes', 'SA', 'CMA-ES', 'Nelder-Mead', 'Hyperopt', 'QGA']  # 添加QGA
    n_vars = 10
    n_iter = 300
    max_order = 3
    density = 0.1
    n_calls = 10
    result_dir = 'three/results_three_end2'
    os.makedirs(result_dir, exist_ok=True)
    all_results = {}
    
    # 生成HUBO问题
    terms_dict = generate_hubo_terms(n_vars, max_order=max_order, density=density)
    with open(os.path.join(result_dir, 'hubo_problem.json'), 'w') as f:
        json.dump(convert_to_python_types(terms_dict), f, indent=4, cls=NumpyEncoder)
    
    # 主进度条
    main_pbar = tqdm(total=len(methods) * len(opt_methods), desc="Overall Progress")
    
    for method in methods:
        print(f"\n=== 优化方法: {method} ===")
        method_dir = os.path.join(result_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        
        # 创建目标函数
        obj_func = partial(objective, terms_dict=terms_dict, n_vars=n_vars, n_iter=n_iter, method_name=method)
        hyperopt_obj_partial = partial(hyperopt_objective, terms_dict=terms_dict, n_vars=n_vars, n_iter=n_iter, method_name=method)
        
        # 定义参数空间
        space = [(0.01, 0.5), (0.01, 0.3)]
        bounds = [(0.01, 0.5), (0.01, 0.3)]
        x0 = [0.1, 0.1]
        
        method_results = {'params': {}, 'scores': {}, 'times': {}, 'energy_histories': {}}
        
        # 对每种优化方法进行优化
        for opt_name in opt_methods:
            # 为每个优化方法创建进度条
            opt_pbar = tqdm(total=n_calls, desc=f"{method} - {opt_name} Progress", leave=False)
            
            # 包装目标函数以包含进度条
            obj_func_with_pbar = partial(obj_func, pbar=opt_pbar)
            hyperopt_obj_with_pbar = partial(hyperopt_obj_partial, pbar=opt_pbar)
            
            result = optimize_with_method(
                opt_name, obj_func_with_pbar, hyperopt_obj_with_pbar, space, bounds, x0, n_calls, method_dir, method, opt_pbar
            )
            
            opt_pbar.close()
            
            if result is not None:
                method_results['params'][opt_name] = result['params']
                method_results['scores'][opt_name] = result['score']
                method_results['times'][opt_name] = result['time']
            
            # 更新主进度条
            main_pbar.update(1)
            main_pbar.set_postfix({
                'method': method,
                'optimizer': opt_name
            })
        
        print("\n运行最优参数并记录能量历史...")
        for opt_name in opt_methods:
            if opt_name in method_results['params']:
                xi, dt = method_results['params'][opt_name]
                try:
                    term_orders, term_coeffs, term_indices = terms_dict_to_tensor(terms_dict, n_vars)
                    solver = HUBOSolverBatch(
                        term_orders, term_coeffs, term_indices, n_vars, 1, 
                        xi=xi, dt=dt, n_iter=n_iter, method=method, device='cpu'
                    )
                    _, _ = solver.solve()
                    
                    # 处理能量历史 - 修复这里的问题
                    energy_history = []
                    if hasattr(solver, 'energy_history') and solver.energy_history:
                        # 确保我们有正确的数据格式
                        for energy_batch in solver.energy_history:
                            if isinstance(energy_batch, np.ndarray) and energy_batch.size > 0:
                                # 取第一个批次的值（因为我们只有一个批次）
                                energy_value = energy_batch[0] if energy_batch.ndim > 0 else energy_batch
                                energy_history.append(float(energy_value))
                            else:
                                energy_history.append(float(energy_batch))
                    
                    method_results['energy_histories'][opt_name] = energy_history
                    print(f"{opt_name} 能量历史长度: {len(energy_history)}")
                except Exception as e:
                    print(f"使用{opt_name}最优参数运行时出错: {e}")
                    import traceback
                    traceback.print_exc()
        
        all_results[method] = convert_to_python_types(method_results)
        
        print(f"绘制 {method} 方法的能量历史...")
        plt.figure(figsize=(12, 8))
        
        # 检查是否有能量历史数据
        has_data = False
        for opt_name, history in method_results['energy_histories'].items():
            if opt_name in method_results['params'] and history:
                xi, dt = method_results['params'][opt_name]
                plt.plot(history, label=f'{opt_name} (xi={xi:.3f}, dt={dt:.3f})', linewidth=2, alpha=0.8)
                has_data = True
                print(f"{opt_name} 有 {len(history)} 个数据点")
        
        if not has_data:
            print(f"警告: {method} 方法没有能量历史数据")
            plt.text(0.5, 0.5, 'No energy history data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=14)
        else:
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Energy', fontsize=12)
            plt.title(f'{method} Method: Energy History Comparison (n_vars={n_vars})', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(method_dir, f'{method}_energy_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    main_pbar.close()
    
    # 保存所有结果
    with open(os.path.join(result_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4, cls=NumpyEncoder)
    
    print("\n所有优化完成！结果已保存到", result_dir)
    
    # 绘制最终比较图
    print("绘制最终比较图...")
    plt.figure(figsize=(14, 10))
    
    # 为每个方法创建一个子图
    for i, method in enumerate(methods):
        plt.subplot(2, 2, i+1)
        
        # 检查是否有能量历史数据
        has_data = False
        if method in all_results and 'energy_histories' in all_results[method]:
            for opt_name, history in all_results[method]['energy_histories'].items():
                if history and opt_name in all_results[method]['params']:
                    params = all_results[method]['params'][opt_name]
                    xi, dt = params[0], params[1]
                    plt.plot(history, label=f'{opt_name} (xi={xi:.3f}, dt={dt:.3f})', linewidth=2, alpha=0.8)
                    has_data = True
        
        if not has_data:
            plt.text(0.5, 0.5, 'No energy history data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12)
        else:
            plt.xlabel('Iteration')
            plt.ylabel('Energy')
            plt.title(f'{method} Method')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'final_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("最终比较图已保存")