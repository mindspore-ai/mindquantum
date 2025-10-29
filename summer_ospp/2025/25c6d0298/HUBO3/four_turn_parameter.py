import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from skopt import gp_minimize
from scipy.optimize import dual_annealing, minimize
import cma
from four_base import HUBOSolver, generate_hubo_terms, square_hubo_terms, combine_hubo_terms, estimate_lambda_range
import time
from functools import partial
import json
import matplotlib as mpl
from json import JSONEncoder
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 设置matplotlib使用非交互式后端（适用于无显示环境）
mpl.use('Agg')

# 自定义JSON编码器处理NumPy数据类型
class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# 量子遗传算法 (QGA) 实现
class QuantumGeneticAlgorithm:
    def __init__(self, objective_func, bounds, n_population=10, n_generations=10, theta=0.05*np.pi):
        self.objective_func = objective_func
        self.bounds = bounds
        self.n_population = n_population
        self.n_generations = n_generations
        self.theta = theta  # 旋转角
        self.n_params = len(bounds)
        
        # 每个参数使用10位，所以总比特数为n_params * 10
        self.n_bits = self.n_params * 10
        
        # 初始化量子种群 (每个量子比特用一对(α, β)表示)
        self.population = np.ones((n_population, 2 * self.n_bits)) * (1/np.sqrt(2))
        
    def measure(self, individual):
        """测量量子个体得到经典二进制表示"""
        prob_0 = individual[::2]**2  # |α|²
        # 确保生成正确数量的比特
        bits = (np.random.random(self.n_bits) > prob_0[:self.n_bits]).astype(int)
        return bits
    
    def decode(self, bits):
        """将二进制表示解码为实际参数值"""
        params = []
        for i, (lower, upper) in enumerate(self.bounds):
            # 确定每个参数使用的比特数
            bits_per_param = 10
            start_idx = i * bits_per_param
            end_idx = (i + 1) * bits_per_param
            
            # 确保不超出bits的范围
            if end_idx > len(bits):
                # 如果分配的比特数不够，使用所有剩余的比特
                param_bits = bits[start_idx:]
            else:
                param_bits = bits[start_idx:end_idx]
            
            # 将二进制转换为十进制
            if len(param_bits) == 0:
                # 如果没有比特，使用范围的中点
                param = (lower + upper) / 2
            else:
                decimal = sum(bit * (2 ** j) for j, bit in enumerate(param_bits))
                max_value = (2 ** len(param_bits)) - 1
                if max_value == 0:
                    param = (lower + upper) / 2
                else:
                    param = lower + (upper - lower) * decimal / max_value
            params.append(param)
        return params
    
    def update_quantum(self, individual, best_individual, bits, best_bits):
        """更新量子个体 using quantum rotation gate"""
        new_individual = individual.copy()
        n_bits = len(bits)
        
        for i in range(n_bits):
            # 确定旋转方向
            if bits[i] == 0 and best_bits[i] == 1:
                # 顺时针旋转
                angle = self.theta
            elif bits[i] == 1 and best_bits[i] == 0:
                # 逆时针旋转
                angle = -self.theta
            else:
                # 不旋转
                angle = 0
            
            # 应用量子旋转门
            if angle != 0:
                idx = i * 2
                # 确保不超出individual的范围
                if idx + 1 < len(individual):
                    alpha, beta = individual[idx], individual[idx+1]
                    new_alpha = alpha * np.cos(angle) - beta * np.sin(angle)
                    new_beta = alpha * np.sin(angle) + beta * np.cos(angle)
                    new_individual[idx] = new_alpha
                    new_individual[idx+1] = new_beta
        
        return new_individual
    
    def optimize(self):
        """运行量子遗传算法优化"""
        best_fitness = float('-inf')
        best_params = None
        best_bits = None
        
        # 主循环
        for generation in range(self.n_generations):
            # 测量种群得到经典表示
            classical_pop = []
            for individual in self.population:
                bits = self.measure(individual)
                classical_pop.append(bits)
            
            # 评估适应度
            fitnesses = []
            params_list = []
            for bits in classical_pop:
                params = self.decode(bits)
                params_list.append(params)
                fitness = -self.objective_func(params)  # 负号是因为我们的目标函数返回负能量
                fitnesses.append(fitness)
            
            # 更新最佳个体
            max_fitness = max(fitnesses)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_idx = np.argmax(fitnesses)
                best_params = params_list[best_idx]
                best_bits = classical_pop[best_idx]
            
            # 更新量子种群
            for i, individual in enumerate(self.population):
                self.population[i] = self.update_quantum(
                    individual, self.population[best_idx], 
                    classical_pop[i], best_bits
                )
        
        return best_params, best_fitness

# 优化目标函数：给定参数，返回最后能量的负值（最大化问题取负）
def objective(params, n_vars, n_iter, method_name, density, problem_cache=None):
    xi, dt, lambda_val = params
    try:
        # 如果提供了问题缓存，使用缓存的问题
        if problem_cache is not None:
            num_terms, den_terms = problem_cache
        else:
            # 生成分子和分母项
            num_terms = generate_hubo_terms(n_vars, max_order=3, density=density)
            den_terms = generate_hubo_terms(n_vars, max_order=3, density=density)
        
        # 将分子和分母项平方
        num_squared = square_hubo_terms(num_terms)
        den_squared = square_hubo_terms(den_terms)
        
        # 构造y4 = A² - λB²
        y4_terms = combine_hubo_terms(num_squared, den_squared, lambda_val)
        
        # 调参功能只运行一次
        solver = HUBOSolver(y4_terms, n_vars, xi=xi, dt=dt, n_iter=n_iter, method=method_name, device='cpu', runs=1)
        _, _ = solver.solve()
        return -solver.energy_history[-1][-1]
    except Exception as e:
        print(f"Error during solver execution: {e}")
        return 1000  # 返回一个较大的值表示失败

# 批量目标函数评估
def batch_objective(params_list, n_vars, n_iter, method_name, density, problem_cache=None):
    """批量评估目标函数，使用并行处理"""
    results = []
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), len(params_list))) as executor:
        # 提交所有任务
        futures = {
            executor.submit(objective, params, n_vars, n_iter, method_name, density, problem_cache): i
            for i, params in enumerate(params_list)
        }
        
        # 收集结果
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                print(f"Task failed: {e}")
                results.append((idx, 1000))  # 失败时返回大值
    
    # 按原始顺序排序结果
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]

# Hyperopt的目标函数包装器
def hyperopt_objective(params, n_vars, n_iter, method_name, density, problem_cache=None):
    xi = params['xi']
    dt = params['dt']
    lambda_val = params['lambda_val']
    # 调用原始目标函数
    return objective((xi, dt, lambda_val), n_vars, n_iter, method_name, density, problem_cache)

# 转换NumPy类型为Python原生类型
def convert_to_python_types(data):
    if isinstance(data, dict):
        return {k: convert_to_python_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_python_types(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_python_types(item) for item in data)
    elif isinstance(data, np.generic):
        return data.item()  # 将NumPy标量转换为Python原生类型
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

# 修复CMA-ES边界设置问题
def create_cma_es(x0, n_calls):
    """创建并配置CMA-ES优化器，处理边界设置问题"""
    try:
        # 新版本的边界设置方式
        es = cma.CMAEvolutionStrategy(x0, 0.1, {
            'bounds': [[0.01, 0.01, 0.5], [0.5, 0.3, 1.5]],
            'maxfevals': n_calls,
            'verbose': -1
        })
        return es
    except Exception as e:
        print(f"新版CMA-ES边界设置失败，尝试旧版设置: {e}")
        try:
            # 旧版本的边界设置方式
            es = cma.CMAEvolutionStrategy(x0, 0.1, {
                'bounds': [0.01, 0.5, 0.01, 0.3, 0.5, 1.5],  # [xi_min, xi_max, dt_min, dt_max, lambda_min, lambda_max]
                'maxfevals': n_calls,
                'verbose': -1
            })
            return es
        except Exception as e2:
            print(f"旧版CMA-ES边界设置也失败: {e2}")
            # 最后尝试无边界设置
            print("尝试无边界CMA-ES")
            return cma.CMAEvolutionStrategy(x0, 0.1, {
                'maxfevals': n_calls,
                'verbose': -1
            })

def run_hyperparameter_tuning():
    """运行超参数调优功能"""
    methods =['bSB','dSB','CFC','CAC']
    opt_methods = ['Bayes', 'SA', 'CMA-ES', 'Nelder-Mead', 'Hyperopt', 'QGA']
    
    n_vars = 10
    n_iter = 200
    density = 0.05
    n_calls = 10  # 每种优化方法的评估次数
    
    # 创建结果目录
    result_dir = './four/result_four_end2'
    os.makedirs(result_dir, exist_ok=True)
    
    # 存储所有结果
    all_results = {}
    
    # 预生成问题缓存，确保所有方法评估相同的问题
    problem_cache = None
    # 生成一次问题并缓存
    num_terms = generate_hubo_terms(n_vars, max_order=3, density=density)
    den_terms = generate_hubo_terms(n_vars, max_order=3, density=density)
    problem_cache = (num_terms, den_terms)

    # 自动估算lambda区间
    print("自动估算lambda区间...")
    lambda_min, lambda_max = estimate_lambda_range(num_terms, den_terms, n_vars)
    print(f"自动推导的lambda区间: [{lambda_min:.4f}, {lambda_max:.4f}]")

    for method in methods:
        print(f"\n=== 优化方法: {method} ===")
        method_dir = os.path.join(result_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # 创建特定于方法的objective函数（使用问题缓存）
        obj_func = partial(objective, n_vars=n_vars, n_iter=n_iter, 
                          method_name=method, density=density, problem_cache=problem_cache)

        # 创建批量评估函数
        batch_obj_func = partial(batch_objective, n_vars=n_vars, n_iter=n_iter,
                               method_name=method, density=density, problem_cache=problem_cache)

        # 创建Hyperopt的目标函数
        hyperopt_obj = partial(hyperopt_objective, n_vars=n_vars, 
                              n_iter=n_iter, method_name=method, density=density, problem_cache=problem_cache)

        # 参数范围 (xi, dt, lambda)
        space = [(0.01, 0.5), (0.01, 0.3), (lambda_min, lambda_max)]  
        bounds = [(0.01, 0.5), (0.01, 0.3), (lambda_min, lambda_max)]
        x0 = [0.1, 0.1, (lambda_min + lambda_max) / 2]  # 初始点 (xi, dt, lambda)

        method_results = {
            'params': {},
            'scores': {},
            'times': {},
            'energy_histories': {},
            'lambda_range': [lambda_min, lambda_max]  # 保存lambda区间
        }

        # 1. 贝叶斯优化
        print("运行贝叶斯优化...")
        # 根据method选择acq_func
        if method in ["bSB", "dSB"]:
            acq_func = "LCB"
        elif method == "CAC":
            acq_func = "EI"
        elif method == "CFC":
            acq_func = "PI"
        else:
            acq_func = "LCB"  # 默认
        start_time = time.time()
        try:
            res_bayes = gp_minimize(
                obj_func,
                space,
                n_calls=40,
                n_initial_points=30,
                acq_func=acq_func,
                random_state=42
            )
            bayes_time = time.time() - start_time
            best_xi_bayes, best_dt_bayes, best_lambda_bayes = res_bayes.x
            best_score_bayes = -res_bayes.fun
            print(f"Bayesian Optimization: xi={best_xi_bayes:.4f}, dt={best_dt_bayes:.4f}, lambda={best_lambda_bayes:.4f}, "
                f"最佳值={best_score_bayes:.4f}, 耗时={bayes_time:.2f}s")
            method_results['params']['Bayes'] = (best_xi_bayes, best_dt_bayes, best_lambda_bayes)
            method_results['scores']['Bayes'] = best_score_bayes
            method_results['times']['Bayes'] = bayes_time
        except Exception as e:
            print(f"贝叶斯优化失败: {e}")
        
        # 2. 模拟退火
        print("运行模拟退火...")
        start_time = time.time()
        try:
            res_sa = dual_annealing(obj_func, bounds, maxiter=n_calls, seed=42)
            sa_time = time.time() - start_time
            best_xi_sa, best_dt_sa, best_lambda_sa = res_sa.x
            best_score_sa = -res_sa.fun
            print(f"Simulated Annealing: xi={best_xi_sa:.4f}, dt={best_dt_sa:.4f}, lambda={best_lambda_sa:.4f}, "
                f"最佳值={best_score_sa:.4f}, 耗时={sa_time:.2f}s")
            
            method_results['params']['SA'] = (best_xi_sa, best_dt_sa, best_lambda_sa)
            method_results['scores']['SA'] = best_score_sa
            method_results['times']['SA'] = sa_time
        except Exception as e:
            print(f"模拟退火失败: {e}")
        
        # 3. CMA-ES优化 - 使用修复后的边界设置
        print("运行CMA-ES...")
        start_time = time.time()
        try:
            es = create_cma_es(x0, n_calls)
            es.optimize(obj_func)
            cma_time = time.time() - start_time
            best_xi_cma, best_dt_cma, best_lambda_cma = es.result.xbest
            best_score_cma = -es.result.fbest
            print(f"CMA-ES: xi={best_xi_cma:.4f}, dt={best_dt_cma:.4f}, lambda={best_lambda_cma:.4f}, "
                f"最佳值={best_score_cma:.4f}, 耗时={cma_time:.2f}s")
            
            method_results['params']['CMA-ES'] = (best_xi_cma, best_dt_cma, best_lambda_cma)
            method_results['scores']['CMA-ES'] = best_score_cma
            method_results['times']['CMA-ES'] = cma_time
        except Exception as e:
            print(f"CMA-ES失败: {e}")
        
        # 4. Nelder-Mead优化
        print("运行Nelder-Mead...")
        start_time = time.time()
        try:
            res_nm = minimize(obj_func, x0, method='Nelder-Mead', 
                            bounds=bounds, options={'maxfev': n_calls})
            nm_time = time.time() - start_time
            best_xi_nm, best_dt_nm, best_lambda_nm = res_nm.x
            best_score_nm = -res_nm.fun
            print(f"Nelder-Mead: xi={best_xi_nm:.4f}, dt={best_dt_nm:.4f}, lambda={best_lambda_nm:.4f}, "
                f"最佳值={best_score_nm:.4f}, 耗时={nm_time:.2f}s")
            
            method_results['params']['Nelder-Mead'] = (best_xi_nm, best_dt_nm, best_lambda_nm)
            method_results['scores']['Nelder-Mead'] = best_score_nm
            method_results['times']['Nelder-Mead'] = nm_time
        except Exception as e:
            print(f"Nelder-Mead失败: {e}")
        
        # 5. Hyperopt优化
        print("运行Hyperopt...")
        start_time = time.time()
        try:
            # 定义Hyperopt的搜索空间 (lambda区间用动态的lambda_min和lambda_max)
            hyperopt_space = {
                'xi': hp.uniform('xi', 0.01, 0.5),
                'dt': hp.uniform('dt', 0.01, 0.3),
                'lambda_val': hp.uniform('lambda_val', lambda_min, lambda_max)
            }
            # 创建Trials对象记录优化过程
            trials = Trials()
            # 运行Hyperopt优化
            best_hyperopt = fmin(
                fn=hyperopt_obj,
                space=hyperopt_space,
                algo=tpe.suggest,
                max_evals=n_calls,
                trials=trials,
                rstate=np.random.default_rng(42)
            )
            hyperopt_time = time.time() - start_time
            # 从trials中提取最佳结果
            best_trial = trials.best_trial
            best_xi_hyperopt = best_trial['misc']['vals']['xi'][0]
            best_dt_hyperopt = best_trial['misc']['vals']['dt'][0]
            best_lambda_hyperopt = best_trial['misc']['vals']['lambda_val'][0]
            best_score_hyperopt = best_trial['result']['loss']
            print(f"Hyperopt: xi={best_xi_hyperopt:.4f}, dt={best_dt_hyperopt:.4f}, lambda={best_lambda_hyperopt:.4f}, "
                f"最佳值={best_score_hyperopt:.4f}, 耗时={hyperopt_time:.2f}s")
            
            method_results['params']['Hyperopt'] = (best_xi_hyperopt, best_dt_hyperopt, best_lambda_hyperopt)
            method_results['scores']['Hyperopt'] = best_score_hyperopt
            method_results['times']['Hyperopt'] = hyperopt_time
            
            # 保存Hyperopt试验数据
            with open(os.path.join(method_dir, f'hyperopt_trials_{method}.json'), 'w') as f:
                json.dump(trials.vals, f, indent=4)
                
        except Exception as e:
            print(f"Hyperopt失败: {e}")
        
        # 6. 量子遗传算法 (QGA)
        print("运行量子遗传算法...")
        start_time = time.time()
        try:
            # 设置QGA参数
            n_population = 5
            n_generations = 2  # 总评估次数 = n_population * n_generations = 10
            
            # 创建QGA实例
            qga = QuantumGeneticAlgorithm(
                obj_func, 
                bounds, 
                n_population=n_population, 
                n_generations=n_generations
            )
            
            # 运行QGA优化
            best_params_qga, best_fitness_qga = qga.optimize()
            qga_time = time.time() - start_time
            
            best_xi_qga, best_dt_qga, best_lambda_qga = best_params_qga
            best_score_qga = best_fitness_qga
            
            print(f"QGA: xi={best_xi_qga:.4f}, dt={best_dt_qga:.4f}, lambda={best_lambda_qga:.4f}, "
                f"最佳值={best_score_qga:.4f}, 耗时={qga_time:.2f}s")
            
            method_results['params']['QGA'] = (best_xi_qga, best_dt_qga, best_lambda_qga)
            method_results['scores']['QGA'] = best_score_qga
            method_results['times']['QGA'] = qga_time
        except Exception as e:
            print(f"量子遗传算法失败: {e}")
        
        # 运行最优参数并记录能量历史
        print("\n运行最优参数并记录能量历史...")
        for opt_name in opt_methods:
            if opt_name in method_results['params']:
                xi, dt, lambda_val = method_results['params'][opt_name]
                try:
                    # 使用缓存的问题
                    num_squared = square_hubo_terms(problem_cache[0])
                    den_squared = square_hubo_terms(problem_cache[1])
                    y4_terms = combine_hubo_terms(num_squared, den_squared, lambda_val)
                    
                    # 使用基础功能运行5次，获取更稳定的结果
                    solver = HUBOSolver(y4_terms, n_vars, xi=xi, dt=dt, 
                                    n_iter=n_iter, method=method, device='cpu', runs=5)
                    _, _ = solver.solve()
                    # 确保能量历史是Python原生类型
                    energy_history = [float(e) for e in solver.energy_history[0]]
                    method_results['energy_histories'][opt_name] = energy_history
                except Exception as e:
                    print(f"使用{opt_name}最优参数运行时出错: {e}")
        
        # 保存方法结果（转换为Python原生类型）
        all_results[method] = convert_to_python_types(method_results)
        
        # 绘制能量历史
        print(f"绘制 {method} 方法的能量历史...")
        plt.figure(figsize=(12, 8))
        
        for opt_name, history in method_results['energy_histories'].items():
            xi, dt, lambda_val = method_results['params'][opt_name]
            plt.plot(history, 
                     label=f'{opt_name} (xi={xi:.3f}, dt={dt:.3f}, λ={lambda_val:.3f})',
                     linewidth=2,
                     alpha=0.8)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Energy', fontsize=12)
        plt.title(f'{method} Method: Energy History Comparison (n_vars={n_vars})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(method_dir, f'{method}_energy_history.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 保存所有结果
    with open(os.path.join(result_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4, cls=NumpyEncoder)
    
    print("\n所有优化完成！结果已保存到", result_dir)

if __name__ == "__main__":
    # 运行调参功能
    run_hyperparameter_tuning()