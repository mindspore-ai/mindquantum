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
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
import math

# 设置matplotlib使用非交互式后端（适用于无显示环境）
mpl.use('Agg')

class LargeSpinRatioExperiment:
    """大规模自旋比值HUBO实验类"""
    
    def __init__(self, n_vars=300, n_terms=10, max_order=3, n_runs=50, 
                 result_dir='./HUBO3/results_large_spin_ratio'):
        """
        初始化大规模自旋比值实验
        
        参数:
        n_vars: 变量个数 (自旋数)
        n_terms: 多项式项数
        max_order: 最大阶数
        n_runs: 运行轮数
        result_dir: 结果保存目录
        """
        self.n_vars = n_vars
        self.n_terms = n_terms
        self.max_order = max_order
        self.n_runs = n_runs
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)
        
        # 实验配置
        self.methods = ['bSB', 'dSB', 'CAC', 'CFC']
        self.opt_methods = ['Bayes', 'CMA-ES', 'Nelder-Mead', 'Hyperopt', 'QGA']
        self.n_iter_values = [10, 100, 300, 500, 800]
        self.n_calls = 10  # 超参数搜索迭代次数
        
        # 存储结果
        self.results = {}
        self.summary_data = []
        self.energy_histories = {}  # 存储能量历史
        
    def generate_random_hubo_terms(self, n_terms=None, ensure_nonzero=True):
        """
        生成随机HUBO项（固定项数），确保分母不为零
        
        参数:
        n_terms: 项数，如果为None则使用self.n_terms
        ensure_nonzero: 是否确保分母项不为零
        
        返回:
        terms_dict: HUBO项字典
        """
        if n_terms is None:
            n_terms = self.n_terms
            
        terms_dict = defaultdict(list)
        generated_terms = set()
        
        while len(generated_terms) < n_terms:
            # 随机选择阶数 (1到max_order)
            order = np.random.randint(1, self.max_order + 1)
            
            # 随机选择不重复的变量
            vars = tuple(sorted(np.random.choice(self.n_vars, size=order, replace=False)))
            
            if vars not in generated_terms:
                generated_terms.add(vars)
                # 生成随机系数 [-10, 10]，但避免系数为零
                coef = np.random.choice([x for x in range(-10, 11) if x != 0])
                terms_dict[order].append((float(coef), vars))
        
        return dict(terms_dict)

    def evaluate_hubo_terms(self, terms_dict, spins):
        """
        计算HUBO项在给定自旋配置下的值
        
        参数:
        terms_dict: HUBO项字典
        spins: 自旋配置 (-1, +1)
        
        返回:
        value: HUBO函数值
        """
        total_value = 0.0
        
        for order, terms in terms_dict.items():
            for coef, var_indices in terms:
                # 计算该项的值：系数 × 所有变量的乘积
                term_value = coef
                for var_idx in var_indices:
                    term_value *= spins[var_idx]  # 自旋取值为±1
                total_value += term_value
            
        return total_value

    def create_ratio_problem(self):
        """
        创建比值HUBO问题，确保分母不会太小
        
        返回:
        problem_data: 包含分子和分母项的元组 (num_terms, den_terms)
        lambda_range: lambda的范围
        """
        max_attempts = 10  # 最大尝试次数
        attempt = 0
        
        while attempt < max_attempts:
            # 生成分子和分母项（各10项）
            num_terms = self.generate_random_hubo_terms(self.n_terms)
            den_terms = self.generate_random_hubo_terms(self.n_terms)
            
            # 估算lambda范围
            lambda_min, lambda_max = estimate_lambda_range(num_terms, den_terms, self.n_vars)
            
            # 检查分母项是否合理
            if self.check_denominator_reasonable(den_terms):
                return (num_terms, den_terms), (lambda_min, lambda_max)
            
            attempt += 1
            print(f"尝试生成合理的分母项，第 {attempt} 次尝试...")
        
        # 如果多次尝试都失败，使用最后一次生成的结果
        print("警告：无法生成理想的分母项，使用最后一次尝试的结果")
        return (num_terms, den_terms), (lambda_min, lambda_max)

    def check_denominator_reasonable(self, den_terms):
        """
        检查分母项是否合理，避免分母过小
        
        参数:
        den_terms: 分母项字典
        
        返回:
        bool: 分母项是否合理
        """
        # 检查分母项是否有足够的非零系数
        total_coef = 0
        for order, terms in den_terms.items():
            for coef, _ in terms:
                total_coef += abs(coef)
        
        # 如果分母项的总系数绝对值太小，认为不合理
        if total_coef < 5:  # 经验阈值
            return False
        
        return True

    def objective_function(self, params, n_iter, method_name, problem_data):
        """
        修正的目标函数：直接返回我们要最大化的平方比值 A²/B²
        
        参数:
        params: (xi, dt, lambda_val)
        n_iter: 迭代次数
        method_name: 方法名称
        problem_data: (num_terms, den_terms)
        
        返回:
        平方比值的正值（因为我们要最大化 A²/B²）
        """
        xi, dt, lambda_val = params
        try:
            num_terms, den_terms = problem_data
            
            # 将分子和分母项平方
            num_squared = square_hubo_terms(num_terms)
            den_squared = square_hubo_terms(den_terms)
            
            # 构造y4 = A² - λB²
            y4_terms = combine_hubo_terms(num_squared, den_squared, lambda_val)
            
            # 运行求解器
            solver = HUBOSolver(y4_terms, self.n_vars, xi=xi, dt=dt, 
                              n_iter=n_iter, method=method_name, device='cpu', runs=1)
            best_spins, _ = solver.solve()
            
            # 计算实际的平方比值 A²/B²
            A_value = self.evaluate_hubo_terms(num_terms, best_spins)
            B_value = self.evaluate_hubo_terms(den_terms, best_spins)
            
            # 计算 A² 和 B²
            A_squared = A_value ** 2
            B_squared = B_value ** 2
            
            # 改进的比值计算，避免分母过小
            if abs(B_squared) < 1e-6:  # 稍微放宽阈值
                # 如果分母很小，返回一个基于分子的值而不是固定的大数值
                ratio = A_squared * 1000  # 放大分子值
            else:
                ratio = A_squared / B_squared
            
            # 限制比值范围，避免过大或过小
            ratio = np.clip(ratio, -1e6, 1e6)
            
            # 确保返回Python标量而不是PyTorch张量
            if hasattr(ratio, 'item'):
                ratio = ratio.item()
            
            return float(ratio)
            
        except Exception as e:
            print(f"Error during solver execution: {e}")
            return 0.0  # 返回最小值表示失败

    def hyperopt_objective_wrapper(self, params, n_iter, method_name, problem_data):
        """Hyperopt目标函数包装器 - 修正版本"""
        xi = params['xi']
        dt = params['dt']
        lambda_val = params['lambda_val']
        result = self.objective_function((xi, dt, lambda_val), n_iter, method_name, problem_data)
        # Hyperopt最小化，所以返回负值
        result = -result  # 关键修改：因为我们想要最大化比值，所以返回负值
        # 确保返回Python标量
        if hasattr(result, 'item'):
            result = result.item()
        return float(result)

    def run_single_optimization(self, opt_name, obj_func, bounds, x0, n_iter, method, problem_data, lambda_range):
        """
        运行单次优化 - 修正版本
        
        返回:
        result: 包含最佳参数和分数的字典
        """
        try:
            start_time = time.time()
            
            if opt_name == 'Bayes':
                space = [(0.01, 0.5), (0.01, 0.3), (lambda_range[0], lambda_range[1])]
                n_initial_points = min(5, self.n_calls // 2)
                
                # 根据方法选择acquisition函数
                if method in ["bSB", "dSB"]:
                    acq_func = "LCB"
                elif method == "CAC":
                    acq_func = "EI"
                elif method == "CFC":
                    acq_func = "PI"
                else:
                    acq_func = "LCB"
                
                # 关键修改：Bayes最小化，所以我们需要最小化负的比值
                def wrapped_obj_func(params):
                    result = obj_func(params)
                    # 确保返回Python标量
                    if hasattr(result, 'item'):
                        result = result.item()
                    # 因为我们想要最大化比值，但Bayes最小化，所以返回负值
                    return float(-result)
                    
                res = gp_minimize(
                    wrapped_obj_func,
                    space,
                    n_calls=self.n_calls,
                    n_initial_points=n_initial_points,
                    acq_func=acq_func,
                    random_state=42
                )
                best_params = res.x
                # 关键修改：恢复正值
                best_score = -res.fun
                
            elif opt_name == 'SA':
                # 包装目标函数以确保返回标量
                def wrapped_obj_func(params):
                    result = obj_func(params)
                    # 确保返回Python标量
                    if hasattr(result, 'item'):
                        result = result.item()
                    # 因为我们想要最大化比值，但SA最小化，所以返回负值
                    return float(-result)
                    
                res = dual_annealing(wrapped_obj_func, bounds, maxiter=self.n_calls//2, seed=42)
                best_params = res.x
                # 关键修改：恢复正值
                best_score = -res.fun
                
            elif opt_name == 'CMA-ES':
                # 关键修改：CMA-ES最小化，所以我们需要最小化负的比值
                def wrapped_obj_func(params):
                    result = obj_func(params)
                    # 确保返回Python标量
                    if hasattr(result, 'item'):
                        result = result.item()
                    # 因为我们想要最大化比值，但CMA-ES最小化，所以返回负值
                    return float(-result)
                
                es = self.create_cma_es(x0, self.n_calls, bounds)
                es.optimize(wrapped_obj_func)
                best_params = es.result.xbest
                # 关键修改：恢复正值
                best_score = -es.result.fbest
                
            elif opt_name == 'Nelder-Mead':
                # 关键修改：Nelder-Mead最小化，所以我们需要最小化负的比值
                def wrapped_obj_func(params):
                    result = obj_func(params)
                    # 确保返回Python标量
                    if hasattr(result, 'item'):
                        result = result.item()
                    # 因为我们想要最大化比值，但Nelder-Mead最小化，所以返回负值
                    return float(-result)
                
                res = minimize(wrapped_obj_func, x0, method='Nelder-Mead', 
                             bounds=bounds, options={'maxfev': self.n_calls})
                best_params = res.x
                # 关键修改：恢复正值
                best_score = -res.fun
                
            elif opt_name == 'Hyperopt':
                hyperopt_space = {
                    'xi': hp.uniform('xi', 0.01, 0.5),
                    'dt': hp.uniform('dt', 0.01, 0.3),
                    'lambda_val': hp.uniform('lambda_val', lambda_range[0], lambda_range[1])
                }
                
                hyperopt_obj = partial(self.hyperopt_objective_wrapper, 
                                     n_iter=n_iter, method_name=method, 
                                     problem_data=problem_data)
                
                trials = Trials()
                best_hyperopt = fmin(
                    fn=hyperopt_obj,
                    space=hyperopt_space,
                    algo=tpe.suggest,
                    max_evals=self.n_calls,
                    trials=trials,
                    rstate=np.random.default_rng(42)
                )
                
                best_trial = trials.best_trial
                best_params = [
                    best_trial['misc']['vals']['xi'][0],
                    best_trial['misc']['vals']['dt'][0],
                    best_trial['misc']['vals']['lambda_val'][0]
                ]
                # 关键修改：Hyperopt返回的是负的比值，所以需要取负得到正值
                best_score = -best_trial['result']['loss']
                
            elif opt_name == 'QGA':
                # 修复：QGA应该直接使用原始目标函数，不包装
                # 因为QGA内部已经处理了最大化问题
                qga = QuantumGeneticAlgorithm(
                    obj_func,  # 直接使用原始目标函数
                    bounds, 
                    n_population=10, 
                    n_generations=10
                )
                best_params, best_score = qga.optimize()
                # 注意：QGA返回的是原始比值，不需要取负
                
            else:
                return None
                
            opt_time = time.time() - start_time
            
            return {
                'params': best_params,
                'score': best_score,  # 现在这个score是正的比值
                'time': opt_time
            }
            
        except Exception as e:
            print(f"{opt_name}优化失败: {e}")
            return None

    def create_cma_es(self, x0, n_calls, bounds):
        """创建CMA-ES优化器"""
        try:
            lower_bounds = [b[0] for b in bounds]
            upper_bounds = [b[1] for b in bounds]
            
            es = cma.CMAEvolutionStrategy(x0, 0.1, {
                'bounds': [lower_bounds, upper_bounds],
                'maxfevals': n_calls,
                'verbose': -1
            })
            return es
        except Exception as e:
            print(f"CMA-ES边界设置失败: {e}")
            return cma.CMAEvolutionStrategy(x0, 0.1, {
                'maxfevals': n_calls,
                'verbose': -1
            })

    def evaluate_final_ratio(self, problem_data, params, n_iter, method):
        """
        评估最终比值性能 - 改进版本：优化 A²/B²
        
        返回:
        best_ratio: 最佳比值（50轮中最好结果）
        avg_last_10_ratio: 最后10轮平均比值
        energy_history: 能量历史
        """
        xi, dt, lambda_val = params
        num_terms, den_terms = problem_data
        
        # 将分子和分母项平方
        num_squared = square_hubo_terms(num_terms)
        den_squared = square_hubo_terms(den_terms)
        
        # 构造y4 = A² - λB²
        y4_terms = combine_hubo_terms(num_squared, den_squared, lambda_val)
        
        ratios = []
        energy_histories = []  # 存储所有轮次的能量历史
        
        # 运行50轮
        for run in range(self.n_runs):
            try:
                solver = HUBOSolver(y4_terms, self.n_vars, xi=xi, dt=dt, 
                                n_iter=n_iter, method=method, device='cpu', runs=1)
                best_spins, _ = solver.solve()
                
                # 计算实际的平方比值 A²/B²
                # 1. 计算分子 A 的值
                A_value = self.evaluate_hubo_terms(num_terms, best_spins)
                # 2. 计算分母 B 的值  
                B_value = self.evaluate_hubo_terms(den_terms, best_spins)
                
                # 计算 A² 和 B²
                A_squared = A_value ** 2
                B_squared = B_value ** 2
                
                # 改进的比值计算，避免分母过小
                if abs(B_squared) < 1e-6:  # 稍微放宽阈值
                    # 如果分母很小，返回一个基于分子的值而不是固定的大数值
                    ratio = A_squared * 1000  # 放大分子值
                else:
                    ratio = A_squared / B_squared
                
                # 限制比值范围，避免过大或过小
                ratio = np.clip(ratio, -1e6, 1e6)
                    
                # 确保比值为Python标量
                if hasattr(ratio, 'item'):
                    ratio = ratio.item()
                    
                ratios.append(float(ratio))
                
                # 记录能量历史（记录所有轮次的能量历史）
                if hasattr(solver, 'energy_history') and solver.energy_history is not None:
                    energy_history = self.process_energy_history(solver.energy_history)
                    energy_histories.append(energy_history)
                
            except Exception as e:
                print(f"第{run+1}轮评估失败: {e}")
                ratios.append(0.0)  # 失败时添加默认值
        
        # 计算指标 - 现在我们直接优化平方比值
        # 过滤掉异常大的值
        valid_ratios = [r for r in ratios if abs(r) < 1e8]
        
        if not valid_ratios:
            best_ratio = 0.0
            last_10_avg_ratio = 0.0
        else:
            best_ratio = max(valid_ratios) if valid_ratios else 0.0  # 寻找最大比值
            # 计算最后10个有效比值
            last_10_valid = [r for r in ratios[-10:] if abs(r) < 1e8]
            last_10_avg_ratio = np.mean(last_10_valid) if last_10_valid else np.mean(valid_ratios)
        
        # 计算平均能量历史
        avg_energy_history = []
        if energy_histories:
            # 找到最长的能量历史长度
            max_length = max(len(eh) for eh in energy_histories)
            
            # 对每个迭代步骤计算平均值
            for i in range(max_length):
                step_energies = []
                for eh in energy_histories:
                    if i < len(eh):
                        step_energies.append(eh[i])
                
                if step_energies:
                    avg_energy_history.append(np.mean(step_energies))
        
        return float(best_ratio), float(last_10_avg_ratio), avg_energy_history

    def process_energy_history(self, energy_history):
        """
        处理能量历史数据，确保所有元素都是浮点数
        
        参数:
        energy_history: 原始能量历史数据
        
        返回:
        processed_history: 处理后的能量历史数据（浮点数列表）
        """
        processed_history = []
        
        if energy_history is None:
            return processed_history
        
        for energy_value in energy_history:
            try:
                # 如果能量值是张量，提取标量值
                if torch.is_tensor(energy_value):
                    if energy_value.dim() == 0:
                        processed_history.append(float(energy_value.item()))
                    else:
                        # 如果是多维张量，取第一个元素
                        processed_history.append(float(energy_value[0].item()))
                # 如果能量值是列表或数组，取第一个元素
                elif isinstance(energy_value, (list, np.ndarray)):
                    if len(energy_value) > 0:
                        first_element = energy_value[0]
                        if torch.is_tensor(first_element):
                            processed_history.append(float(first_element.item()))
                        else:
                            processed_history.append(float(first_element))
                    else:
                        processed_history.append(0.0)
                # 如果已经是标量，直接转换
                else:
                    processed_history.append(float(energy_value))
            except Exception as e:
                print(f"处理能量历史数据时出错: {e}, 值: {energy_value}")
                processed_history.append(0.0)
        
        return processed_history

    def run_comprehensive_experiment(self):
        """运行综合性实验"""
        print("=== 大规模自旋比值HUBO实验 ===")
        print(f"自旋数: {self.n_vars}")
        print(f"项数: {self.n_terms}")
        print(f"运行轮数: {self.n_runs}")
        print(f"算法: {self.methods}")
        print(f"优化方法: {self.opt_methods}")
        print(f"迭代次数: {self.n_iter_values}")
        
        # 生成HUBO问题
        problem_data, lambda_range = self.create_ratio_problem()
        print(f"Lambda范围: [{lambda_range[0]:.4f}, {lambda_range[1]:.4f}]")
        
        # 保存HUBO问题
        with open(os.path.join(self.result_dir, 'hubo_problem.json'), 'w') as f:
            json.dump(convert_to_python_types({
                'num_terms': problem_data[0],
                'den_terms': problem_data[1],
                'lambda_range': lambda_range
            }), f, indent=4)
        
        # 运行实验
        for n_iter in self.n_iter_values:
            print(f"\n--- 迭代次数: {n_iter} ---")
            
            for method in self.methods:
                print(f"\n算法: {method}")
                
                for opt_method in self.opt_methods:
                    print(f"  优化方法: {opt_method}", end="")
                    
                    # 定义目标函数 - 现在返回正值
                    obj_func = partial(self.objective_function, 
                                     n_iter=n_iter, 
                                     method_name=method, 
                                     problem_data=problem_data)
                    
                    # 设置参数空间
                    bounds = [(0.01, 0.5), (0.01, 0.3), (lambda_range[0], lambda_range[1])]
                    x0 = [0.1, 0.1, (lambda_range[0] + lambda_range[1]) / 2]
                    
                    # 运行优化
                    start_time = time.time()
                    opt_result = self.run_single_optimization(
                        opt_method, obj_func, bounds, x0, n_iter, method, problem_data, lambda_range
                    )
                    opt_time = time.time() - start_time
                    
                    if opt_result:
                        # 评估最终性能（50轮）
                        best_ratio, avg_last_10_ratio, energy_history = self.evaluate_final_ratio(
                            problem_data, opt_result['params'], n_iter, method
                        )
                        
                        stats = {
                            'n_iter': n_iter,
                            'method': method,
                            'optimizer': opt_method,
                            'best_energy': best_ratio,  # 现在存储的是比值
                            'avg_last_10': avg_last_10_ratio,  # 现在存储的是比值
                            'optimization_time': opt_result['time'],
                            'total_time': opt_time,
                            'params': opt_result['params'],
                            'score': opt_result['score']
                        }
                        
                        self.summary_data.append(stats)
                        
                        # 存储能量历史
                        key = f"{n_iter}_{method}_{opt_method}"
                        self.energy_histories[key] = energy_history
                        
                        # 检查比值是否异常
                        if abs(best_ratio) > 1e8:
                            print(f" - 警告: 比值异常大，最佳比值: {best_ratio:.4f}, "
                                  f"最后10轮平均: {avg_last_10_ratio:.4f}, "
                                  f"总时间: {opt_time:.2f}s")
                        else:
                            print(f" - 最佳比值: {best_ratio:.4f}, "
                                  f"最后10轮平均: {avg_last_10_ratio:.4f}, "
                                  f"总时间: {opt_time:.2f}s")
                    else:
                        print(f" - 失败")
        
        # 保存结果和生成可视化
        self.save_results()
        self.create_visualizations()
        
        print(f"\n=== 实验完成 ===")
        print(f"结果保存在: {self.result_dir}")

    def save_results(self):
        """保存实验结果"""
        # 保存汇总数据
        summary_df = pd.DataFrame(self.summary_data)
        summary_df.to_csv(os.path.join(self.result_dir, 'experiment_summary.csv'), index=False)
        
        # 保存详细结果
        with open(os.path.join(self.result_dir, 'detailed_results.json'), 'w') as f:
            json.dump(convert_to_python_types({
                'summary_data': self.summary_data,
                'config': {
                    'n_vars': self.n_vars,
                    'n_terms': self.n_terms,
                    'n_runs': self.n_runs,
                    'methods': self.methods,
                    'opt_methods': self.opt_methods,
                    'n_iter_values': self.n_iter_values
                }
            }), f, indent=4)
        
        # 保存能量历史
        with open(os.path.join(self.result_dir, 'energy_histories.json'), 'w') as f:
            json.dump(convert_to_python_types(self.energy_histories), f, indent=4)
        
        print(f"结果已保存到 {self.result_dir}")

    def create_visualizations(self):
        """创建可视化图表"""
        if not self.summary_data:
            print("没有数据可生成可视化")
            return
        
        df = pd.DataFrame(self.summary_data)
        
        # 1. 主要评价指标：最佳比值
        self.plot_main_metrics(df)
        
        # 2. 辅助指标：最后10轮平均比值
        self.plot_secondary_metrics(df)
        
        # 3. 时间分析
        self.plot_time_analysis(df)
        
        # 4. 热力图分析
        self.plot_heatmaps(df)
        
        # 5. 替换为能量值 vs 迭代次数曲线图
        self.plot_energy_vs_iterations(df)
        
        print("可视化图表已生成")

    def plot_energy_vs_iterations(self, df):
        """绘制能量值（比值）随迭代次数变化的曲线图"""
        if len(df) == 0:
            print("没有数据可绘制能量 vs 迭代次数图")
            return
        
        plt.figure(figsize=(16, 12))
        
        # 子图1: 各优化方法在不同迭代次数下的最佳比值
        plt.subplot(2, 2, 1)
        for opt_method in self.opt_methods:
            opt_data = df[df['optimizer'] == opt_method]
            if not opt_data.empty:
                # 按迭代次数排序
                opt_data = opt_data.sort_values('n_iter')
                # 过滤掉异常大的值
                valid_data = opt_data[opt_data['best_energy'] < 1e8]
                if not valid_data.empty:
                    plt.plot(valid_data['n_iter'], valid_data['best_energy'], 
                            label=opt_method, marker='o', linewidth=2, markersize=6)
        
        plt.xlabel('Iterations (n_iter)')
        plt.ylabel('Best Ratio Value')
        plt.title('Ratio Value vs Iterations by Optimizer')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')  # 使用对数坐标更好地显示不同迭代次数
        
        # 子图2: 各HUBO方法在不同迭代次数下的最佳比值
        plt.subplot(2, 2, 2)
        for method in self.methods:
            method_data = df[df['method'] == method]
            if not method_data.empty:
                # 按迭代次数排序
                method_data = method_data.sort_values('n_iter')
                # 过滤掉异常大的值
                valid_data = method_data[method_data['best_energy'] < 1e8]
                if not valid_data.empty:
                    plt.plot(valid_data['n_iter'], valid_data['best_energy'], 
                            label=method, marker='s', linewidth=2, markersize=6)
        
        plt.xlabel('Iterations (n_iter)')
        plt.ylabel('Best Ratio Value')
        plt.title('Ratio Value vs Iterations by HUBO Method')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # 子图3: 最佳性能组合趋势
        plt.subplot(2, 2, 3)
        
        # 找出每个迭代次数下的最佳方法组合
        best_per_iteration = []
        for n_iter in self.n_iter_values:
            iter_data = df[df['n_iter'] == n_iter]
            if not iter_data.empty:
                # 过滤掉异常大的值
                valid_iter_data = iter_data[iter_data['best_energy'] < 1e8]
                if not valid_iter_data.empty:
                    best_idx = valid_iter_data['best_energy'].idxmax()  # 寻找最大比值
                    best_row = valid_iter_data.loc[best_idx]
                    best_per_iteration.append({
                        'n_iter': n_iter,
                        'method': best_row['method'],
                        'optimizer': best_row['optimizer'],
                        'energy': best_row['best_energy']
                    })
        
        if best_per_iteration:
            best_df = pd.DataFrame(best_per_iteration)
            best_df = best_df.sort_values('n_iter')
            
            plt.plot(best_df['n_iter'], best_df['energy'], 
                    marker='*', linewidth=3, markersize=12, 
                    color='red', label='Best Combination')
            
            # 添加标签显示最佳组合
            for _, row in best_df.iterrows():
                plt.annotate(f"{row['method']}-{row['optimizer']}", 
                            (row['n_iter'], row['energy']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.8, 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        
        plt.xlabel('Iterations (n_iter)')
        plt.ylabel('Best Ratio Value')
        plt.title('Best Performance Trend vs Iterations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # 子图4: 收敛速度分析 (比值改善率)
        plt.subplot(2, 2, 4)
        
        # 计算每个方法组合的比值改善率
        improvement_data = []
        for method in self.methods:
            for opt_method in self.opt_methods:
                combo_data = df[(df['method'] == method) & (df['optimizer'] == opt_method)]
                if len(combo_data) > 1:
                    combo_data = combo_data.sort_values('n_iter')
                    # 过滤掉异常大的值
                    valid_combo_data = combo_data[combo_data['best_energy'] < 1e8]
                    if len(valid_combo_data) > 1:
                        # 计算从最小迭代到最大迭代的比值改善
                        min_iter = valid_combo_data['n_iter'].min()
                        max_iter = valid_combo_data['n_iter'].max()
                        min_energy = valid_combo_data[valid_combo_data['n_iter'] == min_iter]['best_energy'].values[0]
                        max_energy = valid_combo_data[valid_combo_data['n_iter'] == max_iter]['best_energy'].values[0]
                        
                        if min_energy > 0:  # 避免除零
                            improvement = (max_energy - min_energy) / min_energy * 100  # 百分比改善
                            improvement_data.append({
                                'method': method,
                                'optimizer': opt_method,
                                'improvement': improvement,
                                'min_energy': min_energy,
                                'max_energy': max_energy
                            })
        
        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)
            
            # 绘制改善率条形图
            labels = [f"{row['method']}-{row['optimizer']}" for _, row in improvement_df.iterrows()]
            plt.barh(range(len(improvement_df)), improvement_df['improvement'])
            plt.yticks(range(len(improvement_df)), labels, fontsize=8)
            plt.xlabel('Ratio Improvement (%)')
            plt.title('Ratio Improvement from Minimum to Maximum Iterations')
            plt.grid(True, alpha=0.3, axis='x')
        else:
            plt.text(0.5, 0.5, 'No valid improvement data', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('No Valid Improvement Data')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'energy_vs_iterations.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建更详细的子图表
        self.plot_detailed_energy_vs_iterations(df)

    def plot_detailed_energy_vs_iterations(self, df):
        """绘制详细的能量值（比值）vs 迭代次数图表"""
        if len(df) == 0:
            return
        
        # 1. 为每个HUBO方法创建单独的子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, method in enumerate(self.methods):
            if i < len(axes):
                ax = axes[i]
                method_data = df[df['method'] == method]
                
                for opt_method in self.opt_methods:
                    opt_data = method_data[method_data['optimizer'] == opt_method]
                    if not opt_data.empty:
                        # 按迭代次数排序
                        opt_data = opt_data.sort_values('n_iter')
                        # 过滤掉异常大的值
                        valid_data = opt_data[opt_data['best_energy'] < 1e8]
                        if not valid_data.empty:
                            ax.plot(valid_data['n_iter'], valid_data['best_energy'], 
                                   marker='o', linewidth=2, label=opt_method, markersize=5)
                
                ax.set_xlabel('Iterations (n_iter)')
                ax.set_ylabel('Best Ratio Value')
                ax.set_title(f'Ratio vs Iterations: {method} Method')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'energy_vs_iterations_by_method.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 为每个优化方法创建单独的子图
        n_cols = 3
        n_rows = math.ceil(len(self.opt_methods) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, opt_method in enumerate(self.opt_methods):
            if i < len(axes):
                ax = axes[i]
                opt_data = df[df['optimizer'] == opt_method]
                
                for method in self.methods:
                    method_data = opt_data[opt_data['method'] == method]
                    if not method_data.empty:
                        # 按迭代次数排序
                        method_data = method_data.sort_values('n_iter')
                        # 过滤掉异常大的值
                        valid_data = method_data[method_data['best_energy'] < 1e8]
                        if not valid_data.empty:
                            ax.plot(valid_data['n_iter'], valid_data['best_energy'], 
                                   marker='s', linewidth=2, label=method, markersize=5)
                
                ax.set_xlabel('Iterations (n_iter)')
                ax.set_ylabel('Best Ratio Value')
                ax.set_title(f'Ratio vs Iterations: {opt_method} Optimizer')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
        
        # 移除多余的子图
        for i in range(len(self.opt_methods), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'energy_vs_iterations_by_optimizer.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 创建热力图显示不同迭代次数下的性能
        self.plot_energy_heatmap_vs_iterations(df)

    def plot_energy_heatmap_vs_iterations(self, df):
        """绘制能量值（比值）vs 迭代次数的热力图"""
        if len(df) == 0:
            return
        
        # 创建组合热力图
        plt.figure(figsize=(15, 8))
        
        # 准备数据：每个方法组合在不同迭代次数下的比值
        heatmap_data = []
        for method in self.methods:
            for opt_method in self.opt_methods:
                row_data = {'Method-Optimizer': f'{method}-{opt_method}'}
                for n_iter in self.n_iter_values:
                    mask = (df['method'] == method) & (df['optimizer'] == opt_method) & (df['n_iter'] == n_iter)
                    subset = df[mask]
                    if not subset.empty:
                        # 过滤掉异常大的值
                        valid_subset = subset[subset['best_energy'] < 1e8]
                        if not valid_subset.empty:
                            row_data[n_iter] = valid_subset['best_energy'].values[0]
                        else:
                            row_data[n_iter] = np.nan
                    else:
                        row_data[n_iter] = np.nan
                heatmap_data.append(row_data)
        
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df = heatmap_df.set_index('Method-Optimizer')
        
        # 绘制热力图
        plt.subplot(1, 2, 1)
        if not heatmap_df.empty and not heatmap_df.isna().all().all():
            sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='viridis_r',
                       cbar_kws={'label': 'Best Ratio Value'})
            plt.title('Ratio Values vs Iterations\n(Method Combinations)')
        else:
            plt.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('No Valid Data')
        plt.xlabel('Iterations')
        plt.ylabel('Method-Optimizer Combination')
        
        # 按方法分组的热力图
        plt.subplot(1, 2, 2)
        method_heatmap_data = []
        for method in self.methods:
            row_data = {'Method': method}
            for n_iter in self.n_iter_values:
                method_data = df[(df['method'] == method) & (df['n_iter'] == n_iter)]
                # 过滤掉异常大的值
                valid_method_data = method_data[method_data['best_energy'] < 1e8]
                if not valid_method_data.empty:
                    row_data[n_iter] = valid_method_data['best_energy'].mean()
                else:
                    row_data[n_iter] = np.nan
            method_heatmap_data.append(row_data)
        
        method_heatmap_df = pd.DataFrame(method_heatmap_data)
        method_heatmap_df = method_heatmap_df.set_index('Method')
        
        if not method_heatmap_df.empty and not method_heatmap_df.isna().all().all():
            sns.heatmap(method_heatmap_df, annot=True, fmt='.3f', cmap='plasma_r',
                       cbar_kws={'label': 'Best Ratio Value'})
            plt.title('Ratio Values vs Iterations\n(Grouped by HUBO Method)')
        else:
            plt.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('No Valid Data')
        plt.xlabel('Iterations')
        plt.ylabel('HUBO Method')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'energy_heatmap_vs_iterations.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_main_metrics(self, df):
        """绘制主要评价指标图（无阴影）"""
        plt.figure(figsize=(15, 10))
        
        # 子图1: 各优化方法在不同迭代次数下的最佳比值
        plt.subplot(2, 2, 1)
        for opt_method in df['optimizer'].unique():
            opt_data = df[df['optimizer'] == opt_method]
            plt.plot(opt_data['n_iter'], opt_data['best_energy'], 
                    label=opt_method, marker='o', linewidth=2)
        plt.xlabel('Iterations')
        plt.ylabel('Best Ratio Value (50 runs)')
        plt.title('Main Metric: Best Ratio vs Iterations by Optimizer')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 各HUBO方法在不同迭代次数下的最佳比值
        plt.subplot(2, 2, 2)
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            plt.plot(method_data['n_iter'], method_data['best_energy'], 
                    label=method, marker='s', linewidth=2)
        plt.xlabel('Iterations')
        plt.ylabel('Best Ratio Value (50 runs)')
        plt.title('Main Metric: Best Ratio vs Iterations by HUBO Method')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图3: 最佳性能组合
        plt.subplot(2, 2, 3)
        # 过滤掉异常大的值
        valid_data = df[df['best_energy'] < 1e8]
        if len(valid_data) > 0:
            best_combinations = valid_data.nlargest(min(10, len(valid_data)), 'best_energy')
            labels = [f"{row['method']}-{row['optimizer']}\n(n_iter={row['n_iter']})" 
                     for _, row in best_combinations.iterrows()]
            plt.barh(range(len(best_combinations)), best_combinations['best_energy'])
            plt.yticks(range(len(best_combinations)), labels, fontsize=8)
            plt.xlabel('Best Ratio Value')
            plt.title('Top Method Combinations (Main Metric)')
        else:
            plt.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('No Valid Combinations')
        
        # 子图4: 参数分布
        plt.subplot(2, 2, 4)
        xi_values = [p[0] for p in df['params']]
        dt_values = [p[1] for p in df['params']]
        plt.scatter(xi_values, dt_values, c=df['best_energy'], cmap='viridis', alpha=0.6)
        plt.colorbar(label='Best Ratio Value')
        plt.xlabel('xi')
        plt.ylabel('dt')
        plt.title('Parameter Distribution vs Performance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'main_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_secondary_metrics(self, df):
        """绘制辅助指标图（无阴影）"""
        plt.figure(figsize=(12, 8))
        
        # 子图1: 最后10轮平均比值 vs 迭代次数
        plt.subplot(1, 2, 1)
        for opt_method in df['optimizer'].unique():
            opt_data = df[df['optimizer'] == opt_method]
            plt.plot(opt_data['n_iter'], opt_data['avg_last_10'], 
                    label=opt_method, marker='o', linewidth=2)
        plt.xlabel('Iterations')
        plt.ylabel('Average Last 10 Runs Ratio Value')
        plt.title('Secondary Metric: Last 10 Average vs Iterations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 主要指标 vs 辅助指标散点图
        plt.subplot(1, 2, 2)
        for opt_method in df['optimizer'].unique():
            opt_data = df[df['optimizer'] == opt_method]
            plt.scatter(opt_data['best_energy'], opt_data['avg_last_10'], 
                       label=opt_method, alpha=0.7, s=60)
        plt.xlabel('Best Ratio Value (Main Metric)')
        plt.ylabel('Last 10 Average (Secondary Metric)')
        plt.title('Main vs Secondary Metrics Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'secondary_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_time_analysis(self, df):
        """绘制时间分析图（无阴影）"""
        plt.figure(figsize=(12, 8))
        
        # 子图1: 总时间 vs 迭代次数
        plt.subplot(1, 2, 1)
        for opt_method in df['optimizer'].unique():
            opt_data = df[df['optimizer'] == opt_method]
            plt.plot(opt_data['n_iter'], opt_data['total_time'], 
                    label=opt_method, marker='o', linewidth=2)
        plt.xlabel('Iterations')
        plt.ylabel('Total Time (s)')
        plt.title('Total Computation Time vs Iterations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 效率分析 (比值/时间)
        plt.subplot(1, 2, 2)
        for i, (_, row) in enumerate(df.iterrows()):
            plt.scatter(row['total_time'], row['best_energy'], 
                       label=f"{row['method']}-{row['optimizer']}", s=80, alpha=0.7)
            plt.annotate(f"{row['n_iter']}", (row['total_time'], row['best_energy']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel('Total Time (s)')
        plt.ylabel('Best Ratio Value')
        plt.title('Efficiency Analysis: Ratio vs Time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'time_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_heatmaps(self, df):
        """绘制热力图分析（无阴影）"""
        plt.figure(figsize=(16, 12))
        
        # 过滤掉异常大的值
        valid_data = df[df['best_energy'] < 1e8]
        
        # 子图1: 优化方法 vs 迭代次数 (最佳比值)
        plt.subplot(2, 2, 1)
        if len(valid_data) > 0:
            pivot_energy = valid_data.pivot_table(values='best_energy', 
                                         index='optimizer', columns='n_iter', aggfunc='mean')
            sns.heatmap(pivot_energy, annot=True, fmt='.3f', cmap='viridis', 
                       cbar_kws={'label': 'Best Ratio Value'})
            plt.title('Optimizer vs Iterations (Main Metric)')
        else:
            plt.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('No Valid Data')
        plt.xlabel('Iterations')
        plt.ylabel('Optimizer')
        
        # 子图2: HUBO方法 vs 迭代次数 (最佳比值)
        plt.subplot(2, 2, 2)
        if len(valid_data) > 0:
            pivot_method = valid_data.pivot_table(values='best_energy', 
                                         index='method', columns='n_iter', aggfunc='mean')
            sns.heatmap(pivot_method, annot=True, fmt='.3f', cmap='plasma', 
                       cbar_kws={'label': 'Best Ratio Value'})
            plt.title('HUBO Method vs Iterations (Main Metric)')
        else:
            plt.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('No Valid Data')
        plt.xlabel('Iterations')
        plt.ylabel('HUBO Method')
        
        # 子图3: 综合性能矩阵
        plt.subplot(2, 2, 3)
        if len(valid_data) > 0:
            combo_performance = valid_data.pivot_table(values='best_energy', 
                                             index='optimizer', columns='method', aggfunc='mean')
            sns.heatmap(combo_performance, annot=True, fmt='.3f', cmap='RdYlGn', 
                       cbar_kws={'label': 'Best Ratio Value'})
            plt.title('Method Combination Performance Matrix')
        else:
            plt.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('No Valid Data')
        plt.xlabel('HUBO Method')
        plt.ylabel('Optimizer')
        
        # 子图4: 时间效率热力图
        plt.subplot(2, 2, 4)
        time_efficiency = df.pivot_table(values='total_time', 
                                       index='optimizer', columns='n_iter', aggfunc='mean')
        sns.heatmap(time_efficiency, annot=True, fmt='.1f', cmap='coolwarm', 
                   cbar_kws={'label': 'Total Time (s)'})
        plt.title('Computation Time Heatmap')
        plt.xlabel('Iterations')
        plt.ylabel('Optimizer')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'performance_heatmaps.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


# QuantumGeneticAlgorithm 和其他辅助类保持不变...

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
                # 修复：直接使用原始目标函数，不需要包装
                fitness = self.objective_func(params)
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


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


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


# 使用示例
if __name__ == "__main__":
    # 创建大规模自旋比值实验
    experiment = LargeSpinRatioExperiment(
        n_vars=300,      # 300个自旋
        n_terms=10,      # 10项多项式
        max_order=3,     # 最大3阶
        n_runs=50,       # 50轮实验
        result_dir='./HUBO3/results_large_spin_ratio'
    )
    
    # 运行实验
    experiment.run_comprehensive_experiment()