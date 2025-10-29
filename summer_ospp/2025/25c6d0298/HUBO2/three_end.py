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
import pandas as pd
import seaborn as sns
from collections import defaultdict

mpl.use('Agg')

class LargeSpinHUBOExperiment:
    """大规模自旋HUBO实验类"""

    def __init__(self, n_vars=300, n_terms=10, max_order=3, n_runs=50, opt_methods=None,
                 result_dir='./HUBO2/results_large_spin'):
        """
        初始化大规模自旋实验
        
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
        self.opt_methods = opt_methods if opt_methods is not None else []
        os.makedirs(result_dir, exist_ok=True)
        
        # 实验配置
        self.methods = ['bSB', 'dSB', 'CAC', 'CFC']
       # self.opt_methods = ['Bayes', 'SA', 'CMA-ES', 'Nelder-Mead', 'Hyperopt', 'QGA']
        self.n_iter_values = [10, 100,300,500,800 ]
        self.n_calls = 10  # 超参数搜索迭代次数
        
        # 存储结果
        self.results = {}
        self.summary_data = []
        self.energy_histories = {}  # 存储能量历史
        
    def generate_sparse_hubo_terms(self):
        """
        生成稀疏HUBO项（固定项数而不是密度）
        
        返回:
        terms_dict: HUBO项字典
        """
        terms_dict = defaultdict(list)
        generated_terms = set()
        
        while len(generated_terms) < self.n_terms:
            # 随机选择阶数 (1到max_order)
            order = np.random.randint(1, self.max_order + 1)
            
            # 随机选择不重复的变量
            vars = tuple(sorted(np.random.choice(self.n_vars, size=order, replace=False)))
            
            if vars not in generated_terms:
                generated_terms.add(vars)
                # 生成随机系数 [-10, 10]
                coef = np.random.uniform(-10, 10)
                terms_dict[order].append((float(coef), vars))
        
        return dict(terms_dict)

    def create_hubo_solver(self, terms_dict, xi, dt, n_iter, method, batch_size=1):
        """创建HUBO求解器"""
        term_orders, term_coeffs, term_indices = self.terms_dict_to_tensor(terms_dict, self.n_vars)
        solver = HUBOSolverBatch(
            term_orders, term_coeffs, term_indices, self.n_vars, batch_size,
            xi=xi, dt=dt, n_iter=n_iter, method=method, device='cpu'
        )
        return solver

    def terms_dict_to_tensor(self, terms_dict, n_vars):
        """将terms_dict转换为张量表示"""
        all_terms = []
        for order, terms in terms_dict.items():
            for coef, vars_list in terms:
                all_terms.append((order, coef, vars_list))
        
        max_order = max([order for order, _, _ in all_terms]) if all_terms else 1
        
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

    def run_single_experiment(self, terms_dict, method, opt_method, n_iter, run_id):
        """
        运行单次实验
        
        返回:
        best_energy: 最佳能量值
        avg_last_10: 最后10次平均能量值
        total_time: 总时间
        params: 找到的最佳参数
        energy_history: 完整的能量历史
        """
        start_time = time.time()
        
        # 定义目标函数
        obj_func = partial(self.objective_function, 
                          terms_dict=terms_dict, 
                          n_iter=n_iter, 
                          method_name=method)
        
        # 运行优化
        if opt_method == 'Bayes':
            result = self.run_bayes_optimization(obj_func, n_iter)
        elif opt_method == 'SA':
            result = self.run_simulated_annealing(obj_func, n_iter)
        elif opt_method == 'CMA-ES':
            result = self.run_cma_es(obj_func, n_iter)
        elif opt_method == 'Nelder-Mead':
            result = self.run_nelder_mead(obj_func, n_iter)
        elif opt_method == 'Hyperopt':
            result = self.run_hyperopt(obj_func, n_iter)
        elif opt_method == 'QGA':
            result = self.run_quantum_genetic(obj_func, n_iter)
        else:
            return None
        
        total_time = time.time() - start_time
        
        if result:
            best_energy = result['best_energy']
            params = result['params']
            
            # 计算最后10次平均能量值
            avg_last_10 = self.calculate_last_10_energy(terms_dict, params, n_iter, method)
            
            # 获取完整的能量历史
            energy_history = self.get_energy_history(terms_dict, params, n_iter, method)
            
            return {
                'best_energy': best_energy,
                'avg_last_10': avg_last_10,
                'total_time': total_time,
                'params': params,
                'energy_history': energy_history
            }
        
        return None

    def get_energy_history(self, terms_dict, params, n_iter, method):
        """获取完整的能量历史"""
        xi, dt = params
        solver = self.create_hubo_solver(terms_dict, xi, dt, n_iter, method)
        _, _ = solver.solve()
        return [float(e[0]) for e in solver.energy_history]

    def objective_function(self, params, terms_dict, n_iter, method_name):
        """目标函数"""
        xi, dt = params
        solver = self.create_hubo_solver(terms_dict, xi, dt, n_iter, method_name)
        _, best_energy = solver.solve()
        return -float(best_energy[0].item())

    def calculate_last_10_energy(self, terms_dict, params, n_iter, method):
        """计算最后10次迭代的平均能量值"""
        xi, dt = params
        energies = []
        
        for _ in range(10):
            solver = self.create_hubo_solver(terms_dict, xi, dt, n_iter, method)
            _, energy = solver.solve()
            energies.append(float(energy[0].item()))
        
        return np.mean(energies)

    def run_bayes_optimization(self, obj_func, n_iter):
        """运行贝叶斯优化"""
        space = [(0.01, 0.5), (0.01, 0.3)]
        n_initial_points = min(5, self.n_calls // 2)
        
        try:
            res = gp_minimize(
                obj_func, space, 
                n_calls=self.n_calls,
                n_initial_points=n_initial_points,
                random_state=42
            )
            return {
                'best_energy': -res.fun,
                'params': res.x
            }
        except Exception as e:
            print(f"Bayes优化失败: {e}")
            return None

    def run_simulated_annealing(self, obj_func, n_iter):
        """运行模拟退火"""
        bounds = [(0.01, 0.5), (0.01, 0.3)]
        
        try:
            res = dual_annealing(obj_func, bounds, maxiter=self.n_calls, seed=42)
            return {
                'best_energy': -res.fun,
                'params': res.x
            }
        except Exception as e:
            print(f"SA优化失败: {e}")
            return None

    def run_cma_es(self, obj_func, n_iter):
        """运行CMA-ES"""
        x0 = [0.1, 0.1]
        
        try:
            es = cma.CMAEvolutionStrategy(x0, 0.1, {'maxfevals': self.n_calls})
            es.optimize(obj_func)
            return {
                'best_energy': -es.result.fbest,
                'params': es.result.xbest
            }
        except Exception as e:
            print(f"CMA-ES优化失败: {e}")
            return None

    def run_nelder_mead(self, obj_func, n_iter):
        """运行Nelder-Mead"""
        x0 = [0.1, 0.1]
        bounds = [(0.01, 0.5), (0.01, 0.3)]
        
        try:
            res = minimize(obj_func, x0, method='Nelder-Mead', 
                          bounds=bounds, options={'maxfev': self.n_calls})
            return {
                'best_energy': -res.fun,
                'params': res.x
            }
        except Exception as e:
            print(f"Nelder-Mead优化失败: {e}")
            return None

    def run_hyperopt(self, obj_func, n_iter):
        """运行Hyperopt"""
        space = {
            'xi': hp.uniform('xi', 0.01, 0.5),
            'dt': hp.uniform('dt', 0.01, 0.3)
        }
        
        def hyperopt_obj(params):
            xi = params['xi']
            dt = params['dt']
            return obj_func((xi, dt))
        
        try:
            trials = Trials()
            best = fmin(
                fn=hyperopt_obj,
                space=space,
                algo=tpe.suggest,
                max_evals=self.n_calls,
                trials=trials,
                rstate=np.random.default_rng(42)
            )
            
            best_trial = trials.best_trial
            params = [
                best_trial['misc']['vals']['xi'][0],
                best_trial['misc']['vals']['dt'][0]
            ]
            
            return {
                'best_energy': -best_trial['result']['loss'],
                'params': params
            }
        except Exception as e:
            print(f"Hyperopt优化失败: {e}")
            return None

    def run_quantum_genetic(self, obj_func, n_iter):
        """运行量子遗传算法"""
        bounds = [(0.01, 0.5), (0.01, 0.3)]
        
        try:
            qga = QuantumGeneticAlgorithm(obj_func, bounds, self.n_calls, pop_size=10, n_qubits=8)
            best_params, best_energy = qga.optimize()
            return {
                'best_energy': -best_energy,
                'params': best_params
            }
        except Exception as e:
            print(f"QGA优化失败: {e}")
            return None

    def run_comprehensive_experiment(self):
        """运行综合性实验"""
        print("=== 大规模自旋HUBO实验 ===")
        print(f"自旋数: {self.n_vars}")
        print(f"项数: {self.n_terms}")
        print(f"运行轮数: {self.n_runs}")
        print(f"算法: {self.methods}")
        print(f"优化方法: {self.opt_methods}")
        print(f"迭代次数: {self.n_iter_values}")
        
        # 生成HUBO问题
        hubo_problem = self.generate_sparse_hubo_terms()
        print(f"生成的HUBO问题包含 {len(hubo_problem)} 个不同阶数的项")
        
        # 保存HUBO问题
        with open(os.path.join(self.result_dir, 'hubo_problem.json'), 'w') as f:
            json.dump(convert_to_python_types(hubo_problem), f, indent=4)
        
        # 运行实验
        for n_iter in self.n_iter_values:
            print(f"\n--- 迭代次数: {n_iter} ---")
            
            for method in self.methods:
                print(f"\n算法: {method}")
                
                for opt_method in self.opt_methods:
                    print(f"  优化方法: {opt_method}", end="")
                    
                    run_results = []
                    energy_histories = []
                    total_start_time = time.time()
                    
                    # 运行多轮
                    for run_id in range(self.n_runs):
                        result = self.run_single_experiment(
                            hubo_problem, method, opt_method, n_iter, run_id
                        )
                        
                        if result:
                            run_results.append(result)
                            energy_histories.append(result['energy_history'])
                    
                    total_time = time.time() - total_start_time
                    
                    if run_results:
                        # 计算统计指标
                        best_energies = [r['best_energy'] for r in run_results]
                        avg_last_10 = [r['avg_last_10'] for r in run_results]
                        
                        stats = {
                            'n_iter': n_iter,
                            'method': method,
                            'optimizer': opt_method,
                            'best_energy_mean': np.mean(best_energies),
                            'best_energy_std': np.std(best_energies),
                            'avg_last_10_mean': np.mean(avg_last_10),
                            'avg_last_10_std': np.std(avg_last_10),
                            'total_time': total_time,
                            'success_rate': len(run_results) / self.n_runs,
                            'n_success': len(run_results)
                        }
                        
                        self.summary_data.append(stats)
                        
                        # 存储能量历史
                        key = f"{n_iter}_{method}_{opt_method}"
                        self.energy_histories[key] = {
                            'histories': energy_histories,
                            'mean_history': np.mean(energy_histories, axis=0),
                            'std_history': np.std(energy_histories, axis=0)
                        }
                        
                        print(f" - 成功: {len(run_results)}/{self.n_runs}, "
                              f"最佳能量: {stats['best_energy_mean']:.4f}±{stats['best_energy_std']:.4f}, "
                              f"总时间: {total_time:.2f}s")
                    else:
                        print(f" - 失败: 0/{self.n_runs}")
        
        # 保存结果
        self.save_results()
        
        # 生成可视化
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
            json.dump(convert_to_python_types(self.results), f, indent=4)
        
        # 保存能量历史
        energy_history_data = {}
        for key, value in self.energy_histories.items():
            energy_history_data[key] = {
                'mean_history': value['mean_history'].tolist(),
                'std_history': value['std_history'].tolist()
            }
        
        with open(os.path.join(self.result_dir, 'energy_histories.json'), 'w') as f:
            json.dump(energy_history_data, f, indent=4)
        
        print(f"结果已保存到 {self.result_dir}")

    def create_visualizations(self):
        """创建可视化图表"""
        if not self.summary_data:
            print("没有数据可生成可视化")
            return
        
        df = pd.DataFrame(self.summary_data)
        
        # 1. 主要评价指标：最佳能量值
        self.plot_main_metrics(df)
        
        # 2. 辅助指标：最后10次平均能量值
        self.plot_secondary_metrics(df)
        
        # 3. 时间分析
        self.plot_time_analysis(df)
        
        # 4. 热力图分析
        self.plot_heatmaps(df)
        
        # 5. 能量收敛曲线
        self.plot_energy_convergence()
        
        print("可视化图表已生成")

    def plot_energy_convergence(self):
        """绘制能量收敛曲线"""
        if not self.energy_histories:
            print("没有能量历史数据")
            return
        
        # 创建能量收敛曲线图
        plt.figure(figsize=(16, 12))
        
        # 选择一些代表性的配置进行展示
        selected_configs = []
        for n_iter in self.n_iter_values:
            for method in self.methods:
                for opt_method in self.opt_methods:
                    key = f"{n_iter}_{method}_{opt_method}"
                    if key in self.energy_histories:
                        selected_configs.append((n_iter, method, opt_method, key))
        
        # 按迭代次数分组绘制
        max_subplots = 4
        for i, n_iter in enumerate(self.n_iter_values):
            if i >= max_subplots:
                break
            plt.subplot(2, 2, i+1)
            iter_configs = [cfg for cfg in selected_configs if cfg[0] == n_iter]
            for n_iter_val, method, opt_method, key in iter_configs:
                if key in self.energy_histories:
                    data = self.energy_histories[key]
                    mean_history = data['mean_history']
                    std_history = data['std_history']
                    iterations = range(1, len(mean_history) + 1)
                    plt.plot(iterations, mean_history, 
                            label=f"{method}-{opt_method}", linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Energy Value')
            plt.title(f'Energy Convergence (n_iter={n_iter})')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # 使用对数坐标更好地显示收敛
        
        # 比较不同HUBO方法的收敛速度
        plt.subplot(2, 2, 3)
        best_opt_method = 'Bayes'  # 选择一个优化方法进行比较
        for method in self.methods:
            for n_iter in self.n_iter_values:
                key = f"{n_iter}_{method}_{best_opt_method}"
                if key in self.energy_histories:
                    data = self.energy_histories[key]
                    mean_history = data['mean_history']
                    iterations = range(1, len(mean_history) + 1)
                    plt.plot(iterations, mean_history, 
                            label=f"{method} (n_iter={n_iter})", linewidth=2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Energy Value')
        plt.title(f'Convergence Comparison (Optimizer: {best_opt_method})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 比较不同优化方法的收敛速度
        plt.subplot(2, 2, 4)
        best_method = 'dSB'  # 选择一个HUBO方法进行比较
        n_iter = 100  # 选择较大的迭代次数
        for opt_method in self.opt_methods:
            key = f"{n_iter}_{best_method}_{opt_method}"
            if key in self.energy_histories:
                data = self.energy_histories[key]
                mean_history = data['mean_history']
                iterations = range(1, len(mean_history) + 1)
                plt.plot(iterations, mean_history, 
                        label=f"{opt_method}", linewidth=2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Energy Value')
        plt.title(f'Optimizer Comparison (Method: {best_method}, n_iter={n_iter})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'energy_convergence.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制详细的单次运行能量曲线
        self.plot_detailed_energy_curves()

    def plot_detailed_energy_curves(self):
        """绘制详细的单次运行能量曲线"""
        if not self.energy_histories:
            return
        
        # 为每个配置创建单独的能量曲线图
        for n_iter in self.n_iter_values:
            for method in self.methods:
                plt.figure(figsize=(12, 8))
                
                has_data = False
                for opt_method in self.opt_methods:
                    key = f"{n_iter}_{method}_{opt_method}"
                    if key in self.energy_histories:
                        has_data = True
                        data = self.energy_histories[key]
                        mean_history = data['mean_history']
                        std_history = data['std_history']
                        iterations = range(1, len(mean_history) + 1)
                        
                        plt.plot(iterations, mean_history, 
                                label=f"{opt_method}", linewidth=2)
                        
                
                if has_data:
                    plt.xlabel('Iteration')
                    plt.ylabel('Energy Value')
                    plt.title(f'Energy Convergence: {method} (n_iter={n_iter})')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.yscale('log')
                    
                    filename = f"energy_convergence_{method}_niter{n_iter}.png"
                    plt.savefig(os.path.join(self.result_dir, filename), 
                               dpi=300, bbox_inches='tight')
                
                plt.close()

    def plot_main_metrics(self, df):
        """绘制主要评价指标图"""
        plt.figure(figsize=(15, 10))
        
        # 子图1: 各优化方法在不同迭代次数下的最佳能量值
        plt.subplot(2, 2, 1)
        for opt_method in df['optimizer'].unique():
            opt_data = df[df['optimizer'] == opt_method]
            plt.plot(opt_data['n_iter'], opt_data['best_energy_mean'], 
                    label=opt_method, marker='o', linewidth=2)
        plt.xlabel('Iterations')
        plt.ylabel('Best Energy Value')
        plt.title('Main Metric: Best Energy vs Iterations by Optimizer')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 各HUBO方法在不同迭代次数下的最佳能量值
        plt.subplot(2, 2, 2)
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            plt.plot(method_data['n_iter'], method_data['best_energy_mean'], 
                    label=method, marker='s', linewidth=2)
        plt.xlabel('Iterations')
        plt.ylabel('Best Energy Value')
        plt.title('Main Metric: Best Energy vs Iterations by HUBO Method')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图3: 最佳性能组合
        plt.subplot(2, 2, 3)
        best_combinations = df.nlargest(10, 'best_energy_mean')
        labels = [f"{row['method']}-{row['optimizer']}\n(n_iter={row['n_iter']})" 
                 for _, row in best_combinations.iterrows()]
        plt.barh(range(len(best_combinations)), best_combinations['best_energy_mean'])
        plt.yticks(range(len(best_combinations)), labels, fontsize=8)
        plt.xlabel('Best Energy Value')
        plt.title('Top 10 Method Combinations (Main Metric)')
        
        # 子图4: 成功率
        plt.subplot(2, 2, 4)
        success_rates = df.groupby('optimizer')['success_rate'].mean().sort_values()
        plt.barh(range(len(success_rates)), success_rates.values)
        plt.yticks(range(len(success_rates)), success_rates.index)
        plt.xlabel('Success Rate')
        plt.title('Optimizer Success Rates')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'main_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_secondary_metrics(self, df):
        """绘制辅助指标图"""
        plt.figure(figsize=(12, 8))
        
        # 子图1: 最后10次平均能量值 vs 迭代次数
        plt.subplot(1, 2, 1)
        for opt_method in df['optimizer'].unique():
            opt_data = df[df['optimizer'] == opt_method]
            plt.plot(opt_data['n_iter'], opt_data['avg_last_10_mean'], 
                    label=opt_method, marker='o', linewidth=2)
        plt.xlabel('Iterations')
        plt.ylabel('Average Last 10 Energy Value')
        plt.title('Secondary Metric: Last 10 Average vs Iterations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 主要指标 vs 辅助指标散点图
        plt.subplot(1, 2, 2)
        for opt_method in df['optimizer'].unique():
            opt_data = df[df['optimizer'] == opt_method]
            plt.scatter(opt_data['best_energy_mean'], opt_data['avg_last_10_mean'], 
                       label=opt_method, alpha=0.7, s=60)
        plt.xlabel('Best Energy Value (Main Metric)')
        plt.ylabel('Last 10 Average (Secondary Metric)')
        plt.title('Main vs Secondary Metrics Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'secondary_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_time_analysis(self, df):
        """绘制时间分析图"""
        plt.figure(figsize=(12, 8))
        
        # 子图1: 总时间 vs 迭代次数
        plt.subplot(1, 2, 1)
        for opt_method in df['optimizer'].unique():
            opt_data = df[df['optimizer'] == opt_method]
            plt.plot(opt_data['n_iter'], opt_data['total_time'], 
                    label=opt_method, marker='o', linewidth=2)
        plt.xlabel('Iterations')
        plt.ylabel('Total Time for 50 Runs (s)')
        plt.title('Total Computation Time vs Iterations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 效率分析 (能量值/时间)
        plt.subplot(1, 2, 2)
        efficiency = df['best_energy_mean'] / df['total_time']
        for i, (_, row) in enumerate(df.iterrows()):
            plt.scatter(row['total_time'], row['best_energy_mean'], 
                       label=f"{row['method']}-{row['optimizer']}", s=80, alpha=0.7)
            plt.annotate(f"{row['n_iter']}", (row['total_time'], row['best_energy_mean']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel('Total Time (s)')
        plt.ylabel('Best Energy Value')
        plt.title('Efficiency Analysis: Energy vs Time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'time_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_heatmaps(self, df):
        """绘制热力图分析"""
        plt.figure(figsize=(16, 12))
        
        # 子图1: 优化方法 vs 迭代次数 (最佳能量值)
        plt.subplot(2, 2, 1)
        pivot_energy = df.pivot_table(values='best_energy_mean', 
                                     index='optimizer', columns='n_iter', aggfunc='mean')
        sns.heatmap(pivot_energy, annot=True, fmt='.3f', cmap='viridis', 
                   cbar_kws={'label': 'Best Energy Value'})
        plt.title('Optimizer vs Iterations (Main Metric)')
        plt.xlabel('Iterations')
        plt.ylabel('Optimizer')
        
        # 子图2: HUBO方法 vs 迭代次数 (最佳能量值)
        plt.subplot(2, 2, 2)
        pivot_method = df.pivot_table(values='best_energy_mean', 
                                     index='method', columns='n_iter', aggfunc='mean')
        sns.heatmap(pivot_method, annot=True, fmt='.3f', cmap='plasma', 
                   cbar_kws={'label': 'Best Energy Value'})
        plt.title('HUBO Method vs Iterations (Main Metric)')
        plt.xlabel('Iterations')
        plt.ylabel('HUBO Method')
        
        # 子图3: 综合性能矩阵
        plt.subplot(2, 2, 3)
        combo_performance = df.pivot_table(values='best_energy_mean', 
                                         index='optimizer', columns='method', aggfunc='mean')
        sns.heatmap(combo_performance, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Best Energy Value'})
        plt.title('Method Combination Performance Matrix')
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

# 其他辅助类保持不变...
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
                    'best_energy': f'{-self.best_fitness:.4f}'
                })
        
        return self.best_solution, self.best_fitness

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
        
        # 初始化变量
        self.x = 0.01 * torch.randn(batch_size, n_vars, device=device)
        self.y = 0.01 * torch.randn(batch_size, n_vars, device=device)
        
        # 固定参数
        self.Delta = 0.5
        
        # 用于记录历史
        self.energy_history = []

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

# 使用示例
if __name__ == "__main__":
    # 创建大规模自旋实验
    experiment = LargeSpinHUBOExperiment(
        n_vars=300,      # 300个自旋
        n_terms=10,      # 10项多项式
        max_order=3,     # 最大3阶
        n_runs=50,       # 50轮实验（测试用）
        result_dir='./HUBO2/results_large_spin',
        opt_methods = ['Bayes', 'CMA-ES', 'Nelder-Mead', 'Hyperopt', 'QGA'] #可选优化方法，有Bayes, SA, CMA-ES, Nelder-Mead, Hyperopt, QGA
    )
    
    # 运行实验
    experiment.run_comprehensive_experiment()