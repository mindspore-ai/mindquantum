"""
批量Optuna调优SB算法的超参数
支持同时优化多个QPLIB文件，寻找通用最优参数或每个文件的最优参数
支持BSB和DSB的12种算法变体（CPU/GPU × fp16/int8/fp8）
"""

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import scipy.sparse as sp
import torch
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from qplib_parser import QPLIBParser, ProblemModel
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import time


class SBOptimizer:
    """SB算法优化器（单文件）"""

    def __init__(self, file_path, device="cuda", target_value=None, algorithm="BSB", precision="fp32", use_gpu=True):
        """
        参数:
            file_path: QPLIB文件路径
            device: 'cuda' 或 'cpu'
            target_value: 已知的最优值
            algorithm: 'BSB' 或 'DSB'
            precision: 'fp32', 'fp16', 'int8', 'fp8'
            use_gpu: 是否使用GPU（仅用于标识，实际由device参数控制）
        """
        self.algorithm = algorithm.upper()
        self.precision = precision
        self.use_gpu = use_gpu

        if self.algorithm not in ["BSB", "DSB"]:
            raise ValueError(f"不支持的算法类型: {algorithm}，请选择 'BSB' 或 'DSB'")

        if self.precision not in ["fp32", "fp16", "int8", "fp8"]:
            raise ValueError(f"不支持的精度类型: {precision}")

        # 算法完整名称
        hw_device = "GPU" if use_gpu else "CPU"
        self.full_algorithm_name = f"{self.algorithm}_{hw_device}_{self.precision}"

        # 解析QPLIB文件
        model = QPLIBParser.parse_qplib_file(file_path)

        self.problem_name = model.problem_name
        self.objective_sense = model.objective_sense
        self.num_vars = model.num_vars
        self.constant_term = model.constant_term
        self.target_value = target_value
        self.device = device
        self.file_path = file_path

        # 构建二次项稀疏矩阵
        if model.quad_terms_indices and model.quad_terms_coeffs:
            row_indices = [idx[0] for idx in model.quad_terms_indices]
            col_indices = [idx[1] for idx in model.quad_terms_indices]
            values = model.quad_terms_coeffs

            quad_terms = np.array(
                [[row_indices[i], col_indices[i], values[i]] for i in range(len(values))], dtype=np.float32
            )

            diag_index = np.nonzero(quad_terms[:, 0] == quad_terms[:, 1])[0]
            diag_terms = None
            if len(diag_index) > 0:
                diag_terms = quad_terms[diag_index]
                quad_terms = np.delete(quad_terms, diag_index, axis=0)

            quad_terms_sparse = sp.csr_matrix(
                (
                    np.concatenate([quad_terms[:, 2] / 2.0, quad_terms[:, 2] / 2.0]),
                    (
                        np.concatenate([quad_terms[:, 0], quad_terms[:, 1]]),
                        np.concatenate([quad_terms[:, 1], quad_terms[:, 0]]),
                    ),
                ),
                shape=(self.num_vars, self.num_vars),
            )
        else:
            quad_terms_sparse = sp.csr_matrix((self.num_vars, self.num_vars))
            diag_terms = None

        # 转换为PyTorch稀疏张量
        crow_indices = torch.tensor(quad_terms_sparse.indptr, dtype=torch.int32)
        col_indices = torch.tensor(quad_terms_sparse.indices, dtype=torch.int32)
        values = torch.tensor(quad_terms_sparse.data, dtype=torch.float)
        self.quad_terms = torch.sparse_csr_tensor(
            crow_indices, col_indices, values, size=quad_terms_sparse.shape, dtype=torch.float, device=device
        )

        # 处理线性项
        if model.linear_terms is not None:
            self.linear_terms = torch.from_numpy(model.linear_terms).float().to(device)
            if diag_terms is not None:
                self.linear_terms[diag_terms[:, 0]] = self.linear_terms[diag_terms[:, 0]] + diag_terms[:, 2] / 2
        else:
            if diag_terms is None:
                self.linear_terms = None
            else:
                self.linear_terms = torch.zeros(self.num_vars, device=device, dtype=torch.float32)
                self.linear_terms[diag_terms[:, 0]] = diag_terms[:, 2] / 2

        # 计算缩放因子
        if self.linear_terms is not None:
            h_max = torch.abs(self.linear_terms).max().cpu().item()
        else:
            h_max = 0
        self.scaling = np.max([np.abs(quad_terms_sparse.data).max() if len(quad_terms_sparse.data) > 0 else 0, h_max])
        if self.scaling == 0:
            self.scaling = 1.0

        # 是否为纯线性问题
        self.is_linear = len(quad_terms_sparse.data) == 0

        if not self.is_linear:
            self.quad_terms.values().div_(self.scaling)
            if self.linear_terms is not None:
                self.linear_terms /= self.scaling

        # 计算默认xi值
        if self.is_linear:
            self.default_xi = 0.5 * np.sqrt(self.num_vars - 1) / torch.sqrt((self.linear_terms**2).sum()).item()
        else:
            if self.linear_terms is not None:
                self.default_xi = (
                    0.5
                    * np.sqrt(self.num_vars - 1)
                    / torch.sqrt((self.quad_terms.values() ** 2).sum() + (self.linear_terms**2).sum()).item()
                )
            else:
                self.default_xi = (
                    0.5 * np.sqrt(self.num_vars - 1) / torch.sqrt((self.quad_terms.values() ** 2).sum()).item()
                )

    def calc_grad(self, x):
        """计算目标函数的梯度"""
        if x.dim() == 1:
            x_matrix = x.unsqueeze(1)
            quad_grad = torch.sparse.mm(self.quad_terms, x_matrix).squeeze(1)
        else:
            quad_grad = torch.sparse.mm(self.quad_terms, x)

        if self.linear_terms is not None:
            if x.dim() == 2:
                grad = quad_grad + self.linear_terms.unsqueeze(1).expand_as(quad_grad)
            else:
                grad = quad_grad + self.linear_terms
        else:
            grad = quad_grad
        return grad

    def calc_energy(self, x):
        """计算目标函数值"""
        x_float = x.float()
        scaled_energy = 0.5 * (self.quad_terms @ x_float * x_float).sum(0)
        if self.linear_terms is not None:
            scaled_energy += self.linear_terms @ x_float

        if not self.is_linear:
            scaled_energy *= self.scaling

        if self.constant_term is not None:
            scaled_energy += self.constant_term
        return scaled_energy

    def solve(self, xi, batch_size, max_iters, dt, init_noise):
        """使用给定参数运行SB算法"""
        x = init_noise * (torch.randn((self.num_vars, batch_size), dtype=torch.float, device=self.device)) + 0.5
        y = init_noise * (torch.randn((self.num_vars, batch_size), dtype=torch.float, device=self.device))

        pt = np.linspace(0, 1, max_iters)

        if self.algorithm == "DSB":
            x_sign = torch.where(x > 0.5, 1.0, 0.0)

        for i in range(max_iters):
            if self.algorithm == "BSB":
                grad = self.calc_grad(x)
            else:
                grad = self.calc_grad(x_sign)

            if self.objective_sense == "maximize":
                grad_term = xi * grad
            else:
                grad_term = -xi * grad

            y += (-(1 - pt[i]) * (x - 0.5) + grad_term) * dt
            x += y * dt

            cond = torch.abs(x - 0.5) > 0.5
            x = torch.where(cond, (torch.sign(x - 0.5) + 1) / 2.0, x)
            y = torch.where(cond, 0, y)

            if self.algorithm == "DSB":
                x_sign = torch.where(x > 0.5, 1.0, 0.0)

        best_value = float("-inf") if self.objective_sense == "maximize" else float("inf")

        for i in range(x.shape[1]):
            binary_solution = (x[:, i] > 0.5).int()
            current_value = self.calc_energy(binary_solution).item()

            if self.objective_sense == "maximize":
                if current_value > best_value:
                    best_value = current_value
            else:
                if current_value < best_value:
                    best_value = current_value

        return best_value


class BatchOptunaOptimizer:
    """批量Optuna优化器（支持多文件）"""

    def __init__(
        self,
        file_paths,
        device="cuda",
        target_values=None,
        algorithms="ALL",
        optimization_mode="shared",
        n_trials=100,
        fixed_sigma=None,
    ):
        """
        批量Optuna优化器

        参数:
            file_paths: QPLIB文件路径列表
            device: 'cuda' 或 'cpu'
            target_values: 每个文件的最优值列表 (与file_paths一一对应)
            algorithms:
                - 'ALL': 所有12种算法（6种BSB + 6种DSB）
                - 'BSB': 所有6种BSB算法
                - 'DSB': 所有6种DSB算法
                - 具体算法列表: ['BSB_GPU_fp16', 'DSB_CPU_fp8', ...]
            optimization_mode:
                - 'shared': 所有文件共享同一组最优参数（推荐）
                - 'individual': 每个文件单独优化参数
                - 'adaptive': 先共享优化，再对效果差的文件单独优化
            n_trials: 每个study的试验次数
            fixed_sigma: 固定的sigma值，如果为None则也会优化sigma
        """
        self.file_paths = file_paths
        self.device = device
        self.target_values = target_values if target_values else [None] * len(file_paths)
        self.optimization_mode = optimization_mode
        self.n_trials = n_trials
        self.fixed_sigma = fixed_sigma

        # 定义所有12种算法
        self.all_algorithms = [
            # BSB算法（6种）
            "BSB_CPU_fp16",
            "BSB_CPU_int8",
            "BSB_CPU_fp8",
            "BSB_GPU_fp16",
            "BSB_GPU_int8",
            "BSB_GPU_fp8",
            # DSB算法（6种）
            "DSB_CPU_fp16",
            "DSB_CPU_int8",
            "DSB_CPU_fp8",
            "DSB_GPU_fp16",
            "DSB_GPU_int8",
            "DSB_GPU_fp8",
        ]

        # 根据参数确定要优化的算法列表
        if algorithms == "ALL":
            self.algorithms_to_optimize = self.all_algorithms
        elif algorithms == "BSB":
            self.algorithms_to_optimize = [a for a in self.all_algorithms if a.startswith("BSB_")]
        elif algorithms == "DSB":
            self.algorithms_to_optimize = [a for a in self.all_algorithms if a.startswith("DSB_")]
        elif isinstance(algorithms, list):
            self.algorithms_to_optimize = algorithms
        else:
            raise ValueError(f"不支持的算法类型: {algorithms}")

        if self.optimization_mode not in ["shared", "individual", "adaptive"]:
            raise ValueError(f"不支持的优化模式: {optimization_mode}")

        print("=" * 80)
        print(f"🚀 初始化批量优化器")
        print("=" * 80)
        print(f"  文件数量: {len(file_paths)}")
        print(f"  待优化算法数量: {len(self.algorithms_to_optimize)}")
        print(f"  优化模式: {self.optimization_mode}")
        print(f"  试验次数: {n_trials}")
        print(f"  固定Sigma: {fixed_sigma if fixed_sigma else '否(自动优化)'}")
        print("=" * 80)
        print(f"待优化的算法:")
        for algo in self.algorithms_to_optimize:
            print(f"  - {algo}")

        # 初始化所有文件和算法的优化器
        self.sb_optimizers = {}
        for i, (file_path, target_val) in enumerate(zip(file_paths, self.target_values)):
            self.sb_optimizers[file_path] = {}
            for algo_name in self.algorithms_to_optimize:
                # 解析算法名称
                parts = algo_name.split("_")
                algorithm = parts[0]  # BSB or DSB
                hw_device = parts[1]  # CPU or GPU
                precision = parts[2]  # fp16, int8, fp8

                use_gpu = hw_device == "GPU" and device == "cuda"

                sb_opt = SBOptimizer(
                    file_path=file_path,
                    device=device,
                    target_value=target_val,
                    algorithm=algorithm,
                    precision=precision,
                    use_gpu=use_gpu,
                )
                self.sb_optimizers[file_path][algo_name] = sb_opt

                if i == 0:  # 只在第一个文件时打印信息
                    print(f"  [{algo_name}] vars={sb_opt.num_vars}, xi_default={sb_opt.default_xi:.6f}")

        # 存储优化结果
        self.results = {}

    def optimize_single_algorithm(self, algo_name):
        """
        优化单个算法的所有文件
        返回该算法的优化结果
        """
        print(f"\n{'='*80}")
        print(f"🔧 优化算法: {algo_name}")
        print(f"{'='*80}")

        # 获取该算法在所有文件中的优化器
        optimizers = {fp: opt[algo_name] for fp, opt in self.sb_optimizers.items() if algo_name in opt}
        optimizer_list = list(optimizers.values())

        if self.optimization_mode == "individual":
            return self.optimize_individual_for_algorithm(algo_name, optimizer_list)
        elif self.optimization_mode == "shared":
            return self.optimize_shared_for_algorithm(algo_name, optimizer_list)
        elif self.optimization_mode == "adaptive":
            return self.optimize_adaptive_for_algorithm(algo_name, optimizer_list)

    def optimize_shared_for_algorithm(self, algo_name, optimizer_list):
        """为单个算法优化共享参数"""
        print(f"\n📊 模式: 共享参数优化")
        study_name = f"{algo_name}_shared_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        def objective_shared(trial):
            xi_multiplier = trial.suggest_float("xi_multiplier", 0.1, 10.0, log=True)
            xi = optimizer_list[0].default_xi * xi_multiplier

            if self.fixed_sigma is not None:
                init_noise = self.fixed_sigma
            else:
                init_noise = trial.suggest_float("init_noise", 0.05, 0.3)

            batch_size = 100
            max_iters = 1000
            dt = 0.1

            total_value = 0.0
            valid_count = 0

            for sb_opt in optimizer_list:
                try:
                    xi_adjusted = xi * sb_opt.default_xi / optimizer_list[0].default_xi
                    energy = sb_opt.solve(xi_adjusted, batch_size, max_iters, dt, init_noise)

                    if sb_opt.objective_sense == "maximize":
                        total_value -= energy
                    else:
                        total_value += energy

                    valid_count += 1
                except Exception as e:
                    print(f"  ❌ {sb_opt.problem_name} 失败: {e}")
                    total_value += float("inf")

            avg_value = total_value / valid_count if valid_count > 0 else float("inf")
            trial.set_user_attr("xi", xi)
            trial.set_user_attr("valid_count", valid_count)
            return avg_value

        study = optuna.create_study(
            study_name=study_name, direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
        )

        print(f"开始优化 {len(optimizer_list)} 个文件的共享参数...")
        study.optimize(objective_shared, n_trials=self.n_trials, show_progress_bar=True)

        best_trial = study.best_trial
        best_params = best_trial.params
        best_xi = optimizer_list[0].default_xi * best_params["xi_multiplier"]
        best_sigma = self.fixed_sigma if self.fixed_sigma else best_params.get("init_noise", 0.1)

        print(f"\n✅ {algo_name} 共享参数优化完成!")
        print(f"  最佳xi: {best_xi:.6f}")
        print(f"  最佳sigma: {best_sigma:.2f}")

        # 评估所有文件使用最佳参数的效果
        results = {}
        for sb_opt in optimizer_list:
            xi_adjusted = best_xi * sb_opt.default_xi / optimizer_list[0].default_xi
            energy = sb_opt.solve(xi_adjusted, 100, 800, 0.1, best_sigma)

            gap = None
            if sb_opt.target_value is not None:
                gap = abs(energy - sb_opt.target_value) / abs(sb_opt.target_value) * 100

            results[sb_opt.problem_name] = {
                "energy": energy,
                "xi": xi_adjusted,
                "sigma": best_sigma,
                "target_value": sb_opt.target_value,
                "gap_percent": gap,
                "optimization_mode": "shared",
            }

        return results

    def optimize_individual_for_algorithm(self, algo_name, optimizer_list):
        """为单个算法优化每个文件的独立参数"""
        print(f"\n📊 模式: 独立参数优化")
        results = {}

        for i, sb_opt in enumerate(optimizer_list):
            print(f"\n[{i+1}/{len(optimizer_list)}] 优化 {sb_opt.problem_name}...")

            study_name = f"{algo_name}_{sb_opt.problem_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            def objective_single(trial):
                xi_multiplier = trial.suggest_float("xi_multiplier", 0.1, 10.0, log=True)
                xi = sb_opt.default_xi * xi_multiplier

                if self.fixed_sigma is not None:
                    init_noise = self.fixed_sigma
                else:
                    init_noise = trial.suggest_float("init_noise", 0.05, 0.3)

                try:
                    energy = sb_opt.solve(xi, 100, 800, 0.1, init_noise)

                    if sb_opt.objective_sense == "maximize":
                        return -energy
                    else:
                        return energy
                except:
                    return float("inf")

            study = optuna.create_study(
                study_name=study_name, direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
            )

            study.optimize(objective_single, n_trials=self.n_trials, show_progress_bar=False)

            best_trial = study.best_trial
            best_xi = sb_opt.default_xi * best_trial.params["xi_multiplier"]
            best_sigma = self.fixed_sigma if self.fixed_sigma else best_trial.params.get("init_noise", 0.1)
            best_energy = -best_trial.value if sb_opt.objective_sense == "maximize" else best_trial.value

            gap = None
            if sb_opt.target_value is not None:
                gap = abs(best_energy - sb_opt.target_value) / abs(sb_opt.target_value) * 100

            results[sb_opt.problem_name] = {
                "energy": best_energy,
                "xi": best_xi,
                "sigma": best_sigma,
                "target_value": sb_opt.target_value,
                "gap_percent": gap,
            }

            print(f"  ✅ 最佳: xi={best_xi:.6f}, sigma={best_sigma:.2f}, energy={best_energy:.4f}", end="")
            if gap is not None:
                print(f" (gap: {gap:.2f}%)")
            else:
                print()

        return results

    def optimize_adaptive_for_algorithm(self, algo_name, optimizer_list):
        """为单个算法进行自适应优化"""
        print(f"\n📊 模式: 自适应优化")
        print("  第一阶段: 共享参数优化")
        print("  第二阶段: 对gap>10%的文件单独优化")

        # 第一阶段：共享优化
        results_shared = self.optimize_shared_for_algorithm(algo_name, optimizer_list)

        # 第二阶段：识别效果差的文件并单独优化
        poor_files = []
        for problem_name, result in results_shared.items():
            if result["gap_percent"] and result["gap_percent"] > 10:
                poor_files.append(problem_name)

        if poor_files:
            print(f"\n📊 第二阶段: 对效果差的文件单独优化...")
            print(f"  需要单独优化的文件: {poor_files}")

            for sb_opt in optimizer_list:
                if sb_opt.problem_name in poor_files:
                    print(f"\n  单独优化 {sb_opt.problem_name}...")

                    study_name = (
                        f"{algo_name}_{sb_opt.problem_name}_adaptive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )

                    def objective_single(trial):
                        xi_multiplier = trial.suggest_float("xi_multiplier", 0.1, 10.0, log=True)
                        xi = sb_opt.default_xi * xi_multiplier

                        if self.fixed_sigma is not None:
                            init_noise = self.fixed_sigma
                        else:
                            init_noise = trial.suggest_float("init_noise", 0.05, 0.3)

                        try:
                            energy = sb_opt.solve(xi, 100, 800, 0.1, init_noise)

                            if sb_opt.objective_sense == "maximize":
                                return -energy
                            else:
                                return energy
                        except:
                            return float("inf")

                    study = optuna.create_study(
                        study_name=study_name, direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
                    )

                    study.optimize(objective_single, n_trials=self.n_trials, show_progress_bar=False)

                    best_trial = study.best_trial
                    best_xi = sb_opt.default_xi * best_trial.params["xi_multiplier"]
                    best_sigma = self.fixed_sigma if self.fixed_sigma else best_trial.params.get("init_noise", 0.1)
                    best_energy = -best_trial.value if sb_opt.objective_sense == "maximize" else best_trial.value

                    gap = None
                    if sb_opt.target_value is not None:
                        gap = abs(best_energy - sb_opt.target_value) / abs(sb_opt.target_value) * 100

                    results_shared[sb_opt.problem_name] = {
                        "energy": best_energy,
                        "xi": best_xi,
                        "sigma": best_sigma,
                        "target_value": sb_opt.target_value,
                        "gap_percent": gap,
                        "optimization_mode": "individual",
                    }

                    print(f"    ✅ 优化后: xi={best_xi:.6f}, sigma={best_sigma:.2f}, energy={best_energy:.4f}", end="")
                    if gap is not None:
                        print(f" (gap: {gap:.2f}%)")
                    else:
                        print()
        else:
            print(f"  ✅ 所有文件使用共享参数效果良好，无需单独优化")

        return results_shared

    def optimize_shared(self):
        """优化共享参数（所有文件使用同一组参数）"""
        print("\n" + "=" * 80)
        print(f"📊 模式1: 共享参数优化")
        print("=" * 80)

        study_name = f"{self.algorithm}_shared_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        study = optuna.create_study(
            study_name=study_name, direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
        )

        print(f"开始优化 {len(self.sb_optimizers)} 个文件的共享参数...")
        study.optimize(self.objective_shared, n_trials=self.n_trials, show_progress_bar=True)

        best_trial = study.best_trial
        best_params = best_trial.params
        best_xi = self.sb_optimizers[0].default_xi * best_params["xi_multiplier"]
        best_sigma = self.fixed_sigma if self.fixed_sigma else best_params.get("init_noise", 0.1)

        print(f"\n✅ 共享参数优化完成!")
        print(f"  最佳xi: {best_xi:.6f}")
        print(f"  最佳sigma: {best_sigma:.2f}")
        print(f"  目标值: {-best_trial.value:.6f}")

        # 评估所有文件使用最佳参数的效果
        print(f"\n📈 评估各文件使用共享参数的效果:")
        results_shared = {}

        for sb_opt in self.sb_optimizers:
            xi_adjusted = best_xi * sb_opt.default_xi / self.sb_optimizers[0].default_xi
            energy = sb_opt.solve(xi_adjusted, 100, 800, 0.1, best_sigma)

            gap = None
            if sb_opt.target_value is not None:
                gap = abs(energy - sb_opt.target_value) / abs(sb_opt.target_value) * 100

            results_shared[sb_opt.problem_name] = {
                "energy": energy,
                "xi": xi_adjusted,
                "sigma": best_sigma,
                "target_value": sb_opt.target_value,
                "gap_percent": gap,
                "optimization_mode": "shared",
            }

            print(f"  {sb_opt.problem_name}: {energy:.4f}", end="")
            if gap is not None:
                print(f" (gap: {gap:.2f}%)")
            else:
                print()

        return results_shared

    def optimize_individual(self):
        """优化每个文件的独立参数"""
        print("\n" + "=" * 80)
        print(f"📊 模式2: 独立参数优化")
        print("=" * 80)

        results_individual = {}

        for i, sb_opt in enumerate(self.sb_optimizers):
            print(f"\n[{i+1}/{len(self.sb_optimizers)}] 优化 {sb_opt.problem_name}...")

            study_name = f"{self.algorithm}_{sb_opt.problem_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            def objective_single(trial, sb=sb_opt):
                xi_multiplier = trial.suggest_float("xi_multiplier", 0.1, 10.0, log=True)
                xi = sb.default_xi * xi_multiplier

                if self.fixed_sigma is not None:
                    init_noise = self.fixed_sigma
                else:
                    init_noise = trial.suggest_float("init_noise", 0.05, 0.3)

                try:
                    energy = sb.solve(xi, 100, 800, 0.1, init_noise)

                    if sb.objective_sense == "maximize":
                        return -energy
                    else:
                        return energy
                except:
                    return float("inf")

            study = optuna.create_study(
                study_name=study_name, direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
            )

            study.optimize(objective_single, n_trials=self.n_trials, show_progress_bar=False)

            best_trial = study.best_trial
            best_xi = sb_opt.default_xi * best_trial.params["xi_multiplier"]
            best_sigma = self.fixed_sigma if self.fixed_sigma else best_trial.params.get("init_noise", 0.1)
            best_energy = -best_trial.value if sb_opt.objective_sense == "maximize" else best_trial.value

            gap = None
            if sb_opt.target_value is not None:
                gap = abs(best_energy - sb_opt.target_value) / abs(sb_opt.target_value) * 100

            results_individual[sb_opt.problem_name] = {
                "energy": best_energy,
                "xi": best_xi,
                "sigma": best_sigma,
                "target_value": sb_opt.target_value,
                "gap_percent": gap,
            }

            print(f"  ✅ 最佳: xi={best_xi:.6f}, sigma={best_sigma:.2f}, energy={best_energy:.4f}", end="")
            if gap is not None:
                print(f" (gap: {gap:.2f}%)")
            else:
                print()

        return results_individual

    def optimize_adaptive(self):
        """自适应优化：先共享优化，再对效果差的文件单独优化"""
        print("\n" + "=" * 80)
        print(f"📊 模式3: 自适应优化")
        print("=" * 80)
        print("  第一阶段: 共享参数优化")
        print("  第二阶段: 对gap>10%的文件单独优化")
        print("=" * 80)

        # 第一阶段：共享优化
        results_shared = self.optimize_shared()

        # 第二阶段：识别效果差的文件并单独优化
        print(f"\n📊 第二阶段: 对效果差的文件单独优化...")
        poor_files = []
        for problem_name, result in results_shared.items():
            if result["gap_percent"] and result["gap_percent"] > 10:
                poor_files.append(problem_name)

        if poor_files:
            print(f"  需要单独优化的文件: {poor_files}")

            # 只对效果差的文件单独优化
            poor_indices = [i for i, sb in enumerate(self.sb_optimizers) if sb.problem_name in poor_files]

            for idx in poor_indices:
                sb_opt = self.sb_optimizers[idx]
                print(f"\n  单独优化 {sb_opt.problem_name}...")

                study_name = (
                    f"{self.algorithm}_{sb_opt.problem_name}_adaptive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

                def objective_single(trial, sb=sb_opt):
                    xi_multiplier = trial.suggest_float("xi_multiplier", 0.1, 10.0, log=True)
                    xi = sb.default_xi * xi_multiplier

                    if self.fixed_sigma is not None:
                        init_noise = self.fixed_sigma
                    else:
                        init_noise = trial.suggest_float("init_noise", 0.05, 0.3)

                    try:
                        energy = sb.solve(xi, 100, 800, 0.1, init_noise)

                        if sb.objective_sense == "maximize":
                            return -energy
                        else:
                            return energy
                    except:
                        return float("inf")

                study = optuna.create_study(
                    study_name=study_name, direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
                )

                study.optimize(objective_single, n_trials=self.n_trials, show_progress_bar=False)

                best_trial = study.best_trial
                best_xi = sb_opt.default_xi * best_trial.params["xi_multiplier"]
                best_sigma = self.fixed_sigma if self.fixed_sigma else best_trial.params.get("init_noise", 0.1)
                best_energy = -best_trial.value if sb_opt.objective_sense == "maximize" else best_trial.value

                gap = None
                if sb_opt.target_value is not None:
                    gap = abs(best_energy - sb_opt.target_value) / abs(sb_opt.target_value) * 100

                # 更新结果
                results_shared[sb_opt.problem_name] = {
                    "energy": best_energy,
                    "xi": best_xi,
                    "sigma": best_sigma,
                    "target_value": sb_opt.target_value,
                    "gap_percent": gap,
                    "optimization_mode": "individual",
                }

                print(f"    ✅ 优化后: xi={best_xi:.6f}, sigma={best_sigma:.2f}, energy={best_energy:.4f}", end="")
                if gap is not None:
                    print(f" (gap: {gap:.2f}%)")
                else:
                    print()
        else:
            print(f"  ✅ 所有文件使用共享参数效果良好，无需单独优化")

        return results_shared

    def run(self):
        """运行批量优化"""
        start_time = time.time()

        # 为每个算法分别优化
        all_results = {}
        for algo_name in self.algorithms_to_optimize:
            print(f"\n{'#'*80}")
            print(f"# 开始优化算法: {algo_name}")
            print(f"{'#'*80}")

            results = self.optimize_single_algorithm(algo_name)
            all_results[algo_name] = results

        end_time = time.time()
        total_time = end_time - start_time

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path("save_json") / f"optuna_batch_12algs_{self.optimization_mode}_{timestamp}.json"

        summary_data = {
            "algorithms": self.algorithms_to_optimize,
            "optimization_mode": self.optimization_mode,
            "n_files": len(self.file_paths),
            "n_algorithms": len(self.algorithms_to_optimize),
            "n_trials": self.n_trials,
            "fixed_sigma": self.fixed_sigma,
            "total_time_seconds": total_time,
            "timestamp": timestamp,
            "results": all_results,
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print(f"\n" + "=" * 80)
        print(f"💾 结果已保存到: {output_file}")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"优化算法数: {len(self.algorithms_to_optimize)}")
        print("=" * 80)

        return summary_data


def save_results_to_csv(all_results, output_file=None):
    """
    将所有算法的优化结果保存到CSV文件

    参数:
        all_results (dict): 包含所有算法优化结果的字典
        output_file (str, optional): 输出文件路径
    """
    csv_rows = []

    for algorithm_name, problem_results in all_results.items():
        for problem_name, result in problem_results.items():
            # 解析算法名称
            parts = algorithm_name.split("_")
            algorithm_type = parts[0]  # BSB or DSB
            device = parts[1]  # CPU or GPU
            precision = parts[2]  # fp16, int8, fp8

            row = {
                "algorithm_name": algorithm_name,
                "algorithm_type": algorithm_type,
                "device": device,
                "precision": precision,
                "problem_name": problem_name,
                "best_energy": result["energy"],
                "best_xi": result["xi"],
                "best_sigma": result["sigma"],
                "target_value": result["target_value"],
                "gap_percent": result["gap_percent"],
                "optimization_mode": result.get("optimization_mode", "N/A"),
            }
            csv_rows.append(row)

    # 保存到CSV
    if output_file is None:
        csv_save_dir = Path("save_csv")
        csv_save_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = csv_save_dir / f"batch_optuna_12algs_{timestamp}.csv"

    df = pd.DataFrame(csv_rows)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n💾 CSV结果已保存到: {output_file}")

    return output_file


if __name__ == "__main__":
    import sys

    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"✓ CUDA设备可用: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠ CUDA不可用，使用CPU")
        device = "cpu"

    print()

    # 获取qplib文件夹下所有.qplib文件
    qplib_dir = Path("qplib")
    qplib_files = sorted(qplib_dir.glob("*.qplib"))

    if not qplib_files:
        print(f"❌ qplib文件夹中没有找到.qplib文件")
        sys.exit(1)

    print(f"找到 {len(qplib_files)} 个QPLIB文件:")
    for i, f in enumerate(qplib_files, 1):
        print(f"  {i}. {f.name}")

    # 配置参数
    if len(sys.argv) > 1:
        N_TRIALS = int(sys.argv[1])
    else:
        N_TRIALS = 20  # 默认试验次数（12种算法 x 23个实例，减少默认试验次数）

    if len(sys.argv) > 2:
        ALGORITHMS_ARG = sys.argv[2].upper()
        if ALGORITHMS_ARG == "ALL":
            ALGORITHMS = "ALL"
        elif ALGORITHMS_ARG == "BSB":
            ALGORITHMS = "BSB"
        elif ALGORITHMS_ARG == "DSB":
            ALGORITHMS = "DSB"
        else:
            ALGORITHMS = "ALL"
    else:
        ALGORITHMS = "ALL"

    if len(sys.argv) > 3:
        MODE = sys.argv[3].lower()
    else:
        MODE = "individual"  # 默认使用独立优化

    if len(sys.argv) > 4:
        sigma_arg = sys.argv[4].lower()
        if sigma_arg == "auto":
            FIXED_SIGMA = None
        else:
            FIXED_SIGMA = float(sigma_arg)
    else:
        FIXED_SIGMA = None

    print()
    print("=" * 80)
    print(f"批量优化配置:")
    print(f"  - 文件数: {len(qplib_files)}")
    print(f"  - 算法: {ALGORITHMS}")
    print(f"  - 模式: {MODE}")
    print(f"  - 试验次数: {N_TRIALS}")
    print(f"  - 固定Sigma: {FIXED_SIGMA}")
    print("=" * 80)

    # 创建批量优化器
    batch_opt = BatchOptunaOptimizer(
        file_paths=qplib_files,
        device=device,
        algorithms=ALGORITHMS,
        optimization_mode=MODE,
        n_trials=N_TRIALS,
        fixed_sigma=FIXED_SIGMA,
    )

    # 运行优化
    results = batch_opt.run()

    # 保存所有结果到CSV
    if results:
        csv_file = save_results_to_csv(results["results"])

        # 生成汇总报告
        print(f"\n{'='*80}")
        print(f"📊 所有算法优化结果汇总")
        print(f"{'='*80}")

        for algo_name, problem_results in results["results"].items():
            print(f"\n🔹 {algo_name}:")

            # 计算统计信息
            gaps = [r["gap_percent"] for r in problem_results.values() if r["gap_percent"] is not None]
            if gaps:
                avg_gap = np.mean(gaps)
                min_gap = np.min(gaps)
                max_gap = np.max(gaps)
                print(f"   平均Gap: {avg_gap:.2f}%, 最小Gap: {min_gap:.2f}%, 最大Gap: {max_gap:.2f}%")

            # 找到最佳参数
            xi_values = [r["xi"] for r in problem_results.values()]
            sigma_values = [r["sigma"] for r in problem_results.values()]
            if xi_values:
                print(f"   xi范围: [{min(xi_values):.6f}, {max(xi_values):.6f}]")
            if sigma_values:
                print(f"   sigma范围: [{min(sigma_values):.2f}, {max(sigma_values):.2f}]")

    print("\n🎉 所有算法批量优化完成!")
    print(f"优化了 {len(results['results'])} 种算法")
    print(f"每个算法优化了 {len(qplib_files)} 个实例")

"""
使用说明:
python SB_optuna_batch_tuner.py [n_trials] [algorithms] [mode] [sigma]

参数说明:
  n_trials      : 每个算法的试验次数（默认20）
  algorithms    : ALL, BSB, DSB（默认ALL）
  mode          : shared, individual, adaptive（默认individual）
  sigma         : auto或具体数值（默认None）

示例:
python SB_optuna_batch_tuner.py 20 ALL individual auto      # 所有12种算法，独立优化
python SB_optuna_batch_tuner.py 50 BSB shared 0.1        # 6种BSB算法，共享参数
python SB_optuna_batch_tuner.py 30 DSB adaptive auto     # 6种DSB算法，自适应优化

说明:
  - ALL模式会优化所有12种算法（6种BSB + 6种DSB）
  - 每种算法都会对每个QPLIB实例进行独立参数优化
  - 结果CSV包含: algorithm_name, algorithm_type, device, precision, problem_name, best_energy, best_xi, best_sigma, gap_percent
"""
