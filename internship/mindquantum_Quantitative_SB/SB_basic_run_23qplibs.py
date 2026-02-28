#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量求解QPLIB文件 - 使用12种BSB和DSB算法

该脚本会依次执行qplib文件夹下的所有.qplib数据文件，
使用12种算法求解目标值，并生成汇总报告。

算法配置：
- BSB算法（6种）：BSB_CPU_fp16, BSB_CPU_int8, BSB_CPU_fp8,
                   BSB_GPU_fp16, BSB_GPU_int8, BSB_GPU_fp8
- DSB算法（6种）：DSB_CPU_fp16, DSB_CPU_int8, DSB_CPU_fp8,
                   DSB_GPU_fp16, DSB_GPU_int8, DSB_GPU_fp8

每种算法都代表不同的硬件（CPU/GPU）和精度配置（fp16/int8/fp8）的组合。
"""

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import scipy.sparse as sp
import torch
import json
import subprocess
import os
import re
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class QBExternal:
    def __init__(self, file_path, device="cuda", fixed_xi=None, problem_db=None, optimized_params_file=None):
        """
        初始化量子退火求解器（基于qplib_parser.h的C++ exe版本）

        参数：
            file_path (str): QPLIB文件路径
            device (str): 计算设备 ('cuda' 或 'cpu')
            fixed_xi (float, optional): 固定的xi值。如果为None，则自动计算
            problem_db (DataFrame, optional): 问题数据库，用于获取精确最优解
            optimized_params_file (str, optional): 优化参数CSV文件路径

        说明：
            不使用qplib_parser.py，直接依赖C++ exe通过qplib_parser.h解析数据
        """
        self.device = device
        self.file_path = file_path
        self.problem_db = problem_db

        # 获取问题名称（从文件路径提取）
        self.problem_name = Path(file_path).stem

        # 缩放因子（将在第一次运行时获取）
        self.scaling_factor = None

        # 加载优化参数
        self.optimized_params = {}
        if optimized_params_file:
            self.load_optimized_params(optimized_params_file)

        # 使用固定xi或自动计算的xi
        if fixed_xi is not None:
            self.xi = fixed_xi
            print(f"✅ 使用固定的优化xi值: {self.xi:.6f}")
        else:
            self.xi = 0.365  # 默认使用优化值
            print(f"⚙️  使用xi值: {self.xi:.6f}")

        # 获取精确最优解（从数据库）
        self.exact_optimal = self._get_exact_optimal_from_db()

        # 大缩放因子阈值
        self.LARGE_SCALING_THRESHOLD = 100.0  # 超过这个值认为是大缩放因子

        # 获取BSB和DSB可执行文件路径
        self.bsb_dir = Path("BSB")
        self.dsb_dir = Path("DSB")

        # BSB算法：6种变体（CPU/GPU × fp16/int8/fp8）
        self.bsb_executables = {
            "BSB_CPU_fp16": self.bsb_dir / "BSB_CPU_fp16.exe",
            "BSB_CPU_int8": self.bsb_dir / "BSB_CPU_int8.exe",
            "BSB_CPU_fp8": self.bsb_dir / "BSB_CPU_fp8.exe",
            "BSB_GPU_fp16": self.bsb_dir / "BSB_GPU_fp16.exe",
            "BSB_GPU_int8": self.bsb_dir / "BSB_GPU_int8.exe",
            "BSB_GPU_fp8": self.bsb_dir / "BSB_GPU_fp8.exe",
        }

        # DSB算法：6种变体（CPU/GPU × fp16/int8/fp8）
        self.dsb_executables = {
            "DSB_CPU_fp16": self.dsb_dir / "DSB_CPU_fp16.exe",
            "DSB_CPU_int8": self.dsb_dir / "DSB_CPU_int8.exe",
            "DSB_CPU_fp8": self.dsb_dir / "DSB_CPU_fp8.exe",
            "DSB_GPU_fp16": self.dsb_dir / "DSB_GPU_fp16.exe",
            "DSB_GPU_int8": self.dsb_dir / "DSB_GPU_int8.exe",
            "DSB_GPU_fp8": self.dsb_dir / "DSB_GPU_fp8.exe",
        }

        # 合并所有可执行文件（共12种算法）
        self.all_executables = {**self.bsb_executables, **self.dsb_executables}

        # 检查可执行文件是否存在
        missing_executables = [name for name, path in self.all_executables.items() if not path.exists()]
        if missing_executables:
            print(f"⚠️  以下算法可执行文件不存在: {missing_executables}")
            print(f"   请先编译相应的CUDA文件:")
            for name in missing_executables:
                if name.startswith("BSB_"):
                    algo_name = name[4:]  # 移除'BSB_'前缀
                    cu_file = self.bsb_dir / f"BSB_{algo_name}.cu"
                else:  # DSB_
                    algo_name = name[4:]  # 移除'DSB_'前缀
                    cu_file = self.dsb_dir / f"DSB_{algo_name}.cu"

                if cu_file.exists():
                    print(f"   nvcc -o {self.all_executables[name]} {cu_file}")
            print(f"   或使用Visual Studio编译项目")
            print(f"   如果可执行文件存在但路径不同，请修改相应的可执行文件字典")

        # 初始化问题数据（将从C++ exe输出中获取）
        self.problem_info = None  # 将从第一次运行中获取
        self.objective_sense = None
        self.num_vars = None
        self.num_constraints = None
        self.scaling_factor = None
        self.quad_terms_count = None
        self.linear_terms_count = None
        self.constant_term = None

        print(f"QPLIB文件: {file_path}")
        print(f"问题名称: {self.problem_name}")
        if self.exact_optimal is not None:
            print(f"精确最优解: {self.exact_optimal}")
        print(f"使用xi值: {self.xi:.6f}")
        print(
            f"可用算法总数: {len(self.all_executables)} (BSB: {len(self.bsb_executables)}, DSB: {len(self.dsb_executables)})"
        )
        print(f"注意: 问题数据将由C++ exe通过qplib_parser.h解析")

    def _get_exact_optimal_from_db(self):
        """从数据库获取精确最优解"""
        if self.problem_db is None:
            return None

        problem_row = self.problem_db[self.problem_db["instance_id"] == self.problem_name]
        if problem_row.empty:
            return None

        solution_str = problem_row.iloc[0]["solution_objective_value"]
        if pd.isna(solution_str) or solution_str == "":
            return None

        match = re.search(r"([+-]?\d*\.?\d+)", str(solution_str))
        if match:
            return float(match.group(1))
        return None

    def load_optimized_params(self, csv_file):
        """
        从CSV文件加载优化参数

        参数：
            csv_file (str): CSV文件路径
        """
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                key = (row["algorithm_name"], row["problem_name"])
                self.optimized_params[key] = {"xi": row["best_xi"], "sigma": row["best_sigma"]}
            print(f"✅ 已加载 {len(self.optimized_params)} 组优化参数")
        except Exception as e:
            print(f"⚠️  加载优化参数失败: {e}")

    def get_optimized_params(self, algorithm_name):
        """
        获取指定算法和问题的优化参数

        参数：
            algorithm_name (str): 算法名称（如 'BSB_CPU_fp16'）

        返回：
            tuple: (xi, sigma) 或 (None, None) 如果找不到
        """
        key = (algorithm_name, self.problem_name)
        params = self.optimized_params.get(key, {})
        return params.get("xi", None), params.get("sigma", None)

    def parse_problem_info_from_output(self, output_text: str) -> Dict:
        """
        从C++ exe输出中解析问题基本信息

        参数：
            output_text: C++ exe的输出文本

        返回：
            包含问题信息的字典
        """
        info = {
            "problem_name": None,
            "objective_sense": None,
            "num_vars": None,
            "num_constraints": None,
            "scaling_factor": None,
            "quad_terms_count": None,
            "linear_terms_count": None,
            "constant_term": None,
        }

        lines = output_text.split("\n")

        for line in lines:
            # 解析问题名称
            if "Problem name:" in line:
                match = re.search(r"Problem name:\s*(\S+)", line)
                if match:
                    info["problem_name"] = match.group(1)

            # 解析优化方向
            elif "Objective sense:" in line:
                match = re.search(r"Objective sense:\s*(\w+)", line, re.IGNORECASE)
                if match:
                    info["objective_sense"] = match.group(1).lower()

            # 解析变量数量
            elif "Number of variables:" in line:
                match = re.search(r"Number of variables:\s*(\d+)", line)
                if match:
                    info["num_vars"] = int(match.group(1))

            # 解析约束数量
            elif "Number of constraints:" in line:
                match = re.search(r"Number of constraints:\s*(\d+)", line)
                if match:
                    info["num_constraints"] = int(match.group(1))

            # 解析缩放因子
            elif "Scaling factor:" in line:
                match = re.search(r"Scaling factor:\s*([\d.]+)", line)
                if match:
                    info["scaling_factor"] = float(match.group(1))

            # 解析二次项数量
            elif "Number of quadratic terms:" in line:
                match = re.search(r"Number of quadratic terms:\s*(\d+)", line)
                if match:
                    info["quad_terms_count"] = int(match.group(1))

            # 解析线性项数量
            elif "Number of linear terms:" in line:
                match = re.search(r"Number of linear terms:\s*(\d+)", line)
                if match:
                    info["linear_terms_count"] = int(match.group(1))

            # 解析常数项
            elif "Constant term:" in line:
                match = re.search(r"Constant term:\s*([-\d.]+)", line)
                if match:
                    info["constant_term"] = float(match.group(1))

        return info

    def run_algorithm_executable(self, algorithm_name, qplib_file_path):
        """
        运行算法可执行文件（BSB或DSB）并解析输出结果

        参数：
            algorithm_name (str): 算法名称（如 'BSB_CPU_fp16', 'DSB_GPU_int8'）
            qplib_file_path (str): QPLIB文件路径

        返回：
            dict: 包含解和性能信息的字典
        """
        # 确定算法类型和可执行文件路径
        if algorithm_name.startswith("BSB_"):
            executable_path = self.bsb_executables[algorithm_name]
            algorithm_dir = self.bsb_dir
            algorithm_type = "BSB"
        elif algorithm_name.startswith("DSB_"):
            executable_path = self.dsb_executables[algorithm_name]
            algorithm_dir = self.dsb_dir
            algorithm_type = "DSB"
        else:
            raise ValueError(f"未知的算法类型: {algorithm_name}")

        if not executable_path.exists():
            raise FileNotFoundError(f"{algorithm_type}可执行文件不存在: {executable_path}")

        # 获取优化参数
        optimized_xi, optimized_sigma = self.get_optimized_params(algorithm_name)

        print(f"\n🔧 运行 {algorithm_name} 算法...")
        print(f"   可执行文件: {executable_path}")
        print(f"   输入文件: {qplib_file_path}")
        if optimized_xi is not None:
            print(f"   优化参数: xi={optimized_xi:.6f}, sigma={optimized_sigma:.4f}")

        # 使用算法期望的路径格式（参考qplib_parser.h中的定义）
        # 算法期望从其目录看到的路径是 "../qplib/文件名"
        qplib_filename = Path(qplib_file_path).name
        algo_qplib_path = f"../qplib/{qplib_filename}"

        print(f"   {algorithm_type}期望路径: {algo_qplib_path}")

        start_time = time.time()

        try:
            # 构建命令行参数
            cmd = [str(executable_path)]

            # 根据算法类型添加参数
            if algorithm_type == "BSB":
                # BSB所有版本: [executable] [file_path] [xi]
                cmd.append(algo_qplib_path)
                if optimized_xi is not None:
                    cmd.append(str(optimized_xi))
            elif algorithm_type == "DSB" and "CPU" in algorithm_name:
                # DSB CPU版本: [executable] [file_path] [xi]
                cmd.append(algo_qplib_path)
                if optimized_xi is not None:
                    cmd.append(str(optimized_xi))
            elif algorithm_type == "DSB" and "GPU" in algorithm_name:
                # DSB GPU版本: [executable] [file_path] -x [xi]
                cmd.extend(["-f", algo_qplib_path])
                if optimized_xi is not None:
                    cmd.extend(["-x", str(optimized_xi)])

            # 运行可执行文件
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(algorithm_dir), timeout=600  # 10分钟超时
            )

            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # 转换为毫秒

            if result.returncode != 0:
                print(f"❌ {algorithm_name} 执行失败:")
                print(f"   错误输出: {result.stderr}")
                return None

            # 解析输出
            output_text = result.stdout

            # 首次运行时，从输出中提取问题信息
            if self.problem_info is None:
                self.problem_info = self.parse_problem_info_from_output(output_text)
                if self.problem_info["objective_sense"]:
                    self.objective_sense = self.problem_info["objective_sense"]
                if self.problem_info["num_vars"]:
                    self.num_vars = self.problem_info["num_vars"]
                if self.problem_info["num_constraints"]:
                    self.num_constraints = self.problem_info["num_constraints"]
                if self.problem_info["scaling_factor"]:
                    self.scaling_factor = self.problem_info["scaling_factor"]
                if self.problem_info["quad_terms_count"]:
                    self.quad_terms_count = self.problem_info["quad_terms_count"]
                if self.problem_info["linear_terms_count"]:
                    self.linear_terms_count = self.problem_info["linear_terms_count"]
                if self.problem_info["constant_term"]:
                    self.constant_term = self.problem_info["constant_term"]

                print(f"\n📊 从C++ exe输出中解析到问题信息:")
                print(f"   问题名称: {self.problem_info['problem_name']}")
                print(f"   优化方向: {self.objective_sense}")
                print(f"   变量数: {self.num_vars}")
                print(f"   约束数: {self.num_constraints}")
                print(f"   缩放因子: {self.scaling_factor}")

            return self.parse_algorithm_output(output_text, algorithm_name, execution_time)

        except subprocess.TimeoutExpired:
            print(f"❌ {algorithm_name} 执行超时")
            return None
        except Exception as e:
            print(f"❌ {algorithm_name} 执行出错: {e}")
            return None

    def parse_algorithm_output(self, output_text, algorithm_name, execution_time):
        """
        解析算法的输出文本（BSB或DSB）

        参数：
            output_text (str): 算法的输出文本
            algorithm_name (str): 算法名称
            execution_time (float): Python端执行时间（毫秒，作为备选）

        返回：
            dict: 包含解和性能信息的字典
        """
        result = {
            "algorithm_name": algorithm_name,
            "execution_time_ms": execution_time,  # 默认使用Python端时间
            "cpp_execution_time_ms": None,  # C++内部报告的时间
            "objective_value": None,
            "solution_vector": [],
            "solution_ones": 0,
            "solution_zeros": 0,
            "solution_density": 0.0,
            "iterations": 800,
            "batch_size": 100,
            "parsing_success": False,
        }

        lines = output_text.split("\n")

        try:
            # 优先解析C++报告的执行时间
            for line in lines:
                if "TIME: Solving completed, total time:" in line:
                    match = re.search(r"time:\s*(\d+\.?\d*)\s*ms", line)
                    if match:
                        result["cpp_execution_time_ms"] = float(match.group(1))
                        # 优先使用C++报告的时间
                        result["execution_time_ms"] = result["cpp_execution_time_ms"]
                        break

            # 解析目标函数值（支持BSB和DSB格式）
            for line in lines:
                if (
                    "BSB GPU OBJECTIVE VALUE:" in line  # 优先级1: BSB GPU版本
                    or "BSB OBJECTIVE VALUE:" in line  # 优先级2: BSB CPU版本
                    or "DSB GPU OBJECTIVE VALUE:" in line  # 优先级3: DSB GPU版本
                    or "DSB OBJECTIVE VALUE:" in line  # 优先级4: DSB CPU版本
                    or "DSB GPU CONTINUOUS ENERGY:" in line  # 优先级5: DSB GPU continuous energy格式
                    or "Optimal objective value:" in line  # 优先级6: 备选方案
                ):
                    match = re.search(r"(\d+\.?\d*)", line)
                    if match:
                        result["objective_value"] = float(match.group(1))
                        break

            # 解析解向量
            solution_started = False
            solution_values = []

            for line in lines:
                if "RESULT: Solution" in line:
                    solution_started = True
                    continue
                elif (
                    solution_started
                    and line.strip()
                    and not any(
                        stat in line
                        for stat in [
                            "Solution statistics",
                            "DEBUG: Solution",
                            "Variables set to",
                            "- Values",
                            "- Positive values",
                            "- Non-positive",
                        ]
                    )
                ):
                    # 解析二进制解（格式：0, 1, 0, 1, ...）
                    numbers = re.findall(r"[01]", line)
                    solution_values.extend([int(n) for n in numbers])
                elif "Solution statistics" in line or "DEBUG: Solution" in line:
                    break

            result["solution_vector"] = solution_values

            # 解析解统计信息（支持多种格式）
            solution_ones_found = False

            # 格式1: "Variables set to 1: xxx/xxx" (旧格式)
            for line in lines:
                if "Variables set to 1:" in line:
                    match = re.search(r"(\d+)/(\d+)", line)
                    if match:
                        result["solution_ones"] = int(match.group(1))
                        result["solution_zeros"] = int(match.group(2))
                        if self.num_vars is not None and self.num_vars > 0:
                            result["solution_density"] = (result["solution_ones"] / self.num_vars) * 100
                        else:
                            result["solution_density"] = 0.0
                        solution_ones_found = True
                    break

            # 格式2: "- Values > 0.5 (→1): xxx/xxx" (新格式)
            if not solution_ones_found:
                for line in lines:
                    if "Values > 0.5" in line or "Values > 0.5" in line.replace("→", ""):
                        match = re.search(r"(\d+)/(\d+)", line)
                        if match:
                            result["solution_ones"] = int(match.group(1))
                            result["solution_zeros"] = int(match.group(2))
                            if self.num_vars is not None and self.num_vars > 0:
                                result["solution_density"] = (result["solution_ones"] / self.num_vars) * 100
                            else:
                                result["solution_density"] = 0.0
                            solution_ones_found = True
                        break

            # 格式3: "- Positive values" (旧格式，保留兼容)
            if not solution_ones_found:
                for line in lines:
                    if "Positive values" in line and "→1" in line:
                        match = re.search(r"(\d+)/(\d+)", line)
                        if match:
                            result["solution_ones"] = int(match.group(1))
                            result["solution_zeros"] = int(match.group(2))
                            if self.num_vars is not None and self.num_vars > 0:
                                result["solution_density"] = (result["solution_ones"] / self.num_vars) * 100
                            else:
                                result["solution_density"] = 0.0
                            solution_ones_found = True
                        break

            # 格式4: "DEBUG: Solution - 1s: xxx, 0s: xxx" (GPU fp8)
            if not solution_ones_found:
                for line in lines:
                    if "DEBUG: Solution - 1s:" in line:
                        match = re.search(r"1s:\s*(\d+),\s*0s:\s*(\d+)", line)
                        if match:
                            result["solution_ones"] = int(match.group(1))
                            result["solution_zeros"] = int(match.group(2))
                            if self.num_vars is not None and self.num_vars > 0:
                                result["solution_density"] = (result["solution_ones"] / self.num_vars) * 100
                            else:
                                result["solution_density"] = 0.0
                            solution_ones_found = True
                        break

            # 如果没有找到统计信息，手动计算
            if not solution_ones_found and solution_values:
                result["solution_ones"] = sum(solution_values)
                result["solution_zeros"] = len(solution_values) - result["solution_ones"]
                # 确保len(solution_values)不为0
                if len(solution_values) > 0:
                    result["solution_density"] = (result["solution_ones"] / len(solution_values)) * 100
                else:
                    result["solution_density"] = 0.0

            # 确定算法类型用于显示
            if algorithm_name.startswith("BSB_"):
                algorithm_type = "BSB"
            else:
                algorithm_type = "DSB"

            result["parsing_success"] = True

            # 打印时间来源信息
            time_source = "C++内部时间" if result["cpp_execution_time_ms"] is not None else "Python端时间"
            print(f"✅ {algorithm_name} 解析成功:")
            print(f"   算法类型: {algorithm_type}")
            print(f"   目标函数值: {result['objective_value']}")
            print(f"   执行时间: {result['execution_time_ms']:.2f} ms (来源: {time_source})")
            if result["cpp_execution_time_ms"] is not None:
                print(f"   Python端时间: {execution_time:.2f} ms (仅供参考)")
            if self.num_vars is not None:
                print(f"   解中1的数量: {result['solution_ones']}/{self.num_vars} ({result['solution_density']:.2f}%)")
            else:
                print(f"   解中1的数量: {result['solution_ones']}/{len(solution_values) if solution_values else 0}")

        except Exception as e:
            print(f"❌ {algorithm_name} 输出解析失败: {e}")
            result["error"] = str(e)

        return result

    def solve_all_variants(self, qplib_file_path=None):
        """
        运行所有12种算法变体（6种BSB + 6种DSB）

        参数：
            qplib_file_path (str, optional): QPLIB文件路径。如果为None，使用初始化时的文件路径

        返回：
            dict: 包含所有算法结果的字典
        """
        if qplib_file_path is None:
            qplib_file_path = self.file_path

        print(f"\n🚀 开始运行所有算法变体（BSB + DSB）")
        print("=" * 80)
        print(f"问题: {self.problem_name}")
        print(f"文件: {qplib_file_path}")
        num_vars_str = str(self.num_vars) if self.num_vars is not None else "未知（将在第一次运行后确定）"
        print(f"变量数: {num_vars_str}, 优化方向: {self.objective_sense}")
        if self.exact_optimal is not None:
            print(f"精确最优解: {self.exact_optimal}")
        print("=" * 80)

        # BSB算法：6种变体
        bsb_algorithms = ["BSB_CPU_fp16", "BSB_CPU_int8", "BSB_CPU_fp8", "BSB_GPU_fp16", "BSB_GPU_int8", "BSB_GPU_fp8"]
        # DSB算法：6种变体
        dsb_algorithms = ["DSB_CPU_fp16", "DSB_CPU_int8", "DSB_CPU_fp8", "DSB_GPU_fp16", "DSB_GPU_int8", "DSB_GPU_fp8"]
        # 所有算法
        all_algorithms = bsb_algorithms + dsb_algorithms

        results = {}

        print(f"\n📋 BSB算法（Ballistic Simulated Bifurcation）:")
        print(f"   共 {len(bsb_algorithms)} 种变体（CPU/GPU × fp16/int8/fp8）")
        for algorithm in bsb_algorithms:
            result = self.run_algorithm_executable(algorithm, qplib_file_path)
            results[algorithm] = result

        print(f"\n📋 DSB算法（Discrete Simulated Bifurcation）:")
        print(f"   共 {len(dsb_algorithms)} 种变体（CPU/GPU × fp16/int8/fp8）")
        for algorithm in dsb_algorithms:
            result = self.run_algorithm_executable(algorithm, qplib_file_path)
            results[algorithm] = result

        # 分析和比较结果
        self.analyze_and_compare_results(results)

        return results

    def analyze_and_compare_results(self, results):
        """
        分析和比较所有算法的结果（BSB + DSB分开比较）

        参数：
            results (dict): 包含所有算法结果的字典
        """
        print(f"\n📊 所有算法变体结果比较（BSB + DSB 分开比较）")
        print("=" * 80)

        valid_results = {k: v for k, v in results.items() if v and v.get("parsing_success", False)}

        if not valid_results:
            print("❌ 没有成功解析的算法结果")
            return

        # 分离BSB和DSB结果
        bsb_results = {k: v for k, v in valid_results.items() if k.startswith("BSB_")}
        dsb_results = {k: v for k, v in valid_results.items() if k.startswith("DSB_")}

        # 计算求解质量（与精确最优解的比率）
        def calculate_quality_ratio(obj_value):
            if self.exact_optimal is None or self.exact_optimal == 0:
                return None
            if obj_value is None:
                return None
            return obj_value / self.exact_optimal

        # 为每个结果添加求解质量信息
        for algo_name, result in valid_results.items():
            result["quality_ratio"] = calculate_quality_ratio(result["objective_value"])

        # 打印详细结果
        print(f"\n📋 所有算法详细结果:")
        for algorithm_name, result in valid_results.items():
            algorithm_type = "BSB" if algorithm_name.startswith("BSB_") else "DSB"
            quality_str = f"{result['quality_ratio']*100:.2f}%" if result["quality_ratio"] is not None else "N/A"
            time_source = "C++内部时间" if result.get("cpp_execution_time_ms") is not None else "Python端时间"
            obj_value_str = f"{result['objective_value']:.6f}" if result.get("objective_value") is not None else "N/A"
            print(
                f"   {algorithm_name} ({algorithm_type}): 目标值={obj_value_str}, 质量={quality_str}, 时间={result['execution_time_ms']:.2f}ms (来源: {time_source})"
            )

        # BSB vs DSB 对比（总体）
        if bsb_results and dsb_results:
            print(f"\n🔍 BSB vs DSB 总体对比:")

            # 找到各类别最佳（只考虑有objective_value的结果）
            bsb_valid = [(k, v) for k, v in bsb_results.items() if v.get("objective_value") is not None]
            dsb_valid = [(k, v) for k, v in dsb_results.items() if v.get("objective_value") is not None]

            if bsb_valid and self.objective_sense == "maximize":
                bsb_best = max(bsb_valid, key=lambda x: x[1]["objective_value"])
            elif bsb_valid:
                bsb_best = min(bsb_valid, key=lambda x: x[1]["objective_value"])
            else:
                bsb_best = None

            if dsb_valid and self.objective_sense == "maximize":
                dsb_best = max(dsb_valid, key=lambda x: x[1]["objective_value"])
            elif dsb_valid:
                dsb_best = min(dsb_valid, key=lambda x: x[1]["objective_value"])
            else:
                dsb_best = None

            if bsb_best:
                print(
                    f"   BSB最佳: {bsb_best[0]} (目标值: {bsb_best[1]['objective_value']:.6f}, 时间: {bsb_best[1]['execution_time_ms']:.2f} ms)"
                )
            if dsb_best:
                print(
                    f"   DSB最佳: {dsb_best[0]} (目标值: {dsb_best[1]['objective_value']:.6f}, 时间: {dsb_best[1]['execution_time_ms']:.2f} ms)"
                )

            # 计算平均性能（过滤掉None值）
            bsb_obj_values = [
                r["objective_value"] for r in bsb_results.values() if r.get("objective_value") is not None
            ]
            dsb_obj_values = [
                r["objective_value"] for r in dsb_results.values() if r.get("objective_value") is not None
            ]

            if bsb_obj_values:
                bsb_avg_obj = sum(bsb_obj_values) / len(bsb_obj_values)
            else:
                bsb_avg_obj = 0

            if dsb_obj_values:
                dsb_avg_obj = sum(dsb_obj_values) / len(dsb_obj_values)
            else:
                dsb_avg_obj = 0

            bsb_times = [r["execution_time_ms"] for r in bsb_results.values() if r.get("execution_time_ms") is not None]
            dsb_times = [r["execution_time_ms"] for r in dsb_results.values() if r.get("execution_time_ms") is not None]

            bsb_avg_time = sum(bsb_times) / len(bsb_times) if bsb_times else 0
            dsb_avg_time = sum(dsb_times) / len(dsb_times) if dsb_times else 0

            print(f"   BSB平均目标值: {bsb_avg_obj:.6f}, 平均时间: {bsb_avg_time:.2f} ms")
            print(f"   DSB平均目标值: {dsb_avg_obj:.6f}, 平均时间: {dsb_avg_time:.2f} ms")

        # ========== BSB算法分析 ==========
        print(f"\n" + "=" * 80)
        print(f"🔹 BSB算法分析")
        print("=" * 80)

        if bsb_results:
            # 筛选出质量>=99%的算法
            bsb_quality_99 = []
            for algo_name, result in bsb_results.items():
                if result["quality_ratio"] is not None:
                    # 对于maximize，quality_ratio >= 0.99 表示质量>=99%
                    # 对于minimize，quality_ratio <= 1.01 表示质量>=99%（相对误差<=1%）
                    if self.objective_sense == "maximize":
                        quality_ok = result["quality_ratio"] >= 0.99
                    else:
                        quality_ok = result["quality_ratio"] <= 1.01

                    if quality_ok:
                        bsb_quality_99.append((algo_name, result))

            if bsb_quality_99:
                print(f"\n✅ 求解质量达到99%精确解的BSB算法（共{len(bsb_quality_99)}个）:")
                # 按求解速度排序
                bsb_quality_99_sorted = sorted(bsb_quality_99, key=lambda x: x[1]["execution_time_ms"])

                for i, (algo_name, result) in enumerate(bsb_quality_99_sorted, 1):
                    quality_str = (
                        f"{result['quality_ratio']*100:.2f}%" if result["quality_ratio"] is not None else "N/A"
                    )
                    obj_value_str = (
                        f"{result['objective_value']:.6f}" if result["objective_value"] is not None else "N/A"
                    )
                    print(
                        f"   {i}. {algo_name}: 质量={quality_str}, 时间={result['execution_time_ms']:.2f}ms, 目标值={obj_value_str}"
                    )

                # 最快的99%质量算法
                fastest_bsb = bsb_quality_99_sorted[0]
                print(f"\n🏆 BSB算法 - 最快的99%质量算法: {fastest_bsb[0]}")
                if fastest_bsb[1]["quality_ratio"] is not None:
                    print(f"   求解质量: {fastest_bsb[1]['quality_ratio']*100:.2f}%")
                else:
                    print(f"   求解质量: N/A")
                print(f"   执行时间: {fastest_bsb[1]['execution_time_ms']:.2f} ms")
            else:
                print(f"\n⚠️ 没有BSB算法达到99%的求解质量")

            # 所有BSB算法按质量排序
            print(f"\n📊 BSB算法按求解质量排序:")
            # 过滤掉objective_value为None的结果
            bsb_valid = [(k, v) for k, v in bsb_results.items() if v.get("objective_value") is not None]
            if bsb_valid:
                bsb_quality_sorted = sorted(
                    bsb_valid, key=lambda x: x[1]["objective_value"], reverse=self.objective_sense == "maximize"
                )
            else:
                bsb_quality_sorted = []

            for i, (algo_name, result) in enumerate(bsb_quality_sorted, 1):
                quality_str = (
                    f"{result['quality_ratio']*100:.2f}%" if result.get("quality_ratio") is not None else "N/A"
                )
                obj_value_str = (
                    f"{result['objective_value']:.6f}" if result.get("objective_value") is not None else "N/A"
                )
                print(
                    f"   {i}. {algo_name}: 目标值={obj_value_str}, 质量={quality_str}, 时间={result['execution_time_ms']:.2f}ms"
                )
        else:
            print(f"❌ 没有BSB算法结果")

        # ========== DSB算法分析 ==========
        print(f"\n" + "=" * 80)
        print(f"🔹 DSB算法分析")
        print("=" * 80)

        if dsb_results:
            # 筛选出质量>=99%的算法
            dsb_quality_99 = []
            for algo_name, result in dsb_results.items():
                if result["quality_ratio"] is not None:
                    if self.objective_sense == "maximize":
                        quality_ok = result["quality_ratio"] >= 0.99
                    else:
                        quality_ok = result["quality_ratio"] <= 1.01

                    if quality_ok:
                        dsb_quality_99.append((algo_name, result))

            if dsb_quality_99:
                print(f"\n✅ 求解质量达到99%精确解的DSB算法（共{len(dsb_quality_99)}个）:")
                # 按求解速度排序
                dsb_quality_99_sorted = sorted(dsb_quality_99, key=lambda x: x[1]["execution_time_ms"])

                for i, (algo_name, result) in enumerate(dsb_quality_99_sorted, 1):
                    quality_str = (
                        f"{result['quality_ratio']*100:.2f}%" if result["quality_ratio"] is not None else "N/A"
                    )
                    obj_value_str = (
                        f"{result['objective_value']:.6f}" if result["objective_value"] is not None else "N/A"
                    )
                    print(
                        f"   {i}. {algo_name}: 质量={quality_str}, 时间={result['execution_time_ms']:.2f}ms, 目标值={obj_value_str}"
                    )

                # 最快的99%质量算法
                fastest_dsb = dsb_quality_99_sorted[0]
                print(f"\n🏆 DSB算法 - 最快的99%质量算法: {fastest_dsb[0]}")
                if fastest_dsb[1]["quality_ratio"] is not None:
                    print(f"   求解质量: {fastest_dsb[1]['quality_ratio']*100:.2f}%")
                else:
                    print(f"   求解质量: N/A")
                print(f"   执行时间: {fastest_dsb[1]['execution_time_ms']:.2f} ms")
            else:
                print(f"\n⚠️ 没有DSB算法达到99%的求解质量")

            # 所有DSB算法按质量排序
            print(f"\n📊 DSB算法按求解质量排序:")
            # 过滤掉objective_value为None的结果
            dsb_valid = [(k, v) for k, v in dsb_results.items() if v.get("objective_value") is not None]
            if dsb_valid:
                dsb_quality_sorted = sorted(
                    dsb_valid, key=lambda x: x[1]["objective_value"], reverse=self.objective_sense == "maximize"
                )
            else:
                dsb_quality_sorted = []

            for i, (algo_name, result) in enumerate(dsb_quality_sorted, 1):
                quality_str = (
                    f"{result['quality_ratio']*100:.2f}%" if result.get("quality_ratio") is not None else "N/A"
                )
                obj_value_str = (
                    f"{result['objective_value']:.6f}" if result.get("objective_value") is not None else "N/A"
                )
                print(
                    f"   {i}. {algo_name}: 目标值={obj_value_str}, 质量={quality_str}, 时间={result['execution_time_ms']:.2f}ms"
                )
        else:
            print(f"❌ 没有DSB算法结果")

        # ========== 总结 ==========
        print(f"\n" + "=" * 80)
        print(f"📈 总结")
        print("=" * 80)

        if bsb_quality_99 and dsb_quality_99:
            # 比较BSB和DSB中最快的99%质量算法
            fastest_bsb = sorted(bsb_quality_99, key=lambda x: x[1]["execution_time_ms"])[0]
            fastest_dsb = sorted(dsb_quality_99, key=lambda x: x[1]["execution_time_ms"])[0]

            # 确保execution_time_ms不为None
            bsb_time = fastest_bsb[1]["execution_time_ms"]
            dsb_time = fastest_dsb[1]["execution_time_ms"]

            if bsb_time is None:
                bsb_time = 0.0
            if dsb_time is None:
                dsb_time = 0.0

            print(f"\n在达到99%求解质量的前提下:")
            print(f"   BSB最快: {fastest_bsb[0]} ({bsb_time:.2f}ms)")
            print(f"   DSB最快: {fastest_dsb[0]} ({dsb_time:.2f}ms)")

            if bsb_time < dsb_time:
                print(f"   ✅ BSB在速度上领先 {dsb_time - bsb_time:.2f}ms")
            else:
                print(f"   ✅ DSB在速度上领先 {bsb_time - dsb_time:.2f}ms")
        elif bsb_quality_99:
            print(f"\n仅有BSB算法达到99%求解质量")
        elif dsb_quality_99:
            print(f"\n仅有DSB算法达到99%求解质量")
        else:
            print(f"\n⚠️ 没有算法达到99%求解质量")

    def save_results_to_json(self, results, output_file=None):
        """
        将结果保存到JSON文件

        参数：
            results (dict): 包含所有算法结果的字典
            output_file (str, optional): 输出文件名。如果为None，自动生成文件名
        """
        if output_file is None:
            # 创建save_json目录（如果不存在）
            json_save_dir = Path("save_json")
            json_save_dir.mkdir(exist_ok=True)
            output_file = json_save_dir / f"qplib_results_{self.problem_name}.json"

        # 准备保存的数据
        save_data = {
            "problem_info": {
                "problem_name": self.problem_name,
                "num_vars": self.num_vars,
                "num_constraints": self.num_constraints,
                "objective_sense": self.objective_sense,
                "scaling_factor": self.scaling_factor,
                "quad_terms_count": self.quad_terms_count,
                "linear_terms_count": self.linear_terms_count,
                "xi_parameter": self.xi,
                "exact_optimal": self.exact_optimal,
            },
            "algorithms": {},
            "summary": {
                "total_algorithms": len(results),
                "successful_algorithms": len([r for r in results.values() if r and r.get("parsing_success", False)]),
            },
        }

        # 保存每个算法的详细结果
        for algorithm_name, result in results.items():
            if result:
                energy_value = result.get("objective_value", 0)
                # 如果objective_value是None，使用0
                if energy_value is None:
                    energy_value = 0
                energy_to_optimal_ratio = None
                if self.exact_optimal is not None and self.exact_optimal != 0:
                    energy_to_optimal_ratio = energy_value / self.exact_optimal

                save_data["algorithms"][algorithm_name] = {
                    "objective_value": result.get("objective_value"),
                    "energy_value": energy_value,
                    "exact_optimal": self.exact_optimal,
                    "energy_to_optimal_ratio": energy_to_optimal_ratio,
                    "execution_time_ms": result.get("execution_time_ms"),
                    "solution_ones": result.get("solution_ones"),
                    "solution_zeros": result.get("solution_zeros"),
                    "solution_density": result.get("solution_density"),
                    "iterations": result.get("iterations"),
                    "batch_size": result.get("batch_size"),
                    "parsing_success": result.get("parsing_success", False),
                    "solution_length": len(result.get("solution_vector", [])),
                }

        # 保存到文件
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 详细结果已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 保存结果文件失败: {e}")

    def save_results_to_csv(self, results, output_file=None):
        """
        将所有算法结果保存到CSV文件（仅统计信息，不对比）

        参数：
            results (dict): 包含所有算法结果的字典
            output_file (str, optional): 输出文件名。如果为None，自动生成文件名
        """
        if output_file is None:
            # 创建save_csv目录（如果不存在）
            csv_save_dir = Path("save_csv")
            csv_save_dir.mkdir(exist_ok=True)
            output_file = csv_save_dir / f"qplib_results_{self.problem_name}.csv"

        # 准备CSV数据
        csv_rows = []

        # 添加每个算法的结果（每个结果行包含完整信息）
        for algorithm_name, result in results.items():
            if result and result.get("parsing_success", False):
                # 解析算法信息
                algorithm_type = "BSB" if algorithm_name.startswith("BSB_") else "DSB"
                device = "CPU" if "CPU" in algorithm_name else "GPU"
                precision = algorithm_name.split("_")[-1]

                objective_value = result.get("objective_value", 0)
                # 如果objective_value是None，使用0
                if objective_value is None:
                    objective_value = 0
                energy_value = objective_value  # 这里目标值和能量值相同

                # 计算与精确最优解的比率
                energy_to_optimal_ratio = None
                if self.exact_optimal is not None and self.exact_optimal != 0:
                    energy_to_optimal_ratio = energy_value / self.exact_optimal

                # 优先使用C++报告的执行时间
                execution_time = result.get("cpp_execution_time_ms", result.get("execution_time_ms", ""))

                row = {
                    "problem_name": self.problem_name,
                    "num_vars": self.num_vars,
                    "num_constraints": self.num_constraints,
                    "objective_sense": self.objective_sense,
                    "scaling_factor": self.scaling_factor,
                    "quad_terms_count": self.quad_terms_count,
                    "linear_terms_count": self.linear_terms_count,
                    "xi_parameter": self.xi,
                    "algorithm_type": algorithm_type,
                    "device": device,
                    "precision": precision,
                    "algorithm_name": algorithm_name,
                    "objective_value": objective_value,
                    "energy_value": energy_value,
                    "exact_optimal": self.exact_optimal,
                    "energy_to_optimal_ratio": energy_to_optimal_ratio,
                    "execution_time_ms": execution_time,
                    "cpp_execution_time_ms": result.get("cpp_execution_time_ms", ""),
                    "python_execution_time_ms": (
                        result.get("execution_time_ms", "") if result.get("cpp_execution_time_ms") is None else ""
                    ),
                    "solution_ones": result.get("solution_ones", ""),
                    "solution_zeros": result.get("solution_zeros", ""),
                    "solution_density": result.get("solution_density", ""),
                    "iterations": result.get("iterations", ""),
                    "batch_size": result.get("batch_size", ""),
                    "parsing_success": result.get("parsing_success", ""),
                }
                csv_rows.append(row)

        # 保存到CSV
        try:
            df = pd.DataFrame(csv_rows)
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\n💾 CSV结果已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 保存CSV文件失败: {e}")


def solve_single_qplib_file(
    qplib_file_path, device="cuda", fixed_xi=None, problem_db=None, all_results=None, optimized_params_file=None
):
    """
    求解单个QPLIB文件

    参数：
        qplib_file_path (str): QPLIB文件路径
        device (str): 计算设备
        fixed_xi (float): 固定的xi值
        problem_db (DataFrame): 问题数据库
        all_results (dict): 存储所有结果的字典
        optimized_params_file (str): 优化参数CSV文件路径

    返回：
        dict: 该文件的结果
    """
    print(f"\n{'='*80}")
    print(f"📂 开始处理文件: {qplib_file_path}")
    print(f"{'='*80}")

    # 创建求解器实例
    ins = QBExternal(
        qplib_file_path,
        device=device,
        fixed_xi=fixed_xi,
        problem_db=problem_db,
        optimized_params_file=optimized_params_file,
    )

    # 求解优化问题（使用所有12种算法变体：6种BSB + 6种DSB）
    print(f"\n🚀 开始调用算法接口（BSB + DSB）...")
    results = ins.solve_all_variants()

    print(f"\n🎉 文件 {Path(qplib_file_path).name} 求解完成！")

    # 保存详细结果到JSON文件
    ins.save_results_to_json(results)

    # 保存结果到CSV文件
    ins.save_results_to_csv(results)

    # 保存到汇总结果中
    if all_results is not None:
        problem_name = Path(qplib_file_path).stem
        all_results[problem_name] = {
            "file_path": str(qplib_file_path),  # 转换为字符串，避免JSON序列化错误
            "problem_name": ins.problem_name,
            "num_vars": ins.num_vars,
            "num_constraints": ins.num_constraints,
            "objective_sense": ins.objective_sense,
            "scaling_factor": ins.scaling_factor,
            "xi": ins.xi,
            "exact_optimal": ins.exact_optimal,
            "results": results,
        }

    return results, ins


def save_batch_csv(all_results, output_file=None):
    """
    保存所有问题的汇总结果到CSV

    参数：
        all_results (dict): 包含所有问题结果的字典
        output_file (str, optional): 输出文件名
    """
    csv_rows = []

    for problem_id, problem_data in all_results.items():
        if problem_data is None:
            continue

        results = problem_data["results"]
        exact_optimal = problem_data.get("exact_optimal")

        for algorithm_name, result in results.items():
            if result and result.get("parsing_success", False):
                objective_value = result.get("objective_value", 0)
                # 如果objective_value是None，使用0
                if objective_value is None:
                    objective_value = 0
                energy_value = objective_value

                # 计算与精确最优解的比率
                energy_to_optimal_ratio = None
                if exact_optimal is not None and exact_optimal != 0:
                    energy_to_optimal_ratio = energy_value / exact_optimal

                # 优先使用C++报告的执行时间
                execution_time = result.get("cpp_execution_time_ms", result.get("execution_time_ms", ""))

                row = {
                    "problem_id": problem_id,
                    "num_vars": problem_data["num_vars"],
                    "objective_sense": problem_data["objective_sense"],
                    "scaling_factor": problem_data["scaling_factor"],
                    "xi_parameter": problem_data["xi"],
                    "algorithm_name": algorithm_name,
                    "algorithm_type": "BSB" if algorithm_name.startswith("BSB_") else "DSB",
                    "precision": algorithm_name.split("_")[-1],
                    "device": "CPU" if "CPU" in algorithm_name else "GPU",
                    "objective_value": objective_value,
                    "energy_value": energy_value,
                    "exact_optimal": exact_optimal,
                    "energy_to_optimal_ratio": energy_to_optimal_ratio,
                    "execution_time_ms": execution_time,
                    "cpp_execution_time_ms": result.get("cpp_execution_time_ms", ""),
                    "python_execution_time_ms": (
                        result.get("execution_time_ms", "") if result.get("cpp_execution_time_ms") is None else ""
                    ),
                    "solution_ones": result.get("solution_ones", ""),
                    "solution_zeros": result.get("solution_zeros", ""),
                    "solution_density": result.get("solution_density", ""),
                    "iterations": result.get("iterations", ""),
                    "batch_size": result.get("batch_size", ""),
                    "parsing_success": result.get("parsing_success", ""),
                }
                csv_rows.append(row)

    # 保存到CSV
    try:
        csv_save_dir = Path("save_csv")
        csv_save_dir.mkdir(exist_ok=True)

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = csv_save_dir / f"batch_summary_{timestamp}.csv"

        df = pd.DataFrame(csv_rows)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n💾 批量汇总CSV已保存到: {output_file}")
    except Exception as e:
        print(f"❌ 保存批量CSV文件失败: {e}")


def main():
    """
    主函数：依次求解qplib文件夹下的所有.qplib文件，使用12种算法（6种BSB + 6种DSB）

    说明：
        - 不使用qplib_parser.py
        - 直接依赖C++ exe通过qplib_parser.h解析数据
        - 从qplib文件夹读取全部23个.qplib数据文件
    """
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA设备可用：{torch.cuda.get_device_name(0)}")
        print(f"CUDA版本：{torch.version.cuda}")
        device = "cuda"
    else:
        print("CUDA不可用，使用CPU")
        device = "cpu"

    # 优化参数CSV文件路径
    OPTIMIZED_PARAMS_FILE = "save_csv/batch_optuna_12algs_20260127_160034.csv"

    # 优化后的xi值（来自optuna调优结果）
    OPTIMIZED_XI = 0.0  # 不再使用，改为从CSV加载

    # 加载问题数据库
    db_path = "data/qplib_results.csv"
    problem_db = None
    if os.path.exists(db_path):
        try:
            problem_db = pd.read_csv(db_path)
            print(f"✅ 加载问题数据库成功: {len(problem_db)} 个问题")
        except Exception as e:
            print(f"⚠️  加载问题数据库失败: {e}")
    else:
        print(f"⚠️  问题数据库不存在: {db_path}")

    # 获取qplib文件夹下所有的.qplib文件
    qplib_dir = Path("qplib")
    if not qplib_dir.exists():
        print(f"❌ qplib文件夹不存在: {qplib_dir}")
        return

    qplib_files = sorted(qplib_dir.glob("*.qplib"))

    if not qplib_files:
        print(f"❌ qplib文件夹中没有找到.qplib文件")
        return

    print(f"\n📋 从 qplib 文件夹找到 {len(qplib_files)} 个QPLIB文件:")
    for i, qplib_file in enumerate(qplib_files, 1):
        print(f"   {i}. {qplib_file.name} ({qplib_file.stat().st_size / 1024:.2f} KB)")

    # 检查优化参数文件是否存在
    optimized_params_available = Path(OPTIMIZED_PARAMS_FILE).exists()
    if optimized_params_available:
        print(f"\n✅ 找到优化参数文件: {OPTIMIZED_PARAMS_FILE}")
    else:
        print(f"\n⚠️  优化参数文件不存在: {OPTIMIZED_PARAMS_FILE}")
        print(f"   将使用默认参数 xi=0.365")

    print(f"\n🔧 将使用以下12种算法进行求解:")
    print(f"   BSB算法 (6种): BSB_CPU_fp16, BSB_CPU_int8, BSB_CPU_fp8,")
    print(f"                  BSB_GPU_fp16, BSB_GPU_int8, BSB_GPU_fp8")
    print(f"   DSB算法 (6种): DSB_CPU_fp16, DSB_CPU_int8, DSB_CPU_fp8,")
    print(f"                  DSB_GPU_fp16, DSB_GPU_int8, DSB_GPU_fp8")
    print(f"\n📌 说明: 数据解析由C++ exe通过qplib_parser.h完成")
    if optimized_params_available:
        print(f"📌 每个算法将使用对应的最优参数（xi和sigma）")

    # 确认是否执行
    user_confirm = input(f"\n是否继续执行所有 {len(qplib_files)} 个文件的求解？(y/n): ").lower().strip()
    if user_confirm != "y":
        print("❌ 用户取消执行")
        return

    # 存储所有结果
    all_results = {}

    # 依次求解每个文件
    print(f"\n{'#'*80}")
    print(f"# 开始批量求解 {len(qplib_files)} 个QPLIB文件")
    print(f"# 算法配置: 12种 (BSB: 6种, DSB: 6种)")
    if optimized_params_available:
        print(f"# 使用优化参数文件: {OPTIMIZED_PARAMS_FILE}")
        print(f"# 每个算法将使用从CSV加载的最优xi参数")
    else:
        print(f"# 使用默认参数 xi=0.365")
    print(f"# 设备: {device}")
    print(f"# 数据解析: 由C++ exe通过qplib_parser.h完成（不使用qplib_parser.py）")
    print(f"{'#'*80}")

    start_time = time.time()

    for i, qplib_file in enumerate(qplib_files, 1):
        try:
            results, ins = solve_single_qplib_file(
                qplib_file,
                device=device,
                fixed_xi=None,  # 不使用固定xi，从CSV加载
                problem_db=problem_db,
                all_results=all_results,
                optimized_params_file=OPTIMIZED_PARAMS_FILE if optimized_params_available else None,
            )
            print(f"\n✅ 进度: {i}/{len(qplib_files)} 文件已完成")
        except Exception as e:
            print(f"\n❌ 文件 {qplib_file.name} 处理失败: {e}")
            continue

    end_time = time.time()
    total_time = end_time - start_time

    # 生成汇总报告
    print(f"\n{'#'*80}")
    print(f"# 批量求解完成汇总报告")
    print(f"{'#'*80}")
    print(f"总文件数: {len(qplib_files)}")
    print(f"成功处理: {len(all_results)}")
    print(f"总耗时: {total_time:.2f} 秒")
    # 避免除以零错误
    if len(qplib_files) > 0:
        print(f"平均每文件: {total_time/len(qplib_files):.2f} 秒")
    else:
        print(f"平均每文件: N/A")

    # 比较所有问题的最佳算法
    print(f"\n📊 所有问题的最佳算法:")
    print(f"{'='*80}")

    for problem_name, problem_data in all_results.items():
        results = problem_data["results"]
        valid_results = {k: v for k, v in results.items() if v and v.get("parsing_success", False)}

        if valid_results:
            # 过滤掉objective_value为None的结果
            valid_with_obj = {k: v for k, v in valid_results.items() if v.get("objective_value") is not None}

            if valid_with_obj:
                if problem_data.get("objective_sense") == "maximize":
                    best = max(valid_with_obj.items(), key=lambda x: x[1]["objective_value"])
                else:
                    best = min(valid_with_obj.items(), key=lambda x: x[1]["objective_value"])

                best_algo = best[0]
                best_obj = best[1]["objective_value"]
                best_time = best[1]["execution_time_ms"]

                print(f"\n{problem_name}:")
                print(
                    f"  变量数: {problem_data.get('num_vars', 'N/A')}, 优化方向: {problem_data.get('objective_sense', 'N/A')}"
                )
                print(f"  最佳算法: {best_algo}")
                print(f"  目标值: {best_obj:.6f}, 耗时: {best_time:.2f} ms")
            else:
                # 没有有效的objective_value，只显示第一个成功解析的算法
                first_valid = list(valid_results.items())[0]
                best_algo = first_valid[0]
                best_time = first_valid[1]["execution_time_ms"]

                print(f"\n{problem_name}:")
                print(
                    f"  变量数: {problem_data.get('num_vars', 'N/A')}, 优化方向: {problem_data.get('objective_sense', 'N/A')}"
                )
                print(f"  最佳算法: {best_algo} (未获得有效目标值)")
                print(f"  目标值: N/A, 耗时: {best_time:.2f} ms")

    # 保存汇总结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"save_json/all_qplib_results_summary_{timestamp}.json"

    # 清理 all_results 中的非JSON序列化对象（如WindowsPath）
    cleaned_all_results = {}
    for problem_name, problem_data in all_results.items():
        cleaned_data = problem_data.copy()
        # 将 WindowsPath 转换为字符串
        if "file_path" in cleaned_data and hasattr(cleaned_data["file_path"], "__fspath__"):
            cleaned_data["file_path"] = str(cleaned_data["file_path"])
        cleaned_all_results[problem_name] = cleaned_data

    summary_data = {
        "summary": {
            "total_files": len(qplib_files),
            "successful_files": len(all_results),
            "total_time_seconds": total_time,
            "timestamp": timestamp,
            "device": device,
            "xi_parameter": OPTIMIZED_XI,
            "parser_source": "C++ exe (qplib_parser.h)",
            "note": "不使用qplib_parser.py，数据由C++ exe通过qplib_parser.h解析",
        },
        "problems": cleaned_all_results,  # 使用清理后的版本
    }

    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 汇总结果已保存到: {summary_file}")
    except Exception as e:
        print(f"❌ 保存汇总结果失败: {e}")

    # 生成CSV格式的汇总报告
    save_batch_csv(all_results)

    print(f"\n{'='*80}")
    print(f"🎉 所有问题求解完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
