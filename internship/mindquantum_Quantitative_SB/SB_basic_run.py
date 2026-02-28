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


class QBExternal:
    def __init__(
        self, file_path=None, device="cuda", fixed_xi=0, problem_id=None, use_optuna_params=False, optuna_csv_path=None
    ):
        """
        初始化量子退火求解器（外部BSB/DSB接口版本）

        参数：
            file_path (str, optional): QPLIB文件路径。如果为None且指定problem_id，则从数据库查找
            device (str): 计算设备 ('cuda' 或 'cpu')
            fixed_xi (float, optional): 固定的xi值，默认为0
            problem_id (str, optional): QPLIB问题ID（如 'QPLIB_3565'）。如果为None，默认使用第一条数据
            use_optuna_params (bool, optional): 是否从Optuna优化结果中获取最优参数，默认为False
            optuna_csv_path (str, optional): Optuna优化结果的CSV文件路径。如果为None且use_optuna_params为True，则自动查找最新的CSV文件
        """
        # 加载问题数据库
        self.problem_db_path = "data/qplib_results.csv"
        self.problem_db = self.load_problem_database()

        # Optuna参数数据库
        self.use_optuna_params = use_optuna_params
        self.optuna_csv_path = optuna_csv_path
        self.optuna_db = None
        if use_optuna_params:
            self.optuna_db = self.load_optuna_database(optuna_csv_path)

        # 确定要处理的问题

        if problem_id is not None:
            # 使用指定的问题ID
            self.problem_name = problem_id.strip().strip("'").strip('"')  # 移除可能的引号和多余字符
            file_path = self.get_qplib_file_from_db(self.problem_name)
            if file_path is None:
                raise ValueError(f"问题ID '{self.problem_name}' 在数据库中未找到")
        elif file_path is not None:
            # 从文件路径提取问题ID
            file_name = Path(file_path).stem
            self.problem_name = file_name
        else:
            # 没有指定，使用数据库中第一个实际存在的文件
            for idx in range(len(self.problem_db)):
                problem = self.problem_db.iloc[idx]
                test_path = f"qplib/{problem['instance_id']}.qplib"
                if os.path.exists(test_path):
                    self.problem_name = problem["instance_id"]
                    file_path = test_path
                    print(f"未指定问题，使用数据库第{idx+1}条: {self.problem_name}")
                    break

        # 验证QPLIB文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"QPLIB文件不存在: {file_path}")

        # 从数据库读取精确最优解（如果有）
        clean_name = self.problem_name.strip().strip("'").strip('"')
        problem_row = self.problem_db[self.problem_db["instance_id"] == clean_name]
        if not problem_row.empty:
            exact_optimal = self._parse_exact_solution(problem_row.iloc[0]["solution_objective_value"])
        else:
            exact_optimal = None

        # 使用C++的qplib_parser.h来解析，这里我们通过exe的输出来获取问题信息
        # 先读取QPLIB文件的基本信息（包括优化方向）
        self.file_path = file_path
        self._parse_qplib_basic_info(file_path)

        # 使用固定的xi值（默认为0）
        self.xi = fixed_xi if fixed_xi is not None else 0
        print(f"✅ 使用固定的xi值: {self.xi:.6f}")

        self.device = device

        # 获取BSB和DSB可执行文件路径
        self.bsb_dir = Path("BSB")
        self.dsb_dir = Path("DSB")

        self.bsb_executables = {
            "BSB_CPU_fp16": self.bsb_dir / "BSB_CPU_fp16.exe",
            "BSB_CPU_int8": self.bsb_dir / "BSB_CPU_int8.exe",
            "BSB_CPU_fp8": self.bsb_dir / "BSB_CPU_fp8.exe",
            "BSB_GPU_fp16": self.bsb_dir / "BSB_GPU_fp16.exe",
            "BSB_GPU_int8": self.bsb_dir / "BSB_GPU_int8.exe",
            "BSB_GPU_fp8": self.bsb_dir / "BSB_GPU_fp8.exe",
        }

        self.dsb_executables = {
            "DSB_CPU_fp16": self.dsb_dir / "DSB_CPU_fp16.exe",
            "DSB_CPU_int8": self.dsb_dir / "DSB_CPU_int8.exe",
            "DSB_CPU_fp8": self.dsb_dir / "DSB_CPU_fp8.exe",
            "DSB_GPU_fp16": self.dsb_dir / "DSB_GPU_fp16.exe",
            "DSB_GPU_int8": self.dsb_dir / "DSB_GPU_int8.exe",
            "DSB_GPU_fp8": self.dsb_dir / "DSB_GPU_fp8.exe",
        }

        # 合并所有可执行文件
        self.all_executables = {**self.bsb_executables, **self.dsb_executables}

        # 检查可执行文件是否存在
        missing_executables = [name for name, path in self.all_executables.items() if not path.exists()]
        if missing_executables:
            print(f"⚠️ 以下算法可执行文件不存在: {missing_executables}")
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

        print(f"QPLIB文件: {file_path}")
        print(f"优化方向: {self.objective_sense}")
        print(f"问题统计: 变量数={self.num_vars}, 二次项={self.quad_terms_count}, 线性项={self.linear_terms_count}")

    def _parse_exact_solution(self, solution_str):
        """解析精确最优解字符串"""
        if pd.isna(solution_str) or solution_str == "":
            return None
        match = re.search(r"([+-]?\d*\.?\d+)", str(solution_str))
        if match:
            return float(match.group(1))
        return None

    def _parse_qplib_basic_info(self, file_path):
        """直接读取QPLIB文件的基本信息（不使用Python解析器）"""
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith("#")]

        # 解析基本信息
        self.problem_name = lines[0]
        self.num_vars = int(lines[3].split("#")[0].strip())
        self.num_constraints = 0  # 默认无约束

        # 第3行：优化方向（minimize 或 maximize）
        self.objective_sense = lines[2].strip().lower()

        # 第5行：二次项数量
        num_quad_terms = int(lines[4].split("#")[0].strip())
        self.quad_terms_count = num_quad_terms

        # 解析线性项数量（需要读取更多行）
        current = 5 + num_quad_terms  # 跳过二次项
        default_linear_value = float(lines[current].split("#")[0].strip())
        num_linear_terms = int(lines[current + 1].split("#")[0].strip())

        # 计算非零线性项数量
        if default_linear_value == 0 and num_linear_terms == 0:
            self.linear_terms_count = 0
        else:
            self.linear_terms_count = num_linear_terms
            if default_linear_value != 0:
                self.linear_terms_count = self.num_vars  # 有默认值

        self.constant_term = 0.0  # 可选读取
        self.scaling_factor = 1.0  # 默认值，实际由C++计算

        print(
            f"QPLIB基本信息: 优化方向={self.objective_sense}, 变量数={self.num_vars}, 二次项={self.quad_terms_count}, 线性项={self.linear_terms_count}"
        )

    def load_problem_database(self):
        """
        加载问题数据库

        返回：
            DataFrame: 问题数据库
        """
        try:
            if os.path.exists(self.problem_db_path):
                df = pd.read_csv(self.problem_db_path)
                print(f"✅ 加载问题数据库成功: {len(df)} 个问题")
                return df
            else:
                print(f"⚠️ 问题数据库不存在: {self.problem_db_path}")
                return pd.DataFrame()
        except Exception as e:
            print(f"❌ 加载问题数据库失败: {e}")
            return pd.DataFrame()

    def load_optuna_database(self, csv_path=None):
        """
        加载Optuna优化结果数据库

        参数：
            csv_path (str, optional): CSV文件路径。如果为None，自动查找最新的CSV文件

        返回：
            DataFrame: Optuna优化结果数据库
        """
        if csv_path is None:
            # 自动查找save_csv目录下最新的batch_optuna_12algs_*.csv文件
            from pathlib import Path

            csv_save_dir = Path("save_csv")
            if csv_save_dir.exists():
                csv_files = sorted(csv_save_dir.glob("batch_optuna_12algs_*.csv"), reverse=True)
                if csv_files:
                    csv_path = csv_files[0]
                    print(f"✅ 自动找到最新的Optuna优化结果文件: {csv_path}")
                else:
                    print(f"⚠️ save_csv目录下没有找到Optuna优化结果文件")
                    return pd.DataFrame()
            else:
                print(f"⚠️ save_csv目录不存在")
                return pd.DataFrame()

        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                print(f"✅ 加载Optuna优化数据库成功: {len(df)} 条记录")
                print(f"   文件路径: {csv_path}")
                return df
            else:
                print(f"⚠️ Optuna优化数据库不存在: {csv_path}")
                return pd.DataFrame()
        except Exception as e:
            print(f"❌ 加载Optuna优化数据库失败: {e}")
            return pd.DataFrame()

    def get_optimal_params_for_algorithm(self, algorithm_name, problem_name):
        """
        从Optuna优化结果中获取指定算法和问题的最优参数

        参数：
            algorithm_name (str): 算法名称（如 'BSB_CPU_fp16', 'DSB_GPU_int8'）
            problem_name (str): 问题名称（如 'QPLIB_3565'）

        返回：
            dict: 包含最优参数的字典 {'xi': float, 'sigma': float}，如果未找到则返回None
        """
        if self.optuna_db is None or self.optuna_db.empty:
            return None

        # 查找匹配的记录
        mask = (self.optuna_db["algorithm_name"] == algorithm_name) & (self.optuna_db["problem_name"] == problem_name)
        matching_rows = self.optuna_db[mask]

        if matching_rows.empty:
            # 尝试模糊匹配问题名称
            mask = (self.optuna_db["algorithm_name"] == algorithm_name) & (
                self.optuna_db["problem_name"].str.contains(problem_name, na=False, case=False)
            )
            matching_rows = self.optuna_db[mask]

        if matching_rows.empty:
            print(f"⚠️ 未在Optuna优化结果中找到 {algorithm_name} - {problem_name} 的参数")
            return None

        # 取第一条匹配的记录（应该只有一条）
        row = matching_rows.iloc[0]

        try:
            params = {
                "xi": float(row["best_xi"]),
                "sigma": float(row["best_sigma"]),
                "energy": float(row["best_energy"]),
                "gap_percent": row["gap_percent"] if pd.notna(row["gap_percent"]) else None,
            }
            print(f"✅ 从Optuna获取最优参数: {algorithm_name} - {problem_name}")
            print(f"   xi = {params['xi']:.6f}, sigma = {params['sigma']:.6f}")
            if params["gap_percent"] is not None:
                print(f"   预期能量 = {params['energy']:.6f}, gap = {params['gap_percent']:.2f}%")
            return params
        except Exception as e:
            print(f"❌ 解析Optuna参数失败: {e}")
            return None

    def get_qplib_file_from_db(self, problem_id):
        """
        从数据库获取QPLIB文件路径

        参数：
            problem_id (str): 问题ID（如 'QPLIB_3565'）

        返回：
            str: QPLIB文件路径，如果未找到则返回None
        """
        # 清理problem_id，移除可能的引号
        clean_id = problem_id.strip().strip("'").strip('"')

        problem_row = self.problem_db[self.problem_db["instance_id"] == clean_id]
        if problem_row.empty:
            # 尝试模糊匹配
            print(f"⚠️ 精确查找失败，尝试模糊匹配...")
            problem_row = self.problem_db[self.problem_db["instance_id"].str.contains(clean_id, na=False, case=False)]
            if problem_row.empty:
                return None
            else:
                print(f"✅ 模糊匹配找到: {problem_row.iloc[0]['instance_id']}")

        # 尝试查找QPLIB文件
        qplib_file = f"qplib/{clean_id}.qplib"
        if os.path.exists(qplib_file):
            return qplib_file
        else:
            print(f"⚠️ QPLIB文件不存在: {qplib_file}")
            return None

    def list_available_problems(self, limit=10):
        """
        列出可用的QPLIB问题

        参数：
            limit (int): 最多显示的问题数量

        返回：
            list: 可用的问题ID列表
        """
        if self.problem_db.empty:
            print("❌ 问题数据库为空")
            return []

        problems = self.problem_db["instance_id"].tolist()[:limit]
        print(f"\n📋 可用的QPLIB问题 (前{len(problems)}个):")
        for i, prob_id in enumerate(problems, 1):
            problem_row = self.problem_db[self.problem_db["instance_id"] == prob_id]
            if not problem_row.empty:
                obj_value = problem_row["solution_objective_value"].iloc[0]
                print(f"   {i}. {prob_id} - 精确最优解: {obj_value}")
            else:
                print(f"   {i}. {prob_id}")

        if len(self.problem_db) > limit:
            print(f"   ... 还有 {len(self.problem_db) - limit} 个问题")
            print(f"   总计: {len(self.problem_db)} 个问题")

        return problems

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

        # 如果启用了Optuna参数，尝试获取最优参数
        optuna_params = None
        if self.use_optuna_params:
            optuna_params = self.get_optimal_params_for_algorithm(algorithm_name, self.problem_name)
            if optuna_params:
                print(f"\n🔧 运行 {algorithm_name} 算法（使用Optuna最优参数）...")
            else:
                print(f"\n🔧 运行 {algorithm_name} 算法（使用默认参数）...")
        else:
            print(f"\n🔧 运行 {algorithm_name} 算法...")

        print(f"   可执行文件: {executable_path}")
        print(f"   输入文件: {qplib_file_path}")

        # 使用算法期望的路径格式（参考qplib_parser.h中的定义）
        # 算法期望从其目录看到的路径是 "../qplib/文件名"
        qplib_filename = Path(qplib_file_path).name
        algo_qplib_path = f"../qplib/{qplib_filename}"

        # 构建命令行参数
        if optuna_params:
            # 使用Optuna优化的参数
            cmd = [
                str(executable_path),
                algo_qplib_path,
                "-x",
                str(optuna_params["xi"]),
                "-s",
                str(optuna_params["sigma"]),
            ]
            print(f"   参数: xi={optuna_params['xi']:.6f}, sigma={optuna_params['sigma']:.6f}")
        else:
            # 使用默认参数
            cmd = [str(executable_path), algo_qplib_path]

        print(f"   {algorithm_type}期望路径: {algo_qplib_path}")

        start_time = time.time()

        try:
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
            parsed_result = self.parse_algorithm_output(output_text, algorithm_name, execution_time)

            # 如果使用了Optuna参数，添加参数信息到结果
            if parsed_result and optuna_params:
                parsed_result["used_optuna_params"] = True
                parsed_result["optuna_xi"] = optuna_params["xi"]
                parsed_result["optuna_sigma"] = optuna_params["sigma"]
                parsed_result["optuna_predicted_energy"] = optuna_params["energy"]
                parsed_result["optuna_gap_percent"] = optuna_params["gap_percent"]

                # 计算实际结果与Optuna预测的差距
                if optuna_params["gap_percent"] is not None:
                    predicted_energy = optuna_params["energy"]
                    actual_energy = parsed_result.get("objective_value", 0)
                    if predicted_energy != 0:
                        energy_diff_pct = abs(actual_energy - predicted_energy) / abs(predicted_energy) * 100
                        parsed_result["optuna_energy_diff_percent"] = energy_diff_pct
                        print(
                            f"   Optuna预测: {predicted_energy:.6f}, 实际: {actual_energy:.6f}, 差异: {energy_diff_pct:.2f}%"
                        )

            return parsed_result

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
                # 匹配多种时间格式
                if "C++_TIME:" in line or "TIME: Total solving time:" in line or "TIME: Solving completed, total time:" in line:
                    match = re.search(r"time:\s*(\d+\.?\d*)\s*ms", line, re.IGNORECASE)
                    if match:
                        result["cpp_execution_time_ms"] = float(match.group(1))
                        # 优先使用C++报告的时间
                        result["execution_time_ms"] = result["cpp_execution_time_ms"]
                        break

            # 解析目标函数值（支持BSB和DSB格式）
            for line in lines:
                if (
                    "BSB OBJECTIVE VALUE:" in line
                    or "BSB GPU OBJECTIVE VALUE:" in line
                    or "DSB OBJECTIVE VALUE:" in line
                    or "DSB GPU OBJECTIVE VALUE:" in line
                    or "Optimal objective value:" in line
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
                        result["solution_density"] = (result["solution_ones"] / self.num_vars) * 100
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
                            result["solution_density"] = (result["solution_ones"] / self.num_vars) * 100
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
                            result["solution_density"] = (result["solution_ones"] / self.num_vars) * 100
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
                            result["solution_density"] = (result["solution_ones"] / self.num_vars) * 100
                            solution_ones_found = True
                        break

            # 如果没有找到统计信息，手动计算
            if not solution_ones_found and solution_values:
                result["solution_ones"] = sum(solution_values)
                result["solution_zeros"] = len(solution_values) - result["solution_ones"]
                result["solution_density"] = (result["solution_ones"] / len(solution_values)) * 100

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
            print(f"   解中1的数量: {result['solution_ones']}/{self.num_vars} ({result['solution_density']:.2f}%)")

        except Exception as e:
            print(f"❌ {algorithm_name} 输出解析失败: {e}")
            result["error"] = str(e)

        return result

    def solve_all_variants(self, qplib_file_path=None, problem_id=None, selected_algorithms=None):
        """
        运行指定的算法变体（可选择性执行特定算法）

        参数：
            qplib_file_path (str, optional): QPLIB文件路径。如果为None，使用初始化时的文件路径
            problem_id (str, optional): QPLIB问题ID
            selected_algorithms (list, optional): 指定要执行的算法列表。如果为None，执行所有12种算法

        返回：
            dict: 包含所有算法结果的字典
        """
        # 如果指定了problem_id，重新初始化求解器
        if problem_id is not None and problem_id != self.problem_name:
            qplib_file_path = self.get_qplib_file_from_db(problem_id)
            if qplib_file_path is None:
                print(f"❌ 问题ID {problem_id} 对应的QPLIB文件不存在")
                return None

            # 重新解析新问题
            print(f"切换到问题: {problem_id}")
            self.problem_name = problem_id
            self._parse_qplib_basic_info(qplib_file_path)

            # 优化方向会通过 _parse_qplib_basic_info 从新文件读取，无需手动更新
            self.file_path = qplib_file_path

        if qplib_file_path is None:
            qplib_file_path = self.file_path

        # 所有12种算法
        bsb_algorithms = ["BSB_CPU_fp16", "BSB_CPU_int8", "BSB_CPU_fp8", "BSB_GPU_fp16", "BSB_GPU_int8", "BSB_GPU_fp8"]
        dsb_algorithms = ["DSB_CPU_fp16", "DSB_CPU_int8", "DSB_CPU_fp8", "DSB_GPU_fp16", "DSB_GPU_int8", "DSB_GPU_fp8"]
        all_algorithms = bsb_algorithms + dsb_algorithms

        # 确定要执行的算法
        if selected_algorithms is None:
            # 未指定，执行所有算法
            algorithms_to_run = all_algorithms
            print(f"\n🚀 开始运行所有算法变体（BSB + DSB）")
        else:
            # 指定了特定算法
            algorithms_to_run = []
            for algo in selected_algorithms:
                if algo in all_algorithms:
                    algorithms_to_run.append(algo)
                else:
                    print(f"⚠️ 警告: 算法 '{algo}' 不存在，跳过")

            if not algorithms_to_run:
                print("❌ 错误: 没有有效的算法可执行")
                return {}

            print(f"\n🚀 开始运行指定的算法: {', '.join(algorithms_to_run)}")

        print("=" * 80)
        print(f"问题: {self.problem_name}")
        print(f"文件: {qplib_file_path}")
        print(f"变量数: {self.num_vars}, 优化方向: {self.objective_sense}")
        print("=" * 80)

        results = {}

        # 执行选定的算法
        for algorithm in algorithms_to_run:
            if algorithm.startswith("BSB_"):
                print(f"\n📋 运行BSB算法: {algorithm}")
            else:
                print(f"\n📋 运行DSB算法: {algorithm}")

            result = self.run_algorithm_executable(algorithm, qplib_file_path)
            results[algorithm] = result

        # 分析和比较结果
        self.analyze_and_compare_results(results)

        return results

    @staticmethod
    def list_available_algorithms():
        """
        列出所有可用的算法

        返回：
            list: 算法名称列表
        """
        bsb_algorithms = ["BSB_CPU_fp16", "BSB_CPU_int8", "BSB_CPU_fp8", "BSB_GPU_fp16", "BSB_GPU_int8", "BSB_GPU_fp8"]
        dsb_algorithms = ["DSB_CPU_fp16", "DSB_CPU_int8", "DSB_CPU_fp8", "DSB_GPU_fp16", "DSB_GPU_int8", "DSB_GPU_fp8"]
        all_algorithms = bsb_algorithms + dsb_algorithms

        print(f"\n📋 可用的算法列表（共{len(all_algorithms)}种）:")
        print("=" * 80)
        print("\n🔹 BSB算法（Ballistic Simulated Bifurcation）:")
        for i, algo in enumerate(bsb_algorithms, 1):
            print(f"   {i}. {algo}")

        print("\n🔹 DSB算法（Discrete Simulated Bifurcation）:")
        for i, algo in enumerate(dsb_algorithms, 7):
            print(f"   {i}. {algo}")

        print("\n🔹 快捷选择:")
        print("   --algo BSB          # 运行所有6种BSB算法")
        print("   --algo DSB          # 运行所有6种DSB算法")
        print("   --algo BSB_CPU_fp16 # 运行特定算法")
        print("   --algo BSB_GPU_fp16,DSB_GPU_int8  # 运行多个指定算法")
        print("=" * 80)

        return all_algorithms

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

        # 获取精确最优解
        clean_name = self.problem_name.strip().strip("'").strip('"')
        problem_row = self.problem_db[self.problem_db["instance_id"] == clean_name]
        exact_optimal = None
        if not problem_row.empty:
            exact_optimal = self._parse_exact_solution(problem_row.iloc[0]["solution_objective_value"])

        # 计算求解质量（与精确最优解的比率）
        def calculate_quality_ratio(obj_value):
            if exact_optimal is None or exact_optimal == 0:
                return None
            try:
                return obj_value / exact_optimal
            except (TypeError, ZeroDivisionError):
                return None

        # 为每个结果添加求解质量信息
        for algo_name, result in valid_results.items():
            result["quality_ratio"] = calculate_quality_ratio(result["objective_value"])

        # 打印详细结果
        print(f"\n📋 所有算法详细结果:")
        for algorithm_name, result in valid_results.items():
            algorithm_type = "BSB" if algorithm_name.startswith("BSB_") else "DSB"
            quality_str = f"{result['quality_ratio']*100:.2f}%" if result["quality_ratio"] is not None else "N/A"
            time_str = f"{result['execution_time_ms']:.2f}ms" if result["execution_time_ms"] is not None else "N/A"
            obj_str = f"{result['objective_value']:.6f}" if result.get('objective_value') is not None else "N/A"
            print(
                f"   {algorithm_name} ({algorithm_type}): 目标值={obj_str}, 质量={quality_str}, 时间={time_str}"
            )

        # BSB vs DSB 对比（总体）
        if bsb_results and dsb_results:
            print(f"\n🔍 BSB vs DSB 总体对比:")

            # 找到各类别最佳（过滤掉None值）
            bsb_valid = {k: v for k, v in bsb_results.items() if v.get("objective_value") is not None}
            dsb_valid = {k: v for k, v in dsb_results.items() if v.get("objective_value") is not None}

            if bsb_valid:
                if self.objective_sense == "maximize":
                    bsb_best = max(bsb_valid.items(), key=lambda x: x[1]["objective_value"])
                else:
                    bsb_best = min(bsb_valid.items(), key=lambda x: x[1]["objective_value"])
            else:
                bsb_best = None

            if dsb_valid:
                if self.objective_sense == "maximize":
                    dsb_best = max(dsb_valid.items(), key=lambda x: x[1]["objective_value"])
                else:
                    dsb_best = min(dsb_valid.items(), key=lambda x: x[1]["objective_value"])
            else:
                dsb_best = None

            if not bsb_best or not dsb_best:
                print("   无法对比：缺少有效结果")
                return

            bsb_obj_str = f"{bsb_best[1]['objective_value']:.6f}" if bsb_best[1].get('objective_value') is not None else "N/A"
            dsb_obj_str = f"{dsb_best[1]['objective_value']:.6f}" if dsb_best[1].get('objective_value') is not None else "N/A"
            bsb_time_str = f"{bsb_best[1].get('execution_time_ms', 0):.2f} ms" if bsb_best[1].get('execution_time_ms') is not None else "N/A"
            dsb_time_str = f"{dsb_best[1].get('execution_time_ms', 0):.2f} ms" if dsb_best[1].get('execution_time_ms') is not None else "N/A"
            print(
                f"   BSB最佳: {bsb_best[0]} (目标值: {bsb_obj_str}, 时间: {bsb_time_str})"
            )
            print(
                f"   DSB最佳: {dsb_best[0]} (目标值: {dsb_obj_str}, 时间: {dsb_time_str})"
            )

            # 计算平均性能（过滤None值）
            bsb_objs = [r["objective_value"] for r in bsb_results.values() if r.get("objective_value") is not None]
            dsb_objs = [r["objective_value"] for r in dsb_results.values() if r.get("objective_value") is not None]
            bsb_times = [r["execution_time_ms"] for r in bsb_results.values() if r.get("execution_time_ms") is not None]
            dsb_times = [r["execution_time_ms"] for r in dsb_results.values() if r.get("execution_time_ms") is not None]

            bsb_avg_obj = sum(bsb_objs) / len(bsb_objs) if bsb_objs else None
            dsb_avg_obj = sum(dsb_objs) / len(dsb_objs) if dsb_objs else None
            bsb_avg_time = sum(bsb_times) / len(bsb_times) if bsb_times else None
            dsb_avg_time = sum(dsb_times) / len(dsb_times) if dsb_times else None

            print(f"   BSB平均目标值: {bsb_avg_obj:.6f}, 平均时间: {bsb_avg_time:.2f} ms" if isinstance(bsb_avg_time, (int, float)) else f"   BSB平均目标值: {bsb_avg_obj:.6f}, 平均时间: N/A")
            print(f"   DSB平均目标值: {dsb_avg_obj:.6f}, 平均时间: {dsb_avg_time:.2f} ms" if isinstance(dsb_avg_time, (int, float)) else f"   DSB平均目标值: {dsb_avg_obj:.6f}, 平均时间: N/A")

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
                    time_str = f"{result.get('execution_time_ms', 0):.2f}ms" if result.get("execution_time_ms") is not None else "N/A"
                    obj_str = f"{result.get('objective_value', 0):.6f}" if result.get('objective_value') is not None else "N/A"
                    print(
                        f"   {i}. {algo_name}: 质量={quality_str}, 时间={time_str}, 目标值={obj_str}"
                    )

                # 最快的99%质量算法
                fastest_bsb = bsb_quality_99_sorted[0]
                print(f"\n🏆 BSB算法 - 最快的99%质量算法: {fastest_bsb[0]}")
                print(f"   求解质量: {fastest_bsb[1]['quality_ratio']*100:.2f}%")
                time_str = f"{fastest_bsb[1].get('execution_time_ms', 0):.2f} ms" if fastest_bsb[1].get('execution_time_ms') is not None else "N/A"
                print(f"   执行时间: {time_str}")
            else:
                print(f"\n⚠️ 没有BSB算法达到99%的求解质量")

            # 所有BSB算法按质量排序（过滤None值）
            print(f"\n📊 BSB算法按求解质量排序:")
            bsb_quality_sorted = sorted(
                bsb_valid.items(), key=lambda x: x[1]["objective_value"],
                reverse=self.objective_sense == "maximize"
            )

            for i, (algo_name, result) in enumerate(bsb_quality_sorted, 1):
                quality_str = f"{result['quality_ratio']*100:.2f}%" if result["quality_ratio"] is not None else "N/A"
                time_str = f"{result.get('execution_time_ms', 0):.2f}ms" if result.get("execution_time_ms") is not None else "N/A"
                obj_str = f"{result['objective_value']:.6f}" if result.get('objective_value') is not None else "N/A"
                print(
                    f"   {i}. {algo_name}: 目标值={obj_str}, 质量={quality_str}, 时间={time_str}"
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
                dsb_quality_99_sorted = sorted(dsb_quality_99, key=lambda x: x[1].get("execution_time_ms", 0) or 0)

                for i, (algo_name, result) in enumerate(dsb_quality_99_sorted, 1):
                    quality_str = (
                        f"{result['quality_ratio']*100:.2f}%" if result["quality_ratio"] is not None else "N/A"
                    )
                    time_str = f"{result.get('execution_time_ms', 0):.2f}ms" if result.get("execution_time_ms") is not None else "N/A"
                    obj_str = f"{result.get('objective_value', 0):.6f}" if result.get('objective_value') is not None else "N/A"
                    print(
                        f"   {i}. {algo_name}: 质量={quality_str}, 时间={time_str}, 目标值={obj_str}"
                    )

                # 最快的99%质量算法
                fastest_dsb = dsb_quality_99_sorted[0]
                print(f"\n🏆 DSB算法 - 最快的99%质量算法: {fastest_dsb[0]}")
                print(f"   求解质量: {fastest_dsb[1]['quality_ratio']*100:.2f}%")
                time_str = f"{fastest_dsb[1].get('execution_time_ms', 0):.2f} ms" if fastest_dsb[1].get('execution_time_ms') is not None else "N/A"
                print(f"   执行时间: {time_str}")
            else:
                print(f"\n⚠️ 没有DSB算法达到99%的求解质量")

            # 所有DSB算法按质量排序（过滤None值）
            print(f"\n📊 DSB算法按求解质量排序:")
            dsb_quality_sorted = sorted(
                dsb_valid.items(), key=lambda x: x[1]["objective_value"],
                reverse=self.objective_sense == "maximize"
            )

            for i, (algo_name, result) in enumerate(dsb_quality_sorted, 1):
                quality_str = f"{result['quality_ratio']*100:.2f}%" if result["quality_ratio"] is not None else "N/A"
                time_str = f"{result.get('execution_time_ms', 0):.2f}ms" if result.get("execution_time_ms") is not None else "N/A"
                obj_str = f"{result['objective_value']:.6f}" if result.get('objective_value') is not None else "N/A"
                print(
                    f"   {i}. {algo_name}: 目标值={obj_str}, 质量={quality_str}, 时间={time_str}"
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

            print(f"\n在达到99%求解质量的前提下:")
            print(f"   BSB最快: {fastest_bsb[0]} ({fastest_bsb[1]['execution_time_ms']:.2f}ms)")
            print(f"   DSB最快: {fastest_dsb[0]} ({fastest_dsb[1]['execution_time_ms']:.2f}ms)")

            if fastest_bsb[1]["execution_time_ms"] < fastest_dsb[1]["execution_time_ms"]:
                print(
                    f"   ✅ BSB在速度上领先 {fastest_dsb[1]['execution_time_ms'] - fastest_bsb[1]['execution_time_ms']:.2f}ms"
                )
            else:
                print(
                    f"   ✅ DSB在速度上领先 {fastest_bsb[1]['execution_time_ms'] - fastest_dsb[1]['execution_time_ms']:.2f}ms"
                )
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
            import os
            from pathlib import Path

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
                save_data["algorithms"][algorithm_name] = {
                    "objective_value": result.get("objective_value"),
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
            from pathlib import Path

            csv_save_dir = Path("save_csv")
            csv_save_dir.mkdir(exist_ok=True)
            output_file = csv_save_dir / f"qplib_results_{self.problem_name}.csv"

        # 获取精确最优解
        clean_name = self.problem_name.strip().strip("'").strip('"')
        problem_row = self.problem_db[self.problem_db["instance_id"] == clean_name]
        if not problem_row.empty:
            exact_optimal = self._parse_exact_solution(problem_row.iloc[0]["solution_objective_value"])
        else:
            exact_optimal = None

        # 准备CSV数据
        csv_rows = []

        # 添加每个算法的结果（每个结果行包含完整信息）
        for algorithm_name, result in results.items():
            if result and result.get("parsing_success", False):
                # 解析算法信息
                algorithm_type = "BSB" if algorithm_name.startswith("BSB_") else "DSB"
                device = "CPU" if "CPU" in algorithm_name else "GPU"
                precision = algorithm_name.split("_")[-1]

                objective_value = result.get("objective_value")
                energy_value = objective_value  # 这里目标值和能量值相同

                # 计算与精确最优解的比率
                energy_to_optimal_ratio = None
                if energy_value is not None and exact_optimal is not None and exact_optimal != 0:
                    energy_to_optimal_ratio = energy_value / exact_optimal

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
            df = pd.DataFrame(csv_rows)
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\n💾 CSV结果已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 保存CSV文件失败: {e}")


def run_single_problem(
    problem_id=None, device="cuda", xi=0, algorithms=None, use_optuna_params=False, optuna_csv_path=None
):
    """
    运行单个问题的指定算法

    参数：
        problem_id (str, optional): QPLIB问题ID
        device (str): 计算设备
        xi (float): xi参数，默认为0
        algorithms (list, optional): 指定要执行的算法列表。如果为None，执行所有12种算法
        use_optuna_params (bool, optional): 是否使用Optuna优化的参数，默认为False
        optuna_csv_path (str, optional): Optuna优化结果CSV文件路径

    返回：
        dict: 算法结果
    """
    # 创建求解器实例（使用固定的xi值）
    if problem_id:
        ins = QBExternal(
            problem_id=problem_id,
            device=device,
            fixed_xi=xi,
            use_optuna_params=use_optuna_params,
            optuna_csv_path=optuna_csv_path,
        )
    else:
        ins = QBExternal(
            device=device, fixed_xi=xi, use_optuna_params=use_optuna_params, optuna_csv_path=optuna_csv_path
        )

    print(f"\n📋 问题信息:")
    print("=" * 60)
    print(f"问题名称: {ins.problem_name}")
    print(f"变量数量: {ins.num_vars}")
    print(f"约束数量: {ins.num_constraints}")
    print(f"优化方向: {ins.objective_sense}")
    print(f"缩放因子: {ins.scaling_factor:.6f}")
    print(f"ξ参数: {ins.xi:.6f} (固定值)")

    # 求解优化问题（使用指定的算法）
    if algorithms:
        print(f"\n🚀 开始调用指定的算法接口...")
    else:
        print(f"\n🚀 开始调用算法接口（BSB + DSB）...")
    results = ins.solve_all_variants(selected_algorithms=algorithms)

    print(f"\n🎉 求解完成！")

    # 保存详细结果到JSON文件
    ins.save_results_to_json(results)

    # 保存结果到CSV文件
    ins.save_results_to_csv(results)

    return results


def run_batch_problems(
    problem_ids=None, device="cuda", xi=0, algorithms=None, use_optuna_params=False, optuna_csv_path=None
):
    """
    批量运行多个问题

    参数：
        problem_ids (list, optional): 问题ID列表。如果为None，从数据库选择
        device (str): 计算设备
        xi (float): xi参数，默认为0
        algorithms (list, optional): 指定要执行的算法列表。如果为None，执行所有12种算法
        use_optuna_params (bool, optional): 是否使用Optuna优化的参数，默认为False
        optuna_csv_path (str, optional): Optuna优化结果CSV文件路径

    返回：
        dict: 所有问题的结果
    """
    # 加载问题数据库
    db_path = "data/qplib_results.csv"
    if os.path.exists(db_path):
        problem_db = pd.read_csv(db_path)
    else:
        print(f"❌ 问题数据库不存在: {db_path}")
        return {}

    # 确定要处理的问题列表
    if problem_ids is None:
        # 没有指定，使用第一条
        problem_ids = [problem_db.iloc[0]["instance_id"]]
        print(f"未指定问题，使用数据库第一条: {problem_ids[0]}")

    all_results = {}

    for i, problem_id in enumerate(problem_ids, 1):
        print(f"\n{'='*80}")
        print(f"处理问题 {i}/{len(problem_ids)}: {problem_id}")
        print(f"{'='*80}")

        try:
            # 创建求解器实例（使用固定的xi值）
            ins = QBExternal(
                problem_id=problem_id,
                device=device,
                fixed_xi=xi,
                use_optuna_params=use_optuna_params,
                optuna_csv_path=optuna_csv_path,
            )

            print(f"\n📋 问题信息:")
            print(f"  问题名称: {ins.problem_name}")
            print(f"  变量数量: {ins.num_vars}")
            print(f"  优化方向: {ins.objective_sense}")
            print(f"  ξ参数: {ins.xi:.6f} (固定值)")

            # 求解
            print(f"\n🚀 开始调用算法接口...")
            results = ins.solve_all_variants(selected_algorithms=algorithms)

            all_results[problem_id] = results

            # 保存当前问题的CSV结果
            ins.save_results_to_csv(results)

            print(f"\n✅ 问题 {problem_id} 求解完成")

        except Exception as e:
            print(f"\n❌ 问题 {problem_id} 求解失败: {e}")
            import traceback

            traceback.print_exc()
            all_results[problem_id] = None

    print(f"\n{'='*80}")
    print(f"🎉 所有问题求解完成！")
    print(f"{'='*80}")

    # 保存所有问题的汇总CSV
    save_batch_csv(all_results)

    return all_results


def save_batch_csv(all_results):
    """
    保存所有问题的汇总结果到CSV

    参数：
        all_results (dict): 包含所有问题结果的字典
    """
    # 加载问题数据库
    db_path = "data/qplib_results.csv"
    if os.path.exists(db_path):
        problem_db = pd.read_csv(db_path)
    else:
        problem_db = pd.DataFrame()

    csv_rows = []

    for problem_id, results in all_results.items():
        if results is None:
            continue

        # 获取精确最优解
        exact_optimal = None
        if not problem_db.empty:
            problem_row = problem_db[problem_db["instance_id"] == problem_id]
            if not problem_row.empty:
                solution_str = problem_row.iloc[0]["solution_objective_value"]
                if pd.notna(solution_str) and solution_str != "":
                    match = re.search(r"([+-]?\d*\.?\d+)", str(solution_str))
                    if match:
                        exact_optimal = float(match.group(1))

        for algorithm_name, result in results.items():
            if result and result.get("parsing_success", False):
                objective_value = result.get("objective_value", 0)
                energy_value = objective_value  # 目标值和能量值相同

                # 计算与精确最优解的比率
                energy_to_optimal_ratio = None
                if exact_optimal is not None and exact_optimal != 0:
                    energy_to_optimal_ratio = energy_value / exact_optimal

                # 优先使用C++报告的执行时间
                execution_time = result.get("cpp_execution_time_ms", result.get("execution_time_ms", ""))

                row = {
                    "problem_id": problem_id,
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
        from pathlib import Path

        csv_save_dir = Path("save_csv")
        csv_save_dir.mkdir(exist_ok=True)
        output_file = csv_save_dir / "batch_summary.csv"

        df = pd.DataFrame(csv_rows)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n💾 批量汇总结果已保存到: {output_file}")
    except Exception as e:
        print(f"❌ 保存批量CSV文件失败: {e}")


if __name__ == "__main__":

    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA设备可用：{torch.cuda.get_device_name(0)}")
        print(f"CUDA版本：{torch.version.cuda}")
        device = "cuda"
    else:
        print("CUDA不可用，使用CPU")
        device = "cpu"

    # xi参数（固定为0）
    XI_VALUE = 0

    # 使用说明
    print("\n" + "=" * 80)
    print("使用方式:")
    print("  python SB_basic_run.py                    # 默认：执行数据库第一条问题，所有算法")
    print("  python SB_basic_run.py --problem QPLIB_3565  # 执行指定问题，所有算法")
    print("  python SB_basic_run.py --list              # 列出可用问题")
    print("  python SB_basic_run.py --list-algo         # 列出可用算法")
    print("  python SB_basic_run.py --algo BSB          # 运行所有BSB算法")
    print("  python SB_basic_run.py --algo DSB          # 运行所有DSB算法")
    print("  python SB_basic_run.py --algo BSB_CPU_fp16 # 运行特定算法")
    print("  python SB_basic_run.py --algo BSB_CPU_fp16,DSB_GPU_fp16  # 运行多个算法")
    print('  python SB_basic_run.py --batch "QPLIB_3565,QPLIB_3642" --algo BSB  # 批量执行')
    print("  python SB_basic_run.py --xi 0.5           # 指定xi值（默认为0）")
    print("  python SB_basic_run.py --use-optuna        # 使用Optuna优化的参数")
    print("  python SB_basic_run.py --use-optuna --optuna-csv path/to/file.csv  # 指定Optuna结果文件")
    print("=" * 80)

    # 解析命令行参数
    import sys

    args = sys.argv[1:]

    xi = XI_VALUE
    selected_algorithms = None
    use_optuna_params = False
    optuna_csv_path = None

    # 解析 --use-optuna 参数
    if "--use-optuna" in args:
        use_optuna_params = True
        print(f"\n✅ 启用Optuna优化参数模式")

    # 解析 --optuna-csv 参数
    if "--optuna-csv" in args:
        try:
            idx = args.index("--optuna-csv")
            if idx + 1 < len(args):
                optuna_csv_path = args[idx + 1]
                if not os.path.exists(optuna_csv_path):
                    print(f"❌ 错误: Optuna CSV文件不存在: {optuna_csv_path}")
                    print("   如果不指定，将自动查找最新的batch_optuna_12algs_*.csv文件")
                    optuna_csv_path = None
                else:
                    print(f"\n✅ 使用指定的Optuna结果文件: {optuna_csv_path}")
            else:
                print("❌ 错误: --optuna-csv 参数需要文件路径")
        except Exception as e:
            print(f"❌ Optuna CSV文件路径解析错误: {e}")

    # 解析 --xi 参数
    if "--xi" in args:
        try:
            idx = args.index("--xi")
            if idx + 1 < len(args):
                xi = float(args[idx + 1])
                print(f"\n✅ 使用指定的xi值: {xi}")
            else:
                print("❌ 错误: --xi 参数需要数值")
        except ValueError as e:
            print(f"❌ xi值解析错误: {e}")

    # 解析 --algo 参数
    if "--algo" in args:
        try:
            idx = args.index("--algo")
            if idx + 1 < len(args):
                algo_str = args[idx + 1].strip()
                # 处理快捷选择
                if algo_str.upper() == "BSB":
                    selected_algorithms = [
                        "BSB_CPU_fp16",
                        "BSB_CPU_int8",
                        "BSB_CPU_fp8",
                        "BSB_GPU_fp16",
                        "BSB_GPU_int8",
                        "BSB_GPU_fp8",
                    ]
                    print(f"\n✅ 选择所有BSB算法（6种）")
                elif algo_str.upper() == "DSB":
                    selected_algorithms = [
                        "DSB_CPU_fp16",
                        "DSB_CPU_int8",
                        "DSB_CPU_fp8",
                        "DSB_GPU_fp16",
                        "DSB_GPU_int8",
                        "DSB_GPU_fp8",
                    ]
                    print(f"\n✅ 选择所有DSB算法（6种）")
                else:
                    # 处理多个算法（逗号分隔）
                    selected_algorithms = [a.strip() for a in algo_str.split(",")]
                    print(f"\n✅ 选择指定算法: {', '.join(selected_algorithms)}")
            else:
                print("❌ 错误: --algo 参数需要算法名称或列表")
        except Exception as e:
            print(f"❌ 算法参数解析错误: {e}")

    # 处理不同的命令
    if not args or "--help" in args:
        # 默认：执行第一条问题，所有算法
        print("\n未指定参数，默认执行数据库第一条问题，运行所有算法...")
        run_single_problem(
            device=device,
            xi=xi,
            algorithms=selected_algorithms,
            use_optuna_params=use_optuna_params,
            optuna_csv_path=optuna_csv_path,
        )
    elif "--list-algo" in args:
        # 列出可用算法
        QBExternal.list_available_algorithms()
    elif "--list" in args:
        # 列出可用问题
        temp_ins = QBExternal(
            device=device, fixed_xi=xi, use_optuna_params=use_optuna_params, optuna_csv_path=optuna_csv_path
        )
        temp_ins.list_available_problems(limit=20)
    elif "--problem" in args:
        # 执行指定问题
        try:
            idx = args.index("--problem")
            if idx + 1 < len(args):
                problem_id = args[idx + 1]
                algo_info = "选定算法" if selected_algorithms else "所有算法"
                optuna_info = "Optuna参数" if use_optuna_params else "默认参数"
                print(f"\n执行指定问题: {problem_id}, 使用{algo_info}, {optuna_info}")
                run_single_problem(
                    problem_id=problem_id,
                    device=device,
                    xi=xi,
                    algorithms=selected_algorithms,
                    use_optuna_params=use_optuna_params,
                    optuna_csv_path=optuna_csv_path,
                )
            else:
                print("❌ 错误: --problem 参数需要问题ID")
                print("   例如: python SB_basic_run.py --problem QPLIB_3565")
        except ValueError as e:
            print(f"❌ 参数错误: {e}")
    elif "--batch" in args:
        # 批量执行
        try:
            idx = args.index("--batch")
            if idx + 1 < len(args):
                problem_ids_str = args[idx + 1]
                problem_ids = [pid.strip() for pid in problem_ids_str.split(",")]
                algo_info = "选定算法" if selected_algorithms else "所有算法"
                optuna_info = "Optuna参数" if use_optuna_params else "默认参数"
                print(f"\n批量执行问题: {problem_ids}, 使用{algo_info}, {optuna_info}")
                run_batch_problems(
                    problem_ids=problem_ids,
                    device=device,
                    xi=xi,
                    algorithms=selected_algorithms,
                    use_optuna_params=use_optuna_params,
                    optuna_csv_path=optuna_csv_path,
                )
            else:
                print("❌ 错误: --batch 参数需要问题ID列表")
                print('   例如: python SB_basic_run.py --batch "QPLIB_3565,QPLIB_3642"')
        except Exception as e:
            print(f"❌ 批量执行错误: {e}")
    else:
        # 默认：执行第一条问题
        algo_info = "选定算法" if selected_algorithms else "所有算法"
        optuna_info = "Optuna参数" if use_optuna_params else "默认参数"
        print(f"\n使用默认设置，执行数据库第一条问题，使用{algo_info}, {optuna_info}...")
        run_single_problem(
            device=device,
            xi=xi,
            algorithms=selected_algorithms,
            use_optuna_params=use_optuna_params,
            optuna_csv_path=optuna_csv_path,
        )
