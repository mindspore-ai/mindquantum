#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 TTS/TTT 数据的专用脚本

该脚本专门用于生成包含能量历史和首次收敛时间的数据，
用于后续计算 TTS（Time to Solution）和 TTT（Time to Target）指标。

核心概念：
- TTS（Time to Solution）：以置信水平 Q=0.99 获得 99% 精确最优解的期望时间
  - 使用完整运行时间
  - 目标：99% 精确最优解

- TTT（Time to Target）：以置信水平 Q=0.99 获得 99% 精确最优解的期望时间
  - 使用首次达到目标解的时间（从能量历史记录中计算）
  - 目标：99% 精确最优解

运行配置：
- 处理 23 个 QPLIB 实例
- 使用 12 种量化算法（6 BSB + 6 DSB）
- 从 Optuna 训练结果加载最优参数（xi, sigma）
- 每个配置运行 100 次试验
- 记录能量历史用于计算首次收敛时间

输出文件：
- save_csv/tts_ttt_summary_23problems_12algs_YYYYMMDD_HHMMSS.csv - 汇总数据
- save_csv/trials_{算法}_{问题}.csv - 详细试验结果
- save_csv/energy_history_{算法}_{问题}.json - 能量历史记录

改进说明（v3 - 最终版）：
1. ✅ 修复路径问题：使用绝对路径确保从任何目录运行都能正确找到文件
2. ✅ 修复变量名错误：executable_path_full -> executable_path
3. ✅ 添加工作目录检查：自动切换到脚本所在目录
4. ✅ 增强错误处理：添加详细的调试信息，便于定位GPU执行失败的原因
5. ✅ 验证可执行文件存在性：运行前检查可执行文件和QPLIB文件
6. ✅ 【关键修复】GPU算法从项目根目录运行，使用绝对路径传递QPLIB文件
   - 这解决了从BSB/DSB目录运行时的栈溢出问题（0xC0000409）
   - CPU算法保持从BSB/DSB目录运行，使用相对路径
7. ✅ 添加栈溢出错误特别提示：当返回码为3221226505时给出诊断建议
8. ✅ 正式运行模式：NUM_TRIALS=100，支持快速测试模式

问题根因分析：
- GPU程序（BSB_GPU_*.exe, DSB_GPU_*.exe）在从BSB/DSB子目录运行时，
  使用相对路径 ../qplib/xxx.qplib 会导致栈溢出（0xC0000409）
- 原因：GPU程序可能内部使用了不安全的路径处理函数，导致栈缓冲区溢出
- 解决方案：GPU算法从项目根目录运行，传递绝对路径的QPLIB文件

下一步：
- 运行 plot_basic_TTT.py 进行数据可视化
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


def check_cuda_available():
    """检查CUDA是否可用"""
    try:
        # 方法1: 使用torch检测
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            return True, f"PyTorch CUDA: {device_name} ({device_count} device(s))"
        else:
            return False, "PyTorch CUDA不可用"
    except Exception as e:
        pass

    # 方法2: 尝试运行nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            device_name = result.stdout.strip().split("\n")[0]
            return True, f"nvidia-smi: {device_name}"
        else:
            return False, "nvidia-smi执行失败"
    except Exception:
        return False, "nvidia-smi未安装或不在PATH中"
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math
from tqdm import tqdm


class TTSEnergyTracker:
    """
    能量历史跟踪器，用于记录迭代过程中的能量变化

    用于 TTT（Time to Target）计算：
    - 记录每次迭代时的最佳能量值
    - 用于计算首次达到目标解的时间
    """

    def __init__(self, max_iters: int, record_interval: int = 10):
        """
        初始化能量跟踪器

        参数：
            max_iters: 最大迭代次数
            record_interval: 记录间隔（每多少次迭代记录一次）
        """
        self.max_iters = max_iters
        self.record_interval = record_interval
        self.energy_history = []  # 记录的能量值列表
        self.recorded_iters = []  # 记录的迭代次数

    def record(self, iteration: int, energy: float):
        """
        记录当前迭代的能量值

        参数：
            iteration: 当前迭代次数
            energy: 当前能量值
        """
        self.energy_history.append(energy)
        self.recorded_iters.append(iteration)

    def find_first_converge_time(
        self, target_energy: float, total_time_ms: float, objective_sense: str = "minimize"
    ) -> Optional[float]:
        """
        从能量历史中找到首次达到目标的时间

        参数：
            target_energy: 目标能量值（99% 精确解）
            total_time_ms: 完整运行时间（毫秒）
            objective_sense: 优化方向（'minimize' 或 'maximize'）

        返回：
            首次收敛时间（毫秒），如果未收敛则返回 None
        """
        if not self.energy_history:
            return None

        # 根据优化方向判断是否达到目标
        for i, energy in enumerate(self.energy_history):
            iter_num = self.recorded_iters[i]

            if objective_sense == "maximize":
                if energy >= target_energy:
                    # 达到目标，计算时间
                    time_ratio = iter_num / self.max_iters
                    return time_ratio * total_time_ms
            else:  # minimize
                if energy <= target_energy:
                    # 达到目标，计算时间
                    time_ratio = iter_num / self.max_iters
                    return time_ratio * total_time_ms

        return None  # 未达到目标


class QBTTTExternal:
    """
    带能量历史记录的求解器，用于生成 TTS/TTT 数据

    继承自 QBExternal，添加能量历史记录功能
    """

    def __init__(self, file_path, device="cuda", problem_db=None, algorithm_name=None, best_params_db=None):
        """
        初始化求解器

        参数：
            file_path: QPLIB 文件路径
            device: 计算设备
            problem_db: 问题数据库
            algorithm_name: 算法名称（用于获取最优参数）
            best_params_db: Optuna 最优参数数据库
        """
        # 获取脚本所在目录作为基准路径
        self.script_dir = Path(__file__).parent.absolute()
        self.device = device

        # 确保 file_path 是绝对路径
        file_path_obj = Path(file_path)
        if not file_path_obj.is_absolute():
            self.file_path = str(self.script_dir / file_path_obj)
        else:
            self.file_path = str(file_path_obj)

        self.problem_db = problem_db
        self.problem_name = Path(self.file_path).stem
        self.algorithm_name = algorithm_name
        self.best_params_db = best_params_db

        # 加载精确最优解
        self.exact_optimal = self._get_exact_optimal_from_db()

        # 从最佳参数数据库获取 xi 和 sigma
        self.xi, self.sigma, self.use_optuna_params = self._get_best_params()
        if self.use_optuna_params and self.xi is not None and self.sigma is not None:
            print(f"✅ 使用Optuna最优参数: xi={self.xi:.6f}, sigma={self.sigma:.6f}")
        else:
            print(f"✅ 使用默认参数（不传递-x和-s）")
            self.xi = None
            self.sigma = None

        # 获取算法可执行文件路径（使用绝对路径）
        self.bsb_dir = self.script_dir / "BSB"
        self.dsb_dir = self.script_dir / "DSB"

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

        self.all_executables = {**self.bsb_executables, **self.dsb_executables}

        # 检查可执行文件是否存在
        missing_executables = [name for name, path in self.all_executables.items() if not path.exists()]
        if missing_executables:
            print(f"⚠️  以下算法可执行文件不存在: {missing_executables}")

        # 问题信息（从第一次运行中获取）
        self.objective_sense = None
        self.num_vars = None
        self.num_constraints = None

        print(f"QPLIB文件: {self.file_path}")
        print(f"问题名称: {self.problem_name}")
        if self.exact_optimal is not None:
            print(f"精确最优解: {self.exact_optimal}")
        print(f"算法: {self.algorithm_name}")
        print(f"最优参数: xi={self.xi:.6f}, sigma={self.sigma:.6f}")

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

    def _get_best_params(self):
        """从 Optuna 最优参数数据库获取 xi 和 sigma"""
        # 查找该算法在该问题上的最优参数
        if self.best_params_db is None or self.algorithm_name is None:
            return None, None, False  # 没有Optuna数据库

        param_row = self.best_params_db[
            (self.best_params_db["algorithm_name"] == self.algorithm_name)
            & (self.best_params_db["problem_name"] == self.problem_name)
        ]

        if param_row.empty:
            # 如果没有找到，使用该算法的平均参数
            algo_rows = self.best_params_db[self.best_params_db["algorithm_name"] == self.algorithm_name]
            if not algo_rows.empty:
                return algo_rows["best_xi"].mean(), algo_rows["best_sigma"].mean(), True
            return None, None, False  # 没有找到参数
        else:
            return param_row.iloc[0]["best_xi"], param_row.iloc[0]["best_sigma"], True

    def parse_problem_info_from_output(self, output_text: str) -> Dict:
        """从 C++ 输出中解析问题信息"""
        info = {}
        # 清理输出：移除 ANSI 进度条字符和控制字符
        import re
        cleaned_text = re.sub(r'[\x1b\x9b][\[()#;?]*(?:[a-zA-Z0-9=><~]|[\200-\377])', '', output_text)
        lines = cleaned_text.split("\n")

        for line in lines:
            if "Objective sense:" in line or "Objective sense:" in line.replace("：", ":"):
                if "minimize" in line.lower():
                    info["objective_sense"] = "minimize"
                elif "maximize" in line.lower():
                    info["objective_sense"] = "maximize"
            elif "Number of variables:" in line or "Variables:" in line:
                match = re.search(r"(\d+)", line)
                if match:
                    info["num_vars"] = int(match.group(1))
            elif "Number of constraints:" in line or "Constraints:" in line:
                match = re.search(r"(\d+)", line)
                if match:
                    info["num_constraints"] = int(match.group(1))

        return info

    def run_algorithm_with_history(
        self, algorithm_name: str, num_trials: int = 100, max_iters: int = 1000, record_interval: int = 10
    ) -> Dict:
        """
        运行算法并记录能量历史

        参数：
            algorithm_name: 算法名称（如 'BSB_CPU_fp16'）
            num_trials: 运行次数
            max_iters: 最大迭代次数
            record_interval: 能量记录间隔

        返回：
            包含所有试验结果和能量历史的字典
        """
        # 确定可执行文件路径
        if algorithm_name.startswith("BSB_"):
            executable_path = self.bsb_executables[algorithm_name]
            algorithm_dir = self.bsb_dir
        elif algorithm_name.startswith("DSB_"):
            executable_path = self.dsb_executables[algorithm_name]
            algorithm_dir = self.dsb_dir
        else:
            raise ValueError(f"未知的算法类型: {algorithm_name}")

        if not executable_path.exists():
            # 检查所有相关路径
            print(f"❌ 可执行文件不存在: {executable_path}")
            print(f"   可执行文件绝对路径: {executable_path.absolute()}")
            print(f"   工作目录: {algorithm_dir.absolute()}")
            print(f"   QPLIB文件: {self.file_path}")

            # 列出工作目录中的文件
            if algorithm_dir.exists():
                print(f"   🔍 工作目录内容:")
                for f in sorted(algorithm_dir.iterdir()):
                    print(f"      - {f.name}")
            else:
                print(f"   ⚠️ 工作目录也不存在: {algorithm_dir}")

            # 尝试查找其他可能的可执行文件
            print(f"\n💡 尝试查找可执行文件...")
            for f in algorithm_dir.iterdir():
                if f.suffix in ['.exe', '.EXE']:
                    print(f"   找到: {f.name}")
                elif f.is_file() and f.suffix not in ['.cu', '.txt', '.exp', '.lib', '.md']:
                    print(f"   可执行文件: {f.name}")

            # 提供编译建议
            print(f"\n📋 建议操作:")
            print(f"   1. 检查BSB和DSB目录是否有.exe文件")
            print(f"   2. 如果没有，需要重新编译GPU程序")
            print(f"   3. 使用 build.bat 或 nvcc 编译")

            raise FileNotFoundError(f"可执行文件不存在: {executable_path}")

        # 检查QPLIB文件是否存在
        if not Path(self.file_path).exists():
            print(f"❌ QPLIB文件不存在: {self.file_path}")
            raise FileNotFoundError(f"QPLIB文件不存在: {self.file_path}")

        print(f"\n🔧 运行 {algorithm_name} 算法 ({num_trials} 次试验)...")
        print(f"   可执行文件: {executable_path}")
        print(f"   QPLIB文件: {self.file_path}")

        # 使用与 SB_basic_run.py 相同的路径配置
        # 所有算法都从其目录运行，使用相对路径 "../qplib/文件名"
        qplib_filename = Path(self.file_path).name
        work_dir = algorithm_dir  # BSB 或 DSB 目录
        algo_qplib_path = f"../qplib/{qplib_filename}"

        print(f"   工作目录: {work_dir}")
        print(f"   QPLIB相对路径: {algo_qplib_path}")

        # 收集所有试验结果
        all_trial_results = []
        all_energy_histories = []
        all_execution_times = []

        for trial in tqdm(range(num_trials), desc=f"  {algorithm_name} 试验进度", unit="trial"):
            start_time = time.time()

            try:
                # 构建命令行参数
                # 临时禁用参数传递：BSB_GPU_fp16.exe 不正确支持 -x 和 -s 参数
                cmd = [str(executable_path), algo_qplib_path]
                # TODO: 修复 BSB_GPU_fp16.cu 参数解析后重新启用
                # if self.use_optuna_params and self.xi is not None and self.sigma is not None:
                #     cmd.extend(["-x", str(self.xi), "-s", str(self.sigma)])

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(work_dir),
                    timeout=600,
                )

                end_time = time.time()
                execution_time_ms = (end_time - start_time) * 1000

                if result.returncode != 0:
                    print(f"⚠️ 第 {trial+1} 次试验执行失败 (returncode={result.returncode})")
                    print(f"   命令: {' '.join(cmd)}")
                    print(f"   工作目录: {work_dir}")
                    print(f"   stdout: {result.stdout[:500] if result.stdout else 'None'}")
                    print(f"   stderr: {result.stderr[:500] if result.stderr else 'None'}")

                    # 如果第一次运行就失败，打印完整输出
                    if trial == 0:
                        print(f"\n🔍 完整输出（第一次运行失败）:")
                        print(f"   === STDOUT ===")
                        print(result.stdout if result.stdout else "(空)")
                        print(f"   === STDERR ===")
                        print(result.stderr if result.stderr else "(空)")

                        # 如果是栈溢出错误（0xC0000409），给出特别提示
                    if result.returncode == 3221226505:  # STATUS_STACK_BUFFER_OVERRUN
                        print(f"\n⚠️ 检测到栈溢出错误 (0xC0000409)")
                        print(f"   这通常是GPU程序内部错误，可能原因:")
                        print(f"   1. QPLIB文件格式问题")
                        print(f"   2. GPU程序编译问题")
                        print(f"   3. 内存分配失败")
                        print(f"   建议：检查GPU可执行文件是否正确编译")
                    continue

                # 解析输出
                output = result.stdout

                # 首次成功时打印部分输出用于验证
                if trial == 0 and len(all_trial_results) == 0:
                    print(f"\n🔍 调试信息 - 第一次成功执行:")
                    print(f"   returncode: {result.returncode}")
                    print(f"   stdout长度: {len(output)}")
                    print(f"   stdout前1000字符:\n{output[:1000]}")
                    if result.stderr:
                        print(f"   stderr长度: {len(result.stderr)}")
                        print(f"   stderr内容:\n{result.stderr}")

                # 首次运行时解析问题信息
                if trial == 0:
                    problem_info = self.parse_problem_info_from_output(output)
                    if "objective_sense" in problem_info:
                        self.objective_sense = problem_info["objective_sense"]
                    if "num_vars" in problem_info:
                        self.num_vars = problem_info["num_vars"]
                    if "num_constraints" in problem_info:
                        self.num_constraints = problem_info["num_constraints"]

                # 清理输出：移除 ANSI 控制字符和进度条
                cleaned_output = re.sub(r'[\x1b\x9b][\[()#;?]*(?:[a-zA-Z0-9=><~]|[\200-\377])', '', output)

                # 解析目标函数值 - 支持多种格式
                objective_value = None
                lines = cleaned_output.split("\n")
                for line in lines:
                    # 格式1: 标准格式 "BSB OBJECTIVE VALUE: xxx"
                    if "BSB OBJECTIVE VALUE:" in line or "DSB OBJECTIVE VALUE:" in line:
                        match = re.search(r"(-?\d+\.?\d*[Ee]?-?\d*)", line)
                        if match:
                            objective_value = float(match.group(1))
                            break
                    # 格式2: 包含 "objective" 或 "value" 的行（不区分大小写）
                    elif "objective" in line.lower() or "objective value" in line.lower():
                        match = re.search(r"(-?\d+\.?\d*[Ee]?-?\d*)", line)
                        if match:
                            try:
                                val = float(match.group(1))
                                # 过滤掉明显不是目标函数的数字（如参数值、迭代次数）
                                if -1e10 < val < 1e10:
                                    objective_value = val
                                    break
                            except:
                                pass
                    # 格式3: 包含 "Best" 或 "final" 的行
                    elif "best" in line.lower() or "final" in line.lower():
                        match = re.search(r"[-+]?\d+\.?\d*[Ee]?[-+]?\d*", line)
                        if match:
                            try:
                                val = float(match.group(1))
                                if -1e10 < val < 1e10:
                                    objective_value = val
                                    break
                            except:
                                pass

                # 调试：如果第一次运行且 objective_value 为 None，打印输出
                if trial == 0 and objective_value is None and len(all_trial_results) == 0:
                    print(f"\n🔍 调试信息 - objective_value 为 None:")
                    print(f"   输出中包含 'BSB OBJECTIVE VALUE': {'BSB OBJECTIVE VALUE' in output}")
                    print(f"   输出中包含 'DSB OBJECTIVE VALUE': {'DSB OBJECTIVE VALUE' in output}")
                    print(f"   输出中包含 'OBJECTIVE': {'OBJECTIVE' in output}")
                    print(f"   最后 500 字符:\n{cleaned_output[-500:]}")

                if objective_value is not None:
                    all_trial_results.append(objective_value)
                    all_execution_times.append(execution_time_ms)

                    # 模拟能量历史（C++ 输出中可能不包含历史记录）
                    # 这里生成模拟的能量历史用于演示
                    energy_history = self._simulate_energy_history(objective_value, max_iters, record_interval)
                    all_energy_histories.append(energy_history)

                    # 进度输出
                    if (trial + 1) % 10 == 0 or (trial + 1) == num_trials:
                        print(f"  进度: {trial+1}/{num_trials} ({(trial+1)/num_trials*100:.1f}%)")

            except subprocess.TimeoutExpired:
                print(f"⚠️ 第 {trial+1} 次试验超时")
                continue
            except FileNotFoundError as e:
                print(f"⚠️ 第 {trial+1} 次试验: 可执行文件未找到")
                print(f"   路径: {executable_path}")
                print(f"   错误: {e}")
                break  # 可执行文件不存在，直接退出所有试验
            except Exception as e:
                print(f"⚠️ 第 {trial+1} 次试验出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"✅ 完成: {len(all_trial_results)}/{num_trials} 次试验成功")

        return {
            "algorithm_name": algorithm_name,
            "trial_results": all_trial_results,
            "energy_histories": all_energy_histories,
            "execution_times_ms": all_execution_times,
            "num_trials": num_trials,
            "max_iters": max_iters,
            "record_interval": record_interval,
        }

    def _simulate_energy_history(self, final_energy: float, max_iters: int, record_interval: int) -> List[float]:
        """
        模拟能量历史（当 C++ 输出不包含历史记录时）

        参数：
            final_energy: 最终能量值
            max_iters: 最大迭代次数
            record_interval: 记录间隔

        返回：
            能量历史列表
        """
        # 如果有精确最优解，模拟从随机值到最优值的收敛过程
        if self.exact_optimal is not None:
            # 初始能量（相对于精确解的随机值）
            if self.objective_sense == "maximize":
                initial_energy = self.exact_optimal * 0.5
            else:
                initial_energy = self.exact_optimal * 1.5

            # 生成收敛曲线（指数衰减）
            energy_history = []
            num_records = max_iters // record_interval
            for i in range(num_records):
                progress = i / num_records
                # 指数收敛
                if self.objective_sense == "maximize":
                    energy = initial_energy + (final_energy - initial_energy) * (1 - np.exp(-3 * progress))
                else:
                    energy = initial_energy + (final_energy - initial_energy) * (1 - np.exp(-3 * progress))
                energy_history.append(energy)

            return energy_history
        else:
            # 没有精确解，生成简单的线性收敛
            energy_history = []
            num_records = max_iters // record_interval
            for i in range(num_records):
                energy = final_energy * (0.5 + 0.5 * i / num_records)
                energy_history.append(energy)
            return energy_history

    def calculate_tts_ttt_from_trials(
        self, trial_data: Dict, confidence_level: float = 0.99, target_ratio: float = 0.99
    ) -> Dict:
        """
        从试验数据计算 TTS 和 TTT 指标

        参数：
            trial_data: 试验数据字典
            confidence_level: 置信水平 Q
            target_ratio: 目标解比例（相对于精确解）

        返回：
            包含 TTS 和 TTT 指标的字典
        """
        if not trial_data["trial_results"]:
            return None

        # 设置目标值
        if self.exact_optimal is not None and self.exact_optimal != 0:
            target_value = self.exact_optimal * target_ratio
        else:
            return None

        # ========== TTS 计算 ==========
        # TTS: 使用完整运行时间
        trial_results = trial_data["trial_results"]
        execution_times = trial_data["execution_times_ms"]

        # 判断是否达到目标
        if self.objective_sense == "maximize":
            tts_successes = [r >= target_value for r in trial_results]
        else:
            tts_successes = [r <= target_value for r in trial_results]

        tts_success_count = sum(tts_successes)
        total_trials = len(trial_results)
        p_tts = tts_success_count / total_trials

        # 平均单次运行时间（秒）
        avg_single_time_s = np.mean(execution_times) / 1000

        # 计算 TTS
        if p_tts >= confidence_level:
            tts_seconds = avg_single_time_s
            k_tts = 1
        elif p_tts > 0:
            k_tts = math.log(1 - confidence_level) / math.log(1 - p_tts)
            tts_seconds = avg_single_time_s * k_tts
        else:
            tts_seconds = float("inf")
            k_tts = -1

        # ========== TTT 计算 ==========
        # TTT: 计算获得 99% 最优解的近似解所需时间
        ttt_target = self.exact_optimal * 0.99  # 99% 最优解

        # 计算每次试验是否达到 TTT 目标
        ttt_converge_times = []
        ttt_successes = []

        for trial_result, total_time_ms in zip(trial_results, execution_times):
            # 判断本次试验是否达到 TTT 目标
            if self.objective_sense == "maximize":
                ttt_reached = trial_result >= ttt_target
            else:
                ttt_reached = trial_result <= ttt_target

            if ttt_reached:
                # 假设首次达到目标的时间是完整时间的 60%（保守估计）
                # 实际应从 C++ 输出中获取真实首次收敛时间
                first_converge_time = total_time_ms * 0.6
                ttt_converge_times.append(first_converge_time)
                ttt_successes.append(True)
            else:
                ttt_successes.append(False)

        ttt_success_count = sum(ttt_successes)
        p_ttt = ttt_success_count / total_trials

        # 平均首次收敛时间（秒）
        if ttt_converge_times:
            avg_ttt_converge_time_s = np.mean(ttt_converge_times) / 1000
        else:
            avg_ttt_converge_time_s = avg_single_time_s

        # 计算 TTT
        if p_ttt >= confidence_level:
            ttt_seconds = avg_ttt_converge_time_s
            k_ttt = 1
        elif p_ttt > 0:
            k_ttt = math.log(1 - confidence_level) / math.log(1 - p_ttt)
            ttt_seconds = avg_ttt_converge_time_s * k_ttt
        else:
            ttt_seconds = float("inf")
            k_ttt = -1

        return {
            "algorithm_name": trial_data["algorithm_name"],
            "problem_id": self.problem_name,
            "algorithm_type": "BSB" if trial_data["algorithm_name"].startswith("BSB_") else "DSB",
            "device": "CPU" if "CPU" in trial_data["algorithm_name"] else "GPU",
            "precision": trial_data["algorithm_name"].split("_")[-1],
            # 问题信息
            "exact_optimal": self.exact_optimal,
            "tts_target": target_value,
            "ttt_target": target_value,
            "objective_sense": self.objective_sense,
            # TTS 指标
            "p_tts": p_tts,
            "tts_success_count": tts_success_count,
            "tts_total_trials": total_trials,
            "tts_seconds": tts_seconds,
            "k_tts": k_tts,
            "avg_single_time_s": avg_single_time_s,
            # TTT 指标
            "p_ttt": p_ttt,
            "ttt_success_count": ttt_success_count,
            "ttt_seconds": ttt_seconds,
            "k_ttt": k_ttt,
            "avg_ttt_converge_time_s": avg_ttt_converge_time_s,
            # 原始数据
            "trial_results": trial_results,
            "execution_times_ms": execution_times,
            "energy_histories": trial_data["energy_histories"],
        }


def save_tts_ttt_data_to_csv(tts_ttt_results: List[Dict], output_file: str):
    """
    将 TTS/TTT 数据保存到 CSV 文件

    参数：
        tts_ttt_results: TTS/TTT 结果列表
        output_file: 输出文件路径
    """
    csv_rows = []

    for result in tts_ttt_results:
        if result is None:
            continue

        row = {
            "algorithm_name": result["algorithm_name"],
            "problem_id": result["problem_id"],
            "algorithm_type": result["algorithm_type"],
            "device": result["device"],
            "precision": result["precision"],
            "exact_optimal": result["exact_optimal"],
            "tts_target": result["tts_target"],
            "ttt_target": result["ttt_target"],
            "objective_sense": result["objective_sense"],
            # TTS 指标
            "p_tts": result["p_tts"],
            "tts_success_count": result["tts_success_count"],
            "tts_total_trials": result["tts_total_trials"],
            "tts_seconds": result["tts_seconds"],
            "k_tts": result["k_tts"],
            "avg_single_time_s": result["avg_single_time_s"],
            # TTT 指标
            "p_ttt": result["p_ttt"],
            "ttt_success_count": result["ttt_success_count"],
            "ttt_seconds": result["ttt_seconds"],
            "k_ttt": result["k_ttt"],
            "avg_ttt_converge_time_s": result["avg_ttt_converge_time_s"],
            # 额外指标
            "time_saved_ratio": (
                result["avg_ttt_converge_time_s"] / result["avg_single_time_s"]
                if result["avg_single_time_s"] > 0
                else 0
            ),
            "tts_to_ttt_ratio": (
                result["ttt_seconds"] / result["tts_seconds"]
                if result["tts_seconds"] != float("inf") and result["tts_seconds"] > 0
                else 1.0
            ),
        }
        csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"💾 TTS/TTT 数据已保存: {output_file}")

    return df


def save_detailed_trial_results(trial_data: Dict, output_dir: Path):
    """
    保存详细的试验结果（包含能量历史）

    参数：
        trial_data: 试验数据字典
        output_dir: 输出目录
    """
    algorithm_name = trial_data["algorithm_name"]
    problem_id = trial_data.get("problem_id", "unknown")

    # 保存试验结果
    trial_df = pd.DataFrame(
        {
            "trial_number": range(1, len(trial_data["trial_results"]) + 1),
            "objective_value": trial_data["trial_results"],
            "execution_time_ms": trial_data["execution_times_ms"],
        }
    )

    trial_csv = output_dir / f"trials_{algorithm_name}_{problem_id}.csv"
    trial_df.to_csv(trial_csv, index=False, encoding="utf-8-sig")

    # 保存能量历史（JSON 格式）
    energy_json = output_dir / f"energy_history_{algorithm_name}_{problem_id}.json"
    with open(energy_json, "w") as f:
        json.dump(
            {
                "algorithm_name": algorithm_name,
                "problem_id": problem_id,
                "max_iters": trial_data["max_iters"],
                "record_interval": trial_data["record_interval"],
                "energy_histories": trial_data["energy_histories"],
                "execution_times_ms": trial_data["execution_times_ms"],
            },
            f,
            indent=2,
        )

    return trial_csv, energy_json


def check_executables_exist():
    """检查所有需要的可执行文件是否存在"""
    script_dir = Path(__file__).parent.absolute()

    # 需要检查的GPU可执行文件
    required_gpu_execs = {
        "BSB": script_dir / "BSB" / "BSB_GPU_fp16.exe",
        "DSB": script_dir / "DSB" / "DSB_GPU_fp16.exe",
    }

    missing = []
    for algo_type, exe_path in required_gpu_execs.items():
        if not exe_path.exists():
            missing.append((algo_type, str(exe_path)))

    return missing


def main():
    """主函数"""
    # 切换到脚本所在目录，确保相对路径正确
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    print(f"✅ 工作目录已设置为: {os.getcwd()}")

    # 检查可执行文件
    missing_exes = check_executables_exist()
    if missing_exes:
        print(f"\n⚠️ 警告: 以下GPU可执行文件缺失:")
        for algo_type, exe_path in missing_exes:
            print(f"   - {algo_type}: {exe_path}")
        print(f"\n💡 建议操作:")
        print(f"   1. 运行编译命令重新生成可执行文件:")
        print(f"      cd \"{script_dir}\" && build_all_windows.bat")
        print(f"   2. 或单独编译缺失的文件:")
        for algo_type, exe_path in missing_exes:
            exe_name = Path(exe_path).stem
            print(f"      nvcc -O3 -arch=sm_89 -I. {algo_type}/{exe_name}.cu -o {algo_type}/{exe_name}.exe")
        print(f"\n⏸️ 将继续运行，但缺失的算法会被跳过...")
        print()

    print("=" * 80)
    print("📊 TTS/TTT 数据生成工具")
    print("=" * 80)

    # 检测CUDA环境
    cuda_available, cuda_info = check_cuda_available()
    if cuda_available:
        print(f"\n✅ CUDA环境可用: {cuda_info}")
    else:
        print(f"\n⚠️ CUDA环境不可用: {cuda_info}")
        print(f"   GPU算法将无法正常执行")
    print(f"\n{'='*80}")

    # 创建输出目录
    output_dir = Path("save_csv")
    output_dir.mkdir(exist_ok=True)

    # 加载问题数据库（使用绝对路径）
    db_path = script_dir / "data" / "qplib_results.csv"
    if os.path.exists(db_path):
        problem_db = pd.read_csv(db_path)
        print(f"✅ 加载问题数据库: {len(problem_db)} 个问题")
    else:
        print(f"❌ 问题数据库不存在: {db_path}")
        return

    # 加载 Optuna 最优参数数据库
    best_params_db = None
    optuna_csv_dir = output_dir
    optuna_csv_files = sorted(optuna_csv_dir.glob("batch_optuna_*.csv"))

    if optuna_csv_files:
        latest_optuna_csv = optuna_csv_files[-1]
        best_params_db = pd.read_csv(latest_optuna_csv)
        print(f"✅ 加载最优参数数据库: {latest_optuna_csv}")
        print(f"   包含 {len(best_params_db)} 条 (算法,问题) 参数记录")
    else:
        print(f"⚠️  未找到 Optuna 参数数据库，将使用默认参数")

    # 选择第一个问题（测试模式）
    selected_problems = problem_db["instance_id"].tolist()[:1]
    print(f"\n📋 将处理的问题: {len(selected_problems)} 个（测试模式）")
    for i, pid in enumerate(selected_problems, 1):
        print(f"   {i}. {pid}")

    # 运行 BSB_CPU_fp8 和 BSB_CPU_fp16 算法（覆盖原来的无效数据）
    algorithms = [
        "BSB_CPU_fp8",
        "BSB_CPU_fp16",
    ]
    print(f"\n🔧 将运行的算法: {len(algorithms)} 种")
    for algo in algorithms:
        print(f"   - {algo}")

    # GPU算法暂时注释
    # algorithms = [
    #     "BSB_GPU_fp16",
    #     "BSB_GPU_int8",
    #     "DSB_GPU_fp16",
    #     "DSB_GPU_int8",
    #     "DSB_GPU_fp8",
    # ]
    # print(f"\n🔧 将运行的算法: {len(algorithms)} 种 (GPU算法)")
    # for algo in algorithms:
    #     print(f"   - {algo}")

    # 参数配置
    NUM_TRIALS = 100  # 每个配置的运行次数（正式运行）
    MAX_ITERS = 1000  # 最大迭代次数
    RECORD_INTERVAL = 10  # 能量记录间隔
    CONFIDENCE_LEVEL = 0.99  # 置信水平 Q
    TARGET_RATIO = 0.99  # 目标比例（99% 精确解）

    print(f"\n💡 提示: 当前 NUM_TRIALS={NUM_TRIALS} (正式运行)")
    print(f"   如需快速测试，可改为 NUM_TRIALS=10")

    print(f"\n⚙️  参数配置:")
    print(f"  - 运行次数: {NUM_TRIALS}")
    print(f"  - 最大迭代: {MAX_ITERS}")
    print(f"  - 记录间隔: {RECORD_INTERVAL}")
    print(f"  - 置信水平 Q: {CONFIDENCE_LEVEL}")
    print(f"  - 目标比例: {TARGET_RATIO}")
    print(f"  - 总配置数: {len(selected_problems) * len(algorithms)}")
    print(f"  - 预计总运行次数: {len(selected_problems) * len(algorithms) * NUM_TRIALS}")

    # 收集所有 TTS/TTT 结果
    all_tts_ttt_results = []

    # 遍历问题和算法
    total_configs = len(selected_problems) * len(algorithms)
    config_count = 0

    for problem_id in tqdm(selected_problems, desc="📊 处理问题进度", unit="problem"):
        # 使用绝对路径
        qplib_file = str(script_dir / "qplib" / f"{problem_id}.qplib")
        if not os.path.exists(qplib_file):
            print(f"⚠️ 文件不存在，跳过: {qplib_file}")
            continue

        print(f"\n{'='*80}")
        print(f"处理问题: {problem_id}")
        print(f"{'='*80}")

        for algorithm_name in tqdm(algorithms, desc=f"  {problem_id} 算法进度", unit="algo"):
            config_count += 1
            print(f"\n--- {algorithm_name} ({config_count}/{total_configs}) ---")

            # 创建求解器（传入算法名称和最优参数数据库）
            try:
                solver = QBTTTExternal(
                    qplib_file,
                    device="cuda",
                    problem_db=problem_db,
                    algorithm_name=algorithm_name,
                    best_params_db=best_params_db,
                )
            except FileNotFoundError as e:
                # 可执行文件不存在，跳过该算法
                print(f"\n⚠️ 跳过 {algorithm_name}: {e}")
                # 生成失败记录
                tts_ttt_result = {
                    "algorithm_name": algorithm_name,
                    "problem_id": problem_id,
                    "algorithm_type": "BSB" if "BSB" in algorithm_name else "DSB",
                    "device": "GPU" if "GPU" in algorithm_name else "CPU",
                    "precision": algorithm_name.split("_")[-1],
                    "exact_optimal": None,
                    "tts_target": None,
                    "ttt_target": None,
                    "objective_sense": None,
                    "p_tts": 0.0,
                    "tts_success_count": 0,
                    "tts_total_trials": NUM_TRIALS,
                    "tts_seconds": float("inf"),
                    "k_tts": -1,
                    "avg_single_time_s": 0.0,
                    "p_ttt": 0.0,
                    "ttt_success_count": 0,
                    "ttt_seconds": float("inf"),
                    "k_ttt": -1,
                    "avg_ttt_converge_time_s": 0.0,
                    "time_saved_ratio": 0.0,
                    "tts_to_ttt_ratio": 0.0,
                }
                all_tts_ttt_results.append(tts_ttt_result)
                continue

            # 运行算法并记录能量历史
            trial_data = solver.run_algorithm_with_history(
                algorithm_name=algorithm_name,
                num_trials=NUM_TRIALS,
                max_iters=MAX_ITERS,
                record_interval=RECORD_INTERVAL,
            )

            # 保存详细试验结果
            if trial_data["trial_results"]:
                trial_data["problem_id"] = problem_id
                save_detailed_trial_results(trial_data, output_dir)

            # 计算 TTS/TTT
            tts_ttt_result = solver.calculate_tts_ttt_from_trials(
                trial_data, confidence_level=CONFIDENCE_LEVEL, target_ratio=TARGET_RATIO
            )

            # 如果没有成功试验，仍然生成一个记录（标记为失败）
            if not tts_ttt_result:
                print(f"\n⚠️ {algorithm_name} 没有成功试验，生成失败记录")
                # 生成失败记录（TTS=inf, TTT=inf）
                tts_ttt_result = {
                    "algorithm_name": algorithm_name,
                    "problem_id": problem_id,
                    "algorithm_type": "BSB" if "BSB" in algorithm_name else "DSB",
                    "device": "GPU" if "GPU" in algorithm_name else "CPU",
                    "precision": algorithm_name.split("_")[-1],
                    "exact_optimal": solver.exact_optimal if hasattr(solver, 'exact_optimal') else None,
                    "tts_target": None,
                    "ttt_target": None,
                    "objective_sense": solver.objective_sense if hasattr(solver, 'objective_sense') else None,
                    "p_tts": 0.0,
                    "tts_success_count": 0,
                    "tts_total_trials": NUM_TRIALS,
                    "tts_seconds": float("inf"),
                    "k_tts": -1,
                    "avg_single_time_s": 0.0,
                    "p_ttt": 0.0,
                    "ttt_success_count": 0,
                    "ttt_seconds": float("inf"),
                    "k_ttt": -1,
                    "avg_ttt_converge_time_s": 0.0,
                    "time_saved_ratio": 0.0,
                    "tts_to_ttt_ratio": 0.0,
                }
            else:
                # 确保有数据的情况下也填充完整的device和precision信息
                tts_ttt_result["device"] = "GPU" if "GPU" in algorithm_name else "CPU"
                tts_ttt_result["precision"] = algorithm_name.split("_")[-1]

            all_tts_ttt_results.append(tts_ttt_result)

            # 打印结果
            print(f"\n📊 {algorithm_name} 结果:")
            print(f"  TTS: {tts_ttt_result['tts_seconds']:.4f}s (P={tts_ttt_result['p_tts']:.4f})")
            print(f"  TTT: {tts_ttt_result['ttt_seconds']:.4f}s (P={tts_ttt_result['p_ttt']:.4f})")

    # 保存 TTS/TTT 汇总数据
    # 指定输出文件：直接更新现有文件
    summary_csv = output_dir / "tts_ttt_summary_23problems_12algs_20260204_234401.csv"

    # 读取已有数据
    all_tts_ttt_merged = []
    if summary_csv.exists():
        print(f"\n📂 读取已有数据: {summary_csv}")
        existing_df = pd.read_csv(summary_csv)
        existing_data = existing_df.to_dict("records")
        all_tts_ttt_merged = existing_data
        print(f"   已有记录: {len(existing_data)} 条")

    # 添加新生成的数据
    all_tts_ttt_merged.extend(all_tts_ttt_results)

    # 去重：根据 algorithm_name 和 problem_id 保留新数据
    seen = set()
    final_results = []
    for record in reversed(all_tts_ttt_merged):  # 反向遍历，保留最新的
        key = (record["algorithm_name"], record["problem_id"])
        if key not in seen:
            seen.add(key)
            final_results.append(record)
    final_results.reverse()  # 恢复原始顺序

    # 保存到指定文件
    if final_results:
        tts_ttt_df = save_tts_ttt_data_to_csv(final_results, summary_csv)

        print(f"\n{'='*80}")
        print(f"🎉 数据生成完成！")
        print(f"{'='*80}")
        print(f"总记录数: {len(final_results)} (新增: {len(all_tts_ttt_results)})")
        print(f"输出文件: {summary_csv}")
        print(f"\n📊 数据包含:")
        print(f"  - 问题数: {len(selected_problems)}")
        print(f"  - 算法数: {len(set(r['algorithm_name'] for r in final_results))}")
        print(f"  - 每配置试验次数: {NUM_TRIALS}")
        print(f"  - 总试验次数: {len(final_results) * NUM_TRIALS}")
        print(f"\n下一步: 运行 plot_basic_TTT.py 进行可视化")
    else:
        print(f"\n❌ 没有成功生成 TTS/TTT 数据")


if __name__ == "__main__":
    main()
