#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QPLIB文件解析器

该模块提供了完整的QPLIB文件解析功能，包括：
1. 问题基本信息解析
2. 二次项、线性项、常数项解析
3. 约束条件处理和分离
4. 问题模型数据结构
"""

import numpy as np
import pandas as pd
import os
import re
from typing import List, Tuple, Optional, Dict, Any
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ProblemModel:
    """
    问题模型数据结构

    该类包含了从QPLIB文件解析出的所有问题信息，
    包括目标函数系数、约束条件、缩放参数等。
    """

    def __init__(self):
        # 基本信息
        self.problem_name: str = ""  # 问题名称
        self.problem_type: str = ""  # 问题类型
        self.objective_sense: str = ""  # 优化方向 (minimize/maximize)
        self.num_vars: int = 0  # 变量数量
        self.num_constraints: int = 0  # 约束数量

        # 目标函数相关
        self.quad_terms_indices: List[List[int]] = []  # 二次项索引对 [i, j]
        self.quad_terms_coeffs: List[float] = []  # 二次项系数
        self.linear_terms: np.ndarray = None  # 线性项系数向量
        self.constant_term: float = 0.0  # 常数项

        # 约束相关
        self.equal_constraint_matrix: np.ndarray = None  # 等式约束矩阵
        self.equal_constraint_bounds: np.ndarray = None  # 等式约束边界
        self.inequal_constraint_matrix: np.ndarray = None  # 不等式约束矩阵
        self.inequal_constraint_bounds: np.ndarray = None  # 不等式约束边界

        # 缩放参数
        self.scaling_factor: float = 1.0  # 缩放因子
        self.xi: float = 1.0  # 耦合强度参数

        # 统计信息
        self.num_quad_terms: int = 0  # 二次项数量
        self.num_linear_terms: int = 0  # 非零线性项数量


class QPLIBParser:
    """
    QPLIB文件解析器

    基于solve_SB_float.py的实现，支持完整的QPLIB格式解析。
    """

    @staticmethod
    def parse_qplib_file(file_path: str) -> ProblemModel:
        """
        解析QPLIB文件并返回问题模型

        参数：
            file_path (str): QPLIB文件路径

        返回：
            ProblemModel: 解析后的问题模型
        """
        model = ProblemModel()

        # 读取文件并清理
        with open(file_path, "r") as f:
            raw_lines = f.readlines()

        # 清理注释与空行
        lines = [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]

        print(f"开始解析QPLIB文件: {file_path}")

        # 1. 读取基本信息
        model.problem_name = lines[0]
        model.problem_type = lines[1]
        model.objective_sense = lines[2].strip().lower()
        model.num_vars = int(lines[3].split("#")[0].strip())

        # 根据实际QPLIB文件格式：
        # 第4行：变量数量（重复）
        # 第5行：二次项数量
        # 对于QPLIB_3565.qplib，似乎没有约束条件
        num_quadratic_terms = int(lines[4].split("#")[0].strip())
        model.num_constraints = 0  # 这个文件没有约束
        current = 5

        print(f"  - 问题名称: {model.problem_name}")
        print(f"  - 问题类型: {model.problem_type}")
        print(f"  - 优化方向: {model.objective_sense}")
        print(f"  - 变量数量: {model.num_vars}")
        print(f"  - 约束数量: {model.num_constraints}")
        print(f"  - 二次项数量: {num_quadratic_terms}")

        # 2. 读取二次项
        quad_terms = []
        diag_terms_indices = []
        diag_terms_coeffs = []

        if num_quadratic_terms > 0:
            print(f"  - 检测到二次项数据，开始解析...")
            for i in range(num_quadratic_terms):
                parts = lines[current + i].split()
                row_idx = int(parts[0]) - 1  # 转换为0-based索引
                col_idx = int(parts[1]) - 1
                coeff = float(parts[2])

                if row_idx == col_idx:
                    # 对角项，也添加到quad_terms中
                    model.quad_terms_indices.append([row_idx, col_idx])
                    model.quad_terms_coeffs.append(coeff)  # 对角项系数保持完整值
                    diag_terms_indices.append(row_idx)
                    diag_terms_coeffs.append(coeff)
                else:
                    # 非对角项，存储对称矩阵（只存上三角）
                    if row_idx > col_idx:
                        row_idx, col_idx = col_idx, row_idx
                    model.quad_terms_indices.append([row_idx, col_idx])
                    model.quad_terms_coeffs.append(coeff)  # 保持原始系数

            # 构建完整的二次项数组（包括对角项）
            quad_terms_data = []
            for i in range(num_quadratic_terms):
                parts = lines[current + i].split()
                row_idx = int(parts[0]) - 1  # 保持0-based索引
                col_idx = int(parts[1]) - 1
                coeff = float(parts[2])
                quad_terms_data.append([row_idx, col_idx, coeff])
            quad_terms = np.array(quad_terms_data, dtype=np.float32)

            current += num_quadratic_terms
        else:
            print(f"  - 未检测到二次项数据")
            quad_terms = np.array([], dtype=np.float32).reshape(0, 3)

        model.num_quad_terms = len(model.quad_terms_indices)

        # 3. 读取默认线性项值
        default_linear_value = float(lines[current].split("#")[0].strip())
        current += 1
        print(f"  - 默认线性项值: {default_linear_value}")

        # 4. 读取指定线性项数量
        num_linear_terms = int(lines[current].split("#")[0].strip())
        current += 1

        # 初始化线性项
        if default_linear_value == 0 and num_linear_terms == 0 and len(diag_terms_indices) == 0:
            model.linear_terms = None
        else:
            model.linear_terms = np.full(model.num_vars, default_linear_value, dtype=np.float64)

            # 读取指定线性项
            for i in range(num_linear_terms):
                parts = lines[current].split()
                var_idx = int(parts[0]) - 1
                coeff = float(parts[1])
                model.linear_terms[var_idx] = coeff
                current += 1

            # 注意：对角项保持原始系数
            # 这里不加到线性项中，避免重复处理

        model.num_linear_terms = (
            np.count_nonzero(np.abs(model.linear_terms) > 1e-12) if model.linear_terms is not None else 0
        )

        # 5. 读取常数项
        model.constant_term = float(lines[current].split("#")[0].strip())
        current += 1
        print(f"  - 常数项: {model.constant_term}")

        # 6. 处理约束（如果有）
        if model.num_constraints > 0:
            # 读取约束线性项数量
            num_constraint_linear_item = int(lines[current].split("#")[0].strip())
            current += 1
            constraint_terms = None
            if num_constraint_linear_item > 0:
                constraint_terms = np.zeros((model.num_constraints, model.num_vars))
                for _ in range(num_constraint_linear_item):
                    parts = lines[current].split()
                    idx_constraints, idx_terms, val = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
                    constraint_terms[idx_constraints, idx_terms] = val
                    current += 1

            # 读取约束边界
            inf = float(lines[current].split("#")[0].strip())
            constraint_lb = float(lines[current + 1].split("#")[0].strip())
            constraint_lb = constraint_lb * np.ones(model.num_constraints)
            num_constraint_lb = int(lines[current + 2].split("#")[0].strip())
            current += 3
            constraint_rb = float(lines[current].split("#")[0].strip())
            constraint_rb = constraint_rb * np.ones(model.num_constraints)
            num_constraint_rb = int(lines[current + 1].split("#")[0].strip())
            current += 2

            # 处理指定的下界
            if num_constraint_lb != 0:
                for _ in range(num_constraint_lb):
                    parts = lines[current].split()
                    idx, val = int(parts[0]) - 1, float(parts[1])
                    constraint_lb[idx] = val
                    current += 1

            # 处理指定的上界
            if num_constraint_rb != 0:
                for _ in range(num_constraint_rb):
                    parts = lines[current].split()
                    idx, val = int(parts[0]) - 1, float(parts[1])
                    constraint_rb[idx] = val
                    current += 1

            # 分离等式约束和不等式约束
            if constraint_terms is not None:
                equal_constraint_ind = np.nonzero(constraint_lb == constraint_rb)[0]
                if len(equal_constraint_ind) > 0 and constraint_rb[equal_constraint_ind] != 0:
                    model.equal_constraint_term = constraint_terms[equal_constraint_ind]
                    model.equal_constraint_bound = constraint_lb[equal_constraint_ind]
                    constraint_lb = np.delete(constraint_lb, equal_constraint_ind)
                    constraint_rb = np.delete(constraint_rb, equal_constraint_ind)
                    constraint_terms = np.delete(constraint_terms, equal_constraint_ind, axis=0)

            # 处理不等式约束
            if constraint_terms is not None:
                if len(constraint_lb) > 0:
                    valid_ind = np.nonzero(constraint_lb != inf)[0]
                    if len(valid_ind) > 0:
                        constraint_terms = np.vstack([constraint_terms, -1 * constraint_terms[valid_ind]])
                        constraint_rb = np.concatenate([constraint_rb, -1 * constraint_lb[valid_ind]])

                if len(constraint_rb) > 0:
                    unvalid_ind = np.nonzero(constraint_rb == inf)[0]
                    if len(unvalid_ind) > 0:
                        constraint_terms = np.delete(constraint_terms, unvalid_ind, axis=0)
                        constraint_rb = np.delete(constraint_rb, unvalid_ind)

                if constraint_terms is not None and len(constraint_rb) > 0:
                    model.inequal_constraint_term = constraint_terms
                    model.inequal_constraint_bound = constraint_rb
        else:
            # 无约束问题
            model.equal_constraint_matrix = None
            model.equal_constraint_bounds = None
            model.inequal_constraint_matrix = None
            model.inequal_constraint_bounds = None

        # 7. 采用不同的缩放策略
        if model.num_quad_terms > 0:
            # 计算最大绝对系数（包括对角项的原始系数）
            max_abs_coeff = 0.0
            for coeff in model.quad_terms_coeffs:
                abs_val = abs(coeff)
                if abs_val > max_abs_coeff:
                    max_abs_coeff = abs_val

            if model.linear_terms is not None:
                for coeff in model.linear_terms:
                    abs_val = abs(coeff)
                    if abs_val > max_abs_coeff:
                        max_abs_coeff = abs_val

            # 不在这里缩放，后续统一处理缩放
            model.scaling_factor = max_abs_coeff if max_abs_coeff > 1e-12 else 1.0

            model.xi = 1.0  # 初始值，会重新计算
            print(f"  - 缩放因子={model.scaling_factor:.6f}, xi将由调用程序重新计算")
        else:
            # 纯线性问题
            model.scaling_factor = 1.0
            model.xi = 1.0
            print(f"  - 线性问题，不应用缩放")

        print(f"解析完成！")
        return model

    @staticmethod
    def save_model_to_json(model: ProblemModel, output_path: str) -> bool:
        """
        将问题模型保存为JSON文件

        参数：
            model (ProblemModel): 问题模型
            output_path (str): 输出文件路径

        返回：
            bool: 保存是否成功
        """
        try:
            import json

            data = {
                "problem_name": model.problem_name,
                "problem_type": model.problem_type,
                "objective_sense": model.objective_sense,
                "num_vars": model.num_vars,
                "num_constraints": model.num_constraints,
                "quad_terms_indices": model.quad_terms_indices,
                "quad_terms_coeffs": model.quad_terms_coeffs,
                "linear_terms": model.linear_terms.tolist() if model.linear_terms is not None else [],
                "constant_term": model.constant_term,
                "scaling_factor": model.scaling_factor,
                "xi": model.xi,
                "num_quad_terms": model.num_quad_terms,
                "num_linear_terms": model.num_linear_terms,
                "num_equal_constraints": (
                    len(model.equal_constraint_bounds) if model.equal_constraint_bounds is not None else 0
                ),
                "num_inequal_constraints": (
                    len(model.inequal_constraint_bounds) if model.inequal_constraint_bounds is not None else 0
                ),
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"问题模型已保存到: {output_path}")
            return True

        except Exception as e:
            print(f"保存文件失败: {e}")
            return False

    @staticmethod
    def get_exact_optimal_value(problem_name: str) -> Optional[float]:
        """
        从data/qplib_results.csv读取问题的精确最优解

        参数：
            problem_name: 问题名称（如 'QPLIB_3565'）
        返回：
            精确最优值，如果未找到则返回None
        """
        try:
            csv_path = "data/qplib_results.csv"
            if not os.path.exists(csv_path):
                print(f"⚠️ qplib_results.csv文件不存在: {csv_path}")
                return None

            df = pd.read_csv(csv_path)
            # 查找对应问题的行
            problem_row = df[df["instance_id"] == problem_name]

            if problem_row.empty:
                print(f"⚠️ 在qplib_results.csv中未找到问题: {problem_name}")
                return None

            # 解析solution_objective_value列
            solution_str = problem_row["solution_objective_value"].iloc[0]
            # 从字符串中提取数值，例如: "282.00000000 (gdx, sol)" -> 282.0
            match = re.search(r"([+-]?\d*\.?\d+)", solution_str)
            if match:
                exact_value = float(match.group(1))
                print(f"📊 从qplib_results.csv解析到精确最优解: {exact_value:.8f}")
                return exact_value
            else:
                print(f"⚠️ 无法解析solution_objective_value: {solution_str}")
                return None

        except Exception as e:
            print(f"❌ 读取qplib_results.csv时出错: {e}")
            return None
