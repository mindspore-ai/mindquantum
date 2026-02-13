#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制13条曲线图：1条精确解 + 12条算法曲线

横轴：QPLIB实例
纵轴：目标函数值（对数刻度）
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from datetime import datetime

# 设置中文字体和样式
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["font.family"] = "sans-serif"


def load_batch_csv(csv_file, exact_solutions_file="data/qplib_results.csv"):
    """
    加载批量求解结果CSV文件和精确解

    参数：
        csv_file: CSV文件路径
        exact_solutions_file: 精确解文件路径

    返回：
        tuple: (df, exact_solutions)
    """
    df = pd.read_csv(csv_file)

    # 加载精确解
    exact_solutions = {}
    try:
        exact_df = pd.read_csv(exact_solutions_file)
        print(f"✅ 加载精确解文件: {exact_solutions_file}")
        print(f"   精确解实例数: {len(exact_df)}")

        # 解析精确值
        def parse_exact_value(value_str):
            import re

            match = re.search(r"([-\d]+\.?\d*)", str(value_str))
            if match:
                return float(match.group(1))
            return None

        exact_df["exact_value"] = exact_df["solution_objective_value"].apply(parse_exact_value)
        exact_solutions = dict(zip(exact_df["instance_id"], exact_df["exact_value"]))

        print(f"   成功匹配精确解: {len([v for v in exact_solutions.values() if v is not None])} 个实例")

    except Exception as e:
        print(f"⚠️  加载精确解文件失败: {e}")

    return df, exact_solutions


def plot_13_curves(df, exact_solutions, output_dir):
    """
    绘制13条曲线图

    参数：
        df: 结果DataFrame
        exact_solutions: 精确解字典
        output_dir: 输出目录
    """
    # 获取所有QPLIB实例
    all_instances = sorted(df["problem_id"].unique())
    print(f"📊 实例数: {len(all_instances)}")
    print(f"   实例列表: {', '.join(all_instances)}")

    # 定义所有12种算法
    all_algorithms = [
        "BSB_CPU_fp16",
        "BSB_CPU_int8",
        "BSB_CPU_fp8",
        "BSB_GPU_fp16",
        "BSB_GPU_int8",
        "BSB_GPU_fp8",
        "DSB_CPU_fp16",
        "DSB_CPU_int8",
        "DSB_CPU_fp8",
        "DSB_GPU_fp16",
        "DSB_GPU_int8",
        "DSB_GPU_fp8",
    ]

    # 分离BSB和DSB算法
    bsb_algorithms = [a for a in all_algorithms if a.startswith("BSB_")]
    dsb_algorithms = [a for a in all_algorithms if a.startswith("DSB_")]

    print(f"\n🔹 BSB算法 ({len(bsb_algorithms)}种): {bsb_algorithms}")
    print(f"🔹 DSB算法 ({len(dsb_algorithms)}种): {dsb_algorithms}")

    # 定义颜色和线型
    colors = [
        "#000000",  # 黑色 - 精确解
        "#e74c3c",
        "#e67e22",
        "#f1c40f",
        "#27ae60",  # 红4种 - BSB
        "#2ecc71",
        "#3498db",
        "#2980b9",
        "#8e44ad",  # 蓝/紫4种 - BSB/GPU
        "#9b59b6",
        "#1abc9c",
        "#16a085",  # 青/绿4种 - DSB
    ]

    line_styles = [
        "-",
        "-",
        "-",
        "-",
        "-",  # BSB CPU 3 + GPU 1
        "--",
        "--",
        "--",
        "--",  # BSB GPU 3 + DSB CPU 1
        ":",
        ":",
        ":",
        ":",  # DSB GPU 3
    ]

    markers = [
        "o",
        "o",
        "o",
        "o",
        "o",  # 圆形 - 精确解 + BSB CPU 3 + BSB GPU 1
        "s",
        "s",
        "s",
        "s",  # 方形 - BSB GPU 2 + DSB CPU 2
        "^",
        "^",
        "^",
        "^",  # 三角形 - DSB GPU 2 + 剩余
    ]

    line_widths = [1.5] + [2.0] * 12  # 精确解更细，算法更粗

    # 创建图形
    fig, ax = plt.subplots(figsize=(20, 12))

    # 1. 先绘制精确解曲线（作为背景参考）
    exact_values = []
    for instance in all_instances:
        exact_val = exact_solutions.get(instance, None)
        exact_values.append(exact_val)

    ax.plot(
        range(len(all_instances)),
        exact_values,
        label="Exact Optimal",
        color=colors[0],
        linestyle=line_styles[0],
        marker=markers[0],
        linewidth=line_widths[0],
        markersize=8,
        alpha=0.3,  # 降低透明度
        zorder=0,
    )  # 置于最底层

    # 2. 绘制12种算法曲线（覆盖在精确解之上）
    for i, algo in enumerate(all_algorithms):
        algo_data = df[df["algorithm_name"] == algo].copy()
        algo_values = []

        for instance in all_instances:
            instance_data = algo_data[algo_data["problem_id"] == instance]
            if len(instance_data) > 0:
                obj_val = instance_data.iloc[0]["objective_value"]
                algo_values.append(obj_val if pd.notna(obj_val) else np.nan)
            else:
                algo_values.append(np.nan)

        color_idx = i + 1  # 跳过第一个颜色（精确解）
        ax.plot(
            range(len(all_instances)),
            algo_values,
            label=algo,
            color=colors[color_idx % len(colors)],
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)],
            linewidth=line_widths[i % len(line_widths)],
            markersize=7,
            alpha=0.9,
            zorder=10 + i,
        )  # 算法曲线在上方

    # 设置坐标轴
    ax.set_xlabel("QPLIB Instance", fontsize=14, fontweight="bold")
    ax.set_ylabel("Objective Value (log scale)", fontsize=14, fontweight="bold")
    ax.set_title("13 Curves: Exact Optimal + 12 Algorithm Variants", fontsize=16, fontweight="bold", pad=15)

    ax.set_xticks(range(len(all_instances)))
    ax.set_xticklabels(all_instances, rotation=45, ha="right", fontsize=9)
    ax.set_yscale("log")

    # 设置图例
    ax.legend(
        loc="upper left", fontsize=8, bbox_to_anchor=(1.02, 1), ncol=1, framealpha=0.9, fancybox=True, shadow=True
    )

    # 添加网格
    ax.grid(True, alpha=0.3, which="both", linestyle="--", linewidth=0.5)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # 右侧留出空间给图例

    output_file = output_dir / "Figure_13_Curves_All_Algorithms.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ 13条曲线图已保存: {output_file}")
    plt.close()

    # 2. 分开绘制BSB和DSB
    plot_bsb_dsb_separate(df, exact_solutions, all_instances, bsb_algorithms, dsb_algorithms, output_dir, colors)


def plot_bsb_dsb_separate(df, exact_solutions, all_instances, bsb_algorithms, dsb_algorithms, output_dir, colors):
    """
    分别绘制BSB和DSB的曲线图

    参数：
        df: 结果DataFrame
        exact_solutions: 精确解字典
        all_instances: 所有实例
        bsb_algorithms: BSB算法列表
        dsb_algorithms: DSB算法列表
        output_dir: 输出目录
        colors: 颜色列表
    """
    # BSB子图
    fig_bsb, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 子图1: BSB算法
    exact_values = [exact_solutions.get(instance, None) for instance in all_instances]
    ax1.plot(
        range(len(all_instances)),
        exact_values,
        label="Exact Optimal",
        color=colors[0],
        linewidth=1.5,
        marker="o",
        markersize=8,
        alpha=0.3,
        zorder=0,
    )  # 精确解作为背景

    for i, algo in enumerate(bsb_algorithms):
        algo_data = df[df["algorithm_name"] == algo].copy()
        algo_values = []

        for instance in all_instances:
            instance_data = algo_data[algo_data["problem_id"] == instance]
            if len(instance_data) > 0:
                obj_val = instance_data.iloc[0]["objective_value"]
                algo_values.append(obj_val if pd.notna(obj_val) else np.nan)
            else:
                algo_values.append(np.nan)

        color_idx = i + 1
        marker_style = "s" if "CPU" in algo else "^"
        line_style = "--" if "CPU" in algo else ":"

        ax1.plot(
            range(len(all_instances)),
            algo_values,
            label=algo,
            color=colors[color_idx % len(colors)],
            linestyle=line_style,
            marker=marker_style,
            linewidth=2.5,
            markersize=7,
            alpha=0.9,
        )  # 算法曲线更清晰

    ax1.set_xlabel("QPLIB Instance", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Objective Value (log scale)", fontsize=12, fontweight="bold")
    ax1.set_title("(a) BSB Algorithms: 6 Variants", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xticks(range(len(all_instances)))
    ax1.set_xticklabels(all_instances, rotation=45, ha="right", fontsize=8)
    ax1.set_yscale("log")
    ax1.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax1.grid(True, alpha=0.3)

    # 子图2: DSB算法
    ax2.plot(
        range(len(all_instances)),
        exact_values,
        label="Exact Optimal",
        color=colors[0],
        linewidth=1.5,
        marker="o",
        markersize=8,
        alpha=0.3,
        zorder=0,
    )  # 精确解作为背景

    for i, algo in enumerate(dsb_algorithms):
        algo_data = df[df["algorithm_name"] == algo].copy()
        algo_values = []

        for instance in all_instances:
            instance_data = algo_data[algo_data["problem_id"] == instance]
            if len(instance_data) > 0:
                obj_val = instance_data.iloc[0]["objective_value"]
                algo_values.append(obj_val if pd.notna(obj_val) else np.nan)
            else:
                algo_values.append(np.nan)

        color_idx = i + 4  # 偏移颜色索引
        marker_style = "s" if "CPU" in algo else "^"
        line_style = "--" if "CPU" in algo else ":"

        ax2.plot(
            range(len(all_instances)),
            algo_values,
            label=algo,
            color=colors[color_idx % len(colors)],
            linestyle=line_style,
            marker=marker_style,
            linewidth=2.5,
            markersize=7,
            alpha=0.9,
        )  # 算法曲线更清晰

    ax2.set_xlabel("QPLIB Instance", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Objective Value (log scale)", fontsize=12, fontweight="bold")
    ax2.set_title("(b) DSB Algorithms: 6 Variants", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xticks(range(len(all_instances)))
    ax2.set_xticklabels(all_instances, rotation=45, ha="right", fontsize=8)
    ax2.set_yscale("log")
    ax2.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "Figure_13_Curves_BSB_DSB_Separate.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ BSB/DSB分开曲线图已保存: {output_file}")
    plt.close()


def main():
    """
    主函数
    """
    # 查找最新的batch_summary CSV文件
    csv_dir = Path("save_csv")
    csv_files = sorted(csv_dir.glob("batch_summary_*.csv"))

    if not csv_files:
        print(f"❌ 未找到 batch_summary CSV 文件")
        return

    # 使用最新的文件
    csv_file = csv_files[-1]
    print(f"📂 使用批量结果文件: {csv_file}")

    # 加载数据
    df, exact_solutions = load_batch_csv(csv_file)
    print(f"✅ 加载数据: {len(df)} 条记录")
    print(f"   问题数: {df['problem_id'].nunique()}")
    print(f"   算法数: {df['algorithm_name'].nunique()}")

    # 创建输出目录
    output_dir = Path("save_pic")
    output_dir.mkdir(exist_ok=True)

    print(f"\n🎨 开始生成13条曲线图...")
    print("=" * 80)

    # 生成曲线图
    try:
        plot_13_curves(df, exact_solutions, output_dir)

        print("\n" + "=" * 80)
        print(f"🎉 所有曲线图生成完成！")
        print(f"📁 输出目录: {output_dir}")
        print("=" * 80)

    except Exception as e:
        print(f"❌ 生成曲线图时出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
