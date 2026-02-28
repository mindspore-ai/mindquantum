#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量求解结果可视化 - BSB和DSB算法对比分析

基于 batch_summary_*.csv 文件，对23个QPLIB实例的求解结果进行可视化。
包括：
1. 求解质量对比（objective_value vs exact_optimal）
2. 执行时间对比
3. 算法性能排名
4. BSB vs DSB 综合对比
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from pathlib import Path
import re
from datetime import datetime
import warnings

# 抑制numpy的RuntimeWarning（奇异矩阵行列式计算警告）
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.linalg")


# 设置中文字体和样式
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["font.family"] = "sans-serif"
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_batch_csv(csv_file, exact_solutions_file="data/qplib_results.csv"):
    """
    加载批量求解结果CSV文件和精确解

    参数：
        csv_file: CSV文件路径
        exact_solutions_file: 精确解文件路径

    返回：
        DataFrame: 求解结果
    """
    df = pd.read_csv(csv_file)

    # 计算求解质量百分比
    df["quality_percent"] = df["energy_to_optimal_ratio"] * 100

    # 分类算法
    df["algorithm_category"] = df["algorithm_type"]  # 'BSB' or 'DSB'

    # 加载精确解
    try:
        exact_df = pd.read_csv(exact_solutions_file)
        print(f"✅ 加载精确解文件: {exact_solutions_file}")
        print(f"   精确解实例数: {len(exact_df)}")

        # 解析精确解的目标函数值
        def parse_exact_value(value_str):
            """解析精确值字符串，提取数字部分"""
            import re

            match = re.search(r"([-\d]+\.?\d*)", str(value_str))
            if match:
                return float(match.group(1))
            return None

        exact_df["exact_value"] = exact_df["solution_objective_value"].apply(parse_exact_value)

        # 创建精确解字典
        exact_solutions = dict(zip(exact_df["instance_id"], exact_df["exact_value"]))

        # 将精确解合并到结果数据中
        df["exact_optimal_from_qplib"] = df["problem_id"].map(exact_solutions)

        # 如果CSV中的exact_optimal为空，使用qplib_results.csv中的值
        df["exact_optimal"] = df["exact_optimal"].fillna(df["exact_optimal_from_qplib"])

        # 重新计算求解质量
        df["energy_to_optimal_ratio"] = None
        df.loc[df["exact_optimal"] != 0, "energy_to_optimal_ratio"] = (
            df.loc[df["exact_optimal"] != 0, "objective_value"] / df.loc[df["exact_optimal"] != 0, "exact_optimal"]
        )
        df["quality_percent"] = df["energy_to_optimal_ratio"] * 100

        print(f"   成功匹配精确解: {df['exact_optimal_from_qplib'].notna().sum()} 个实例")

    except Exception as e:
        print(f"⚠️  加载精确解文件失败: {e}")
        print(f"   将使用CSV中的exact_optimal值")

    return df


def plot_quality_comparison(df, output_dir):
    """
    绘制求解质量对比图（分BSB和DSB），包含精确解信息

    参数：
        df: 结果DataFrame
        output_dir: 输出目录
    """
    # 分离BSB和DSB数据
    bsb_df = df[df["algorithm_type"] == "BSB"]
    dsb_df = df[df["algorithm_type"] == "DSB"]

    # 计算每个问题的平均质量
    bsb_avg_quality = bsb_df.groupby("problem_id")["quality_percent"].mean().reset_index()
    dsb_avg_quality = dsb_df.groupby("problem_id")["quality_percent"].mean().reset_index()

    # 获取精确解信息
    exact_info = (
        df.groupby("problem_id").agg({"exact_optimal": "first", "exact_optimal_from_qplib": "first"}).reset_index()
    )

    # 合并数据 - 使用外连接确保所有实例都包含
    quality_comparison = (
        bsb_avg_quality.merge(dsb_avg_quality, on="problem_id", suffixes=("_BSB", "_DSB"), how="outer")
        .merge(exact_info, on="problem_id", how="outer")
        .sort_values("problem_id")
    )

    # 检查缺少数据的实例
    missing_bsb_problems = quality_comparison[quality_comparison["quality_percent_BSB"].isna()]["problem_id"].tolist()
    missing_dsb_problems = quality_comparison[quality_comparison["quality_percent_DSB"].isna()]["problem_id"].tolist()
    if missing_bsb_problems:
        print(f"⚠️  以下实例缺少BSB数据: {missing_bsb_problems}")
    if missing_dsb_problems:
        print(f"⚠️  以下实例缺少DSB数据: {missing_dsb_problems}")

    # 绘制对比图 - 改为2x2布局
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # 按实例ID从小到大排序
    quality_comparison = quality_comparison.sort_values("problem_id")

    # 子图1: 所有问题平均质量对比 - 统一颜色方案
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(quality_comparison))
    width = 0.35

    # 统一颜色：BSB使用绿色，DSB使用蓝色（与Figure_2一致）
    bars1 = ax1.bar(
        x - width / 2,
        quality_comparison["quality_percent_BSB"],
        width,
        label="BSB",
        color="#2ecc71",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax1.bar(
        x + width / 2,
        quality_comparison["quality_percent_DSB"],
        width,
        label="DSB",
        color="#3498db",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # 添加参考线
    ax1.axhline(y=99, color="red", linestyle="--", linewidth=2, alpha=0.7, label="99% Threshold")
    ax1.axhline(y=100, color="darkgreen", linestyle=":", linewidth=2, alpha=0.7, label="100% Optimal")

    ax1.set_xlabel("QPLIB Instance", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Solution Quality (%)", fontsize=12, fontweight="bold")
    ax1.set_title("(a) BSB vs DSB: Average Solution Quality per Problem", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(quality_comparison["problem_id"], rotation=45, ha="right", fontsize=8)

    # 动态设置y轴范围，确保能看到所有数据
    min_quality = min(quality_comparison["quality_percent_BSB"].min(), quality_comparison["quality_percent_DSB"].min())
    if pd.notna(min_quality) and min_quality < 95:
        ax1.set_ylim(min_quality - 2, 101)
    else:
        ax1.set_ylim(90, 101)

    ax1.legend(fontsize=9, loc="lower left")
    ax1.grid(axis="y", alpha=0.3)

    # 在柱状图上显示数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if pd.notna(height) and height > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    fontweight="bold",
                )

    # 子图2: 精确解数值对比
    ax2 = fig.add_subplot(gs[0, 1])
    x2 = np.arange(len(quality_comparison))
    exact_values = quality_comparison["exact_optimal"].values

    # 从原始数据获取平均目标函数值 - 直接使用quality_comparison中的数据
    bsb_obj_avg = bsb_df.groupby("problem_id")["objective_value"].mean()
    dsb_obj_avg = dsb_df.groupby("problem_id")["objective_value"].mean()

    # 将平均目标函数值映射到quality_comparison
    bsb_obj_values = quality_comparison["problem_id"].map(bsb_obj_avg).fillna(0).values
    dsb_obj_values = quality_comparison["problem_id"].map(dsb_obj_avg).fillna(0).values

    # 使用对数刻度（因为数值范围很大）
    ax2.bar(x2 - 0.3, exact_values, 0.2, label="Exact Optimal", color="gold", alpha=0.9)
    ax2.bar(x2 - 0.1, bsb_obj_values, 0.2, label="BSB", color="#2ecc71", alpha=0.8)
    ax2.bar(x2 + 0.1, dsb_obj_values, 0.2, label="DSB", color="#3498db", alpha=0.8)

    ax2.set_xlabel("QPLIB Instance", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Objective Value (log scale)", fontsize=12, fontweight="bold")
    ax2.set_title("(b) Objective Value Comparison (log scale)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(quality_comparison["problem_id"], rotation=45, ha="right", fontsize=8)
    ax2.set_yscale("log")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # 子图3: BSB热图 - BSB算法的质量表现
    ax3 = fig.add_subplot(gs[1, 0])
    bsb_problems = sorted(bsb_df["problem_id"].unique())
    algorithms_bsb = bsb_df["algorithm_name"].unique()

    # BSB热图
    heatmap_data_bsb = bsb_df.pivot(index="algorithm_name", columns="problem_id", values="quality_percent")
    heatmap_data_bsb = heatmap_data_bsb[bsb_problems]  # 按问题顺序排列

    # 转换为数值类型并处理NaN
    heatmap_values_bsb = heatmap_data_bsb.values.astype(float)

    # 绘制热图
    im_bsb = ax3.imshow(heatmap_values_bsb, aspect="auto", cmap="RdYlGn", vmin=90, vmax=100)

    # 设置刻度
    ax3.set_xticks(np.arange(len(bsb_problems)))
    ax3.set_yticks(np.arange(len(algorithms_bsb)))

    ax3.set_xticklabels(bsb_problems, rotation=45, ha="right", fontsize=8)
    ax3.set_yticklabels([a.replace("_", "\n") for a in algorithms_bsb], fontsize=7)

    ax3.set_xlabel("QPLIB Instance (BSB)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("BSB Algorithm", fontsize=12, fontweight="bold")
    ax3.set_title("(c) BSB Algorithms Solution Quality Heatmap", fontsize=13, fontweight="bold", pad=10)

    # 在热图单元格中添加数值
    for i in range(len(algorithms_bsb)):
        for j in range(len(bsb_problems)):
            value = heatmap_values_bsb[i, j]
            if not np.isnan(value):
                text_color = "white" if value < 95 else "black"
                ax3.text(
                    j, i, f"{value:.0f}", ha="center", va="center", fontsize=5, color=text_color, fontweight="bold"
                )

    # 添加颜色条
    cbar_bsb = plt.colorbar(im_bsb, ax=ax3)
    cbar_bsb.set_label("Quality (%)", rotation=270, labelpad=15, fontsize=10)

    # 子图4: DSB热图 - DSB算法的质量表现
    ax4 = fig.add_subplot(gs[1, 1])
    dsb_problems = sorted(dsb_df["problem_id"].unique())
    algorithms_dsb = dsb_df["algorithm_name"].unique()

    # DSB热图
    heatmap_data_dsb = dsb_df.pivot(index="algorithm_name", columns="problem_id", values="quality_percent")
    heatmap_data_dsb = heatmap_data_dsb[dsb_problems]  # 按问题顺序排列

    # 转换为数值类型并处理NaN
    heatmap_values_dsb = heatmap_data_dsb.values.astype(float)

    # 绘制热图
    im_dsb = ax4.imshow(heatmap_values_dsb, aspect="auto", cmap="RdYlGn", vmin=90, vmax=100)

    # 设置刻度
    ax4.set_xticks(np.arange(len(dsb_problems)))
    ax4.set_yticks(np.arange(len(algorithms_dsb)))

    ax4.set_xticklabels(dsb_problems, rotation=45, ha="right", fontsize=8)
    ax4.set_yticklabels([a.replace("_", "\n") for a in algorithms_dsb], fontsize=7)

    ax4.set_xlabel("QPLIB Instance (DSB)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("DSB Algorithm", fontsize=12, fontweight="bold")
    ax4.set_title("(d) DSB Algorithms Solution Quality Heatmap", fontsize=13, fontweight="bold", pad=10)

    # 在热图单元格中添加数值
    for i in range(len(algorithms_dsb)):
        for j in range(len(dsb_problems)):
            value = heatmap_values_dsb[i, j]
            if not np.isnan(value):
                text_color = "white" if value < 95 else "black"
                ax4.text(
                    j, i, f"{value:.0f}", ha="center", va="center", fontsize=5, color=text_color, fontweight="bold"
                )

    # 添加颜色条
    cbar_dsb = plt.colorbar(im_dsb, ax=ax4)
    cbar_dsb.set_label("Quality (%)", rotation=270, labelpad=15, fontsize=10)

    # 调整子图间距
    plt.subplots_adjust(hspace=0.4, wspace=0.25, top=0.95, bottom=0.08, left=0.08, right=0.95)

    output_file = output_dir / "Figure_1_Quality_Comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Figure 1 saved: {output_file}")
    plt.close()


def plot_execution_time_comparison(df, output_dir):
    """
    绘制执行时间对比图（分BSB和DSB）

    参数：
        df: 结果DataFrame
        output_dir: 输出目录
    """
    # 分离BSB和DSB数据
    bsb_df = df[df["algorithm_type"] == "BSB"]
    dsb_df = df[df["algorithm_type"] == "DSB"]

    # 计算平均时间
    bsb_avg_time = bsb_df.groupby("problem_id")["execution_time_ms"].mean().reset_index()
    dsb_avg_time = dsb_df.groupby("problem_id")["execution_time_ms"].mean().reset_index()

    # 合并数据
    time_comparison = bsb_avg_time.merge(dsb_avg_time, on="problem_id", suffixes=("_BSB", "_DSB")).sort_values(
        "execution_time_ms_BSB"
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 子图1: BSB算法执行时间对比
    x = np.arange(len(time_comparison))
    width = 0.35

    bars1 = axes[0, 0].bar(
        x - width / 2, time_comparison["execution_time_ms_BSB"], width, label="BSB", color="#2ecc71", alpha=0.8
    )
    bars2 = axes[0, 0].bar(
        x + width / 2, time_comparison["execution_time_ms_DSB"], width, label="DSB", color="#3498db", alpha=0.8
    )

    axes[0, 0].set_xlabel("QPLIB Instance", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Execution Time (ms)", fontsize=12, fontweight="bold")
    axes[0, 0].set_title("BSB vs DSB: Average Execution Time per Problem", fontsize=14, fontweight="bold", pad=10)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(time_comparison["problem_id"], rotation=45, ha="right", fontsize=9)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(axis="y", alpha=0.3)

    # 子图2: BSB内部算法时间对比
    bsb_time_by_algo = bsb_df.groupby("algorithm_name")["execution_time_ms"].mean().sort_values()
    axes[0, 1].barh(range(len(bsb_time_by_algo)), bsb_time_by_algo.values, color="#2ecc71", alpha=0.8)
    axes[0, 1].set_yticks(range(len(bsb_time_by_algo)))
    axes[0, 1].set_yticklabels(bsb_time_by_algo.index, fontsize=10)
    axes[0, 1].set_xlabel("Average Execution Time (ms)", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("BSB Algorithms: Average Execution Time", fontsize=14, fontweight="bold", pad=10)
    axes[0, 1].grid(axis="x", alpha=0.3)

    # 子图3: DSB内部算法时间对比
    dsb_time_by_algo = dsb_df.groupby("algorithm_name")["execution_time_ms"].mean().sort_values()
    axes[1, 0].barh(range(len(dsb_time_by_algo)), dsb_time_by_algo.values, color="#3498db", alpha=0.8)
    axes[1, 0].set_yticks(range(len(dsb_time_by_algo)))
    axes[1, 0].set_yticklabels(dsb_time_by_algo.index, fontsize=10)
    axes[1, 0].set_xlabel("Average Execution Time (ms)", fontsize=12, fontweight="bold")
    axes[1, 0].set_title("DSB Algorithms: Average Execution Time", fontsize=14, fontweight="bold", pad=10)
    axes[1, 0].grid(axis="x", alpha=0.3)

    # 子图4: 时间分布小提琴图
    # 准备数据
    bsb_times = bsb_df["execution_time_ms"].values
    dsb_times = dsb_df["execution_time_ms"].values

    # 创建数据列表和标签
    violin_data = [bsb_times, dsb_times]
    violin_labels = ["BSB", "DSB"]

    # 绘制小提琴图
    parts = axes[1, 1].violinplot(
        violin_data, positions=range(len(violin_data)), showmeans=True, showmedians=True, showextrema=True
    )

    # 设置颜色
    for pc, color in zip(parts["bodies"], ["#2ecc71", "#3498db"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    parts["cmeans"].set_color("red")
    parts["cmeans"].set_linewidth(2)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    parts["cmaxes"].set_color("black")
    parts["cmins"].set_color("black")
    parts["cbars"].set_color("black")

    axes[1, 1].set_xticks(range(len(violin_labels)))
    axes[1, 1].set_xticklabels(violin_labels, fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Execution Time (ms)", fontsize=12, fontweight="bold")
    axes[1, 1].set_title(
        "BSB vs DSB: Execution Time Distribution (Violin Plot)", fontsize=14, fontweight="bold", pad=10
    )
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "Figure_2_Execution_Time_Comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Figure 2 saved: {output_file}")
    plt.close()


def plot_algorithm_ranking(df, output_dir):
    """
    绘制算法性能排名图

    参数：
        df: 结果DataFrame
        output_dir: 输出目录
    """
    # 分离BSB和DSB
    bsb_df = df[df["algorithm_type"] == "BSB"]
    dsb_df = df[df["algorithm_type"] == "DSB"]

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # 子图1: BSB算法性能排名
    # 按平均质量和平均时间综合排名
    bsb_perf = (
        bsb_df.groupby("algorithm_name")
        .agg({"quality_percent": "mean", "execution_time_ms": "mean", "objective_value": "mean"})
        .reset_index()
    )

    # 计算综合得分（质量 * 0.7 - 时间归一化 * 0.3）
    bsb_perf["time_normalized"] = bsb_perf["execution_time_ms"] / bsb_perf["execution_time_ms"].max()
    bsb_perf["score"] = bsb_perf["quality_percent"] * 0.7 - bsb_perf["time_normalized"] * 30
    bsb_perf = bsb_perf.sort_values("score", ascending=False)

    x = np.arange(len(bsb_perf))
    width = 0.25

    axes[0].bar(x - width, bsb_perf["quality_percent"], width, label="Quality (%)", color="#2ecc71", alpha=0.8)
    axes[0].bar(
        x, bsb_perf["time_normalized"] * 100, width, label="Time (normalized × 100)", color="#e74c3c", alpha=0.8
    )
    axes[0].bar(x + width, bsb_perf["score"], width, label="Score", color="#9b59b6", alpha=0.8)

    axes[0].set_xlabel("BSB Algorithm", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Score / Value", fontsize=12, fontweight="bold")
    axes[0].set_title("BSB Algorithms: Performance Ranking", fontsize=14, fontweight="bold", pad=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bsb_perf["algorithm_name"], rotation=45, ha="right", fontsize=10)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)

    # 子图2: DSB算法性能排名
    dsb_perf = (
        dsb_df.groupby("algorithm_name")
        .agg({"quality_percent": "mean", "execution_time_ms": "mean", "objective_value": "mean"})
        .reset_index()
    )

    dsb_perf["time_normalized"] = dsb_perf["execution_time_ms"] / dsb_perf["execution_time_ms"].max()
    dsb_perf["score"] = dsb_perf["quality_percent"] * 0.7 - dsb_perf["time_normalized"] * 30
    dsb_perf = dsb_perf.sort_values("score", ascending=False)

    x = np.arange(len(dsb_perf))

    axes[1].bar(x - width, dsb_perf["quality_percent"], width, label="Quality (%)", color="#3498db", alpha=0.8)
    axes[1].bar(
        x, dsb_perf["time_normalized"] * 100, width, label="Time (normalized × 100)", color="#e74c3c", alpha=0.8
    )
    axes[1].bar(x + width, dsb_perf["score"], width, label="Score", color="#9b59b6", alpha=0.8)

    axes[1].set_xlabel("DSB Algorithm", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Score / Value", fontsize=12, fontweight="bold")
    axes[1].set_title("DSB Algorithms: Performance Ranking", fontsize=14, fontweight="bold", pad=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dsb_perf["algorithm_name"], rotation=45, ha="right", fontsize=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "Figure_3_Algorithm_Ranking.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Figure 3 saved: {output_file}")
    plt.close()


def plot_scatter_quality_vs_time(df, output_dir):
    """
    绘制质量-时间散点图

    参数：
        df: 结果DataFrame
        output_dir: 输出目录
    """
    # 分离BSB和DSB
    bsb_df = df[df["algorithm_type"] == "BSB"]
    dsb_df = df[df["algorithm_type"] == "DSB"]

    # 定义符号和颜色
    markers = ["o", "s", "^", "v", "D", "p"]  # 圆形、方形、三角形、倒三角、菱形、五边形
    colors = ["#27ae60", "#2ecc71", "#16a085", "#1abc9c", "#9b59b6", "#8e44ad"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 子图1: BSB
    bsb_algorithms = sorted(bsb_df["algorithm_name"].unique())
    for i, algo in enumerate(bsb_algorithms):
        algo_data = bsb_df[bsb_df["algorithm_name"] == algo]
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        axes[0].scatter(
            algo_data["execution_time_ms"],
            algo_data["quality_percent"],
            label=algo.replace("BSB_", ""),
            alpha=0.7,
            s=100,
            marker=marker,
            edgecolors="black",
            linewidth=1.5,
        )

    axes[0].axhline(y=99, color="red", linestyle="--", linewidth=2, alpha=0.7, label="99% Threshold")
    axes[0].set_xlabel("Execution Time (ms)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Solution Quality (%)", fontsize=12, fontweight="bold")
    axes[0].set_title("BSB: Quality vs Execution Time", fontsize=14, fontweight="bold", pad=10)
    axes[0].legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].grid(alpha=0.3)

    # 子图2: DSB
    dsb_algorithms = sorted(dsb_df["algorithm_name"].unique())
    for i, algo in enumerate(dsb_algorithms):
        algo_data = dsb_df[dsb_df["algorithm_name"] == algo]
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        axes[1].scatter(
            algo_data["execution_time_ms"],
            algo_data["quality_percent"],
            label=algo.replace("DSB_", ""),
            alpha=0.7,
            s=100,
            marker=marker,
            edgecolors="black",
            linewidth=1.5,
        )

    axes[1].axhline(y=99, color="red", linestyle="--", linewidth=2, alpha=0.7, label="99% Threshold")
    axes[1].set_xlabel("Execution Time (ms)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Solution Quality (%)", fontsize=12, fontweight="bold")
    axes[1].set_title("DSB: Quality vs Execution Time", fontsize=14, fontweight="bold", pad=10)
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "Figure_4_Quality_vs_Time_Scatter.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Figure 4 saved: {output_file}")
    plt.close()


# def plot_solution_density(df, output_dir):
#     """
#     绘制解密度分布图

#     参数：
#         df: 结果DataFrame
#         output_dir: 输出目录
#     """
#     # 分离BSB和DSB
#     bsb_df = df[df['algorithm_type'] == 'BSB']
#     dsb_df = df[df['algorithm_type'] == 'DSB']

#     fig, axes = plt.subplots(1, 2, figsize=(18, 8))

#     # 子图1: BSB
#     bsb_density = bsb_df.groupby('algorithm_name')['solution_density'].mean().sort_values()
#     colors_bsb = ['#27ae60' if d > 40 and d < 60 else '#e74c3c' for d in bsb_density.values]

#     bars = axes[0].barh(range(len(bsb_density)), bsb_density.values,
#                        color=colors_bsb, alpha=0.8, edgecolor='black', linewidth=1)
#     axes[0].set_yticks(range(len(bsb_density)))
#     axes[0].set_yticklabels(bsb_density.index, fontsize=10)
#     axes[0].set_xlabel('Solution Density (%)', fontsize=12, fontweight='bold')
#     axes[0].set_title('BSB Algorithms: Solution Density Distribution', fontsize=14, fontweight='bold', pad=10)
#     axes[0].axvline(x=50, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='50% Balanced')
#     axes[0].legend(fontsize=10)
#     axes[0].grid(axis='x', alpha=0.3)

#     # 添加数值标注
#     for i, (bar, val) in enumerate(zip(bars, bsb_density.values)):
#         axes[0].text(val + 2, i, f"{val:.1f}%", va='center', fontsize=9, fontweight='bold')

#     # 子图2: DSB
#     dsb_density = dsb_df.groupby('algorithm_name')['solution_density'].mean().sort_values()
#     colors_dsb = ['#3498db' if d > 40 and d < 60 else '#e74c3c' for d in dsb_density.values]

#     bars = axes[1].barh(range(len(dsb_density)), dsb_density.values,
#                        color=colors_dsb, alpha=0.8, edgecolor='black', linewidth=1)
#     axes[1].set_yticks(range(len(dsb_density)))
#     axes[1].set_yticklabels(dsb_density.index, fontsize=10)
#     axes[1].set_xlabel('Solution Density (%)', fontsize=12, fontweight='bold')
#     axes[1].set_title('DSB Algorithms: Solution Density Distribution', fontsize=14, fontweight='bold', pad=10)
#     axes[1].axvline(x=50, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='50% Balanced')
#     axes[1].legend(fontsize=10)
#     axes[1].grid(axis='x', alpha=0.3)

#     # 添加数值标注
#     for i, (bar, val) in enumerate(zip(bars, dsb_density.values)):
#         axes[1].text(val + 2, i, f"{val:.1f}%", va='center', fontsize=9, fontweight='bold')

#     plt.tight_layout()

#     output_file = output_dir / 'Figure_5_Solution_Density.png'
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     print(f"✅ Figure 5 saved: {output_file}")
#     plt.close()


# def plot_summary_statistics(df, output_dir):
#     """
#     绘制汇总统计图

#     参数：
#         df: 结果DataFrame
#         output_dir: 输出目录
#     """
#     # 计算统计指标
#     bsb_df = df[df['algorithm_type'] == 'BSB']
#     dsb_df = df[df['algorithm_type'] == 'DSB']

#     summary_data = {
#         'BSB': {
#             'Avg Quality (%)': bsb_df['quality_percent'].mean(),
#             'Quality Std (%)': bsb_df['quality_percent'].std(),
#             'Min Quality (%)': bsb_df['quality_percent'].min(),
#             'Max Quality (%)': bsb_df['quality_percent'].max(),
#             'Avg Time (ms)': bsb_df['execution_time_ms'].mean(),
#             'Time Std (ms)': bsb_df['execution_time_ms'].std(),
#             'Avg Density (%)': bsb_df['solution_density'].mean()
#         },
#         'DSB': {
#             'Avg Quality (%)': dsb_df['quality_percent'].mean(),
#             'Quality Std (%)': dsb_df['quality_percent'].std(),
#             'Min Quality (%)': dsb_df['quality_percent'].min(),
#             'Max Quality (%)': dsb_df['quality_percent'].max(),
#             'Avg Time (ms)': dsb_df['execution_time_ms'].mean(),
#             'Time Std (ms)': dsb_df['execution_time_ms'].std(),
#             'Avg Density (%)': dsb_df['solution_density'].mean()
#         }
#     }

#     # 绘制雷达图
#     fig = plt.figure(figsize=(16, 8))

#     # 标准化指标（值越大越好）
#     metrics = ['Avg Quality', 'Max Quality', 'Time Inverse', 'Balance']

#     # 归一化数据
#     def normalize(value, min_val, max_val):
#         if max_val == min_val:
#             return 1.0
#         return (value - min_val) / (max_val - min_val)

#     bsb_norm = [
#         normalize(summary_data['BSB']['Avg Quality (%)'], 95, 100),
#         normalize(summary_data['BSB']['Max Quality (%)'], 95, 100),
#         normalize(1000 / summary_data['BSB']['Avg Time (ms)'], 0, 100),
#         100 - abs(50 - summary_data['BSB']['Avg Density (%)'])  # 平衡度
#     ]

#     dsb_norm = [
#         normalize(summary_data['DSB']['Avg Quality (%)'], 95, 100),
#         normalize(summary_data['DSB']['Max Quality (%)'], 95, 100),
#         normalize(1000 / summary_data['DSB']['Avg Time (ms)'], 0, 100),
#         100 - abs(50 - summary_data['DSB']['Avg Density (%)'])
#     ]

#     # 绘制雷达图
#     angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
#     bsb_norm += bsb_norm[:1]
#     dsb_norm += dsb_norm[:1]
#     angles += angles[:1]

#     ax = plt.subplot(111, projection='polar')

#     ax.plot(angles, bsb_norm, 'o-', linewidth=2, label='BSB', color='#2ecc71')
#     ax.fill(angles, bsb_norm, alpha=0.25, color='#2ecc71')

#     ax.plot(angles, dsb_norm, 'o-', linewidth=2, label='DSB', color='#3498db')
#     ax.fill(angles, dsb_norm, alpha=0.25, color='#3498db')

#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
#     ax.set_ylim(0, 1)
#     ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
#     ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
#     ax.grid(True)
#     ax.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1.3, 1.1))

#     plt.title('BSB vs DSB: Comprehensive Performance Comparison', fontsize=16, fontweight='bold', pad=20)

#     plt.tight_layout()

#     output_file = output_dir / 'Figure_6_Summary_Radar.png'
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     print(f"✅ Figure 6 saved: {output_file}")
#     plt.close()

#     # 打印统计表格
#     print("\n📊 Summary Statistics:")
#     print("=" * 80)
#     for metric in summary_data['BSB'].keys():
#         bsb_val = summary_data['BSB'][metric]
#         dsb_val = summary_data['DSB'][metric]
#         print(f"{metric:25s}: BSB={bsb_val:.2f}, DSB={dsb_val:.2f}")


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

    # 加载数据（包含精确解）
    df = load_batch_csv(csv_file)
    print(f"✅ 加载数据: {len(df)} 条记录")
    print(f"   问题数: {df['problem_id'].nunique()}")
    print(f"   算法数: {df['algorithm_name'].nunique()}")
    print(f"   BSB算法: {df[df['algorithm_type'] == 'BSB']['algorithm_name'].nunique()} 种")
    print(f"   DSB算法: {df[df['algorithm_type'] == 'DSB']['algorithm_name'].nunique()} 种")

    # 显示精确解使用情况
    exact_count = df["exact_optimal"].notna().sum()
    exact_from_qplib = df["exact_optimal_from_qplib"].notna().sum()
    print(f"   精确解可用: {exact_count}/{len(df)} ({exact_count/len(df)*100:.1f}%)")
    print(f"   从qplib_results.csv加载: {exact_from_qplib}")

    # 创建输出目录
    output_dir = Path("save_pic")
    output_dir.mkdir(exist_ok=True)

    print(f"\n🎨 开始生成图表...")
    print("=" * 80)

    # 生成所有图表
    try:
        plot_quality_comparison(df, output_dir)
        plot_execution_time_comparison(df, output_dir)
        plot_algorithm_ranking(df, output_dir)
        plot_scatter_quality_vs_time(df, output_dir)

        print("\n" + "=" * 80)
        print(f"🎉 所有图表生成完成！")
        print(f"📁 输出目录: {output_dir}")
        print("=" * 80)

    except Exception as e:
        print(f"❌ 生成图表时出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
