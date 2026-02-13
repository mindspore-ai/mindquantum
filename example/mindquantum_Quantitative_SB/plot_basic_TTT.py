#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTT (Time to Target) 指标可视化

基于 SB_basic_ttt_data.py 生成的 tts_ttt_summary_*.csv 文件，
绘制 TTT 相关图表：
1. TTT vs TTS 对比图：展示 TTT 相比 TTS 的时间节省
2. TTT 成功率分析：展示各算法的成功率分布
3. TTT 排名图：展示算法在所有问题上的综合排名

数据特点：
- 可能有多个问题实例（23个或仅1个）
- 有12种量化算法（6 BSB + 6 DSB）
- TTT使用首次收敛时间（约TTS的60%）
- 包含成功率和时间节省比例
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime

# 设置中文字体和样式
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["font.family"] = "sans-serif"
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_tts_ttt_data(csv_file):
    """
    加载 TTS/TTT 数据 CSV 文件

    参数：
        csv_file: CSV 文件路径

    返回：
        DataFrame: TTS/TTT 结果数据
    """
    df = pd.read_csv(csv_file)
    print(f"✅ 加载 TTS/TTT 数据: {len(df)} 条记录")

    # 过滤掉 TTT 为无穷大的数据（可选）
    # df = df[df["ttt_seconds"] != float("inf")]

    return df


def plot_ttt_vs_tts_comparison(df, output_file):
    """
    绘制 TTT vs TTS 对比图

    参数：
        df: TTS/TTT 数据
        output_file: 输出文件路径
    """
    # 根据数据量调整布局
    num_problems = df["problem_id"].nunique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 散点图：TTT vs TTS
    valid_data = df[(df["ttt_seconds"] != float("inf")) & (df["tts_seconds"] != float("inf"))].copy()

    if not valid_data.empty:
        # 为 BSB 和 DSB 使用不同颜色
        bsb_data = valid_data[valid_data["algorithm_type"] == "BSB"]
        dsb_data = valid_data[valid_data["algorithm_type"] == "DSB"]

        ax1.scatter(bsb_data["tts_seconds"], bsb_data["ttt_seconds"],
                   alpha=0.7, s=150, label="BSB", edgecolors="darkblue", linewidth=1)
        ax1.scatter(dsb_data["tts_seconds"], dsb_data["ttt_seconds"],
                   alpha=0.7, s=150, label="DSB", edgecolors="darkred", linewidth=1)

        # 绘制 y=x 线
        max_time = max(valid_data["tts_seconds"].max(), valid_data["ttt_seconds"].max())
        ax1.plot([0, max_time], [0, max_time], 'k--', alpha=0.5, label="TTT = TTS", linewidth=2)

        # 添加算法标签（单问题时）
        if num_problems == 1:
            for _, row in valid_data.iterrows():
                ax1.annotate(row["algorithm_name"],
                            (row["tts_seconds"], row["ttt_seconds"]),
                            xytext=(5, 5), textcoords="offset points",
                            fontsize=7, alpha=0.7)

        ax1.set_xlabel("TTS (seconds)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("TTT (seconds)", fontsize=12, fontweight="bold")
        ax1.set_title("(a) TTT vs TTS Comparison", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 时间节省比例柱状图
    if not df.empty:
        time_ratio = df[(df["ttt_seconds"] != float("inf")) &
                       (df["tts_seconds"] != float("inf")) &
                       (df["tts_seconds"] > 0)].copy()

        if not time_ratio.empty:
            time_ratio["time_saved_percent"] = (1 - time_ratio["ttt_seconds"] / time_ratio["tts_seconds"]) * 100

            # 按算法分组
            if num_problems == 1:
                # 单问题：直接按算法名称排序
                algo_stats = time_ratio.sort_values("algorithm_name")
                x_pos = np.arange(len(algo_stats))
                colors = ["steelblue" if atype == "BSB" else "coral" for atype in algo_stats["algorithm_type"]]
                labels = algo_stats["algorithm_name"].tolist()
            else:
                # 多问题：按算法分组计算平均
                algo_stats = time_ratio.groupby(["algorithm_name", "algorithm_type"])["time_saved_percent"].agg(
                    ["mean", "std", "count"]
                ).reset_index().sort_values("algorithm_name")
                x_pos = np.arange(len(algo_stats))
                colors = ["steelblue" if atype == "BSB" else "coral" for atype in algo_stats["algorithm_type"]]
                labels = algo_stats["algorithm_name"].tolist()

            # 使用平均值或直接值
            y_values = algo_stats["mean"] if "mean" in algo_stats.columns else algo_stats["time_saved_percent"]
            y_err = algo_stats["std"] if "std" in algo_stats.columns else 0

            bars = ax2.bar(x_pos, y_values, yerr=y_err,
                          capsize=5, alpha=0.7, color=colors, edgecolor="black", linewidth=0.5)

            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([name.replace("_", "\n") for name in labels],
                               rotation=45, ha="right", fontsize=8)
            ax2.set_ylabel("Time Saved (%)", fontsize=12, fontweight="bold")
            ax2.set_title("(b) TTT Time Saving vs TTS", fontsize=14, fontweight="bold")
            ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
            ax2.grid(True, alpha=0.3, axis="y")

            # 添加数值标注
            for i, bar in enumerate(bars):
                height = y_values.iloc[i] if hasattr(y_values, 'iloc') else y_values[i]
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f"{height:.1f}%", ha="center", va="bottom" if height >= 0 else "top", fontsize=7)

            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor="steelblue", edgecolor="black", label="BSB"),
                             Patch(facecolor="coral", edgecolor="black", label="DSB")]
            ax2.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ Figure 2 saved: {output_file}")


def plot_ttt_success_rate(df, output_file):
    """
    绘制 TTT 成功率分析图

    参数：
        df: TTS/TTT 数据
        output_file: 输出文件路径
    """
    num_problems = df["problem_id"].nunique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 成功率柱状图
    success_df = df.copy()
    success_df["success_rate"] = success_df["ttt_success_count"] / success_df["tts_total_trials"] * 100

    if num_problems == 1:
        # 单问题：显示所有算法的成功率
        ax1.bar(range(len(success_df)), success_df["success_rate"],
                color=["steelblue" if atype == "BSB" else "coral" for atype in success_df["algorithm_type"]],
                alpha=0.7, edgecolor="black", linewidth=0.5)

        ax1.set_xticks(range(len(success_df)))
        ax1.set_xticklabels(success_df["algorithm_name"], rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("TTT Success Rate (%)", fontsize=12, fontweight="bold")
        ax1.set_title(f"(a) TTT Success Rate - {success_df.iloc[0]['problem_id']}", fontsize=14, fontweight="bold")
        ax1.set_ylim([0, 105])
        ax1.grid(True, alpha=0.3, axis="y")

        # 添加数值标注
        for i, (_, row) in enumerate(success_df.iterrows()):
            ax1.text(i, row["success_rate"] + 1, f"{row['success_rate']:.0f}%",
                    ha="center", va="bottom", fontsize=7)
    else:
        # 多问题：选择5个代表性实例
        problem_stats = success_df.groupby(["problem_id", "algorithm_type"])["success_rate"].mean().reset_index()
        pivot_data = problem_stats.pivot(index="problem_id", columns="algorithm_type", values="success_rate")

        if len(pivot_data) > 5:
            # 计算每个问题的平均成功率
            pivot_data["avg_success"] = pivot_data.mean(axis=1)
            pivot_data_sorted = pivot_data.sort_values("avg_success")
            indices = [0, len(pivot_data_sorted)//4, len(pivot_data_sorted)//2,
                      len(pivot_data_sorted)*3//4, len(pivot_data_sorted)-1]
            pivot_data = pivot_data_sorted.iloc[indices].drop(columns=["avg_success"])

        x_pos = np.arange(len(pivot_data))
        width = 0.35

        if "BSB" in pivot_data.columns:
            ax1.bar(x_pos - width/2, pivot_data["BSB"], width, label="BSB", color="steelblue", alpha=0.7)
        if "DSB" in pivot_data.columns:
            ax1.bar(x_pos + width/2, pivot_data["DSB"], width, label="DSB", color="coral", alpha=0.7)

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(pivot_data.index, rotation=45, ha="right", fontsize=10)
        ax1.set_ylabel("TTT Success Rate (%)", fontsize=12, fontweight="bold")
        ax1.set_title("(a) TTT Success Rate by Problem (5 Representative Instances)", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.set_ylim([0, 105])
        ax1.grid(True, alpha=0.3, axis="y")

    # 算法平均成功率
    algo_stats = success_df.groupby(["algorithm_name", "algorithm_type"])["success_rate"].agg(
        ["mean", "std", "count"]
    ).reset_index()

    algo_stats_sorted = algo_stats.sort_values("mean", ascending=False)
    x_pos_algo = np.arange(len(algo_stats_sorted))
    colors = ["steelblue" if atype == "BSB" else "coral" for atype in algo_stats_sorted["algorithm_type"]]

    bars = ax2.barh(x_pos_algo, algo_stats_sorted["mean"],
                   xerr=algo_stats_sorted["std"], capsize=3,
                   color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)

    ax2.set_yticks(x_pos_algo)
    ax2.set_yticklabels(algo_stats_sorted["algorithm_name"])
    ax2.set_xlabel("TTT Success Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_title("(b) TTT Success Rate by Algorithm", fontsize=14, fontweight="bold")
    ax2.set_xlim([0, 105])
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ Figure 3 saved: {output_file}")


def plot_ttt_ranking(df, output_file):
    """
    绘制 TTT 排名图

    参数：
        df: TTS/TTT 数据
        output_file: 输出文件路径
    """
    num_problems = df["problem_id"].nunique()
    fig, ax = plt.subplots(figsize=(14, 10))

    # 定义全部12种算法
    all_algorithms = [
        ("BSB_CPU_fp16", "BSB", "CPU", "fp16"),
        ("BSB_CPU_int8", "BSB", "CPU", "int8"),
        ("BSB_CPU_fp8", "BSB", "CPU", "fp8"),
        ("BSB_GPU_fp16", "BSB", "GPU", "fp16"),
        ("BSB_GPU_int8", "BSB", "GPU", "int8"),
        ("BSB_GPU_fp8", "BSB", "GPU", "fp8"),
        ("DSB_CPU_fp16", "DSB", "CPU", "fp16"),
        ("DSB_CPU_int8", "DSB", "CPU", "int8"),
        ("DSB_CPU_fp8", "DSB", "CPU", "fp8"),
        ("DSB_GPU_fp16", "DSB", "GPU", "fp16"),
        ("DSB_GPU_int8", "DSB", "GPU", "int8"),
        ("DSB_GPU_fp8", "DSB", "GPU", "fp8"),
    ]

    # 计算每个算法的平均 TTT（排除无穷大）
    valid_ttt = df[df["ttt_seconds"] != float("inf")].copy()

    # 初始化结果字典
    algo_results = {}
    for algo_name, algo_type, device, precision in all_algorithms:
        algo_results[algo_name] = {
            "algorithm_name": algo_name,
            "algorithm_type": algo_type,
            "device": device,
            "precision": precision,
            "mean": np.nan,
            "std": 0,
            "count": 0
        }

    if not valid_ttt.empty:
        # 计算每个算法的统计信息
        for algo_name, group in valid_ttt.groupby("algorithm_name"):
            if num_problems == 1:
                # 单问题：直接使用该值
                algo_results[algo_name]["mean"] = group["ttt_seconds"].iloc[0]
                algo_results[algo_name]["std"] = 0
                algo_results[algo_name]["count"] = 1
            else:
                # 多问题：计算平均值
                stats = group["ttt_seconds"].agg(["mean", "std", "count"])
                if algo_name in algo_results:
                    algo_results[algo_name]["mean"] = stats["mean"]
                    algo_results[algo_name]["std"] = stats["std"]
                    algo_results[algo_name]["count"] = int(stats["count"])

    # 转换为DataFrame
    algo_avg_ttt = pd.DataFrame([
        {
            "algorithm_name": algo_name,
            "algorithm_type": v["algorithm_type"],
            "device": v["device"],
            "precision": v["precision"],
            "mean": v["mean"],
            "std": v["std"],
            "count": v["count"]
        }
        for algo_name, v in algo_results.items()
    ])

    # 排序：有数据的按mean升序，没有数据的放到最后
    has_data = algo_avg_ttt[algo_avg_ttt["count"] > 0].copy()
    no_data = algo_avg_ttt[algo_avg_ttt["count"] == 0].copy()

    has_data_sorted = has_data.sort_values("mean", ascending=True)
    # 对于没有数据的算法，设置一个小的非零值以便绘制（显示为N/A）
    no_data["mean"] = 0.001  # 设置一个小的值以便绘制
    no_data["std"] = 0
    algo_avg_ttt = pd.concat([has_data_sorted, no_data], ignore_index=True)

    y_pos = np.arange(len(algo_avg_ttt))
    colors = ["steelblue" if atype == "BSB" else "coral" for atype in algo_avg_ttt["algorithm_type"]]

    # 对于没有数据的算法，使用灰色
    final_colors = []
    for i, (idx, row) in enumerate(algo_avg_ttt.iterrows()):
        if row["count"] == 0:
            final_colors.append("lightgray")
        else:
            final_colors.append("steelblue" if row["algorithm_type"] == "BSB" else "coral")

    bars = ax.barh(y_pos, algo_avg_ttt["mean"],
                  xerr=algo_avg_ttt["std"], capsize=3,
                  color=final_colors, alpha=0.7, edgecolor="black", linewidth=0.5)

    # 在柱状图右侧添加数值标注
    for i, (idx, row) in enumerate(algo_avg_ttt.iterrows()):
        if row["count"] > 0 and not np.isnan(row["mean"]):
            ax.text(row["mean"] * 1.05, y_pos[i], f"{row['mean']:.3f}s",
                   ha="left", va="center", fontsize=9, fontweight="bold")
        else:
            ax.text(0.01, y_pos[i], "N/A",
                   ha="left", va="center", fontsize=9, color="gray")

    ax.set_yticks(y_pos)
    # 增加横轴字体大小
    ax.set_yticklabels([f"{row['algorithm_name']} ({row['device']}, {row['precision']})"
                       for idx, row in algo_avg_ttt.iterrows()],
                      fontsize=9, fontweight="normal")
    ax.set_xlabel("Average TTT (seconds)", fontsize=12, fontweight="bold")
    title = f"TTT Algorithm Ranking ({num_problems} Problem{'s' if num_problems > 1 else ''}) - Lower is Better"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="steelblue", edgecolor="black", label="BSB"),
                     Patch(facecolor="coral", edgecolor="black", label="DSB")]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ Figure 4 saved: {output_file}")


def main():
    """主函数"""
    print("=" * 80)
    print("📊 TTT (Time to Target) 指标可视化")
    print("=" * 80)

    # 创建输出目录
    output_dir = Path("save_pic")
    output_dir.mkdir(exist_ok=True)

    # 查找最新的 TTS/TTT 数据文件
    csv_dir = Path("save_csv")
    ttt_files = sorted(csv_dir.glob("tts_ttt_summary_*.csv"))

    if not ttt_files:
        print("❌ 未找到 TTS/TTT 数据文件")
        print("   请先运行 SB_basic_ttt_data.py 生成数据")
        return

    latest_file = ttt_files[-1]
    print(f"\n📂 加载数据文件: {latest_file}")

    # 加载数据
    df = load_tts_ttt_data(latest_file)

    num_problems = df["problem_id"].nunique()
    num_algorithms = df["algorithm_name"].nunique()

    print(f"\n📊 数据统计:")
    print(f"  - 总记录数: {len(df)}")
    print(f"  - 算法数: {num_algorithms}")
    print(f"  - 问题数: {num_problems}")
    print(f"  - 问题列表: {', '.join(sorted(df['problem_id'].unique()))}")
    print(f"  - TTT 有效数据: {len(df[df['ttt_seconds'] != float('inf')])}")
    print(f"  - TTT 无穷大: {len(df[df['ttt_seconds'] == float('inf')])}")

    # 添加数据统计
    valid_ttt = df[df["ttt_seconds"] != float("inf")]
    if not valid_ttt.empty:
        print(f"\n📈 TTT 统计:")
        print(f"  - 最小值: {valid_ttt['ttt_seconds'].min():.4f}s")
        print(f"  - 最大值: {valid_ttt['ttt_seconds'].max():.4f}s")
        print(f"  - 平均值: {valid_ttt['ttt_seconds'].mean():.4f}s")
        print(f"  - 中位数: {valid_ttt['ttt_seconds'].median():.4f}s")

    # 生成图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n🎨 生成图表...")

    # 图1: TTT vs TTS 对比
    comparison_file = output_dir / f"Figure_TTS_TTT_Comparison_{timestamp}.png"
    plot_ttt_vs_tts_comparison(df, comparison_file)

    # 图2: 成功率分析
    success_file = output_dir / f"Figure_TTT_Success_Rate_{timestamp}.png"
    plot_ttt_success_rate(df, success_file)

    # 图3: 排名图
    ranking_file = output_dir / f"Figure_TTT_Ranking_{timestamp}.png"
    plot_ttt_ranking(df, ranking_file)

    print(f"\n{'='*80}")
    print(f"🎉 可视化完成！")
    print(f"{'='*80}")
    print(f"生成的图表:")
    print(f"  1. {comparison_file.name}")
    print(f"  2. {success_file.name}")
    print(f"  3. {ranking_file.name}")


if __name__ == "__main__":
    main()
