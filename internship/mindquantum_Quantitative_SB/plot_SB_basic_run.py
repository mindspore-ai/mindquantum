"""
可视化 BSB/DSB 算法在 QPLIB_3506 问题上的性能对比
基于 save_csv/qplib_results_QPLIB_3506.csv 生成
改进版：修复中文字体显示问题，优化图表样式
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib
import warnings

# 抑制字体警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 创建 save_pic 文件夹
Path("save_pic").mkdir(exist_ok=True)


# =============================================================================
# 设置中文字体（多平台支持）
# =============================================================================
def setup_chinese_font():
    """设置中文字体，支持Windows/Linux/Mac"""
    platform_fonts = {
        "Windows": ["Microsoft YaHei", "SimHei", "KaiTi", "FangSong"],
        "Linux": ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Droid Sans Fallback"],
        "Darwin": ["PingFang SC", "Heiti TC", "STHeiti", "Arial Unicode MS"],
    }

    import platform as pltf

    system = pltf.system()

    if system in platform_fonts:
        fonts = platform_fonts[system]
        plt.rcParams["font.sans-serif"] = fonts + ["Arial", "DejaVu Sans"]
    else:
        # 默认字体列表
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Arial Unicode MS",
            "DejaVu Sans",
            "WenQuanYi Micro Hei",
            "Droid Sans Fallback",
        ]

    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 测试字体是否可用
    test_chinese_font()


def test_chinese_font():
    """测试中文字体是否可用"""
    test_text = "测试中文"
    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.text(0, 0, test_text)
    try:
        fig.savefig("temp_font_test.png", bbox_inches="tight")
        import os

        if os.path.exists("temp_font_test.png"):
            os.remove("temp_font_test.png")
        plt.close(fig)
        print("✅ 中文字体设置成功")
    except:
        print("⚠️ 中文字体设置可能有问题，将使用备用方案")
        plt.close(fig)
        return False
    return True


# 执行字体设置
setup_chinese_font()

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

# 颜色主题
COLORS = {
    "high_quality": "#2ecc71",  # 绿色
    "medium_quality": "#f39c12",  # 橙色
    "low_quality": "#e74c3c",  # 红色
    "cpu": "#3498db",  # 蓝色
    "gpu": "#e74c3c",  # 红色
    "bsb": "#3498db",  # 蓝色
    "dsb": "#9b59b6",  # 紫色
}

# 读取CSV文件
csv_file = "save_csv/qplib_results_QPLIB_3506.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"❌ 错误: 找不到文件 {csv_file}")
    exit(1)

print("=" * 80)
print("Data Loading Successful!")
print(f"Problem Name: {df['problem_name'].iloc[0]}")
print(f"Number of Variables: {df['num_vars'].iloc[0]}")
print(f"Exact Optimal: {df['exact_optimal'].iloc[0]}")
print(f"Objective Sense: {df['objective_sense'].iloc[0]}")
print(f"Number of Algorithms: {len(df)}")
print("=" * 80)

# 分离BSB和DSB结果
bsb_df = df[df["algorithm_type"] == "BSB"].copy()
dsb_df = df[df["algorithm_type"] == "DSB"].copy()

# 计算质量百分比（相对于精确最优解）
bsb_df["quality_percent"] = bsb_df["energy_to_optimal_ratio"] * 100
dsb_df["quality_percent"] = dsb_df["energy_to_optimal_ratio"] * 100

# =============================================================================
# 图1: 求解质量对比（水平柱状图）
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# BSB求解质量
colors_bsb = [
    COLORS["high_quality"] if q >= 99 else COLORS["low_quality"] if q < 95 else COLORS["medium_quality"]
    for q in bsb_df["quality_percent"]
]
bars1 = axes[0].barh(
    bsb_df["algorithm_name"], bsb_df["quality_percent"], color=colors_bsb, alpha=0.8, edgecolor="black", linewidth=1.2
)
axes[0].set_xlabel("Solution Quality (%)", fontsize=13, fontweight="bold")
axes[0].set_title("BSB Algorithm Solution Quality", fontsize=15, fontweight="bold", pad=15)
axes[0].set_xlim(95, 101)
axes[0].axvline(x=100, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Exact Optimal")
axes[0].axvline(x=99, color="orange", linestyle="--", linewidth=1.5, alpha=0.6, label="99% Threshold")
axes[0].legend(fontsize=11, loc="lower right")
axes[0].grid(axis="x", alpha=0.3)

# 在柱状图上显示数值
for i, (bar, val) in enumerate(zip(bars1, bsb_df["quality_percent"])):
    text_color = "white" if val < 98 else "black"
    axes[0].text(
        val - 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}%",
        ha="right",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=text_color,
    )

# DSB求解质量
colors_dsb = [
    COLORS["high_quality"] if q >= 99 else COLORS["low_quality"] if q < 95 else COLORS["medium_quality"]
    for q in dsb_df["quality_percent"]
]
bars2 = axes[1].barh(
    dsb_df["algorithm_name"], dsb_df["quality_percent"], color=colors_dsb, alpha=0.8, edgecolor="black", linewidth=1.2
)
axes[1].set_xlabel("Solution Quality (%)", fontsize=13, fontweight="bold")
axes[1].set_title("DSB Algorithm Solution Quality", fontsize=15, fontweight="bold", pad=15)
axes[1].set_xlim(95, 101)
axes[1].axvline(x=100, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Exact Optimal")
axes[1].axvline(x=99, color="orange", linestyle="--", linewidth=1.5, alpha=0.6, label="99% Threshold")
axes[1].legend(fontsize=11, loc="lower right")
axes[1].grid(axis="x", alpha=0.3)

# 在柱状图上显示数值
for i, (bar, val) in enumerate(zip(bars2, dsb_df["quality_percent"])):
    text_color = "white" if val < 98 else "black"
    axes[1].text(
        val - 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}%",
        ha="right",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=text_color,
    )

plt.tight_layout()
plt.savefig("save_pic/QPLIB_3506_quality_comparison.png", bbox_inches="tight", facecolor="white")
print("✅ Figure 1 saved: save_pic/QPLIB_3506_quality_comparison.png")
plt.close()

# =============================================================================
# 图2: 求解质量 vs 执行时间（散点图）
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# BSB散点图
for idx, row in bsb_df.iterrows():
    color = (
        COLORS["high_quality"]
        if row["quality_percent"] >= 99
        else COLORS["low_quality"] if row["quality_percent"] < 95 else COLORS["medium_quality"]
    )
    marker_size = 300 if row["quality_percent"] >= 99 else 150
    axes[0].scatter(
        row["execution_time_ms"],
        row["quality_percent"],
        s=marker_size,
        c=color,
        alpha=0.7,
        edgecolors="black",
        linewidth=1.5,
        label=row["algorithm_name"],
    )

    # 添加标签
    axes[0].annotate(
        row["algorithm_name"],
        (row["execution_time_ms"], row["quality_percent"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="bottom",
    )

axes[0].set_xlabel("Execution Time (ms)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Solution Quality (%)", fontsize=13, fontweight="bold")
axes[0].set_title("BSB: Solution Quality vs Execution Time", fontsize=15, fontweight="bold", pad=15)
axes[0].axhline(y=99, color="orange", linestyle="--", linewidth=2, alpha=0.7)
axes[0].axhline(y=100, color="red", linestyle="--", linewidth=2, alpha=0.7)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, max(bsb_df["execution_time_ms"]) * 1.1)

# DSB散点图
for idx, row in dsb_df.iterrows():
    color = (
        COLORS["high_quality"]
        if row["quality_percent"] >= 99
        else COLORS["low_quality"] if row["quality_percent"] < 95 else COLORS["medium_quality"]
    )
    marker_size = 300 if row["quality_percent"] >= 99 else 150
    axes[1].scatter(
        row["execution_time_ms"],
        row["quality_percent"],
        s=marker_size,
        c=color,
        alpha=0.7,
        edgecolors="black",
        linewidth=1.5,
        label=row["algorithm_name"],
    )

    # 添加标签
    axes[1].annotate(
        row["algorithm_name"],
        (row["execution_time_ms"], row["quality_percent"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="bottom",
    )

axes[1].set_xlabel("Execution Time (ms)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Solution Quality (%)", fontsize=13, fontweight="bold")
axes[1].set_title("DSB: Solution Quality vs Execution Time", fontsize=15, fontweight="bold", pad=15)
axes[1].axhline(y=99, color="orange", linestyle="--", linewidth=2, alpha=0.7)
axes[1].axhline(y=100, color="red", linestyle="--", linewidth=2, alpha=0.7)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, max(dsb_df["execution_time_ms"]) * 1.1)

plt.tight_layout()
plt.savefig("save_pic/QPLIB_3506_quality_time_scatter.png", bbox_inches="tight", facecolor="white")
print("✅ Figure 2 saved: save_pic/QPLIB_3506_quality_time_scatter.png")
plt.close()

# =============================================================================
# 图3: CPU vs GPU 性能对比（按精度分组）
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# BSB CPU vs GPU
precision_order = ["fp16", "int8", "fp8"]
cpu_bsbs = [
    (
        bsb_df[(bsb_df["device"] == "CPU") & (bsb_df["precision"] == prec)]["execution_time_ms"].values[0]
        if len(bsb_df[(bsb_df["device"] == "CPU") & (bsb_df["precision"] == prec)]) > 0
        else np.nan
    )
    for prec in precision_order
]
gpu_bsbs = [
    (
        bsb_df[(bsb_df["device"] == "GPU") & (bsb_df["precision"] == prec)]["execution_time_ms"].values[0]
        if len(bsb_df[(bsb_df["device"] == "GPU") & (bsb_df["precision"] == prec)]) > 0
        else np.nan
    )
    for prec in precision_order
]

x = np.arange(len(precision_order))
width = 0.35

bars1 = axes[0].bar(
    x - width / 2, cpu_bsbs, width, label="CPU", color=COLORS["cpu"], alpha=0.8, edgecolor="black", linewidth=1.2
)
bars2 = axes[0].bar(
    x + width / 2, gpu_bsbs, width, label="GPU", color=COLORS["gpu"], alpha=0.8, edgecolor="black", linewidth=1.2
)

axes[0].set_xlabel("Data Precision", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Execution Time (ms)", fontsize=13, fontweight="bold")
axes[0].set_title("BSB: CPU vs GPU Performance Comparison", fontsize=15, fontweight="bold", pad=15)
axes[0].set_xticks(x)
axes[0].set_xticklabels([p.upper() for p in precision_order], fontsize=11, fontweight="bold")
axes[0].legend(fontsize=12, loc="upper right")
axes[0].grid(axis="y", alpha=0.3)

# 添加加速比标注
for i, (cpu, gpu) in enumerate(zip(cpu_bsbs, gpu_bsbs)):
    if not np.isnan(cpu) and not np.isnan(gpu):
        speedup = cpu / gpu
        # 标注在GPU柱子上方
        axes[0].annotate(
            f"{speedup:.1f}x",
            xy=(i + width / 2, gpu),
            xytext=(0, 20),
            textcoords="offset points",
            fontsize=11,
            ha="center",
            va="bottom",
            fontweight="bold",
            color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7, edgecolor="darkred"),
        )

# DSB CPU vs GPU
cpu_dsbs = [
    (
        dsb_df[(dsb_df["device"] == "CPU") & (dsb_df["precision"] == prec)]["execution_time_ms"].values[0]
        if len(dsb_df[(dsb_df["device"] == "CPU") & (dsb_df["precision"] == prec)]) > 0
        else np.nan
    )
    for prec in precision_order
]
gpu_dsbs = [
    (
        dsb_df[(dsb_df["device"] == "GPU") & (dsb_df["precision"] == prec)]["execution_time_ms"].values[0]
        if len(dsb_df[(dsb_df["device"] == "GPU") & (dsb_df["precision"] == prec)]) > 0
        else np.nan
    )
    for prec in precision_order
]

bars3 = axes[1].bar(
    x - width / 2, cpu_dsbs, width, label="CPU", color=COLORS["cpu"], alpha=0.8, edgecolor="black", linewidth=1.2
)
bars4 = axes[1].bar(
    x + width / 2, gpu_dsbs, width, label="GPU", color=COLORS["gpu"], alpha=0.8, edgecolor="black", linewidth=1.2
)

axes[1].set_xlabel("Precision Type", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Execution Time (ms)", fontsize=13, fontweight="bold")
axes[1].set_title("DSB: CPU vs GPU Performance Comparison", fontsize=15, fontweight="bold", pad=15)
axes[1].set_xticks(x)
axes[1].set_xticklabels([p.upper() for p in precision_order], fontsize=11, fontweight="bold")
axes[1].legend(fontsize=12, loc="upper right")
axes[1].grid(axis="y", alpha=0.3)

# 添加加速比标注
for i, (cpu, gpu) in enumerate(zip(cpu_dsbs, gpu_dsbs)):
    if not np.isnan(cpu) and not np.isnan(gpu):
        speedup = cpu / gpu
        # 标注在GPU柱子上方
        axes[1].annotate(
            f"{speedup:.1f}x",
            xy=(i + width / 2, gpu),
            xytext=(0, 20),
            textcoords="offset points",
            fontsize=11,
            ha="center",
            va="bottom",
            fontweight="bold",
            color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7, edgecolor="darkred"),
        )

plt.tight_layout()
plt.savefig("save_pic/QPLIB_3506_cpu_gpu_comparison.png", bbox_inches="tight", facecolor="white")
print("✅ Figure 3 saved: save_pic/QPLIB_3506_cpu_gpu_comparison.png")
plt.close()

# =============================================================================
# 图4: 求解质量 vs 精度类型（分组柱状图）
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# 准备数据 - 按精度顺序提取数据，确保与X轴标签一致
bsb_qualities_list = []
dsb_qualities_list = []
for prec in precision_order:
    bsb_avg = bsb_df[bsb_df["precision"] == prec]["quality_percent"].mean()
    dsb_avg = dsb_df[dsb_df["precision"] == prec]["quality_percent"].mean()
    bsb_qualities_list.append(bsb_avg)
    dsb_qualities_list.append(dsb_avg)

x = np.arange(len(precision_order))
width = 0.35

bars1 = ax.bar(
    x - width / 2,
    bsb_qualities_list,
    width,
    label="BSB",
    color=COLORS["bsb"],
    alpha=0.8,
    edgecolor="black",
    linewidth=1.2,
)
bars2 = ax.bar(
    x + width / 2,
    dsb_qualities_list,
    width,
    label="DSB",
    color=COLORS["dsb"],
    alpha=0.8,
    edgecolor="black",
    linewidth=1.2,
)

ax.set_xlabel("Data Precision", fontsize=13, fontweight="bold")
ax.set_ylabel("Average Solution Quality (%)", fontsize=13, fontweight="bold")
ax.set_title("BSB vs DSB: Solution Quality by Precision", fontsize=15, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels([p.upper() for p in precision_order], fontsize=11, fontweight="bold")
ax.legend(fontsize=12, loc="lower right")
ax.axhline(y=99, color="orange", linestyle="--", linewidth=2, alpha=0.7, label="99% Threshold")
ax.axhline(y=100, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Exact Optimal")
ax.set_ylim(95, 101)
ax.grid(axis="y", alpha=0.3)

# 在柱状图上显示数值
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height - 0.3,
            f"{height:.2f}%",
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold",
        )

plt.tight_layout()
plt.savefig("save_pic/QPLIB_3506_precision_quality_comparison.png", bbox_inches="tight", facecolor="white")
print("✅ Figure 4 saved: save_pic/QPLIB_3506_precision_quality_comparison.png")
plt.close()

# =============================================================================
# 图5: 解密度分布对比
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# BSB解密度 - 重置索引以确保位置正确
bsb_df_reset = bsb_df.reset_index(drop=True)
for idx in range(len(bsb_df_reset)):
    row = bsb_df_reset.iloc[idx]
    color = (
        COLORS["high_quality"]
        if row["quality_percent"] >= 99
        else COLORS["low_quality"] if row["quality_percent"] < 95 else COLORS["medium_quality"]
    )
    # 如果solution_density为0，使用灰色表示异常
    if row["solution_density"] == 0:
        color = "#95a5a6"  # 灰色
    axes[0].barh(idx, row["solution_density"], color=color, alpha=0.8, edgecolor="black", linewidth=1.2, height=0.6)
    # 添加数值标注
    axes[0].text(
        row["solution_density"] + 2,
        idx,
        f"{row['solution_density']:.1f}%",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

axes[0].set_yticks(range(len(bsb_df_reset)))
axes[0].set_yticklabels(bsb_df_reset["algorithm_name"], fontsize=9.5)
axes[0].set_xlabel("Solution Density (%)", fontsize=13, fontweight="bold")
axes[0].set_title("BSB: Solution Density Distribution", fontsize=15, fontweight="bold", pad=15)
axes[0].axvline(x=50, color="blue", linestyle="--", linewidth=2, alpha=0.6, label="50% Balanced Line")
axes[0].legend(fontsize=10, loc="upper right")
axes[0].grid(axis="x", alpha=0.3)
axes[0].set_xlim(0, 100)

# 添加异常数据说明
if (bsb_df_reset["solution_density"] == 0).any():
    axes[0].text(
        0.98,
        0.02,
        "Gray bar: Invalid solution data\n(solution_ones=solution_zeros=0)",
        transform=axes[0].transAxes,
        fontsize=8,
        color="darkred",
        style="italic",
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

# DSB解密度 - 重置索引以确保位置正确
dsb_df_reset = dsb_df.reset_index(drop=True)
for idx in range(len(dsb_df_reset)):
    row = dsb_df_reset.iloc[idx]
    color = (
        COLORS["high_quality"]
        if row["quality_percent"] >= 99
        else COLORS["low_quality"] if row["quality_percent"] < 95 else COLORS["medium_quality"]
    )
    axes[1].barh(idx, row["solution_density"], color=color, alpha=0.8, edgecolor="black", linewidth=1.2, height=0.6)
    # 添加数值标注
    axes[1].text(
        row["solution_density"] + 2,
        idx,
        f"{row['solution_density']:.1f}%",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

axes[1].set_yticks(range(len(dsb_df_reset)))
axes[1].set_yticklabels(dsb_df_reset["algorithm_name"], fontsize=9.5)
axes[1].set_xlabel("Solution Density (%)", fontsize=13, fontweight="bold")
axes[1].set_title("DSB: Solution Density Distribution", fontsize=15, fontweight="bold", pad=15)
axes[1].axvline(x=50, color="blue", linestyle="--", linewidth=2, alpha=0.6, label="50% Balanced Line")
axes[1].legend(fontsize=10, loc="upper right")
axes[1].grid(axis="x", alpha=0.3)
axes[1].set_xlim(0, 100)

plt.tight_layout()
plt.savefig("save_pic/QPLIB_3506_solution_density.png", bbox_inches="tight", facecolor="white")
print("✅ Figure 5 saved: save_pic/QPLIB_3506_solution_density.png")
plt.close()

# =============================================================================
# 生成汇总报告
# =============================================================================
print("\n" + "=" * 80)
print("Performance Summary Report")
print("=" * 80)

# BSB汇总
print("\n[BSB Algorithm Summary]")
best_quality_bsb = bsb_df.loc[bsb_df["quality_percent"].idxmax()]
fastest_bsb = bsb_df.loc[bsb_df["execution_time_ms"].idxmin()]
best_quality_99_bsb = bsb_df[bsb_df["quality_percent"] >= 99].nsmallest(1, "execution_time_ms")

print(f"  Best Quality: {best_quality_bsb['algorithm_name']} ({best_quality_bsb['quality_percent']:.2f}%)")
print(f"  Fastest Speed: {fastest_bsb['algorithm_name']} ({fastest_bsb['execution_time_ms']:.2f}ms)")
if len(best_quality_99_bsb) > 0:
    print(
        f"  Fastest 99% Quality: {best_quality_99_bsb.iloc[0]['algorithm_name']} ({best_quality_99_bsb.iloc[0]['execution_time_ms']:.2f}ms)"
    )
else:
    print(f"  ⚠️ No BSB algorithm reached 99% solution quality")

# DSB汇总
print("\n[DSB Algorithm Summary]")
best_quality_dsb = dsb_df.loc[dsb_df["quality_percent"].idxmax()]
fastest_dsb = dsb_df.loc[dsb_df["execution_time_ms"].idxmin()]
best_quality_99_dsb = dsb_df[dsb_df["quality_percent"] >= 99].nsmallest(1, "execution_time_ms")

print(f"  Best Quality: {best_quality_dsb['algorithm_name']} ({best_quality_dsb['quality_percent']:.2f}%)")
print(f"  Fastest Speed: {fastest_dsb['algorithm_name']} ({fastest_dsb['execution_time_ms']:.2f}ms)")
if len(best_quality_99_dsb) > 0:
    print(
        f"  Fastest 99% Quality: {best_quality_99_dsb.iloc[0]['algorithm_name']} ({best_quality_99_dsb.iloc[0]['execution_time_ms']:.2f}ms)"
    )
else:
    print(f"  ⚠️ No DSB algorithm reached 99% solution quality")

# GPU加速效果
print("\n[GPU Acceleration Effect]")
avg_cpu_bsb = bsb_df[bsb_df["device"] == "CPU"]["execution_time_ms"].mean()
avg_gpu_bsb = bsb_df[bsb_df["device"] == "GPU"]["execution_time_ms"].mean()
avg_cpu_dsb = dsb_df[dsb_df["device"] == "CPU"]["execution_time_ms"].mean()
avg_gpu_dsb = dsb_df[dsb_df["device"] == "GPU"]["execution_time_ms"].mean()

print(f"  BSB Average Speedup: {avg_cpu_bsb / avg_gpu_bsb:.2f}x")
print(f"  DSB Average Speedup: {avg_cpu_dsb / avg_gpu_dsb:.2f}x")
print(f"  Overall Average Speedup: {(avg_cpu_bsb + avg_cpu_dsb) / (avg_gpu_bsb + avg_gpu_dsb):.2f}x")

# 精度对比
print("\n[Precision Type Comparison]")
for prec in precision_order:
    bsb_avg = bsb_df[bsb_df["precision"] == prec]["quality_percent"].mean()
    dsb_avg = dsb_df[dsb_df["precision"] == prec]["quality_percent"].mean()
    print(f"  {prec.upper()}: BSB={bsb_avg:.2f}%, DSB={dsb_avg:.2f}%")

print("=" * 80)
print("🎉 All charts have been generated and saved to save_pic/ directory")
print("=" * 80)
