# Plot_fig5.py

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed

# 导入所有需要的核心函数
from SDP import GW_algorithm, maxcut_interior_point

# ==============================================================================
# 1.工作单元函数
# ==============================================================================
def run_full_r_sweep_for_one_graph(seed, config):
    """
    对一个新生成的图，执行一次包含所有投影的计算，复用结果。
    """
    n = config['n']
    R_values = config['R_values']
    
    print(f"  > 开始处理第 {seed+1}/{config['realizations']} 个独立实验...")

    # a. 生成一个图并只求解一次SDP
    graph = nx.random_regular_graph(3, n, seed=seed)
    num_edges = graph.number_of_edges()
    pro_info = np.array([[u, v, 1.0] for u, v in graph.edges()])
    _, _, X_opt = maxcut_interior_point(pro_info)
    X_opt_final = X_opt[-1]

    # b. ---一次性执行最大次数的投影 ---
    max_r = max(R_values)
    max_projections = 1 + max_r
    
    print(f"    - 正在为 Exp #{seed+1} 执行 {max_projections} 次投影...")
    np.random.seed(seed) # 固定随机种子以保证可复现性
    v_ran = np.random.uniform(-1, 1, (max_projections, n))
    # GW_algorithm的第三个返回值E是包含了每一次投影能量的列表
    _, _, all_single_projection_energies = GW_algorithm(pro_info, X_opt_final, v_ran)

    # c. --- 通过切片和求最大值来复用结果 ---
    gw_ratios_for_this_graph = []
    print(f"    - 正在为 Exp #{seed+1} 生成所有R值的结果...")
    for r_extra in R_values:
        total_projections = 1 + r_extra
        # 从所有投影结果中，取出前 total_projections 个，并找到其中的最大值
        best_energy_for_this_r = np.max(all_single_projection_energies[:total_projections])
        gw_ratios_for_this_graph.append(best_energy_for_this_r / num_edges)

    print(f"  > 第 {seed+1}/{config['realizations']} 个独立实验完成。")
    return gw_ratios_for_this_graph

# ==============================================================================
# --- 控制与配置 ---
# ==============================================================================
RECALCULATE = False
DATA_FILENAME = 'data/figure5_data.npz'
FIG3_DATA_FILENAME = 'data/figure3_data.npz' 

# ==============================================================================
# --- 模拟参数设置 ---
# ==============================================================================
if RECALCULATE:
    config = {
        'node_sizes': [200,400,600,800,1000], 
        'realizations': 40,
        # R值列表
        'R_values': [0, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
    }

# ==============================================================================
# --- 核心计算逻辑  ---
# ==============================================================================
if RECALCULATE:
    print("模式: 重新计算数据...")
    
    # --- 1. 检查并加载已有进度 ---
    plot_data = defaultdict(dict)
    nodes_to_process = config['node_sizes'] # 默认处理所有n
    
    if os.path.exists(DATA_FILENAME):
        print(f"发现已有的进度文件: {DATA_FILENAME}，尝试恢复...")
        try:
            data = np.load(DATA_FILENAME)
            # 检查哪些n已经完成
            nodes_already_done = []
            for n_check in config['node_sizes']:
                # 如果一个n的结果存在于文件中，我们就认为它已完成
                if f'n_{n_check}_gw' in data:
                    nodes_already_done.append(n_check)
                    # 将已有的数据加载到plot_data中
                    plot_data[n_check]['R'] = data['R_values']
                    plot_data[n_check]['gw_ratios'] = data[f'n_{n_check}_gw']
                    plot_data[n_check]['qaoa_baseline'] = data[f'n_{n_check}_qaoa_baseline'].item()

            if nodes_already_done:
                print(f"已完成的n: {nodes_already_done}")
                # 从总任务列表中移除已完成的n
                nodes_to_process = [n for n in config['node_sizes'] if n not in nodes_already_done]
            
            if not nodes_to_process:
                print("所有n均已计算完成，无需重新计算。")
            else:
                 print(f"将要处理的n: {nodes_to_process}")

        except Exception as e:
            print(f"加载进度文件失败: {e}。将重新开始计算所有任务。")
            plot_data = defaultdict(dict) # 重置
            nodes_to_process = config['node_sizes']

    # --- 2. 加载QAOA基线数据 ---
    print(f"\n--- 正在从 {FIG3_DATA_FILENAME} 加载QAOA基线数据 ---")
    if not os.path.exists(FIG3_DATA_FILENAME):
        print(f"错误: 依赖的数据文件 {FIG3_DATA_FILENAME} 不存在。")
        exit()
    fig3_data = np.load(FIG3_DATA_FILENAME)
    qaoa_baselines = {n: ratio for n, ratio in zip(fig3_data['node_sizes'], fig3_data['avg_bias_ratios'])}
    print("QAOA基线数据加载成功。")

    # --- 3. 循环处理剩余的n值 ---
    if nodes_to_process:
        start_time = time.time()
        for n in nodes_to_process:
            n_start_time = time.time()
            print(f"\n--- 开始并行处理 n={n} 的 {config['realizations']} 个独立实验 ---")
            
            temp_config = config.copy()
            temp_config['n'] = n
            
            all_results_for_n = Parallel(n_jobs=2)( # 限制并行数以防止内存问题
                delayed(run_full_r_sweep_for_one_graph)(i, temp_config)
                for i in tqdm(range(config['realizations']), desc=f"Processing n={n}")
            )
            
            all_gw_ratios_np = np.array(all_results_for_n)
            avg_gw_ratios = np.mean(all_gw_ratios_np, axis=0)
            
            plot_data[n]['R'] = config['R_values']
            plot_data[n]['gw_ratios'] = avg_gw_ratios
            if n in qaoa_baselines:
                plot_data[n]['qaoa_baseline'] = qaoa_baselines[n]
            else:
                plot_data[n]['qaoa_baseline'] = 0
            
            n_total_time = time.time() - n_start_time
            print(f"--- n={n} 计算完成，耗时: {n_total_time:.2f} 秒 ---")

            # --- 4. 增量保存数据 ---
            print(f"--- 正在增量保存进度到 {DATA_FILENAME} ---")
            save_dict = {'R_values': config['R_values'], 'node_sizes': config['node_sizes']}
            for existing_n in plot_data:
                if 'gw_ratios' in plot_data[existing_n]:
                    save_dict[f'n_{existing_n}_gw'] = plot_data[existing_n]['gw_ratios']
                    save_dict[f'n_{existing_n}_qaoa_baseline'] = plot_data[existing_n]['qaoa_baseline']
            np.savez(DATA_FILENAME, **save_dict)
            print("--- 进度保存完毕 ---")
            
        total_time = time.time() - start_time
        print(f"\n本次运行计算完成，总耗时: {total_time:.2f} 秒")

# ==============================================================================
# --- 加载与绘图部分 ---
# ==============================================================================
else:
    print(f"模式: 从文件 {DATA_FILENAME} 加载数据...")
    if not os.path.exists(DATA_FILENAME):
        print(f"错误: 数据文件 {DATA_FILENAME} 不存在。")
        exit()
    data = np.load(DATA_FILENAME)
    plot_data = defaultdict(dict)
    config = {'node_sizes': data['node_sizes']}
    R_values_loaded = data['R_values']
    for n in config['node_sizes']:
        plot_data[n]['R'] = R_values_loaded
        plot_data[n]['gw_ratios'] = data[f'n_{n}_gw']
        plot_data[n]['qaoa_baseline'] = data[f'n_{n}_qaoa_baseline'].item()
    print("数据加载成功。")

print("开始绘制图形...")

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(config['node_sizes'])))

for i, n in enumerate(config['node_sizes']):
    r_axis = plot_data[n]['R']
    r_axis_plot = [r if r > 0 else 0.5 for r in r_axis]

    ax.plot(r_axis_plot, plot_data[n]['gw_ratios'], marker='o', linestyle='-', color=colors[i], label=f'GW, n={n}')
    qaoa_level = plot_data[n]['qaoa_baseline']
    ax.axhline(y=qaoa_level, color=colors[i], linestyle='--', label=f'WS-ab-QAOA (R_gw=1), n={n}')

ax.set_xscale('log')
ax.set_xlabel('$R$ Rounding number', fontsize=14)
ax.set_ylabel('$N_{cut}^{alg} / |E|$', fontsize=14)
ax.set_title('Efficiency v.s. Rounding times', fontsize=16)
ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, which="both", ls="-")
plt.savefig('figure5.png', dpi=300, bbox_inches='tight')

print("绘图完成。")