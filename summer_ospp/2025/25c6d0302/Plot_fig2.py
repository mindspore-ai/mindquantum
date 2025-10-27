# reproduce_figure2.py
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import time
import os
import glob
from collections import defaultdict
from joblib import Parallel, delayed

# 导入所有需要的核心函数
from SDP import GW_algorithm, maxcut_interior_point
from BMMK import BMMK_optR
from Adaptive_para import AdaptiveQaoaInput, adaptive_qaoa

# ==============================================================================
# 1. 定义工作单元
# ==============================================================================
def run_all_variants_for_one_graph(dataset_file_path, config):
    """
    对单个从数据集中加载的图实例，运行所有需要对比的算法。
    """
    # a. 从文件加载图和精确解
    try:
        data = np.load(dataset_file_path, allow_pickle=True).item()
        n = data['num_vertices']
        exact_cut = data['exact_max_cut']
        # 将图转换为pro_info格式
        graph_adj = data['graph_adj_list']
        pro_info = []
        for u, neighbors in graph_adj.items():
            for v in neighbors:
                if u < v: # 避免重复边
                    pro_info.append([u, v, 1.0])
        pro_info = np.array(pro_info)
    except Exception as e:
        print(f"错误: 无法加载或处理文件 {dataset_file_path}: {e}")
        return None

    print(f"  > 开始处理 n={n} 的图 (文件: {os.path.basename(dataset_file_path)})...")

    # b. 运行GW算法 (单次投影)
    primal, _, X_opt = maxcut_interior_point(pro_info)
    v_ran_gw = np.random.uniform(-1, 1, (1, n))
    gw_energy, gw_solution, _ = GW_algorithm(pro_info, X_opt[-1], v_ran_gw)

    # c. 运行BMMK算法 (100次)
    theta0 = np.random.uniform(-np.pi, np.pi, (100, n))
    bmmk_energy, bmmk_solution, _ = BMMK_optR(theta0, pro_info, config['max_iters'], k=2)
    ind = random.randint(0,n-1)
    s_ws_warmest = bmmk_solution - bmmk_solution[ind]

    # --- 运行不同的QAOA变体 ---
    results = {'n': n, 'exact': exact_cut, 'gw': gw_energy}

    # d. Standard QAOA (s) - 冷启动
    qaoa_variant_s = {'type': 'standard', 'solution_WS': np.zeros(n), 'energy_WS': 0}
    para_input_s = AdaptiveQaoaInput(config['qaoa_initial_point2'], n, pro_info, qaoa_variant_s, {'max_iters': config['max_iters']})
    res_s = adaptive_qaoa(para_input_s)
    results['s_qaoa'] = res_s.energy

    # e. WS-QAOA (WS)
    qaoa_variant_ws = {'type': 'warm-start', 'epsilon_ws': config['epsilon_ws'], 'solution_WS': gw_solution, 'energy_WS': gw_energy}
    para_input_ws = AdaptiveQaoaInput(config['qaoa_initial_point'], n, pro_info, qaoa_variant_ws, {'max_iters': config['max_iters']})
    res_ws = adaptive_qaoa(para_input_ws)
    results['ws_qaoa'] = res_ws.energy
    
    # f. QAOA-warmest (warm)
    qaoa_variant_warmest = {'type': 'warmest', 'solution_WS': s_ws_warmest, 'energy_WS': bmmk_energy}
    para_input_warmest = AdaptiveQaoaInput(config['qaoa_initial_point2'], n, pro_info, qaoa_variant_warmest, {'max_iters': config['max_iters']})
    res_warmest = adaptive_qaoa(para_input_warmest)
    results['warmest_qaoa'] = res_warmest.energy

    # g. WS-ab-QAOA (bias)
    qaoa_variant_ab = {'type': 'ab', 'ell': config['learning_rate'], 'solution_WS': gw_solution / np.sqrt(3), 'energy_WS': gw_energy}
    para_input_ab = AdaptiveQaoaInput(config['qaoa_initial_point2'], n, pro_info, qaoa_variant_ab, {'max_iters': config['max_iters']})
    res_ab = adaptive_qaoa(para_input_ab)
    results['ab_qaoa'] = res_ab.bias_energy[0] # bias-state method

    return results

# ==============================================================================
# --- 控制与配置 ---
# ==============================================================================
DATASET_DIR = "u3r_dataset"
RECALCULATE = False
DATA_FILENAME = 'data/figure2_data.npz'

# ==============================================================================
# --- 模拟参数设置 ---
# ==============================================================================
if RECALCULATE:
    config = {
        'node_sizes': [40, 60, 80, 100, 120, 140], # 与create_dataset.py中的设置匹配
        'qaoa_initial_point': np.array([[5.7665, 4.4898]]), # 论文中提到的另一个好的初始点
        'qaoa_initial_point2': np.array([[4.2315, 1.0002]]),
        'max_iters': 500,
        'learning_rate': 0.4,
        'epsilon_ws': 0.25,
    }

# ==============================================================================
# --- 核心计算逻辑 ---
# ==============================================================================
if RECALCULATE:
    print("模式: 重新计算数据...")
    start_time = time.time()

    # 1. 发现所有数据集文件作为任务列表
    if not os.path.isdir(DATASET_DIR):
        print(f"错误: 数据集文件夹 '{DATASET_DIR}' 不存在。请先运行 create_dataset.py。")
        exit()
    tasks = glob.glob(f'{DATASET_DIR}/n*/graph_*.npy')
    # tasks = glob.glob(f'{DATASET_DIR}/n40/graph_0.npy')
    print(f"发现 {len(tasks)} 个图实例需要处理...")

    # 2. 并行执行所有任务
    all_results = Parallel(n_jobs=-1)(
        delayed(run_all_variants_for_one_graph)(task_file, config) for task_file in tasks
    )
    # 过滤掉可能失败的任务
    all_results = [res for res in all_results if res is not None]

    # 3. 聚合结果
    grouped_results = defaultdict(list)
    for res in all_results:
        grouped_results[res['n']].append(res)
        
    # 4. 计算每个n的平均性能
    plot_data = defaultdict(dict)
    for n in sorted(grouped_results.keys()):
        # 将字典列表转换为更易处理的结构
        df = {key: [d[key] for d in grouped_results[n]] for key in grouped_results[n][0]}
        
        # 计算差值 (N_alg - N_exact)
        plot_data[n]['diff_gw'] = np.mean(-np.array(df['gw']) + np.array(df['exact']))
        # print(np.array(df['gw']),np.array(df['exact']))
        # print(np.array(df['exact'])-np.mean(np.array(df['gw'])))
        plot_data[n]['diff_s'] = np.mean(np.array(df['exact']) -np.array(df['s_qaoa']))
        plot_data[n]['diff_ws'] = np.mean(-np.array(df['ws_qaoa']) + np.array(df['exact']))
        plot_data[n]['diff_warmest'] = np.mean(-np.array(df['warmest_qaoa']) + np.array(df['exact']))
        plot_data[n]['diff_ab'] = np.mean(-np.array(df['ab_qaoa']) + np.array(df['exact']))
        
        # 计算近似比 (N_alg / N_exact)
        plot_data[n]['ratio_gw'] = np.mean(np.array(df['gw']) / np.array(df['exact']))
        plot_data[n]['ratio_s'] = np.mean(np.array(df['s_qaoa']) / np.array(df['exact']))
        plot_data[n]['ratio_ws'] = np.mean(np.array(df['ws_qaoa']) / np.array(df['exact']))
        plot_data[n]['ratio_warmest'] = np.mean(np.array(df['warmest_qaoa']) / np.array(df['exact']))
        plot_data[n]['ratio_ab'] = np.mean(np.array(df['ab_qaoa']) / np.array(df['exact']))

    total_time = time.time() - start_time
    print(f"\n所有计算完成，总耗时: {total_time:.2f} 秒")
    
    # 5. 保存数据
    np.savez(DATA_FILENAME, node_sizes=sorted(grouped_results.keys()), plot_data=dict(plot_data))
    print(f"数据已保存到文件: {DATA_FILENAME}")

# ==============================================================================
# --- 加载与绘图部分 ---
# ==============================================================================
else:
    print(f"模式: 从文件 {DATA_FILENAME} 加载数据...")
    if not os.path.exists(DATA_FILENAME):
        print(f"错误: 数据文件 {DATA_FILENAME} 不存在。")
        exit()
    data = np.load(DATA_FILENAME, allow_pickle=True)
    plot_data = data['plot_data'].item()
    node_sizes = data['node_sizes']
    print("数据加载成功。")

print("开始绘制图形...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
node_sizes = sorted(plot_data.keys())

# --- 绘图 (a): 差值 ---
ax1.axhline(y=0, color='black', linestyle='-')
ax1.plot(node_sizes, [plot_data[n]['diff_gw'] for n in node_sizes], marker='o', label='GW')
ax1.plot(node_sizes, [plot_data[n]['diff_s'] for n in node_sizes], marker='s', label='s-QAOA')
ax1.plot(node_sizes, [plot_data[n]['diff_ws'] for n in node_sizes], marker='D', label='WS-QAOA')
ax1.plot(node_sizes, [plot_data[n]['diff_warmest'] for n in node_sizes], marker='x', label='QAOA-warmest')
ax1.plot(node_sizes, [plot_data[n]['diff_ab'] for n in node_sizes], marker='^', color='red', label='WS-ab-QAOA (bias)')
ax1.set_ylabel('$N_{cut}^{alg} - N_{cut}$', fontsize=14)
ax1.set_title('Cut value difference', fontsize=16)
ax1.legend()
ax1.grid(True)

# --- 绘图 (b): 近似比 ---
ax2.plot(node_sizes, [plot_data[n]['ratio_gw'] for n in node_sizes], marker='o', label='GW')
ax2.plot(node_sizes, [plot_data[n]['ratio_s'] for n in node_sizes], marker='s', label='s-QAOA')
ax2.plot(node_sizes, [plot_data[n]['ratio_ws'] for n in node_sizes], marker='D', label='WS-QAOA')
ax2.plot(node_sizes, [plot_data[n]['ratio_warmest'] for n in node_sizes], marker='x', label='QAOA-warmest')
ax2.plot(node_sizes, [plot_data[n]['ratio_ab'] for n in node_sizes], marker='^', color='red', label='WS-ab-QAOA (bias)')
ax2.set_xlabel('$n$', fontsize=14)
ax2.set_ylabel('Approximation ratio $r$', fontsize=14)
ax2.set_title('Approximation ratio', fontsize=16)
ax2.legend()
ax2.grid(True)

# fig.tight_layout()
plt.savefig('figure2.png', dpi=300)
# plt.show()

print("绘图完成。")