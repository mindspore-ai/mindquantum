# Plot_fig4.py

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import os
from collections import defaultdict
from joblib import Parallel, delayed

# 导入所有需要的核心函数
from SDP import GW_algorithm, maxcut_interior_point
from Adaptive_para import AdaptiveQaoaInput, adaptive_qaoa

# ==============================================================================
# 1. 定义新的“工作单元”函数
# 这个函数将执行一次完整的独立实验
# ==============================================================================
def run_full_experiment_for_one_graph(seed, config):
    """
    对一个新生成的图，执行所有R值下的计算。
    这是并行计算的基本单元。
    """
    n = config['n']
    print(f"  > 开始处理第 {seed+1}/{config['realizations']} 个独立实验...")

    # a. 生成一个图并只求解一次SDP
    graph = nx.random_regular_graph(3, n, seed=seed)
    num_edges = graph.number_of_edges()
    pro_info = np.array([[u, v, 1.0] for u, v in graph.edges()])
    _, _, X_opt = maxcut_interior_point(pro_info)
    X_opt_final = X_opt[-1] # 获取最终的SDP解矩阵

    # b. 准备R值列表
    m_values = range(11)
    R_values = [int(5 * (m + 1) * m / 2) for m in m_values]
    
    # c. 准备用于存储本次实验结果的列表
    gw_ratios_for_this_graph = []
    bias_ratios_for_this_graph = []

    # d. 串行遍历所有R值，因为它们依赖于同一个X_opt
    for r_extra in R_values:
        total_projections = 1 + r_extra
        
        # --- 运行Goemans-Williamson (GW)算法 ---
        # 复用同一个X_opt_final，只改变随机投影次数
        # 为保证每次的随机向量不同，可以使用r_extra作为随机种子的一部分
        np.random.seed(seed + r_extra) 
        v_ran = np.random.uniform(-1, 1, (total_projections, n))
        gw_energy, gw_solution, _ = GW_algorithm(pro_info, X_opt_final, v_ran)

        # --- 运行WS-ab-QAOA ---
        qaoa_variant = {
            'type': 'ab', 
            'ell': config['learning_rate'],
            'solution_WS': gw_solution / np.sqrt(3), 
            'energy_WS': gw_energy
        }
        para_input = AdaptiveQaoaInput(
            initial_pointR=config['qaoa_initial_point'], 
            n=n, 
            pro_solve=pro_info,
            qaoa_variant=qaoa_variant, 
            sys_para={'max_iters': config['max_iters']}
        )
        res = adaptive_qaoa(para_input)
        bias_energy = res.bias_energy[0]
        
        # 计算并记录本次R值下的性能比率
        gw_ratios_for_this_graph.append(gw_energy / num_edges)
        bias_ratios_for_this_graph.append(bias_energy / num_edges)

    print(f"  > 第 {seed+1}/{config['realizations']} 个独立实验完成。")
    # 返回本次完整实验得到的一系列比率结果
    return gw_ratios_for_this_graph, bias_ratios_for_this_graph

# ==============================================================================
# --- 控制与配置 ---
# ==============================================================================
RECALCULATE = False
DATA_FILENAME = 'data/figure4_data.npz'

# ==============================================================================
# --- 模拟参数设置 ---
# ==============================================================================
config = {
    'n': 1000,
    'realizations': 40,
    'qaoa_initial_point': np.array([[4.2315, 1.0002]]),
    'max_iters': 2000, # 适当减少以加速
    'learning_rate': 0.4,
}
    
# ==============================================================================
# --- 核心计算逻辑 (修改为并行处理40次独立实验) ---
# ==============================================================================
if RECALCULATE:
    print("模式: 重新计算数据 (并行处理40次独立实验)...")
    start_time = time.time()

    # 并行执行40次独立的完整实验
    # 每个任务由其种子 'i' 定义
    all_results = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(run_full_experiment_for_one_graph)(i, config)
        for i in range(config['realizations'])
    )

    # all_results 是一个列表，包含了40个 (gw_ratios_list, bias_ratios_list) 元组
    # 我们需要对这些结果取平均
    # 将结果转换为NumPy数组方便计算
    all_gw_ratios_np = np.array([res[0] for res in all_results]) # 形状为 (40, 11)
    all_bias_ratios_np = np.array([res[1] for res in all_results]) # 形状为 (40, 11)

    # 沿realizations轴(axis=0)取平均值
    avg_gw_ratios = np.mean(all_gw_ratios_np, axis=0)
    avg_bias_ratios = np.mean(all_bias_ratios_np, axis=0)
    
    # 获取R值列表用于保存和绘图
    m_values = range(11)
    R_values = [int(5 * (m + 1) * m / 2) for m in m_values]

    total_time = time.time() - start_time
    print(f"\n所有计算完成，总耗时: {total_time:.2f} 秒")
    
    # --- 保存最终的平均结果 ---
    np.savez(DATA_FILENAME, 
             R_values=R_values, 
             avg_gw_ratios=avg_gw_ratios, 
             avg_bias_ratios=avg_bias_ratios)
    print(f"数据已保存到文件: {DATA_FILENAME}")

# ==============================================================================
# --- 加载与绘图部分 (无需改动) ---
# ==============================================================================
else:
    print(f"模式: 从文件 {DATA_FILENAME} 加载数据...")
    if not os.path.exists(DATA_FILENAME):
        print(f"错误: 数据文件 {DATA_FILENAME} 不存在。")
        exit()
    data = np.load(DATA_FILENAME)
    R_values = data['R_values']
    avg_gw_ratios = data['avg_gw_ratios']
    avg_bias_ratios = data['avg_bias_ratios']
    print("数据加载成功。")

print("开始绘制图形...")

plt.figure(figsize=(10, 7))
plt.plot(R_values, avg_gw_ratios, marker='s', linestyle='-', label='GW')
plt.plot(R_values, avg_bias_ratios, marker='^', linestyle='-', label='WS-ab-QAOA (bias-state)')
plt.xlabel('$R$ The rounding number', fontsize=14)
plt.ylabel('$N_{cut}^{alg} / |E|$', fontsize=14)
plt.title(f'(n={config["n"]})', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('figure4.png', dpi=300)

print("绘图完成。")