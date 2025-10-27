# Plot_fig3.py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import os
# 1. 导入并行计算所需的库
from joblib import Parallel, delayed
# 导入核心函数
from SDP import GW_algorithm, maxcut_interior_point
from Adaptive_para import AdaptiveQaoaInput, adaptive_qaoa

# ==============================================================================
# 2. 创建一个“工作单元”函数
# 这个函数包含了对单个图实例的所有计算，以便并行调用。
# ==============================================================================
def process_single_graph(n, R_gw, qaoa_initial_point, max_iters, seed):
    """
    处理单个图实例的完整计算流程。
    Args:
        n (int): 顶点数。
        R_gw (int): GW算法的投影次数。
        qaoa_initial_point (np.ndarray): QAOA的初始(γ,β)点。
        max_iters (int): QAOA的最大迭代次数。
        seed (int): 用于生成图的随机种子，确保并行时每个图都不同。
    Returns:
        tuple: (gw_energy, bias_energy, num_edges)
    """
    print(f"  > 开始处理 n={n} 的图实例 (seed={seed})...")
    
    # a. 生成一个 3-正则无权图
    graph = nx.random_regular_graph(3, n, seed=seed)
    num_edges = graph.number_of_edges()
    pro_info = np.array([[u, v, 1.0] for u, v in graph.edges()])

    # b. 运行 Goemans-Williamson (GW) 算法
    _, _, X_opt = maxcut_interior_point(pro_info)
    # 使用与图种子相同的种子确保随机性，但行为可复现
    np.random.seed(seed)
    v_ran = np.random.uniform(-1, 1, (R_gw, n))
    gw_energy, gw_solution, _ = GW_algorithm(pro_info, X_opt[-1], v_ran)

    # c. 运行 WS-ab-QAOA
    qaoa_variant = {
        'type': 'ab',
        'ell': 0.4,
        'solution_WS': gw_solution / np.sqrt(3),
        'energy_WS': gw_energy
    }
    para_input = AdaptiveQaoaInput(
        initial_pointR=qaoa_initial_point,
        n=n,
        pro_solve=pro_info,
        qaoa_variant=qaoa_variant,
        sys_para={'max_iters': max_iters}
    )
    res = adaptive_qaoa(para_input)
    bias_energy = res.bias_energy[0]
    
    return gw_energy, bias_energy, num_edges

# ==============================================================================
# --- 控制与配置 ---
# ==============================================================================
RECALCULATE = False
DATA_FILENAME = 'data/figure3_data.npz'
# ==============================================================================
# --- 模拟参数设置 ---
# ==============================================================================
if RECALCULATE:
    node_sizes = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    realizations = 40
    R_gw = 1
    qaoa_initial_point = np.array([[4.2315, 1.0002]])
    max_iters = 4000
    avg_gw_ratios = []
    avg_bias_ratios = []

# ==============================================================================
# --- 并行 ---
# ==============================================================================
if RECALCULATE:
    print("模式: 重新计算数据 (已启用并行加速)...")
    start_time = time.time()

    for n in node_sizes:
        print(f"\n并行处理 n = {n} 的 {realizations} 个图实例...")

        # 3. 将内层循环替换为 Parallel 调用
        # n_jobs=-1 表示使用所有可用的CPU核心
        # delayed(process_single_graph) 将我们的工作函数及其参数打包成一个任务
        # for i in range(realizations) 创建了40个独立的任务
        results = Parallel(n_jobs=-1)(
            delayed(process_single_graph)(n, R_gw, qaoa_initial_point, max_iters, i)
            for i in range(realizations)
        )

        # 4. 处理并行计算返回的结果
        # results 是一个列表，每个元素是 (gw_energy, bias_energy, num_edges)
        # 我们需要解包这些结果
        gw_results_for_n = [res[0] for res in results]
        bias_results_for_n = [res[1] for res in results]
        # 假设所有3-正则图的边数都一样，取第一个即可
        num_edges = results[0][2] 
        
        # d. 计算平均值
        avg_gw_ratio = np.mean(np.array(gw_results_for_n) / num_edges)
        avg_bias_ratio = np.mean(np.array(bias_results_for_n) / num_edges)
        avg_gw_ratios.append(avg_gw_ratio)
        avg_bias_ratios.append(avg_bias_ratio)

        print(f"处理 n = {n} 完成. GW 平均比率: {avg_gw_ratio:.4f}, Bias 平均比率: {avg_bias_ratio:.4f}")

    total_time = time.time() - start_time
    print(f"\n所有计算完成，总耗时: {total_time:.2f} 秒")
    
    # --- e. 保存数据 ---
    np.savez(DATA_FILENAME,
             node_sizes=node_sizes,
             avg_gw_ratios=avg_gw_ratios,
             avg_bias_ratios=avg_bias_ratios)
    print(f"数据已保存到文件: {DATA_FILENAME}")

# ==============================================================================
# --- 加载与绘图部分 ---
# ==============================================================================
else:
    print(f"模式: 从文件 {DATA_FILENAME} 加载数据...")
    if not os.path.exists(DATA_FILENAME):
        print(f"错误: 数据文件 {DATA_FILENAME} 不存在。请先将 RECALCULATE 设置为 True 运行一次。")
        exit()
        
    data = np.load(DATA_FILENAME)
    node_sizes = data['node_sizes']
    avg_gw_ratios = data['avg_gw_ratios']
    avg_bias_ratios = data['avg_bias_ratios']
    print("数据加载成功。")

print("开始绘制图形...")

plt.figure(figsize=(8, 6))
plt.plot(node_sizes, avg_gw_ratios, marker='s', linestyle='-', label='$N_{cut}^{GW}/|E|$')
plt.plot(node_sizes, avg_bias_ratios, marker='^', linestyle='-', label='$N_{cut}^{bias}/|E|$')
plt.xlabel('$n$', fontsize=14)
plt.ylabel('$N_{cut}^{alg}/|E|$', fontsize=14)
plt.title('Figure 3: GW vs. WS-ab-QAOA on u3r', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.ylim(0.87, 0.92)
plt.show()
plt.savefig('figure3.png', dpi=300)
print("绘图完成。")