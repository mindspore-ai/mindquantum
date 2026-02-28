import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import time
from mindquantum.algorithm.qaia import BSB, DSB


def read_gset(filename, negate=True):
    """
    读取Gset数据文件并转换为图矩阵

    参数:
        filename: Gset数据文件路径
        negate: 是否对矩阵取负 (默认True)

    返回:
        G: 稀疏矩阵
    """
    # 读取图表
    graph = pd.read_csv(filename, sep=' ')
    # 节点的数量
    n_v = int(graph.columns[0])
    # 边的数量
    n_e = int(graph.columns[1])

    # 如果节点和边不匹配，会抛出错误
    assert n_e == graph.shape[0], 'The number of edges is not matched'

    # 将读取的数据转换为一个COO矩阵（Coordinate List Format），并返回一个稀疏矩阵
    G = coo_matrix((np.concatenate([graph.iloc[:, -1], graph.iloc[:, -1]]),
                    (np.concatenate([graph.iloc[:, 0]-1, graph.iloc[:, 1]-1]),
                     np.concatenate([graph.iloc[:, 1]-1, graph.iloc[:, 0]-1]))), shape=(n_v, n_v))
    if negate:
        G = -G

    return G


def solve_gset_bsb_dsb(gset_file, sample_size=100, n_iter_list=None, save_plot=True):
    """
    使用BSB和DSB算法求解MAX-CUT问题并绘制切割值收敛图和能量-时间图

    参数:
        gset_file: Gset数据文件路径
        sample_size: 采样次数 (默认100)
        n_iter_list: 迭代次数列表 (默认None，使用[10, 25, 50, 75, 100, 250, 500, 750, 1000])
        save_plot: 是否保存切割值收敛图 (默认True)

    返回:
        bsb_max_cut_list: BSB算法最大切割值列表
        dsb_max_cut_list: DSB算法最大切割值列表
        bsb_time: BSB算法总运行时间
        dsb_time: DSB算法总运行时间
    """
    if n_iter_list is None:
        n_iter_list = [10, 25, 50, 75, 100, 250, 500, 750, 1000]

    print(f'正在读取数据文件: {gset_file}')
    G = read_gset(gset_file, negate=True)  # 对矩阵取负（参考 notebook）
    print(f'图矩阵维度: {G.shape}')

    bsb_max_cut_list = []
    dsb_max_cut_list = []

    # BSB算法求解
    print('\n开始BSB算法求解...')
    start_time = time.time()
    for n_iter in n_iter_list:
        s = BSB(G, batch_size=sample_size, n_iter=n_iter, dt=0.5)
        s.update()
        cut_value = s.calc_cut()
        max_cut = np.max(cut_value)
        bsb_max_cut_list.append(max_cut)
        print(f'  n_iter={n_iter}, cut_max={max_cut:.2f}')
    bsb_time = time.time() - start_time
    print(f'BSB算法完成，运行时间: {bsb_time*1000:.2f}ms')

    # DSB算法求解
    print('\n开始DSB算法求解...')
    start_time = time.time()
    for n_iter in n_iter_list:
        s = DSB(G, batch_size=sample_size, n_iter=n_iter, dt=0.5)
        s.update()
        cut_value = s.calc_cut()
        max_cut = np.max(cut_value)
        dsb_max_cut_list.append(max_cut)
        print(f'  n_iter={n_iter}, cut_max={max_cut:.2f}')
    dsb_time = time.time() - start_time
    print(f'DSB算法完成，运行时间: {dsb_time*1000:.2f}ms')

    # 绘制切割值收敛图和能量-时间图
    if save_plot:
        plot_cut_convergence(n_iter_list, bsb_max_cut_list, dsb_max_cut_list, gset_file)

    return bsb_max_cut_list, dsb_max_cut_list, bsb_time, dsb_time


def plot_cut_convergence(n_iter_list, bsb_max_cut_list, dsb_max_cut_list, gset_file, show: bool = False):
    """
    绘制切割值收敛图（BSB和DSB算法在同一图中，只显示最大值）

    参数:
        n_iter_list: 迭代次数列表
        bsb_max_cut_list: BSB算法最大切割值列表
        dsb_max_cut_list: DSB算法最大切割值列表
        gset_file: Gset数据文件路径
    """
    plt.figure(figsize=(10, 6))

    # 绘制BSB算法收敛曲线
    plt.plot(np.log10(n_iter_list), bsb_max_cut_list, 'o-', color='tab:blue',
             linewidth=2.5, markersize=8, label='BSB', markeredgecolor='white', markeredgewidth=1.5)

    # 绘制DSB算法收敛曲线
    plt.plot(np.log10(n_iter_list), dsb_max_cut_list, 's-', color='tab:red',
             linewidth=2.5, markersize=8, label='DSB', markeredgecolor='white', markeredgewidth=1.5)

    # 标注最终收敛值
    bsb_final_max = bsb_max_cut_list[-1]
    dsb_final_max = dsb_max_cut_list[-1]

    # 使用相对坐标（0-1范围）确保不超出图表
    plt.text(0.97, 0.97,
             f'BSB Max: {bsb_final_max:.2f}\nDSB Max: {dsb_final_max:.2f}',
             transform=plt.gca().transAxes,
             fontsize=11, color='black', fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7))

    # 设置图表样式（参考 notebook）
    plt.xlim([1.05, 3.15])
    plt.xlabel(r'$N_{step}$', fontsize=14)
    plt.ylabel('Cut value', fontsize=14)
    plt.title(f'Cut Value Convergence - {os.path.splitext(os.path.basename(gset_file))[0]}',
              fontsize=16, fontweight='bold')

    plt.legend(fontsize=12, loc='best')

    plt.grid(True, alpha=0.3, linestyle='--')

    # 设置x轴刻度
    plt.xticks(np.log10(n_iter_list), ["$10^2$", "", "", "", "$10^3$", "", "", "", "$10^4$"])
    plt.tight_layout()

    # 保存图片到png文件夹
    gset_name = os.path.splitext(os.path.basename(gset_file))[0]
    png_dir = os.path.join(os.path.dirname(__file__), 'png')
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    save_path = os.path.join(png_dir, f'{gset_name}_python_cut_convergence.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n切割值收敛图已保存至: {save_path}')

    if show:
        plt.show()
    else:
        plt.close()


def plot_energy_time_convergence(n_iter_list, bsb_max_cut_list, dsb_max_cut_list, bsb_time, dsb_time, gset_file, show: bool = False):
    """
    绘制能量（切割值）随时间变化的收敛图

    参数:
        n_iter_list: 迭代次数列表
        bsb_max_cut_list: BSB算法最大切割值列表
        dsb_max_cut_list: DSB算法最大切割值列表
        bsb_time: BSB算法总运行时间（秒）
        dsb_time: DSB算法总运行时间（秒）
        gset_file: Gset数据文件路径
    """
    plt.figure(figsize=(12, 7))

    # 计算时间轴（累计时间，单位：毫秒）
    # 假设每个迭代点的时间与迭代次数成正比
    bsb_cum_time_ms = np.cumsum([n_iter * bsb_time / sum(n_iter_list) * 1000 for n_iter in n_iter_list])
    dsb_cum_time_ms = np.cumsum([n_iter * dsb_time / sum(n_iter_list) * 1000 for n_iter in n_iter_list])

    # 绘制BSB算法能量-时间曲线
    plt.plot(bsb_cum_time_ms, bsb_max_cut_list, 'o-', color='tab:blue',
             linewidth=2, markersize=8, label='BSB')

    # 绘制DSB算法能量-时间曲线
    plt.plot(dsb_cum_time_ms, dsb_max_cut_list, 's-', color='tab:red',
             linewidth=2, markersize=8, label='DSB')

    # 标注最终收敛值和时间
    bsb_final = bsb_max_cut_list[-1]
    dsb_final = dsb_max_cut_list[-1]
    bsb_final_time = bsb_cum_time_ms[-1]
    dsb_final_time = dsb_cum_time_ms[-1]

    # 在图表右上角显示最终收敛值和收敛时间
    info_text = (f'BSB: Cut={bsb_final:.2f}, Time={bsb_final_time:.0f}ms\n'
                f'DSB: Cut={dsb_final:.2f}, Time={dsb_final_time:.0f}ms')
    plt.text(0.98, 0.98, info_text,
             transform=plt.gca().transAxes,
             fontsize=10, color='black', fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))

    plt.xlabel('Time (ms)', fontsize=14)
    plt.ylabel('Cut Value (Energy)', fontsize=14)
    gset_name = os.path.splitext(os.path.basename(gset_file))[0]
    plt.title(f'Energy vs Time Convergence - {gset_name}',
              fontsize=16, fontweight='bold')

    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')

    # 设置x轴为毫秒
    plt.tick_params(axis='x', which='major', labelsize=11)
    plt.tick_params(axis='y', which='major', labelsize=11)

    plt.tight_layout()

    # 保存图片到png文件夹
    png_dir = os.path.join(os.path.dirname(__file__), 'png')
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    save_path = os.path.join(png_dir, f'{gset_name}_python_energy_time_convergence.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n能量-时间收敛图已保存至: {save_path}')

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    # 设置Gset文件路径
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    gset_file = os.path.join(data_dir, 'G22.txt')

    # 检查文件是否存在
    if not os.path.exists(gset_file):
        print(f'错误: 文件 {gset_file} 不存在，请先运行 get_Gsets.py 下载数据')
    else:
        # 使用BSB和DSB算法求解G22
        bsb_max_cut, dsb_max_cut, bsb_time, dsb_time = solve_gset_bsb_dsb(
            gset_file=gset_file,
            sample_size=100,
            n_iter_list=[10, 25, 50, 75, 100, 250, 500, 750, 1000],
            save_plot=True
        )

        # 绘制能量-时间收敛图（不显示，只保存）
        plot_energy_time_convergence([10, 25, 50, 75, 100, 250, 500, 750, 1000],
                                     bsb_max_cut, dsb_max_cut, bsb_time, dsb_time, gset_file, show=False)
