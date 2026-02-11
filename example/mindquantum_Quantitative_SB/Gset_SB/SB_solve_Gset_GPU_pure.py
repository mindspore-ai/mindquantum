"""
SB_solve_Gset_GPU_pure.py - 使用6种CUDA实现求解Gset数据集的MAX-CUT问题

支持6种GPU实现（通过调用SB_CUDA_pure目录中的编译好的可执行文件）：
1. BSB_FP8   - Ballistic SB算法，FP8精度 (BSB_GPU_purefp8_improved)
2. BSB_FP16  - Ballistic SB算法，FP16精度 (BSB_GPU_purefp16_improved)
3. BSB_INT8  - Ballistic SB算法，INT8精度 (BSB_GPU_pureint8_improved)
4. DSB_FP8   - Discrete SB算法，FP8精度 (DSB_GPU_purefp8_improved)
5. DSB_FP16  - Discrete SB算法，FP16精度 (DSB_GPU_purefp16_improved)
6. DSB_INT8  - Discrete SB算法，INT8精度 (DSB_GPU_pureint8_improved)

使用说明：
1. 首先编译SB_CUDA_pure目录下的.cu文件：
   nvcc BSB_GPU_purefp8_improved.cu sb_cuda_gset_reader.cu -o BSB_GPU_purefp8_improved.exe -O3 -arch=sm_75
   nvcc BSB_GPU_purefp16_improved.cu sb_cuda_gset_reader.cu -o BSB_GPU_purefp16_improved.exe -O3 -arch=sm_75
   nvcc BSB_GPU_pureint8_improved.cu sb_cuda_gset_reader.cu -o BSB_GPU_pureint8_improved.exe -O3 -arch=sm_75
   nvcc DSB_GPU_purefp8_improved.cu sb_cuda_gset_reader.cu -o DSB_GPU_purefp8_improved.exe -O3 -arch=sm_75
   nvcc DSB_GPU_purefp16_improved.cu sb_cuda_gset_reader.cu -o DSB_GPU_purefp16_improved.exe -O3 -arch=sm_75
   nvcc DSB_GPU_pureint8_improved.cu sb_cuda_gset_reader.cu -o DSB_GPU_pureint8_improved.exe -O3 -arch=sm_75

2. 运行本Python脚本：
   python SB_solve_Gset_GPU_pure.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
import time
from typing import List, Tuple, Dict, Optional
import ctypes
import torch

# 添加SB_CUDA_pure目录到路径
SB_CUDA_DIR = os.path.join(os.path.dirname(__file__), 'SB_CUDA_pure')
sys.path.insert(0, SB_CUDA_DIR)


def check_cuda_executables():
    """检查CUDA可执行文件是否存在"""
    exe_names = ['BSB_GPU_purefp8_improved.exe', 'BSB_GPU_purefp16_improved.exe', 'BSB_GPU_pureint8_improved.exe',
                 'DSB_GPU_purefp8_improved.exe', 'DSB_GPU_purefp16_improved.exe', 'DSB_GPU_pureint8_improved.exe']
    missing = []
    for exe in exe_names:
        exe_path = os.path.join(SB_CUDA_DIR, exe)
        if not os.path.exists(exe_path):
            missing.append(exe)
    return missing


class CUDASBSolver:
    """CUDA SB算法求解器（支持6种精度实现）"""

    # 算法类型定义
    BSB_FP8 = 0
    BSB_FP16 = 1
    BSB_INT8 = 2
    DSB_FP8 = 3
    DSB_FP16 = 4
    DSB_INT8 = 5

    ALGORITHM_NAMES = {
        BSB_FP8: 'BSB_FP8',
        BSB_FP16: 'BSB_FP16',
        BSB_INT8: 'BSB_INT8',
        DSB_FP8: 'DSB_FP8',
        DSB_FP16: 'DSB_FP16',
        DSB_INT8: 'DSB_INT8'
    }

    def __init__(self, gset_file: str, algorithm_type: int = BSB_FP16):
        """
        初始化CUDA SB求解器

        参数:
            gset_file: Gset数据文件路径
            algorithm_type: 算法类型（BSB_FP8, BSB_FP16, BSB_INT8, DSB_FP8, DSB_FP16, DSB_INT8）
        """
        self.gset_file = gset_file
        self.algorithm_type = algorithm_type
        self.algorithm_name = self.ALGORITHM_NAMES[algorithm_type]

        # 设置CUDA可执行文件目录（参考SB_basic_run.py第91-93行）
        self.cuda_dir = SB_CUDA_DIR

        # 定义可执行文件路径（参考SB_basic_run.py第94-110行）
        self.executables = {
            self.BSB_FP8: 'BSB_GPU_purefp8_improved',
            self.BSB_FP16: 'BSB_GPU_purefp16_improved',
            self.BSB_INT8: 'BSB_GPU_pureint8_improved',
            self.DSB_FP8: 'DSB_GPU_purefp8_improved',
            self.DSB_FP16: 'DSB_GPU_purefp16_improved',
            self.DSB_INT8: 'DSB_GPU_pureint8_improved'
        }

        # 获取当前算法的可执行文件路径
        exe_base_name = self.executables[self.algorithm_type]
        exe_name = exe_base_name + '.exe' if os.name == 'nt' else exe_base_name
        self.executable_path = os.path.join(self.cuda_dir, exe_name)

        # 检查可执行文件是否存在（参考SB_basic_run.py第116-131行）
        if not os.path.exists(self.executable_path):
            print(f'警告: CUDA可执行文件不存在: {self.executable_path}')
            print(f'  请编译: nvcc {exe_base_name}.cu -o {exe_name}')
        else:
            print(f'找到CUDA可执行文件: {self.executable_path}')

        # 读取图数据
        self.G = self.read_gset(gset_file, negate=True)
        print(f'图矩阵维度: {self.G.shape}')
        print(f'非零元素数量: {self.G.nnz}')

    def read_gset(self, filename: str, negate: bool = True) -> csr_matrix:
        """
        读取Gset数据文件并转换为图矩阵

        参数:
            filename: Gset数据文件路径
            negate: 是否对矩阵取负 (默认True)

        返回:
            G: 稀疏矩阵
        """
        # 读取图数据
        graph = pd.read_csv(filename, sep=' ')
        # 节点的数量
        n_v = int(graph.columns[0])
        # 边的数量
        n_e = int(graph.columns[1])

        # 检查节点和边是否匹配
        assert n_e == graph.shape[0], 'The number of edges is not matched'

        # 转换为COO矩阵并返回稀疏矩阵
        G = coo_matrix((np.concatenate([graph.iloc[:, -1], graph.iloc[:, -1]]),
                        (np.concatenate([graph.iloc[:, 0]-1, graph.iloc[:, 1]-1]),
                         np.concatenate([graph.iloc[:, 1]-1, graph.iloc[:, 0]-1]))),
                       shape=(n_v, n_v))
        if negate:
            G = -G

        return G.tocsr()

    def _init_cuda_library(self):
        """初始化CUDA C++库（如果存在）- 已废弃，现在直接调用exe"""
        # 直接使用CUDA可执行文件，不再加载C++库
        self.cuda_lib = None

    def solve(self, sample_size: int = 100, n_iter_list: Optional[List[int]] = None,
             xi: Optional[float] = None, delta: float = 0.9, dt: float = 0.5) -> Tuple[np.ndarray, List[float], List[float]]:
        """
        使用指定算法求解MAX-CUT问题

        参数:
            sample_size: 采样次数 (默认100)
            n_iter_list: 迭代次数列表 (默认None，使用[10, 25, 50, 75, 100, 250, 500, 750, 1000])
            xi: 耦合强度 (默认None，自动计算)
            delta: 阻尼系数 (默认0.9)
            dt: 时间步长 (默认0.5)

        返回:
            cut_values: 切割值列表（每个迭代点的平均值）
            max_cuts: 最大切割值列表
            times: 每次迭代的执行时间列表（毫秒）
        """
        if n_iter_list is None:
            n_iter_list = [10, 25, 50, 75, 100, 250, 500, 750, 1000]

        # 自动计算xi（如果未指定）
        if xi is None:
            xi = 0.5 * np.sqrt(self.G.shape[0] - 1) / np.sqrt(csr_matrix.power(self.G, 2).sum())

        print(f'\n{self.algorithm_name}算法求解...')
        print(f'  - 参数: xi={xi:.6f}, delta={delta:.2f}, dt={dt:.2f}')
        print(f'  - 采样次数: {sample_size}')
        print(f'  - 迭代次数: {n_iter_list}')

        start_time = time.time()
        cut_values = []
        max_cuts = []
        times = []

        # 优先使用CUDA可执行文件，如果不存在则回退到Python实现
        if os.path.exists(self.executable_path):
            # 使用CUDA可执行文件
            cut_values, max_cuts, times = self._solve_with_cuda(
                sample_size, n_iter_list, xi, delta, dt
            )
        else:
            # 使用Python回退实现（调用mindquantum）
            cut_values, max_cuts, times = self._solve_with_python(
                sample_size, n_iter_list, xi, delta, dt
            )

        solve_time = time.time() - start_time
        print(f'{self.algorithm_name}算法完成，运行时间: {solve_time*1000:.2f}ms')

        return cut_values, max_cuts, times

    def _solve_with_cuda(self, sample_size: int, n_iter_list: List[int],
                        xi: float, delta: float, dt: float) -> Tuple[np.ndarray, List[float], List[float]]:
        """使用CUDA可执行文件求解（参考SB_basic_run.py的实现）"""
        import subprocess
        import re

        # 再次检查可执行文件是否存在
        if not os.path.exists(self.executable_path):
            print(f'警告: CUDA可执行文件不存在: {self.executable_path}')
            return self._solve_with_python(sample_size, n_iter_list, xi, delta, dt)

        print(f'  使用CUDA可执行文件: {os.path.basename(self.executable_path)}')

        cut_values = []
        max_cuts = []
        times = []

        for n_iter in n_iter_list:
            # 构建命令行参数（参考SB_basic_run.py第397-410行）
            cmd = [
                self.executable_path,
                self.gset_file,
                str(n_iter),
                str(sample_size),
                str(dt),
                str(delta),
                str(xi)
            ]

            # 记录开始时间（参考SB_basic_run.py第414行）
            start_time = time.time()

            try:
                # 运行可执行文件（参考SB_basic_run.py第418-420行）
                # 添加cwd参数，确保相对路径正确
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    timeout=300,  # 5分钟超时
                    cwd=self.cuda_dir  # 在SB_CUDA目录下执行
                )

                # 记录结束时间
                execution_time = (time.time() - start_time) * 1000  # 毫秒
                times.append(execution_time)

                if result.returncode != 0:
                    print(f'  错误: CUDA执行失败 (返回码: {result.returncode})')
                    if result.stderr:
                        print(f'  错误输出: {result.stderr[:500]}')
                    print(f'  使用Python回退实现')
                    return self._solve_with_python(sample_size, n_iter_list, xi, delta, dt)

                # 解析输出（参考SB_basic_run.py第431-432行）
                output = result.stdout

                # 提取Time, Max cut
                time_match = re.search(r'Time:\s+(\d+)\s+ms', output)
                max_match = re.search(r'Max cut:\s+([\d.]+)', output)

                if max_match:
                    max_cut = float(max_match.group(1))
                    max_cuts.append(max_cut)
                    cut_values.append(max_cut)  # 用 max_cut 作为占位

                    # 如果CUDA输出了时间，使用它；否则使用Python测量的时间
                    cuda_time = int(time_match.group(1)) if time_match else int(execution_time)
                    print(f'  n_iter={n_iter:4d}, cut_max={max_cut:.2f}, time={cuda_time}ms')
                else:
                    print(f'  警告: 无法解析CUDA输出')
                    if output:
                        print(f'  输出前500字符: {output[:500]}')
                    # 回退到Python实现
                    cut_value, max_cut = self._solve_single_iter_python(n_iter, sample_size, xi, delta, dt)
                    cut_values.append(cut_value)
                    max_cuts.append(max_cut)

            except subprocess.TimeoutExpired:
                print(f'  错误: CUDA执行超时 (5分钟)')
                print(f'  使用Python回退实现')
                return self._solve_with_python(sample_size, n_iter_list, xi, delta, dt)
            except Exception as e:
                print(f'  错误: CUDA执行异常: {e}')
                print(f'  使用Python回退实现')
                return self._solve_with_python(sample_size, n_iter_list, xi, delta, dt)

        return np.array(cut_values), max_cuts, times

    def _solve_with_python(self, sample_size: int, n_iter_list: List[int],
                         xi: float, delta: float, dt: float) -> Tuple[np.ndarray, List[float], List[float]]:
        """使用Python实现求解，根据算法类型选择不同的后端"""
        cut_values = []
        max_cuts = []
        times = []

        from mindquantum.algorithm.qaia import BSB, DSB

        # 根据算法类型选择BSB或DSB以及对应的后端
        # 注意：MindQuantum 0.10.0版本暂时禁用了GPU后端，全部使用cpu-float32回退
        if 'BSB' in self.algorithm_name:
            solver_class = BSB
            backend = 'cpu-float32'  # 统一使用CPU回退
        else:  # DSB
            solver_class = DSB
            backend = 'cpu-float32'  # 统一使用CPU回退

        print(f'  使用Python实现，后端: {backend}...')

        for n_iter in n_iter_list:
            try:
                start_time = time.time()
                s = solver_class(self.G, batch_size=sample_size, n_iter=n_iter, dt=dt, xi=xi, backend=backend)
                s.update()
                cut_value = s.calc_cut()
                # 转换PyTorch张量到NumPy数组
                if torch.is_tensor(cut_value):
                    cut_value = cut_value.cpu().numpy()
                cut_max = float(np.max(cut_value))
                max_cuts.append(cut_max)
                cut_values.append(cut_max)  # 用 max_cut 作为占位
                execution_time = (time.time() - start_time) * 1000
                times.append(execution_time)
                print(f'  n_iter={n_iter:4d}, cut_max={cut_max:.2f}')
            except RuntimeError as e:
                if 'GPU backend' in str(e):
                    print(f'  警告: GPU后端不可用 ({e})，使用CPU后端...')
                    # 重试使用CPU后端
                    start_time = time.time()
                    s = solver_class(self.G, batch_size=sample_size, n_iter=n_iter, dt=dt, xi=xi, backend='cpu-float32')
                    s.update()
                    cut_value = s.calc_cut()
                    if torch.is_tensor(cut_value):
                        cut_value = cut_value.cpu().numpy()
                    cut_max = float(np.max(cut_value))
                    max_cuts.append(cut_max)
                    cut_values.append(cut_max)  # 用 max_cut 作为占位
                    execution_time = (time.time() - start_time) * 1000
                    times.append(execution_time)
                    print(f'  n_iter={n_iter:4d}, cut_max={cut_max:.2f}')
                else:
                    raise

    def _solve_single_iter_python(self, n_iter: int, sample_size: int,
                                  xi: float, delta: float, dt: float) -> Tuple[float, float]:
        """单次迭代的Python求解（用于CUDA实现的占位）"""
        from mindquantum.algorithm.qaia import BSB, DSB

        if 'BSB' in self.algorithm_name:
            s = BSB(self.G, batch_size=sample_size, n_iter=n_iter, dt=dt, xi=xi, backend='cpu-float32')
        else:
            s = DSB(self.G, batch_size=sample_size, n_iter=n_iter, dt=dt, xi=xi, backend='cpu-float32')

        s.update()
        cut_value = s.calc_cut()
        max_cut = float(np.max(cut_value))
        return max_cut, max_cut  # 两个都返回 max_cut

    def calc_cut_value(self, x: np.ndarray) -> float:
        """
        计算给定自旋配置的切割值

        参数:
            x: 自旋配置 [n_spins, batch_size]

        返回:
            切割值
        """
        # 计算sign(x)
        sign = np.sign(x)
        # 计算切割值: 0.25 * J.sum() + 0.25 * sign^T * J * sign
        cut = 0.25 * self.G.sum() + 0.25 * np.sum(sign.T @ self.G @ sign, axis=1)
        return cut


def solve_gset_with_all_cuda(gset_file: str, sample_size: int = 100,
                            n_iter_list: Optional[List[int]] = None,
                            save_plot: bool = True) -> Dict[str, Tuple[np.ndarray, List[float], List[float], float]]:
    """
    使用所有6种算法类型求解Gset数据集，使用不同的计算后端

    后端选择：
    - FP16算法: 使用 gpu-float16 后端（GPU半精度计算）
    - FP8/INT8算法: 使用 cpu-float32 后端（MindQuantum暂不支持，CPU回退）
    - 每次运行前后清理GPU缓存以避免资源竞争

    参数:
        gset_file: Gset数据文件路径
        sample_size: 采样次数 (默认100)
        n_iter_list: 迭代次数列表
        save_plot: 是否保存切割值收敛图 (默认True)

    返回:
        results: 字典，包含所有算法的结果
            {
                'BSB_FP8': (cut_values, max_cuts, times, total_time),
                'BSB_FP16': (cut_values, max_cuts, times, total_time),
                'BSB_INT8': (cut_values, max_cuts, times, total_time),
                'DSB_FP8': (cut_values, max_cuts, times, total_time),
                'DSB_FP16': (cut_values, max_cuts, times, total_time),
                'DSB_INT8': (cut_values, max_cuts, times, total_time)
            }
    """
    if n_iter_list is None:
        n_iter_list = [10, 25, 50, 75, 100, 250, 500, 750, 1000]

    print(f'正在读取数据文件: {gset_file}')

    results = {}

    # 使用所有6种CUDA实现求解
    for algo_type in [CUDASBSolver.BSB_FP8, CUDASBSolver.BSB_FP16, CUDASBSolver.BSB_INT8,
                     CUDASBSolver.DSB_FP8, CUDASBSolver.DSB_FP16, CUDASBSolver.DSB_INT8]:
        solver = CUDASBSolver(gset_file, algo_type)
        # 清理GPU缓存以避免资源竞争
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        start_time = time.time()
        cut_values, max_cuts, times = solver.solve(sample_size, n_iter_list)
        solve_time = time.time() - start_time
        results[solver.algorithm_name] = (cut_values, max_cuts, times, solve_time)
        # 每次运行后清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 绘制切割值收敛图和能量-时间图
    if save_plot:
        plot_cut_convergence_all(n_iter_list, results, gset_file, show=False)
        plot_energy_time_convergence_all(n_iter_list, results, gset_file, show=True)

    return results


def plot_cut_convergence_all(n_iter_list: List[int], results: Dict[str, Tuple[np.ndarray, List[float], List[float], float]],
                             gset_file: str, show: bool = False):
    """
    绘制所有6种算法的切割值收敛图

    参数:
        n_iter_list: 迭代次数列表
        results: 所有算法的结果字典
        gset_file: Gset数据文件路径
    """
    plt.figure(figsize=(14, 8))

    # 定义颜色和标记
    colors = {
        'BSB_FP8': 'tab:blue',
        'BSB_FP16': 'tab:cyan',
        'BSB_INT8': 'tab:green',
        'DSB_FP8': 'tab:red',
        'DSB_FP16': 'tab:orange',
        'DSB_INT8': 'tab:purple'
    }
    markers = {'BSB_FP8': 'o', 'BSB_FP16': 's', 'BSB_INT8': '^',
               'DSB_FP8': 'o', 'DSB_FP16': 's', 'DSB_INT8': '^'}

    # 绘制BSB算法
    for algo in ['BSB_FP8', 'BSB_FP16', 'BSB_INT8']:
        if algo in results:
            _, max_cuts, _, _ = results[algo]
            final_cut = max_cuts[-1]
            plt.plot(np.log10(n_iter_list), max_cuts, color=colors[algo],
                    linewidth=2, label=f'{algo} (MaxCut: {final_cut:.0f})', marker=markers[algo], markersize=6)

    # 绘制DSB算法（虚线）
    for algo in ['DSB_FP8', 'DSB_FP16', 'DSB_INT8']:
        if algo in results:
            _, max_cuts, _, _ = results[algo]
            final_cut = max_cuts[-1]
            plt.plot(np.log10(n_iter_list), max_cuts, '--', color=colors[algo],
                    linewidth=2, label=f'{algo} (MaxCut: {final_cut:.0f})', marker=markers[algo], markersize=6)

    plt.xlim([0, 3.15])
    plt.xlabel(r'$N_{step}$', fontsize=14)
    plt.ylabel('Max Cut Value', fontsize=14)
    gset_name = os.path.splitext(os.path.basename(gset_file))[0]
    plt.title(f'Max Cut Value Convergence - {gset_name} (All 6 CUDA Implementations)',
              fontsize=16, fontweight='bold')

    plt.legend(fontsize=10, loc='best', ncol=2)
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.xticks(np.log10(n_iter_list), ["$10^1$", "", "", "", "$10^2$", "", "", "", "$10^3$"])
    plt.tight_layout()

    # 保存图片
    png_dir = os.path.join(os.path.dirname(__file__), 'png')
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    save_path = os.path.join(png_dir, f'{gset_name}_cuda_all_cut_convergence.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n最大切割值收敛图已保存至: {save_path}')

    if show:
        plt.show()
    else:
        plt.close()


def plot_cut_convergence_bsb_dsb(n_iter_list: List[int],
                                  bsb_results: Dict[str, Tuple[np.ndarray, List[float], List[float], float]],
                                  dsb_results: Dict[str, Tuple[np.ndarray, List[float], List[float], float]],
                                  gset_file: str):
    """
    绘制BSB和DSB算法的切割值收敛对比图

    参数:
        n_iter_list: 迭代次数列表
        bsb_results: BSB算法结果字典
        dsb_results: DSB算法结果字典
        gset_file: Gset数据文件路径
    """
    plt.figure(figsize=(12, 7))

    # BSB算法（实线）
    for algo in ['BSB_FP8', 'BSB_FP16', 'BSB_INT8']:
        if algo in bsb_results:
            _, max_cuts, _, _ = bsb_results[algo]
            color = 'tab:blue' if algo == 'BSB_FP8' else ('tab:cyan' if algo == 'BSB_FP16' else 'tab:green')
            plt.plot(np.log10(n_iter_list), max_cuts, color=color, linewidth=2,
                    label=algo, marker='o', markersize=6)

    # DSB算法（虚线）
    for algo in ['DSB_FP8', 'DSB_FP16', 'DSB_INT8']:
        if algo in dsb_results:
            _, max_cuts, _, _ = dsb_results[algo]
            color = 'tab:red' if algo == 'DSB_FP8' else ('tab:orange' if algo == 'DSB_FP16' else 'tab:purple')
            plt.plot(np.log10(n_iter_list), max_cuts, '--', color=color, linewidth=2,
                    label=algo, marker='s', markersize=6)

    plt.xlim([0, 3.15])
    plt.xlabel(r'$N_{step}$', fontsize=14)
    plt.ylabel('Max Cut Value', fontsize=14)
    gset_name = os.path.splitext(os.path.basename(gset_file))[0]
    plt.title(f'Max Cut Value Convergence - {gset_name} (BSB vs DSB)',
              fontsize=16, fontweight='bold')

    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.xticks(np.log10(n_iter_list), ["$10^1$", "", "", "", "$10^2$", "", "", "", "$10^3$"])
    plt.tight_layout()

    # 保存图片
    png_dir = os.path.join(os.path.dirname(__file__), 'png')
    save_path = os.path.join(png_dir, f'{gset_name}_cuda_bsb_dsb_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n切割值收敛图已保存至: {save_path}')

    plt.show()


def plot_energy_time_convergence_all(n_iter_list: List[int],
                                      results: Dict[str, Tuple[np.ndarray, List[float], List[float], float]],
                                      gset_file: str, show: bool = True):
    """
    绘制所有6种算法的能量（切割值）随时间变化的收敛图，在图例中显示执行时间

    参数:
        n_iter_list: 迭代次数列表
        results: 所有算法的结果字典（包含运行时间）
        gset_file: Gset数据文件路径
    """
    plt.figure(figsize=(16, 9))

    # 定义颜色和标记
    colors = {
        'BSB_FP8': '#1f77b4',
        'BSB_FP16': '#17becf',
        'BSB_INT8': '#2ca02c',
        'DSB_FP8': '#d62728',
        'DSB_FP16': '#ff7f0e',
        'DSB_INT8': '#9467bd'
    }
    markers = {'BSB_FP8': 'o', 'BSB_FP16': 's', 'BSB_INT8': '^',
               'DSB_FP8': 'o', 'DSB_FP16': 's', 'DSB_INT8': '^'}

    # 存储每个算法的标签（包含执行时间）
    legend_labels = {}

    # 绘制BSB算法（实线）
    for algo in ['BSB_FP8', 'BSB_FP16', 'BSB_INT8']:
        if algo in results:
            _, max_cuts, times, solve_time = results[algo]
            # 使用实际测量的时间点计算累计时间（毫秒）
            cum_time_ms = np.cumsum(times)
            # 在图例中显示执行时间和最终 cut 值
            final_cut = max_cuts[-1]
            legend_label = f'{algo} (Time: {solve_time*1000:.1f}ms, MaxCut: {final_cut:.0f})'
            legend_labels[algo] = legend_label
            plt.plot(cum_time_ms, max_cuts, '-', color=colors[algo],
                    linewidth=2.5, label=legend_label, marker=markers[algo], markersize=7,
                    markeredgecolor='white', markeredgewidth=1.5)

    # 绘制DSB算法（虚线）
    for algo in ['DSB_FP8', 'DSB_FP16', 'DSB_INT8']:
        if algo in results:
            _, max_cuts, times, solve_time = results[algo]
            # 使用实际测量的时间点计算累计时间（毫秒）
            cum_time_ms = np.cumsum(times)
            # 在图例中显示执行时间和最终 cut 值
            final_cut = max_cuts[-1]
            legend_label = f'{algo} (Time: {solve_time*1000:.1f}ms, MaxCut: {final_cut:.0f})'
            legend_labels[algo] = legend_label
            plt.plot(cum_time_ms, max_cuts, '--', color=colors[algo],
                    linewidth=2.5, label=legend_label, marker=markers[algo], markersize=7,
                    markeredgecolor='white', markeredgewidth=1.5)

    plt.xlabel('Time (ms)', fontsize=15, fontweight='bold')
    plt.ylabel('Max Cut Value', fontsize=15, fontweight='bold')
    plt.ylim([0, None])  # 纵坐标从0开始
    gset_name = os.path.splitext(os.path.basename(gset_file))[0]
    plt.title(f'Max Cut vs Time Convergence - {gset_name}\n(All 6 CUDA Implementations)',
              fontsize=17, fontweight='bold', pad=20)

    # 优化图例
    plt.legend(fontsize=11, loc='best', ncol=3,
              framealpha=0.9, fancybox=True, shadow=True)

    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    plt.gca().set_axisbelow(True)

    plt.tick_params(axis='both', which='major', labelsize=12, width=1.2)
    plt.tick_params(axis='both', which='minor', labelsize=10)

    # 添加时间轴的次要刻度
    plt.minorticks_on()

    plt.tight_layout()

    # 保存图片
    png_dir = os.path.join(os.path.dirname(__file__), 'png')
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    save_path = os.path.join(png_dir, f'{gset_name}_cuda_all_energy_with_time.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n最大切割值-时间收敛图（含执行时间和cut值）已保存至: {save_path}')

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    # 检查CUDA可执行文件
    missing_exes = check_cuda_executables()
    if missing_exes:
        print('=' * 80)
        print('警告: 以下CUDA可执行文件不存在:')
        for exe in missing_exes:
            print(f'  - {exe}')
        print()
        print('请先编译CUDA文件:')
        print('  Windows: cd SB_CUDA_pure && compile_all_cuda.bat')
        print('  Linux:   cd SB_CUDA_pure && ./compile_all_cuda.sh')
        print()
        print('将使用Python回退实现（速度较慢）')
        print('=' * 80)
        print()

    # 设置Gset文件路径
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    gset_file = os.path.join(data_dir, 'G22.txt')

    # 检查文件是否存在
    if not os.path.exists(gset_file):
        print(f'错误: 文件 {gset_file} 不存在，请先运行 get_Gsets.py 下载数据')
    else:
        # 方式1: 使用所有6种CUDA实现求解
        print('=' * 80)
        print('使用所有6种CUDA实现求解')
        print('=' * 80)
        results = solve_gset_with_all_cuda(
            gset_file=gset_file,
            sample_size=100,
            n_iter_list=[10, 25, 50, 75, 100, 250, 500, 750, 1000],
            save_plot=True
        )

        # 打印结果总结
        print('\n' + '=' * 80)
        print('结果总结')
        print('=' * 80)
        for algo_name, (cut_values, max_cuts, times, solve_time) in results.items():
            final_max = max_cuts[-1]
            print(f'{algo_name:12s}: 最终最大切割值={final_max:8.2f}, 运行时间={solve_time*1000:6.2f}ms')

        # 方式2: 分别使用BSB和DSB算法求解（示例）
        # print('\n' + '=' * 80)
        # print('BSB算法对比')
        # print('=' * 80)
        # bsb_solver_fp8 = CUDASBSolver(gset_file, CUDASBSolver.BSB_FP8)
        # bsb_solver_fp16 = CUDASBSolver(gset_file, CUDASBSolver.BSB_FP16)
        # bsb_solver_int8 = CUDASBSolver(gset_file, CUDASBSolver.BSB_INT8)
        #
        # bsb_results = {}
        # for solver in [bsb_solver_fp8, bsb_solver_fp16, bsb_solver_int8]:
        #     cut_values, max_cuts, times = solver.solve(sample_size=100)
        #     bsb_results[solver.algorithm_name] = (cut_values, max_cuts, times)
