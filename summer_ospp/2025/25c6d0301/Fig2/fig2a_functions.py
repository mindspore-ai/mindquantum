"""
Fig2a 核心函数模块
包含 Spherical t-design 可视化的所有核心逻辑
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


def load_design(path):
    """
    从文件中加载 spherical design 坐标数据
    
    Args:
        path: 文件路径
        
    Returns:
        coords: 坐标数组，shape=(N, 3)
    """
    text = Path(path).read_text()
    match = re.search(r'[Yy]\s*=\s*\[(.*?)\]\s*\';', text, re.S)
    if not match:
        raise ValueError(f'Could not find data block in {path}')
    block = match.group(1)
    numbers = [float(x) for x in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+', block)]
    coords = np.asarray(numbers).reshape(-1, 3)
    return coords


def plot_spherical_design(title, coords, view_elev=20, view_azim=35):
    """
    绘制 spherical design 的 3D 可视化
    
    Args:
        title: 图形标题
        coords: 坐标数组，shape=(N, 3)
        view_elev: 仰角
        view_azim: 方位角
        
    Returns:
        fig: matplotlib figure 对象
    """
    phi = np.linspace(0, np.pi, 120)
    theta = np.linspace(0, 2 * np.pi, 120)
    X = np.outer(np.sin(phi), np.cos(theta))
    Y = np.outer(np.sin(phi), np.sin(theta))
    Z = np.outer(np.cos(phi), np.ones_like(theta))
    
    light_rgba = (0.93, 0.86, 0.95, 0.55)
    front_color = '#0b1f66'
    back_color = '#96b4ff'
    edge_color = '#0b1f66'
    
    view_dir = np.array([
        np.cos(np.deg2rad(view_elev)) * np.cos(np.deg2rad(view_azim)),
        np.cos(np.deg2rad(view_elev)) * np.sin(np.deg2rad(view_azim)),
        np.sin(np.deg2rad(view_elev)),
    ])
    view_dir /= np.linalg.norm(view_dir)
    
    visibility = coords @ view_dir
    front_mask = visibility >= 0
    
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(X, Y, Z, color=light_rgba, linewidth=0, antialiased=True)
    # front points use a darker shade
    ax.scatter(coords[front_mask, 0], coords[front_mask, 1], coords[front_mask, 2],
               s=36, c=front_color, edgecolor=edge_color, linewidth=0.2, depthshade=False)
    # back-facing points use a lighter shade
    ax.scatter(coords[~front_mask, 0], coords[~front_mask, 1], coords[~front_mask, 2],
               s=36, c=back_color, edgecolor=edge_color, linewidth=0.2, depthshade=False)
    
    ax.set_title(f"{title}", pad=12)
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    
    return fig


def generate_all_figures(data_dir='data/spherical_designs'):
    """
    生成所有 spherical design 的可视化图形
    
    Args:
        data_dir: 数据文件目录
        
    Returns:
        figs: 图形列表
    """
    data_dir = Path(data_dir)
    designs = [
        ('3-design', data_dir / 'P3_16.m'),
        ('6-design', data_dir / 'P6_49.m'),
        ('9-design', data_dir / 'P9_100.m'),
        ('12-design', data_dir / 'P12_169.m'),
    ]
    
    figs = []
    for title, file_path in designs:
        coords = load_design(file_path)
        fig = plot_spherical_design(title, coords)
        figs.append(fig)
    
    return figs

