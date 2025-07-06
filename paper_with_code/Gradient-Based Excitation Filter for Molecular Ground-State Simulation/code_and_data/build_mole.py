"""生成各种分子的结构"""

import numpy as np


def get_HF_geometry(dist):
    """HF 分子"""
    geometry = [['H', [0.0, 0.0, 0.0 * dist]],
                ['F', [0.0, 0.0, 1.0 * dist]]]
    return geometry


def get_LiH_geometry(dist):
    """LiH 分子"""
    geometry = [['H', [0.0, 0.0, 0.0 * dist]],
                ['Li', [0.0, 0.0, 1.0 * dist]]]
    return geometry



def get_H4_geometry(dist):
    """H4 链式分子"""
    geometry = [['H', [0.0, 0.0, 0.0 * dist]],
                ['H', [0.0, 0.0, 1.0 * dist]],
                ['H', [0.0, 0.0, 2.0 * dist]],
                ['H', [0.0, 0.0, 3.0 * dist]],]
    return geometry


def get_H2O_geometry(dist):
    """H2O 水分子"""
    angle = np.radians(104.5)
    geometry = [['O', [0.0, 0.0, 0.0]],  # 氧原子位于原点
                ['H', [dist, 0.0, 0.0]],  # 第一个氢原子在 x 轴上
                ['H', [dist * np.cos(angle), dist * np.sin(angle), 0.0]]]
    return geometry


def get_BeH2_geometry(dist):
    """BeH2 分子"""
    geometry = [['Be', [0.0, 0.0, 0.0]],  # 铍原子位于原点
                ['H', [0.0, 0.0, dist]],  # 第一个氢原子在 z 轴上
                ['H', [0.0, 0.0, -dist]]]  # 第二个氢原子在 z 轴负方向上
    return geometry


def get_NH3_geometry(dist):
    """NH3 分子"""
    theta = np.radians(107.0)  # 将角度转换为弧度
    x = dist * np.sin(theta)
    z = dist * np.cos(theta)

    geometry = [['N', [0.0, 0.0, 0.0]],  # 氮原子位于原点
                ['H', [x, 0.0, z]],       # 第一个氢原子
                ['H', [-0.5 * x, np.sqrt(3)/2 * x, z]],  # 第二个氢原子
                ['H', [-0.5 * x, -np.sqrt(3)/2 * x, z]]]  # 第三个氢原子
    return geometry


def get_CH4_geometry(dist):
    """CH4 分子"""

    geometry = [['C', [0.0, 0.0, 0.0]],
                ['H', [dist, dist, dist]],
                ['H', [dist, -dist, -dist]],
                ['H', [-dist, dist, -dist]],
                ['H', [-dist, -dist, dist]]]
    return geometry







