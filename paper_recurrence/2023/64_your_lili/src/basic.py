"""模块化设计量子线路"""

import numpy as np
from mindquantum.core.gates import X, H, RX, RY, RZ, CNOT
from mindquantum.core.gates import Measure
from mindquantum.core.circuit import Circuit


def create_epr_state(p: int, q: int) -> Circuit:
    """制备EPR Pair
    Args:
        p: 贝尔态作用第1个位置
        q: 贝尔态作用第2个位置
    Return:
        制备贝尔态线路
    """
    return Circuit([
        H(p),
        X(q, p)
    ])


def create_random_state(p: int) -> Circuit:
    """在1个比特上制备随机态，通过随机旋转初态制备
    Args:
        p: 随机值作用比特
    Return:
        处于随机的单量子态
    """
    # 使用RX, RY, RZ随机旋转角度作用在 |0> 上实现随机状态制备
    t1, t2, t3 = 2 * np.pi * np.random.random(size=3)
    return Circuit([
        RX(t1).on(p),  # 不使用函数 on() 直接使用 RX(t1)(p) 也可
        RY(t2).on(p),
        RZ(t3).on(p)
    ])


def create_basic_module(p: int, q: int) -> Circuit:
    """CNOT,H,Measure 的组合在本位量子通信经常用到，综合为一个接口
    Args:
        p: 第1个量子位置
        q: 第2个量子位置
    Return:
        量子线路模块
    """
    return Circuit([
        CNOT(q, p),
        H(p),
        Measure(f'q{p}').on(p),
        Measure(f'q{q}').on(q)
    ])


def get_measure_result(ket_str: str, idx: int or list) -> int or list:
    """获取指定状态的测量结果，通过解析 get_qs(ket=True) 的返回值实现
    Args:
        ket_str: get_qs(ket=True)获取的量子态
        idx: 需要获取的量子位
    Return:
        指定量子位的测量结果，
    """
    ket_str2 = ket_str.split('\n')[0]
    if isinstance(idx, int):
        new_idx = -idx-2
        return ket_str2[new_idx]
    if isinstance(idx, list):
        return [ket_str2[-i-2] for i in idx]
    print("Error: idx should be int or list!")
    return -1
