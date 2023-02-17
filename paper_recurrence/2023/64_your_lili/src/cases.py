"""复现论文中线路"""

import numpy as np
from mindquantum.core.gates import X, Z
from basic import create_random_state, create_EPR_state, \
    create_basic_module, get_measure_result


def simulate_fig3():
    """复现论文 Fig3 的线路，实现 EPR 传输
    """
    cir_send = create_random_state(0)     # 初始需要传输的态，对应论文中 |y>
    cir_epr12 = create_EPR_state(1, 2)   # 制备 EPR
    cir_mod01 = create_basic_module(0, 1)  # 线路

    # 使用同一个随机数种子，避免各次测量结果不同
    seed = np.random.randint(0, 0xff)  # 随机数种子
    cir_all = cir_send + cir_epr12 + cir_mod01  # 量子线路
    # get_qs 不会改变系统状态
    ket_str = cir_all.get_qs(ket=True, seed=seed)
    # 获取q0, q1测量结果，保存在c0, c1
    c0, c1 = get_measure_result(ket_str, [0, 1])
    # 根据测量结果判断是否加入 X 或 Z 门
    if c1 == '1':
        cir_all += X(2)
    if c0 == '1':
        cir_all += Z(2)

    send_state = cir_send.get_qs(ket=True, seed=seed)
    recv_state = cir_all.get_qs(ket=True, seed=seed)

    print(f'Fig.3\ncomplete circuit:\n{cir_all}\n')
    print(f'Measure result:\nq0={c0}, q1={c1}\n')
    print(f'Send state ¦qn,...,q0⟩:\n{send_state}\n')
    print(f'Recv state ¦qn,...,q0⟩:\n{recv_state}\n')


def simulate_fig5():
    """复现 Fig.5 线路"""
    cir_send = create_random_state(0)     # the send state |y>
    cir_epr1 = create_EPR_state(1, 2)     # EPR Pair [A2, C1]
    cir_epr2 = create_EPR_state(3, 4)     # EPR Pair [C2, B]
    cir_mod01 = create_basic_module(0, 1)
    cir_mod23 = create_basic_module(2, 3)

    cir_all = cir_send + cir_epr1 + cir_epr2 + cir_mod01

    # 使用同一个随机数种子，避免各次测量结果不同
    seed = np.random.randint(0, 0xff)  # 随机数种子

    # 第一次测量
    # get_qs 不会改变系统状态
    ket_str = cir_all.get_qs(ket=True, seed=seed)
    # 获取q0, q1测量结果，保存在c0, c1
    c0, c1 = get_measure_result(ket_str, [0, 1])
    # 根据测量结果判断是否加入 X 或 Z 门
    if c1 == '1':
        cir_all += X(2)
    if c0 == '1':
        cir_all += Z(2)

    # 第二次测量
    cir_all += cir_mod23  # 加上第2部分测量
    # get_qs 不会改变系统状态
    ket_str = cir_all.get_qs(ket=True, seed=seed)
    # 获取q2, q3测量结果，保存在c2, c3
    c2, c3 = get_measure_result(ket_str, [2, 3])
    # 根据测量结果判断是否加入 X 或 Z 门
    if c3 == '1':
        cir_all += X(4)
    if c2 == '1':
        cir_all += Z(4)

    print(f'Fig.5\ncomplete circuit:\n{cir_all}\n')
    print(f'measure result:\nq0 = {c0}, q1 = {c1}\n')
    print(f'send state ¦qn,...,q0⟩:\n{cir_send.get_qs(ket=True, seed=seed)}\n')
    print(f'recv state ¦qn,...,q0⟩:\n{cir_all.get_qs(ket=True, seed=seed)}\n')


def simulate_fig6():
    """复现 Fig.6 线路"""
    cir_epr1 = create_EPR_state(0, 1)  # EPR Pair [A, C1]
    cir_epr2 = create_EPR_state(2, 3)  # EPR Pair [C2, B]
    cir_mod12 = create_basic_module(1, 2)
    cir_all = cir_epr1 + cir_epr2 + cir_mod12

    # 使用同一个随机数种子，避免各次测量结果不同
    seed = np.random.randint(0, 0xff)  # 随机数种子

    # get_qs 不会改变系统状态
    ket_str = cir_all.get_qs(ket=True, seed=seed)
    # 获取q0, q1测量结果，保存在c0, c1
    c1, c2 = get_measure_result(ket_str, [1, 2])
    # 根据测量结果判断是否加入 X 或 Z 门
    if c2 == '1':
        cir_all += X(3)
    if c1 == '1':
        cir_all += Z(3)

    print(f'Fig.6\ncomplete circuit:\n{cir_all}\n')
    print(f'measure result:\nq1 = {c1}, q2 = {c2}\n')
    print(f'last state ¦qn,...,q0⟩:\n{cir_all.get_qs(ket=True, seed=seed)}\n')


def simulate_fig8():
    """复现 Fig.8 线路"""
    cir_send = create_random_state(0)
    cir_epr12 = create_EPR_state(1, 2)
    cir_epr34 = create_EPR_state(3, 4)
    cir_mod23 = create_basic_module(2, 3)
    cir_mod01 = create_basic_module(0, 1)

    cir_all = cir_send + cir_epr12 + cir_epr34 + cir_mod23

    # 使用同一个随机数种子，避免各次测量结果不同
    seed = np.random.randint(0, 0xff)  # 随机数种子

    # 第一次测量
    # get_qs 不会改变系统状态
    ket_str = cir_all.get_qs(ket=True, seed=seed)
    # 获取q2, q3测量结果，保存在c2, c3
    c2, c3 = get_measure_result(ket_str, [2, 3])
    # 根据测量结果判断是否加入 X 或 Z 门
    if c3 == '1':
        cir_all += X(4)
    if c2 == '1':
        cir_all += Z(4)

    # 第二次测量
    cir_all += cir_mod01  # 加上第2部分测量
    # get_qs 不会改变系统状态
    ket_str = cir_all.get_qs(ket=True, seed=seed)
    # 获取q0, q2测量结果，保存在c0, c1
    c0, c1 = get_measure_result(ket_str, [0, 1])
    # 根据测量结果判断是否加入 X 或 Z 门
    if c1 == '1':
        cir_all += X(4)
    if c0 == '1':
        cir_all += Z(4)

    print(f'Fig.8\ncomplete circuit:\n{cir_all}\n')
    print(f'measure result:\nq3={c3}, q2={c2}, q1={c1}, q0={c0}\n')
    print(f'send state ¦qn,...,q0⟩:\n{cir_send.get_qs(ket=True, seed=seed)}\n')
    print(f'recv state ¦qn,...,q0⟩:\n{cir_all.get_qs(ket=True, seed=seed)}\n')


def simulate_fig9():
    """复现 Fig.9 线路"""
    cir_send = create_random_state(0)
    cir_epr12 = create_EPR_state(1, 2)
    cir_epr34 = create_EPR_state(3, 4)
    cir_mod01 = create_basic_module(0, 1)
    cir_mod23 = create_basic_module(2, 3)

    cir_all = cir_send + cir_epr12 + cir_epr34 + cir_mod01 + cir_mod23

    # 使用同一个随机数种子，避免各次测量结果不同
    seed = np.random.randint(0, 0xff)  # 随机数种子

    # get_qs 不会改变系统状态
    ket_str = cir_all.get_qs(ket=True, seed=seed)
    # 获取q0-q3测量结果，保存在c0-c3
    c0, c1, c2, c3 = get_measure_result(ket_str, [0, 1, 2, 3])
    a1, a2, cc1, cc2 = c0, c1, c2, c3  # 对应公式表示
    # 根据测量结果判断是否加入 X 或 Z 门
    if a2 != cc2:  # XOR
        cir_all += X(4)
    if a1 != cc1:  # XOR
        cir_all += Z(4)

    print(f'Fig.9\ncomplete circuit:\n{cir_all}\n')
    print(f'Measure result:\nq3={c3}, q2={c2}, q1={c1}, q0={c0}\n')
    print(f'Send state ¦qn,...,q0⟩:\n{cir_send.get_qs(ket=True, seed=seed)}\n')
    print(f'Recv state ¦qn,...,q0⟩:\n{cir_all.get_qs(ket=True, seed=seed)}\n')


def simulate_fig13():
    """复现 Fig.13 线路, same to fig9"""
    cir_send = create_random_state(0)
    cir_epr12 = create_EPR_state(1, 2)
    cir_epr34 = create_EPR_state(3, 4)
    cir_mod01 = create_basic_module(0, 1)
    cir_mod23 = create_basic_module(2, 3)

    cir_all = cir_send + cir_epr12 + cir_epr34 + cir_mod01 + cir_mod23

    # 使用同一个随机数种子，避免各次测量结果不同
    seed = np.random.randint(0, 0xff)  # 随机数种子

    # get_qs 不会改变系统状态
    ket_str = cir_all.get_qs(ket=True, seed=seed)
    # 获取q0-q3测量结果，保存在c0-c3
    c0, c1, c2, c3 = get_measure_result(ket_str, [0, 1, 2, 3])
    a1, a2, cc1, cc2 = c0, c1, c2, c3  # 对应公式表示
    # 根据测量结果判断是否加入 X 或 Z 门
    if a2 != cc2:  # XOR
        cir_all += X(4)
    if a1 != cc1:  # XOR
        cir_all += Z(4)

    print(f'Fig.13\ncomplete circuit:\n{cir_all}\n')
    print(f'Measure result:\nq3={c3}, q2={c2}, q1={c1}, q0={c0}\n')
    print(f'Send state ¦qn,...,q0⟩:\n{cir_send.get_qs(ket=True, seed=seed)}\n')
    print(f'Recv state ¦qn,...,q0⟩:\n{cir_all.get_qs(ket=True, seed=seed)}\n')


def simulate_fig14():
    """复现 Fig.14 线路"""
    cir_send = create_random_state(0)  # the send state
    cir_epr12 = create_EPR_state(1, 2)
    cir_epr34 = create_EPR_state(3, 4)
    cir_epr56 = create_EPR_state(5, 6)

    cir_mod01 = create_basic_module(0, 1)
    cir_mod23 = create_basic_module(2, 3)
    cir_mod45 = create_basic_module(4, 5)

    cir_all = cir_send + cir_epr12 + cir_epr34 + cir_epr56 +\
        cir_mod01 + cir_mod23 + cir_mod45

    # 使用同一个随机数种子，避免各次测量结果不同
    seed = np.random.randint(0, 0xff)  # 随机数种子

    # get_qs 不会改变系统状态
    ket_str = cir_all.get_qs(ket=True, seed=seed)
    # 获取q0-q3测量结果，保存在c0-c3
    c0, c1, c2, c3, c4, c5 = get_measure_result(ket_str, [0, 1, 2, 3, 4, 5])
    # 根据测量结果判断是否加入 X 或 Z 门
    i0, i1, i2, i3, i4, i5 = int(c0), int(c1), int(c2), \
        int(c3), int(c4), int(c5)
    if i1+i3+i5 in [1, 3]:  # XOR
        cir_all += X(6)
    if i0+i2+i4 in [1, 3]:  # XOR
        cir_all += Z(6)

    print(f'Fig.14\ncomplete circuit:\n{cir_all}\n')
    print(
        f'Measure result:\nq5={c5}, q4={c4}, q3={c3}, q2={c2}, q1={c1}, q0={c0}\n')
    print(f'Send state ¦qn,...,q0⟩:\n{cir_send.get_qs(ket=True, seed=seed)}\n')
    print(f'Recv state ¦qn,...,q0⟩:\n{cir_all.get_qs(ket=True, seed=seed)}\n')
