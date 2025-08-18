import numpy as np
import random

from mindquantum.core.gates import I, X, H, Z, Y, S
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.operators import Hamiltonian


# 分解Rzy门 (Z作用在idx1上, Y作用在idx2上)
def Rzy(circ, idx1, idx2):
    circ += S.hermitian().on(idx2)
    circ += H.on(idx2)
    circ += X.on(idx2, idx1)
    circ += S.hermitian().on(idx2)
    circ += X.on(idx2, idx1)
    circ += H.on(idx2)
    circ += S.on(idx2)
    return circ


# 分解Ryx门 (Y作用在idx1上, X作用在idx2上)
def Ryx(circ, idx1, idx2):
    circ += H.on(idx2)
    circ += S.hermitian().on(idx1)
    circ += H.on(idx1)
    circ += X.on(idx2, idx1)
    circ += S.hermitian().on(idx2)
    circ += X.on(idx2, idx1)
    circ += H.on(idx2)
    circ += H.on(idx1)
    circ += S.on(idx1)
    return circ


# 分解Ry门 (作用在idx2上, 参数为pi/4)
def Ry1(circ, idx2):
    circ += H.on(idx2)
    circ += Z.on(idx2)
    return circ


# 分解Ry门 (作用在idx2上, 参数为-pi/4)
def Ry2(circ, idx2):
    circ += Z.on(idx2)
    circ += H.on(idx2)
    return circ


# 预处理的ADAPT-Clifford算法, k为起始节点, type=0作用YjXk
# type=1作用YkXj, r > 1时作用两比特门
def ADAPT_circ_pre(nqubit, Q_triu, k, type):
    J = np.triu(Q_triu, k=1) + np.triu(Q_triu, k=1).T
    h = np.diag(Q_triu)
    # 初始化量子态
    circ = Circuit(UN(I, nqubit))
    for i in range(nqubit):
        circ += H.on(i)
    circ += Z.on(k)
    active = [k]
    active_j = []  # active_j 用来存储与j取值相同的节点
    active_k = []  # active_k 用来存储与k取值相同的节点
    inactive = [i for i in range(nqubit)]
    inactive.remove(k)
    pre_grad_kb = [0 for i in range(nqubit)]  # 用来存储预处理梯度

    grad_jk, pre, energy = 0, -1, 0
    while len(inactive) > 0:
        if len(active) == 1:  # r = 1时的操作
            grad, idx = -100, -1
            # 选择梯度最大的节点j，做Rzy操作
            for j in inactive:
                grad_jk = 4 * J[j, k]
                if grad < grad_jk:
                    grad = grad_jk
                    idx = j
            active.append(idx)
            inactive.remove(idx)
            circ = Rzy(circ, idx, k)
            energy += grad
            # 对节点k和j做Ryx或者Rxy操作
            if type == 0:
                circ = Ryx(circ, active[1], k)
                energy += 2 * h[active[1]] - 2 * h[k]
            else:
                circ = Ryx(circ, k, active[1])
                energy -= 2 * h[active[1]] - 2 * h[k]
            # 预处理梯度信息
            for b in inactive:
                if type == 0:
                    grad_kb = -2 * h[b] - 4 * J[k, b] \
                              + 4 * J[active[1], b]
                    pre_grad_kb[b] = grad_kb
                else:
                    grad_kb = 2 * h[b] - 4 * J[k, b] \
                              + 4 * J[active[1], b]
                    pre_grad_kb[b] = grad_kb
        else:  # r > 1时的操作
            idx1, idx2 = -1, -1
            grad = -100
            # 计算所有不活跃节点梯度
            for b in inactive:
                grad_kb = pre_grad_kb[b]
                # 根据上次节点加入的集合，对预处理过的梯度进行修正
                if pre in active_k:
                    grad_kb -= 4 * J[pre, b]
                elif pre in active_j:
                    grad_kb += 4 * J[pre, b]
                pre_grad_kb[b] = grad_kb
                grad_jb = -grad_kb
                if grad < grad_jb:
                    grad = grad_jb
                    idx1, idx2 = active[1], b
                if grad < grad_kb:
                    grad = grad_kb
                    idx1, idx2 = k, b
            inactive.remove(idx2)
            pre = idx2
            energy += grad
            # 更新active_k，active_j集合与量子线路
            if idx1 == k:
                active_k.append(idx2)
            else:
                active_j.append(idx2)
            circ = Rzy(circ, idx1, idx2)

    active_k.append(k)
    active_j.append(active[1])
    energy = -0.5 * energy
    # 调整active_k与active_j的相对位置
    # 保证在前的是+1的集合，在后的是-1的集合
    if type == 0:
        return circ, active_k, active_j, energy
    else:
        return circ, active_j, active_k, energy


# 预处理的ADAPT-Clifford算法, k为起始节点, type=0作用YjXk
# type=1作用YkXj, r > 1时作用单比特门
def ADAPT_circ_pre_single(nqubit, Q_triu, k, type):
    J = np.triu(Q_triu, k=1) + np.triu(Q_triu, k=1).T
    h = np.diag(Q_triu)
    # 初始化量子态
    circ = Circuit(UN(I, nqubit))
    for i in range(nqubit):
        circ += H.on(i)
    circ += Z.on(k)
    active = [k]
    active_j = []  # active_j 用来存储与j取值相同的节点
    active_k = []  # active_k 用来存储与k取值相同的节点
    inactive = [i for i in range(nqubit)]
    inactive.remove(k)
    pre_grad_kb = [0 for i in range(nqubit)]  # 用来存储预处理梯度

    grad_jk, pre, energy = 0, -1, 0
    while len(inactive) > 0:
        if len(active) == 1:  # r = 1时的操作
            grad, idx = -100, -1
            # 选择梯度最大的节点j，做Rzy操作
            for j in inactive:
                grad_jk = 4 * J[j, k]
                if grad < grad_jk:
                    grad = grad_jk
                    idx = j
            active.append(idx)
            inactive.remove(idx)
            circ = Rzy(circ, idx, k)
            energy += grad
            # 对节点k和j做Ryx或者Rxy操作
            if type == 0:
                circ = Ryx(circ, active[1], k)
                energy += 2 * h[active[1]] - 2 * h[k]
            else:
                circ = Ryx(circ, k, active[1])
                energy -= 2 * h[active[1]] - 2 * h[k]
            # 预处理梯度信息
            for b in inactive:
                if type == 0:
                    grad_kb = -2 * h[b] - 4 * J[k, b] \
                              + 4 * J[active[1], b]
                    pre_grad_kb[b] = grad_kb
                else:
                    grad_kb = 2 * h[b] - 4 * J[k, b] \
                              + 4 * J[active[1], b]
                    pre_grad_kb[b] = grad_kb
        else:  # r > 1时的操作
            idx1, idx2 = -1, -1
            grad = -100
            # 计算所有不活跃节点梯度
            for b in inactive:
                grad_kb = pre_grad_kb[b]
                # 根据上次节点加入的集合，对预处理过的梯度进行修正
                if pre in active_k:
                    grad_kb -= 4 * J[pre, b]
                elif pre in active_j:
                    grad_kb += 4 * J[pre, b]
                pre_grad_kb[b] = grad_kb
                grad_jb = -grad_kb
                if grad < grad_jb:
                    grad = grad_jb
                    idx1, idx2 = active[1], b
                if grad < grad_kb:
                    grad = grad_kb
                    idx1, idx2 = k, b
            inactive.remove(idx2)
            pre = idx2
            energy += grad
            # 更新active_k，active_j集合与量子线路
            if idx1 == k:
                active_k.append(idx2)
                if type == 0:
                    circ = Ry1(circ, idx2)
                else:
                    circ = Ry2(circ, idx2)
            else:
                active_j.append(idx2)
                if type == 0:
                    circ = Ry2(circ, idx2)
                else:
                    circ = Ry1(circ, idx2)

    active_k.append(k)
    active_j.append(active[1])
    energy = -0.5 * energy
    # 调整active_k与active_j的相对位置
    # 保证在前的是+1的集合，在后的是-1的集合
    if type == 0:
        return circ, active_k, active_j, energy
    else:
        return circ, active_j, active_k, energy


# 单比特翻转优化，同样用到了预处理
def one_flip(nqubit, Q_triu, circ, ap, am, energy):
    J = np.triu(Q_triu, k=1) + np.triu(Q_triu, k=1).T
    h = np.diag(Q_triu)

    det = 0
    dif = [0 for i in range(nqubit)]  # 预处理梯度信息
    for i in range(nqubit):
        for k in ap:
            dif[i] += 4 * J[i, k]
        for j in am:
            dif[i] -= 4 * J[i, j]

    while det >= 0:
        det, flip = 0, 0
        for i in ap:
            det_i = 2 * h[i] + dif[i]
            if det < det_i:
                det = det_i
                flip = i
        for i in am:
            det_i = -2 * h[i] - dif[i]
            if det < det_i:
                det = det_i
                flip = i
        if det == 0:
            break
        # flip为要翻转的比特
        circ += X.on(flip)
        energy -= det
        # 对预处理的梯度信息进行修正
        if flip in ap:
            am.append(flip)
            ap.remove(flip)
            for i in range(nqubit):
                dif[i] -= 8 * J[i, flip]
        else:
            ap.append(flip)
            am.remove(flip)
            for i in range(nqubit):
                dif[i] += 8 * J[i, flip]

    return circ, ap, am, energy


# 双比特翻转优化，所选择的两个比特来自同一个集合
def two_flip(nqubit, Q_triu, circ, ap, am, energy):
    J = np.triu(Q_triu, k=1) + np.triu(Q_triu, k=1).T
    h = np.diag(Q_triu)
    det = 0
    dif = [0 for i in range(nqubit)]  # 预处理梯度信息
    for i in range(nqubit):
        for k in ap:
            dif[i] += 4 * J[i, k]
        for j in am:
            dif[i] -= 4 * J[i, j]

    while det >= 0:
        det, ik, ij = 0, 0, 0
        for k in ap:
            for j in ap:
                if j == k:
                    continue
                det_jk = 2 * (h[k] + h[j]) - 8 * J[j, k]
                det_jk += dif[k] + dif[j]
                if det < det_jk:
                    det = det_jk
                    ik, ij = k, j
        for k in am:
            for j in am:
                if j == k:
                    continue
                det_jk = -2 * (h[k] + h[j]) - 8 * J[j, k]
                det_jk -= (dif[k] + dif[j])
                if det < det_jk:
                    det = det_jk
                    ik, ij = k, j
        if det == 0:
            break
        energy -= det
        # ij, ik为需要翻转的比特
        circ += X.on(ik)
        circ += X.on(ij)
        if ik in ap:
            ap.remove(ik)
            ap.remove(ij)
            am.append(ik)
            am.append(ij)
            # 对预处理的梯度信息进行修正
            for i in range(nqubit):
                dif[i] = dif[i] - 8 * J[i, ik] - 8 * J[i, ij]
        else:
            am.remove(ik)
            am.remove(ij)
            ap.append(ik)
            ap.append(ij)
            # 对预处理的梯度信息进行修正
            for i in range(nqubit):
                dif[i] = dif[i] + 8 * J[i, ik] + 8 * J[i, ij]
    return circ, ap, am, energy


# 双比特翻转优化，所选择的两个比特来自不同集合
def swap_flip(nqubit, Q_triu, circ, ap, am, energy):
    J = np.triu(Q_triu, k=1) + np.triu(Q_triu, k=1).T
    h = np.diag(Q_triu)
    det = 0
    dif = [0 for i in range(nqubit)]  # 预处理梯度信息
    for i in range(nqubit):
        for k in ap:
            dif[i] += 4 * J[i, k]
        for j in am:
            dif[i] -= 4 * J[i, j]

    while det >= 0:
        det, ik, ij = 0, 0, 0
        for k in ap:
            for j in am:
                det_jk = 2 * (h[k] - h[j]) + 8 * J[j, k]
                det_jk += dif[k] - dif[j]
                if det < det_jk:
                    det = det_jk
                    ik, ij = k, j
        if det == 0:
            break
        energy -= det
        # ij, ik为需要翻转的比特
        circ += X.on(ik)
        circ += X.on(ij)
        ap.remove(ik)
        am.remove(ij)
        ap.append(ij)
        am.append(ik)
        # 对预处理的梯度信息进行修正
        for i in range(nqubit):
            dif[i] = dif[i] - 8 * J[i, ik] + 8 * J[i, ij]
    return circ, ap, am, energy


# 随机比特翻转优化，对量子态进行扰动，尝试跳出局部最优
def random_Xflip(nqubit, Q_triu, circ, ap, am, energy):
    J = np.triu(Q_triu, k=1) + np.triu(Q_triu, k=1).T
    h = np.diag(Q_triu)
    select_num = random.sample(range(nqubit), int(0.04 * nqubit))  # 随机选择一定比例的比特进行翻转
    circ_tmp = circ.copy()
    ap_tmp, am_tmp, energy_tmp = ap.copy(), am.copy(), energy
    for idx in select_num:
        circ_tmp += X.on(idx)
        det_i = 2 * h[idx]
        for k in ap_tmp:
            det_i += 4 * J[idx, k]
        for j in am_tmp:
            det_i -= 4 * J[idx, j]
        if idx in am_tmp:
            det_i *= -1
        energy_tmp -= det_i
        if idx in ap_tmp:
            ap_tmp.remove(idx)
            am_tmp.append(idx)
        else:
            ap_tmp.append(idx)
            am_tmp.remove(idx)

    # 随机翻转后，再进行比特翻转优化，看是否能找到更好的结果
    circ_tmp, ap_tmp, am_tmp, val_tmp = \
        swap_flip(nqubit, Q_triu, circ_tmp, ap_tmp, am_tmp, energy_tmp)
    circ_tmp, ap_tmp, am_tmp, val_tmp = \
        two_flip(nqubit, Q_triu, circ_tmp, ap_tmp, am_tmp, val_tmp)
    circ_tmp, ap_tmp, am_tmp, val_tmp = \
        one_flip(nqubit, Q_triu, circ_tmp, ap_tmp, am_tmp, val_tmp)

    return circ_tmp, val_tmp, ap_tmp, am_tmp


def solve(nqubit, Q_triu):
    num = max(20, int(0.035 * nqubit))  # 选取的随机初始点数量
    select_k = random.sample(range(nqubit), num)  # 选取的随机初始点list
    circ_best, energy_best = None, 0
    ap, am = [], []

    # 对于每个起点，分别计算两种type下的结果，保留最好的
    for k in select_k:
        # 预处理优化的ADAPT-Clifford算法(单比特版)
        circ1, ap1, am1, val1 = \
            ADAPT_circ_pre_single(nqubit, Q_triu, k, 0)
        circ2, ap2, am2, val2 = \
            ADAPT_circ_pre_single(nqubit, Q_triu, k, 1)
        if energy_best > val1:
            energy_best = val1
            circ_best = circ1
            ap, am = ap1, am1
        if energy_best > val2:
            energy_best = val2
            circ_best = circ2
            ap, am = ap2, am2

    # 比特翻转优化
    circ_best, ap, am, energy_best = \
        swap_flip(nqubit, Q_triu, circ_best, ap, am, energy_best)
    circ_best, ap, am, energy_best = \
        two_flip(nqubit, Q_triu, circ_best, ap, am, energy_best)
    circ_best, ap, am, energy_best = \
        one_flip(nqubit, Q_triu, circ_best, ap, am, energy_best)

    # 随机比特翻转优化
    for _ in range(6):
        circ_tmp, val_tmp, ap_tmp, am_tmp = \
            random_Xflip(nqubit, Q_triu, circ_best, ap, am, energy_best)
        if val_tmp < energy_best:
            energy_best = val_tmp
            circ_best = circ_tmp
            ap, am = ap_tmp, am_tmp

    return circ_best

