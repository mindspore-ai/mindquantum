import os

os.environ['OMP_NUM_THREADS'] = '2'
from itertools import count
import numpy as np
import time
import copy
import sys
import random
import matplotlib.pyplot as plt
import pickle
import math
import networkx as nx
from utils import *

'''
本赛题旨在引领选手探索，在NISQ时代规模有限的量子计算机上，求解真实场景中的大规模图分割问题。
min-cut: 使得ZiZj项的和最大，而不是最大割里使其最小化
Hamiltonian: 最小化
    -Sum_{ij in g.edges} ZiZj + penalty * (Sum Zi)^2
    ~ -Sum_{ij in g.edges} ZiZj + penalty * (Sum_{i,j in N} ZiZj)
'''

import resource

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 4000, hard))  # 4G
N_sub = 15  # 量子比特的规模限制

filelist = ['graphs/weight_p0.5_n40_cut238.txt', 'graphs/regular_d3_n80_cut111.txt', 'graphs/partition_n100_cut367.txt']


def build_sub_qubo(g, solution, N_sub, J, s, h=None, C=0.0):
    """
    自定义函数选取subqubo子问题。
    例如可简单选择对cost影响最大的前N_sub个变量组合成为subqubo子问题。
    【注】本函数非必须，仅供参考

    返回
    subqubo问题变量的指标，和对应的J，h，C。
    """
    c = 0
    size = len(solution)
    g1 = copy.deepcopy(g)
    copy_x = copy.deepcopy(solution)
    indexs = list(range(size))
    sub_index = []
    for i in range(N_sub):
        if i > 0:
            copy_x[index] = 1 - copy_x[index]
        delta_L = []
        pro_mat = []
        for k in indexs:
            pro_set = np.zeros(size + c)
            neighbors = list(g1.neighbors(k))
            if len(neighbors) == 0:
                pro = 0
            else:
                pro = 1 / len(neighbors)
            pro_set[neighbors] = pro
            pro_set = np.delete(pro_set, sub_index)
            pro_mat.append(pro_set)
        E = np.mat(np.eye(size))
        pro_mat = np.mat(pro_mat).T
        pro_mat = s * E + (1 - s) * pro_mat
        for j in indexs:
            copy_xx = copy.deepcopy(copy_x)
            copy_xx[j] = 1 - copy_xx[j]
            x_flip = copy_xx
            sol_qubo = calc_qubo_x(J, copy_x, h=h, C=C)
            x_flip_qubo = calc_qubo_x(J, x_flip, h=h, C=C)
            delta = x_flip_qubo - sol_qubo
            delta_L.append(delta)
        delta_L = np.array(delta_L)
        if np.min(delta_L) < 0:
            delta_L = delta_L - np.min(delta_L)
        init = delta_L / np.sum(delta_L)
        state = np.mat(init).T
        for j in range(2):
            state = pro_mat * state
        state = state.T.tolist()[0]
        delta_dict = dict(zip(indexs, state))
        index = max(delta_dict, key=delta_dict.get)
        sub_index.append(index)
        g1.remove_node(index)
        indexs.remove(index)
        size = size - 1
        c = c + 1
    # sub_index = np.argpartition(delta_L, -N_sub)[-N_sub:]  # subqubo子问题的变量们
    # print('subindex:',sub_index)
    J_sub, h_sub, C_sub = calc_subqubo(sub_index, solution, J, h=h, C=C)
    # print(len(sub_index))
    return sub_index, J_sub, h_sub, C_sub


def solve(g, sol, G_mincut, G):
    """
    自定义求解函数。
    例如可简单通过不断抽取N_sub个变量组成subqubo问题并对子问题进行求解，最终收敛到一个固定值。
    或者可采取其他方法...

    【注】可任意改变求解过程，但不可使用经典算法如模拟退火，禁忌搜索等提升解质量。请保持输入输出一致。

    输入：
    sol （numpy.array）：初始随机0/1比特串，从左到右分别对应从第1到第N个问题变量。
    G （matrix): QUBO问题矩阵

    输出：
    sol （numpy.array）：求解结果的0/1比特串
    cut_temp （float）：最终结果的cut值
    """
    i = 0
    while i < 10:
        print(f'---{i}---')
        sub_index, J_sub, h_sub, C_sub = build_sub_qubo(g, sol, N_sub, G_mincut, s=0.9, h=None, C=0)
        qubo_t = calc_qubo_x(G_mincut, sol)
        plus_num = np.sum(sol == 1)
        minus_num = np.sum(sol == 0)
        unbalance = abs(plus_num - minus_num) / 2
        # print('before subqubo:',qubo_t,sol,'unb:',unbalance)
        sol = solve_QAOA(J_sub, h_sub, C_sub, sub_index, sol, depth=3, tol=1e-6)
        qubo_temp = calc_qubo_x(G_mincut, sol)
        cut_temp = calc_cut_x(G, sol)
        print('after subqubo:', qubo_temp, '|cut:', cut_temp)
        plus_num = np.sum(sol == 1)
        minus_num = np.sum(sol == 0)
        unbalance = abs(plus_num - minus_num) / 2
        # print('solution:',sol)
        print('unbalance:', unbalance)
        i += 1
    return sol, cut_temp, unbalance


def build_mincut_G(G, penalty=1.):
    """
    Hamiltonian (minimize)
    ~ -Sum_{ij in g.edges} ZiZj + penalty * (Sum Zi)^2
    ~ -Sum_{ij in g.edges} ZiZj + penalty * (Sum_{i,j in N} ZiZj)
    """
    n, _ = G.shape
    G_mincut = G.copy()
    for i in range(n):
        for j in range(n):
            if j > i:
                G_mincut[i, j] += penalty / 2
                G_mincut[j, i] += penalty / 2
    # print(G_mincut[0,:])
    return -G_mincut


def run():
    """
    Main run function, for each graph need to run for 20 times to get the mean result.
    Please do not change this function, we use this function to score your algorithm. 
    """
    cut_list = []
    unb_list = []
    for filename in filelist[:]:
        print(f'--------- File: {filename}--------')
        g, G = read_graph(filename)
        G_mincut = build_mincut_G(G, penalty=1.)  # 得到整体的Ising问题的矩阵，penlaty是哈密顿量中惩罚项前的系数
        n = len(g.nodes)  # 图整体规模
        cuts = []
        unbs = []
        for turn in range(20):
            print(f'------turn {turn}------')
            sol = init_solution(n)  # 随机初始化解
            qubo_start = calc_qubo_x(G_mincut, sol)
            cut_start = calc_cut_x(G, sol)
            print('qubo start:', qubo_start, 'cut start:', cut_start)
            solution, cut, unbalance = solve(g, sol, G_mincut, G)
            cuts.append(cut)
            unbs.append(unbalance)
        cut_list.append(cuts)
        unb_list.append(unbs)
    return np.array(cut_list), np.array(unb_list)


if __name__ == "__main__":
    # 计算分数
    cut_list, unb_list = run()
    print(cut_list, unb_list)
    edge_arr = np.array([384, 120, 534])
    score = (edge_arr - (np.mean(cut_list, axis=1) + np.array([10, 4, 4]) * np.mean(unb_list, axis=1)))
    print('score:', np.sum(score))
