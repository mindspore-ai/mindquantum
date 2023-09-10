import resource
from utils import *
import pickle
import matplotlib.pyplot as plt
import random
import sys
import copy
import time
import numpy as np
from itertools import count
import os
os.environ['OMP_NUM_THREADS'] = '2'

import math

'''
本赛题旨在引领选手探索，在NISQ时代规模有限的量子计算机上，求解真实场景中的大规模图分割问题。
min-cut: 使得ZiZj项的和最大，而不是最大割里使其最小化
Hamiltonian: 最小化
    -Sum_{ij in g.edges} ZiZj + penalty * (Sum Zi)^2
    ~ -Sum_{ij in g.edges} ZiZj + penalty * (Sum_{i,j in N} ZiZj)
'''

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (1024*1024*4000, hard))  # 4G
N_sub = 15  # 量子比特的规模限制

filelist = ['graphs/weight_p0.5_n40_cut238.txt',
            'graphs/regular_d3_n80_cut111.txt', 'graphs/partition_n100_cut367.txt']


N_START = 8



def find_max_d(g):
    maxx = 0
    d = [0 for _ in range(len(g.nodes))]
    for edge in g.edges:
        u = edge[0]
        v = edge[1]
        d[u] += 1
        d[v] += 1
        maxx = max(maxx, d[u])
        maxx = max(maxx, d[v])
    return maxx

def cal_score(origin_G, solution, penalty_i):
    '''
    返回cut+penalty*unb
    '''
    cut_temp =calc_cut_x(origin_G, solution)
    plus_num=np.sum(solution==1)
    minus_num=np.sum(solution==0)
    unbalance=abs(plus_num-minus_num)/2
    return cut_temp + penalty_i*unbalance

def build_sub_qubo(solution,N_sub,G,G_mincut, heads,penalty_i, h=None,C=0.0):
    '''
    自定义函数选取subqubo子问题。
    例如可简单选择对cost影响最大的前N_sub个变量组合成为subqubo子问题。
    【注】本函数非必须，仅供参考
    
    返回
    subqubo问题变量的指标，和对应的J，h，C。
    '''
    delta_L = []
    plus_num=np.sum(solution==1)
    minus_num=np.sum(solution==0)
    cut = calc_cut_x(G, solution)
    for j in range(len(solution)):
        zj = solution[j]
        delta_score = 0.0
        for v in heads[j]:
            if solution[v] == zj:
                delta_score += 1
            else:
                delta_score -= 1
        if zj == 0:
            delta_score += penalty_i * abs((plus_num+1) - (minus_num-1)) /2 - penalty_i * abs(plus_num - minus_num) /2
        else:
            delta_score += penalty_i * abs((plus_num-1) - (minus_num+1)) /2 - penalty_i * abs(plus_num - minus_num) /2
        delta_L.append(delta_score)

    sub_index = np.argpartition(delta_L, N_sub)[:N_sub] # subqubo子问题的变量们
    J_sub,h_sub,C_sub = calc_subqubo(sub_index, solution, G_mincut, h=h,C=C )
    return sub_index,J_sub,h_sub,C_sub

def rebuild_graph(G):
    G_dict = dict(G.todok())
    g = nx.Graph()
    for node, Jij in G_dict.items():
        if node[0] < node[1]:
            g.add_edge(node[0], node[1])
    return g



def solve(sol,G_mincut, G):
    '''
    自定义求解函数。
    例如可简单通过不断抽取20个变量组成subqubo问题并对子问题进行求解，最终收敛到一个固定值。
    或者可采取其他方法...
    
    【注】可任意改变求解过程，但不可使用经典算法如模拟退火，禁忌搜索等提升解质量。请保持输入输出一致。
    
    输入：
    sol （numpy.array）：初始随机0/1比特串，从左到右分别对应从第1到第N个问题变量。
    G （matrix): QUBO问题矩阵
    
    输出：
    sol （numpy.array）：求解结果的0/1比特串
    cut_temp （float）：最终结果的cut值
    '''
    n = len(sol)
    g = rebuild_graph(G)
    
    # max_d = find_max_d(g)
    # my_penalty = 1.0/max_d

    # penalty参数等于乘上的数的两倍
    if n == 40:
        penalty_i = 10.0
    else:
        penalty_i = 4.0

    G_mincut = build_mincut_G(G, penalty=0.5)

    it = 0
    fin_result_solution = sol.copy()
    fin_result_score = cal_score(G, fin_result_solution, penalty_i)

    # 建立点->边对应关系
    heads = [[] for _ in range(n)]
    for edge in g.edges:
        u, v = edge
        heads[u].append(v)
        heads[v].append(u)

    while it < N_START:
        sol= init_solution(n)
        step = 15.0

        result_solution = sol.copy()
        result_score = cal_score(G, result_solution, penalty_i)
        
        while int(step) > 1:
            # print(result_solution)
            n_sub = int(step)
            sub_index,J_sub,h_sub,C_sub=build_sub_qubo(result_solution,n_sub,G,G_mincut, heads, penalty_i)
            tmp_sol=solve_QAOA(J_sub,h_sub,C_sub,sub_index,sol,depth=3,tol=1e-5)
            tmp_score = cal_score(G, tmp_sol, penalty_i)

            if tmp_score < result_score:
                result_solution = tmp_sol.copy()
                result_score = tmp_score
            
            step *= 0.9
        
        if result_score < fin_result_score:
            fin_result_solution = result_solution
            fin_result_score = result_score

        it += 1

    # print("-----------------------------")
    cut_temp = calc_cut_x(G, fin_result_solution)
    plus_num=np.sum(fin_result_solution==1)
    minus_num=np.sum(fin_result_solution==0)
    unbalance=abs(plus_num-minus_num)/2

    return fin_result_solution, cut_temp, unbalance


def build_mincut_G(G, penalty=1.):
    '''
    Hamiltonian (minimize)
    ~ -Sum_{ij in g.edges} ZiZj + penalty * (Sum Zi)^2
    ~ -Sum_{ij in g.edges} ZiZj + penalty * (Sum_{i,j in N} ZiZj)
    '''
    n, _ = G.shape
    G_mincut = G.copy()
    for i in range(n):
        for j in range(n):
            if j > i:
                G_mincut[i, j] += penalty/2
                G_mincut[j, i] += penalty/2
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
        # 得到整体的Ising问题的矩阵，penlaty是哈密顿量中惩罚项前的系数
        G_mincut = build_mincut_G(G, penalty=0.3)
        n = len(g.nodes)  # 图整体规模
        cuts = []
        unbs = []
        for turn in range(20):
            print(f'------turn {turn}------')
            sol = init_solution(n)  # 随机初始化解
            qubo_start = calc_qubo_x(G_mincut, sol)
            cut_start = calc_cut_x(G, sol)
            print('qubo start:', qubo_start, 'cut start:', cut_start)
            solution, cut, unbalance = solve(sol, G_mincut, G)
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
    score = (edge_arr-(np.mean(cut_list, axis=1) +
             np.array([10, 4, 4])*np.mean(unb_list, axis=1)))
    print('score:', np.sum(score))
