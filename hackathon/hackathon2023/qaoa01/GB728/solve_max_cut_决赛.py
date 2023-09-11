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
import numpy as np
from utils import *

import math
from scipy.sparse import csr_matrix

'''
本赛题旨在引领选手探索，在NISQ时代规模有限的量子计算机上，求解真实场景中的大规模图分割问题。
本代码为求解最大割
'''
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (1024*1024*4000, hard)) #4G
N_sub=15 # 量子比特的规模限制

filelist=[ 'graphs/weight_p0.5_n40_cut238.txt', 'graphs/regular_d3_n80_cut111.txt','graphs/partition_n100_cut367.txt']


N_START = 8

def build_sub_qubo(solution, n_sub, J,heads, h=None, C=0.0):
    '''
    自定义函数选取subqubo子问题。
    例如可简单选择对cost影响最大的前N_sub个变量组合成为subqubo子问题。
    【注】本函数非必须，仅供参考
    
    返回
    subqubo问题变量的指标，和对应的J，h，C。
    '''
    delta_L = []
    for j in range(len(solution)):
        zj = solution[j]
        delta_score = 0.0
        for v in heads[j]:
            if solution[v] == zj:
                delta_score += 1
            else:
                delta_score -= 1
        delta_L.append(delta_score)

    sub_index = np.argpartition(delta_L, -n_sub)[-n_sub:] # subqubo子问题的变量们
    J_sub,h_sub,C_sub = calc_subqubo(sub_index, solution, J, h=h,C=C )
    return sub_index,J_sub,h_sub,C_sub

def rebuild_graph(G):
    G_dict = dict(G.todok())
    g = nx.Graph()
    for node, Jij in G_dict.items():
        if node[0] < node[1]:
            g.add_edge(node[0], node[1])
    return g


def solve(sol, G):
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

    it = 0
    fin_result_solution = sol.copy()
    fin_result_score = calc_cut_x(G, fin_result_solution)

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
        result_score = calc_cut_x(G, result_solution)

        while int(step) > 1:
            # print(result_solution)
            n_sub = int(step)
            sub_index,J_sub,h_sub,C_sub=build_sub_qubo(result_solution,n_sub,G,heads,h=None,C=0)
            tmp_sol=solve_QAOA(J_sub,h_sub,C_sub,sub_index,sol,depth=5,tol=1e-5)
            cut_temp =calc_cut_x(G, tmp_sol)

            if cut_temp > result_score:
                result_solution = tmp_sol.copy()
                result_score = cut_temp
            
            step *= 0.9
        
        if result_score > fin_result_score:
            fin_result_solution = result_solution
            fin_result_score = result_score

        it += 1

    # print("-----------------------------")
    cut_temp = calc_cut_x(G, fin_result_solution)
    return fin_result_solution, cut_temp


def run():
    """
    Main run function, for each graph need to run for 20 times to get the mean result.
    Please do not change this function, we use this function to score your algorithm. 
    """
    cut_list=[]
    for filename in filelist[:]:
        print(f'--------- File: {filename}--------')
        g,G=read_graph(filename)
        n=len(g.nodes) # 图整体规模
        cuts=[]
        for turn in range(20):
            print(f'------turn {turn}------')
            sol=init_solution(n) # 随机初始化解   
            qubo_start = calc_qubo_x(G, sol)
            cut_start =calc_cut_x(G, sol)
            print('origin qubo:',qubo_start,'|cut:',cut_start)
            solution,cut=solve(sol, G) #主要求解函数, 主要代码实现
            cuts.append(cut)
        cut_list.append(cuts)    
    return np.array(cut_list)  



if __name__== "__main__": 
    #计算分数
    cut_list=run()
    print(cut_list)
    score=np.array(np.mean(cut_list,axis=1))
    print('score:',np.sum(score))
 