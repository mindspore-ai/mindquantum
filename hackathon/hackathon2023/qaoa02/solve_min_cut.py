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


'''
本赛题旨在引领选手探索，在NISQ时代规模有限的量子计算机上，求解真实场景中的大规模图分割问题。
min-cut: 使得ZiZj项的和最大，而不是最大割里使其最小化
Hamiltonian: 最小化
    -Sum_{ij in g.edges} ZiZj + penalty * (Sum Zi)^2
    ~ -Sum_{ij in g.edges} ZiZj + penalty * (Sum_{i,j in N} ZiZj)
'''

import resource
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (1024*1024*4000, hard)) #4G
N_sub=5 # 量子比特的规模限制

filelist=[ 'graphs/weight_p0.5_n40_cut238.txt', 'graphs/regular_d3_n80_cut111.txt','graphs/partition_n100_cut367.txt']







def build_sub_qubo(solution,N_sub,J,h=None,C=0.0):
    '''
    自定义函数选取subqubo子问题。
    例如可简单选择对cost影响最大的前N_sub个变量组合成为subqubo子问题。
    【注】本函数非必须，仅供参考
    
    返回
    subqubo问题变量的指标，和对应的J，h，C。
    '''
    delta_L=[]
    for j in range(len(solution)):
        copy_x=copy.deepcopy(solution)
        copy_x[j]=int(1)-copy_x[j]
        x_flip=copy_x
        sol_qubo=calc_qubo_x(J,solution,h=h,C=C)
        x_flip_qubo=calc_qubo_x(J,x_flip,h=h,C=C)
        delta=x_flip_qubo-sol_qubo
        delta_L.append(delta)
    delta_L=np.array(delta_L)
    
    sub_index = np.argpartition(delta_L, -N_sub)[-N_sub:] # subqubo子问题的变量们
    #print(delta_L)
    #print(sub_index)
    J_sub,h_sub,C_sub = calc_subqubo(sub_index, solution, J, h=h,C=C )
    return sub_index,J_sub,h_sub,C_sub,delta_L

def solve(sol,g, G):
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
    
    n_e=len(g.edges)
    n_v=len(g.nodes)
    radio=2*n_e/(n_v**2)
    penalty=2*radio/(n_v/(n_v+40))
    G_mincut= build_mincut_G(G, penalty) # 得到整体的Ising问题的矩阵，penlaty是哈密顿量中惩罚项前的系数
    
    
    n = len(sol)
    NUM1=10/radio
    NUM1=min(NUM1,70)
    NUM1 = np.floor(NUM1)
  
    NUM2 = 20
    if radio<0.1:
        NUM2=30
    sol_1=sol
    cut_temp_1=1000
    unbalance_1=1000
    performance_1=cut_temp_1+4*unbalance_1
    for j in range(int(NUM1)):
        sol_0=sol
        cut_temp_0=1000
        unbalance_0=1000
        performance_0=cut_temp_0+4*unbalance_0
        i=0
        R = 0 # 表示禁忌表的长度. 
        sol = np.random.randint(2,size=n)
        tabuList = np.zeros((NUM2+1,N_sub))
        while(i<NUM2):
            i = i+1
            sub_index,J_sub,h_sub,C_sub,delta_L=build_sub_qubo(sol,N_sub,G_mincut,h=None,C=0)
            sorted_indices = np.argsort(delta_L)[::-1]
            # 下面看看 sub_index 是不是合适的. 
            A = copy.deepcopy(sub_index)
            chongfu = 1
            new_index = N_sub
            while chongfu == 1:
                chongfu = 0

                for index in range(0,R):
                    B = tabuList[index,:]
                    B = B.reshape(-1)
                    a_sorted = np.sort(A)
                    b_sorted = np.sort(B)
                    are_equal = np.array_equal(a_sorted, b_sorted)
                    # print("循环",index,"当中的 A 的取值是",A,"B 的取值是:",B)
                    if np.linalg.norm(A - B)<0.001:# np.linalg.norm(a_sorted - b_sorted)
                        chongfu = 1
                        A = np.random.choice(sorted_indices[0:new_index*2], size=N_sub, replace=False)
                        break
                        
            tabuList[R,:] = copy.deepcopy(A.reshape(1,-1))
            #print("当前的Tabu表示:\n", tabuList)
            R = R+1
            sub_index = A.reshape(-1)
            J_sub,h_sub,C_sub = calc_subqubo(sub_index, sol, G_mincut, h=None,C=0.0 )
        
            sol=solve_QAOA(J_sub,h_sub,C_sub,sub_index,sol,depth=3,tol=1e-5) 
            qubo_temp=calc_qubo_x(G_mincut, sol)
            cut_temp =calc_cut_x(G, sol)
            #print("第"+str(i)+"次cut_temp表现效果如下:",cut_temp)

            plus_num=np.sum(sol==1)
            minus_num=np.sum(sol==0)
            unbalance=abs(plus_num-minus_num)/2
            
            performance=cut_temp+4*unbalance
            #print("第"+str(i)+"次perf0的表现效果如下:",performance)
            if performance<performance_0:
                performance_0 = performance
                sol_0=sol
                cut_temp_0=cut_temp
                unbalance_0=unbalance
        
        if performance_0 < performance_1:
            performance_1 = performance_0
            cut_temp_1 = cut_temp_0
            unbalance_1 = unbalance_0
            sol_1 = sol_0
        
        
    return sol_1,cut_temp_1,unbalance_1
            
    


def build_mincut_G(G, penalty=4.):
    '''
    Hamiltonian (minimize)
    ~ -Sum_{ij in g.edges} ZiZj + penalty * (Sum Zi)^2
    ~ -Sum_{ij in g.edges} ZiZj + penalty * (Sum_{i,j in N} ZiZj)
    '''
    n,_=G.shape
    G_mincut=G.copy()
    for i in range(n):
        for j in range(n):
            if j>i:
                G_mincut[i,j]+=penalty/2
                G_mincut[j,i]+=penalty/2
    #print(G_mincut[0,:])
    return -G_mincut

def run():
    """
    Main run function, for each graph need to run for 20 times to get the mean result.
    Please do not change this function, we use this function to score your algorithm. 
    """           
    cut_list=[]
    unb_list=[]
    for filename in filelist[:]:
        print(f'--------- File: {filename}--------')
        g,G=read_graph(filename)
        
        n=len(g.nodes) # 图整体规模
        cuts=[]
        unbs=[]
        for turn in range(20):
            #print(f'------turn {turn}------')
            sol=init_solution(n) # 随机初始化解 
            #qubo_start = calc_qubo_x(G_mincut, sol)
            #cut_start =calc_cut_x(G, sol)  
            #print('qubo start:',qubo_start,'cut start:',cut_start)
            solution, cut, unbalance = solve(sol, g,G) 
            cuts.append(cut)
            unbs.append(unbalance)
        cut_list.append(cuts)
        unb_list.append(unbs)
    return np.array(cut_list), np.array(unb_list) 
        
        

if __name__== "__main__":   
    #计算分数
    cut_list, unb_list=run()
    print(cut_list,unb_list)
    edge_arr=np.array([384, 120, 534])
    score=(edge_arr-(np.mean(cut_list,axis=1)+np.array([10,4,4])*np.mean(unb_list,axis=1)))
    print('score:',np.sum(score))
 

    