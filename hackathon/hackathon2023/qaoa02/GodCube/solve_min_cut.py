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
#from utils_classical import *
from utils_laplace_rotate import *



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
N_sub=15 # 量子比特的规模限制

filelist=[ 'graphs/weight_p0.5_n40_cut238.txt', 'graphs/regular_d3_n80_cut111.txt','graphs/partition_n100_cut367.txt']


def build_sub_qubo(solution,N_sub,J,h=None,C=0.0,eigvec=None,use_rotate=False,is_balance=False):
    '''
    自定义函数选取subqubo子问题。
    例如可简单选择对cost影响最大的前N_sub个变量组合成为subqubo子问题。
    【注】本函数非必须，仅供参考
    
    返回
    subqubo问题变量的指标，和对应的J，h，C。
    '''
    if not(eigvec is None):
        cos_lap = (solution-0.5)*(eigvec[:,1]-np.median(eigvec[:,1]))
        sub_index = np.argpartition(-cos_lap,-N_sub)[-N_sub:]
        J_sub,h_sub,C_sub = calc_subqubo(sub_index, solution, J, h=h,C=C )
        return sub_index,J_sub,h_sub,C_sub
    
    if use_rotate:
        D = np.diag(np.sum(-J.toarray()*2,axis=1))
        M = -J.toarray()*2
        L = D - M
        z0 = np.sign(solution-0.5)
        eigval_,eigvec_ = rotate_to_z0(M,L,z0)
        eigvec_proj = get_eigvec_sub_proj(eigval_,eigvec_)
        #grad = M.dot(z0)*z0/2.
        #degree_avg = np.sum(D)/len(z0)
        #eigvec_proj = eigvec_proj*grad
        #eigvec_proj = 0.5*np.sqrt(degree_avg)*eigvec_proj + grad
        sub_index = np.argpartition(eigvec_proj,-N_sub)[-N_sub:]
        J_sub,h_sub,C_sub = calc_subqubo(sub_index, solution, J, h=h,C=C )
        return sub_index,J_sub,h_sub,C_sub

    delta_L=[]
    for j in range(len(solution)):
        copy_x=copy.deepcopy(solution)
        copy_x[j]=1-copy_x[j]
        x_flip=copy_x
        sol_qubo=calc_qubo_x(J,solution,h=h,C=C)
        x_flip_qubo=calc_qubo_x(J,x_flip,h=h,C=C)
        delta=x_flip_qubo-sol_qubo  -0*np.sum(x_flip-0.5) # grad with unbalance
        delta_L.append(delta)
    delta_L=np.array(delta_L)

    if is_balance:
        plus_num=np.sum(solution==1)
        minus_num=np.sum(solution==0)
        if plus_num > minus_num:
            plus_index = np.where(solution==1)[0]
            N_uba = int(plus_num - minus_num)//2
            plus_sub_index = np.argpartition(delta_L[plus_index], -N_uba)[-N_uba:]
            sub_index = plus_index[plus_sub_index]
            J_sub,h_sub,C_sub = calc_subqubo(sub_index, solution, J, h=h,C=C )
            return sub_index,J_sub,h_sub,C_sub
        elif plus_num < minus_num:
            minus_index = np.where(solution==0)[0]
            N_uba = int(minus_num - plus_num)//2
            minus_sub_index = np.argpartition(delta_L[minus_index], -N_uba)[-N_uba:]
            sub_index = minus_index[minus_sub_index]
            J_sub,h_sub,C_sub = calc_subqubo(sub_index, solution, J, h=h,C=C )
            return sub_index,J_sub,h_sub,C_sub

    sub_index = np.argpartition(delta_L, -N_sub)[-N_sub:] # subqubo子问题的变量们
    #print(delta_L)
    #print(sub_index)

    for iturn in range(2):
        Jcopy = J.toarray()
        Jarr_temp = np.zeros((len(solution),len(solution)))
        Jarr_temp[:,sub_index] = Jcopy[:,sub_index]
        Jarr = csr_matrix(Jarr_temp)
        delta_Lsub=[]
        for j in range(len(solution)):
            copy_x=copy.deepcopy(solution)
            copy_x[j]=1-copy_x[j]
            x_flip=copy_x
            sol_qubo=calc_qubo_x(Jarr,solution,h=h,C=C)
            x_flip_qubo=calc_qubo_x(Jarr,x_flip,h=h,C=C)
            delta=x_flip_qubo-sol_qubo
            delta_Lsub.append(delta)
        delta_Lsub=np.array(delta_Lsub)
        sub_index = np.argpartition(delta_L - delta_Lsub, -N_sub)[-N_sub:]

    J_sub,h_sub,C_sub = calc_subqubo(sub_index, solution, J, h=h,C=C )
    return sub_index,J_sub,h_sub,C_sub

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
    D = np.diag(np.sum(-G.toarray()*2,axis=1))
    M = -G.toarray()*2
    L = D - M
    eigval, eigvec = np.linalg.eigh(L)

    i=0
    sol_best = sol.copy()
    score_best = 1e12
    cut_best   = 1e12
    unbal_best = 1e12
    while(i<6):
        #sol=np.sign(eigvec[:,-1])
        sub_index,J_sub,h_sub,C_sub=build_sub_qubo(sol,N_sub,G_mincut,h=None,C=0,eigvec=eigvec)
        sol=solve_QAOA(J_sub,h_sub,C_sub,sub_index,sol,depth=2,tol=1e-5)
        #sol=solve_QAOA_classical(J_sub,h_sub,C_sub,sub_index,sol)
        qubo_temp=calc_qubo_x(G_mincut, sol)
        cut_temp =calc_cut_x(G, sol)
        #print('after subqubo:',qubo_temp,'|cut:',cut_temp)
        plus_num=np.sum(sol==1)
        minus_num=np.sum(sol==0)
        unbalance=abs(plus_num-minus_num)/2
        print('after subqubo: %.1f'%(qubo_temp,),'|cut:',cut_temp,'|unbalance:',unbalance)
        if (score_best>cut_temp+10*unbalance):
            sol_best = sol.copy()
            score_best = cut_temp+10*unbalance
            cut_best   = cut_temp
            unbal_best = unbalance
        if (score_best<cut_temp+10*unbalance):
            sol = sol_best.copy()
            cut_temp = cut_best
            unbalance = unbal_best
        i+=1

    i=0
    while(i<20):
        #print(f'---{i}---')
        sub_index,J_sub,h_sub,C_sub=build_sub_qubo(sol,N_sub,G_mincut,h=None,C=0,use_rotate=i%2)
        qubo_t=calc_qubo_x(G_mincut, sol)
        plus_num=np.sum(sol==1)
        minus_num=np.sum(sol==0)
        unbalance=abs(plus_num-minus_num)/2
        #print('before subqubo:',qubo_t,sol,'unb:',unbalance)
        sol=solve_QAOA(J_sub,h_sub,C_sub,sub_index,sol,depth=3,tol=1e-5) 
        #sol=solve_QAOA_classical(J_sub,h_sub,C_sub,sub_index,sol)
        qubo_temp=calc_qubo_x(G_mincut, sol)
        cut_temp =calc_cut_x(G, sol)
        #print('after subqubo:',qubo_temp,'|cut:',cut_temp)
        plus_num=np.sum(sol==1)
        minus_num=np.sum(sol==0)
        unbalance=abs(plus_num-minus_num)/2
        print('after subqubo: %.1f'%(qubo_temp,),'|cut:',cut_temp,'|unbalance:',unbalance)
        if (score_best>cut_temp+10*unbalance):
            sol_best = sol.copy()
            score_best = cut_temp+10*unbalance
            cut_best   = cut_temp
            unbal_best = unbalance
        if (score_best<cut_temp+10*unbalance):
            sol = sol_best.copy()
            cut_temp = cut_best
            unbalance = unbal_best
        #print('solution:',sol)
        #print('unbalance:', unbalance)
        i+=1

    if unbalance>0:
        sub_index,J_sub,h_sub,C_sub=build_sub_qubo(sol,N_sub,101*G_mincut+100*G,h=None,C=0,is_balance=True)
        sol=solve_QAOA(J_sub,h_sub,C_sub,sub_index,sol,depth=5,tol=1e-5)
        qubo_temp=calc_qubo_x(G_mincut, sol)
        cut_temp =calc_cut_x(G, sol)
        plus_num=np.sum(sol==1)
        minus_num=np.sum(sol==0)
        unbalance=abs(plus_num-minus_num)/2
        print('after subqubo: %.1f'%(qubo_temp,),'|cut:',cut_temp,'|unbalance:',unbalance)
        if (score_best>cut_temp+10*unbalance):
            sol_best = sol.copy()
            score_best = cut_temp+10*unbalance
            cut_best   = cut_temp
            unbal_best = unbalance
        if (score_best<cut_temp+10*unbalance):
            sol = sol_best.copy()
            cut_temp = cut_best
            unbalance = unbal_best
        
    return sol, cut_temp, unbalance


def build_mincut_G(G, penalty=1.):
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
        G_mincut= build_mincut_G(G, penalty=0.8) # 得到整体的Ising问题的矩阵，penlaty是哈密顿量中惩罚项前的系数
        n=len(g.nodes) # 图整体规模
        cuts=[]
        unbs=[]
        for turn in range(20):
            print(f'------turn {turn}------')
            sol=init_solution(n) # 随机初始化解 
            qubo_start = calc_qubo_x(G_mincut, sol)
            cut_start =calc_cut_x(G, sol)  
            print('qubo start:',qubo_start,'cut start:',cut_start)
            solution, cut, unbalance = solve(sol, G_mincut,G) 
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
 
