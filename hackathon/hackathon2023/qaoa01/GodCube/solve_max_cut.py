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
本代码为求解最大割
'''
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (1024*1024*4000, hard)) #4G
N_sub=15 # 量子比特的规模限制

filelist=[ 'graphs/weight_p0.5_n40_cut238.txt', 'graphs/regular_d3_n80_cut111.txt','graphs/partition_n100_cut367.txt']


def build_sub_qubo(solution,N_sub,J,h=None,C=0.0,eigvec=None,use_rotate=False):
    '''
    自定义函数选取subqubo子问题。
    例如可简单选择对cost影响最大的前N_sub个变量组合成为subqubo子问题。
    【注】本函数非必须，仅供参考
    
    返回
    subqubo问题变量的指标，和对应的J，h，C。
    '''
    if not(eigvec is None):
        cos_lap = (solution-0.5)*eigvec[:,-1]
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
        delta=x_flip_qubo-sol_qubo
        delta_L.append(delta)
    delta_L=np.array(delta_L)
    #delta_L = delta_L + 0.5*np.sum(-J.toarray()*2,axis=1) *np.random.randint(2)
    #print(delta_L)
    sub_index = np.argpartition(delta_L, -N_sub)[-N_sub:] # subqubo子问题的变量们
    #print('subindex:',sub_index)
    
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

def solve(sol,G):
    '''
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
    '''
    D = np.diag(np.sum(-G.toarray()*2,axis=1))
    M = -G.toarray()*2
    L = D - M
    eigval, eigvec = np.linalg.eigh(L)

    i=0
    sol_best = sol.copy()
    cut_best = 0.0
    print(len(sol))
    while(i<6):
        #sol=np.sign(eigvec[:,-1])
        sub_index,J_sub,h_sub,C_sub=build_sub_qubo(sol,N_sub,G,h=None,C=0,eigvec=eigvec)
        sol=solve_QAOA(J_sub,h_sub,C_sub,sub_index,sol,depth=2,tol=1e-5)
        #sol=solve_QAOA_classical(J_sub,h_sub,C_sub,sub_index,sol)
        qubo_temp=calc_qubo_x(G, sol)
        cut_temp =calc_cut_x(G, sol)
        print('after subqubo:',qubo_temp,'|cut:',cut_temp)
        if (cut_temp>cut_best): sol_best = sol.copy(); cut_best = cut_temp
        if (cut_temp<cut_best): sol = sol_best.copy(); cut_temp = cut_best
        i+=1

    i=0
    while(i<20):
        sub_index,J_sub,h_sub,C_sub=build_sub_qubo(sol,N_sub,G,h=None,C=0,use_rotate=i%2)#np.random.randint(2))
        #sub_index,J_sub,h_sub,C_sub=build_sub_qubo(sol,N_sub,G,h=None,C=0)
        #print('before sol:',calc_qubo_x(G, sol))
        sol=solve_QAOA(J_sub,h_sub,C_sub,sub_index,sol,depth=3,tol=1e-5) # You can change the depth and tolerance of QAOA solver
        #sol=solve_QAOA_classical(J_sub,h_sub,C_sub,sub_index,sol)
        qubo_temp=calc_qubo_x(G, sol)
        cut_temp =calc_cut_x(G, sol)
        print('after subqubo:',qubo_temp,'|cut:',cut_temp)
        if (cut_temp>cut_best): sol_best = sol.copy(); cut_best = cut_temp
        if (cut_temp<cut_best): sol = sol_best.copy(); cut_temp = cut_best
        i+=1
    return sol, cut_temp


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
    #best =np.array([238,111,367])
    #print('ratio:',score*(1./best))
    print('score:',np.sum(score))
    
        
        
