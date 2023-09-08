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
本代码为求解最大割
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
        
        copy_x[j]=1-copy_x[j]
        x_flip=copy_x
        sol_qubo=calc_qubo_x(J,solution,h=h,C=C)
        x_flip_qubo=calc_qubo_x(J,x_flip,h=h,C=C)
        delta=x_flip_qubo-sol_qubo
        delta_L.append(delta)
    delta_L=np.array(delta_L)
    # delta_L = np.abs(delta_L)
    # print(delta_L)
    sub_index = np.argpartition(delta_L, -N_sub)[-N_sub:] # subqubo子问题的变量们
    sorted_indices = np.argsort(delta_L)[::-1]
    # print("为什么argsort和argpartition得到的结果是不一致的:",sub_index,sorted_indices)
    # print("delta_L",delta_L)
    # print('subindex:',sub_index)
    J_sub,h_sub,C_sub = calc_subqubo(sub_index, solution, J, h=h,C=C )
    return sub_index,J_sub,h_sub,C_sub,delta_L

def solve(sol,g,G):
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
    # 我们在下面要加入一个大循环, 运行大概20次到40次左右, 而且里面的循环的次数也要进行动态调整. 
    n = len(sol)
    cut_best = 0
    edge_n = 0.5 * G.nnz
    # 针对图的大小, 对于下面参数进行动调调整.
    
    n_e=len(g.edges)
    n_v=len(g.nodes)
    radio=2*n_e/(n_v**2)
    NUM1=20
    NUM2=60
    if radio < 0.1:
        NUM1 = 30
       
    if n_v>50:
        NUM2=150
    
        
        
            
                
        
    # NUM1 = 5
    # NUM2 = 60
    for jndex in range(0,NUM1):
        sol = np.random.randint(2,size=n)
        i=0
        R = 0  
        tabuList = np.zeros((NUM2+1,N_sub))
        cut_temp_best = 0
        while(i<NUM2):
            sub_index,J_sub,h_sub,C_sub,delta_L=build_sub_qubo(sol,N_sub,G,h=None,C=0)
            sorted_indices = np.argsort(delta_L)[::-1]
            # print("核对一下我们的计算结果对不对:",sub_index,sorted_indices)
            # 我们加入一个 Tabu 表, 避免陷入局部最优. 
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





            tabuList[R,:] = A.reshape(1,-1)
            R = R+1
            sub_index = A.reshape(-1)
            J_sub,h_sub,C_sub = calc_subqubo(sub_index, sol, G, h=None,C=0.0 )
            











            #print('before sol:',calc_qubo_x(G, sol))
            sol=solve_QAOA(J_sub,h_sub,C_sub,sub_index,sol,depth=3,tol=1e-5) # You can change the depth and tolerance of QAOA solver

            # print("sol2的结果是:",sol2)
            # print("sol的结果是:",sol)


            qubo_temp=calc_qubo_x(G, sol)
            cut_temp =calc_cut_x(G, sol)
            if cut_temp>cut_temp_best:
                cut_temp_best = cut_temp
                sol_best = copy.deepcopy(sol)
            # print("第"+str(i)+"阶段计算的最大值:",cut_temp)


     #       print('after subqubo:',qubo_temp,'|cut:',cut_temp)
            i+=1
        if cut_best<cut_temp_best:
            cut_best = cut_temp_best
            sol_bbest = copy.deepcopy(sol_best)
    return sol_bbest, cut_best


def run():
    """
    Main run function, for each graph need to run for 20 times to get the mean result.
    Please do not change this function, we use this function to score your algorithm. 
    """
    cut_list=[]
    for filename in filelist[:]:
        #print(f'--------- File: {filename}--------')
        g,G=read_graph(filename)
        n=len(g.nodes) # 图整体规模
        cuts=[]
        for turn in range(20):
            #print(f'------turn {turn}------')
            sol=init_solution(n) # 随机初始化解   
            #qubo_start = calc_qubo_x(G, sol)
            #cut_start =calc_cut_x(G, sol)
            #print('origin qubo:',qubo_start,'|cut:',cut_start)
            solution,cut=solve(sol,g, G) #主要求解函数, 主要代码实现
            cuts.append(cut)
        cut_list.append(cuts)    
    return np.array(cut_list)  



if __name__== "__main__": 
    #计算分数
    cut_list=run()
    print(cut_list)
    score=np.array(np.mean(cut_list,axis=1))
    print('score:',np.sum(score))
        
        