# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 09:39:33 2021
用于量子线路比特量子模型
单量子比特任意态之间的相互制备

动作选用三种策略：

先采用第一种：要么最优，要么选次优

如果效果不好就选第二种：动作要么选最优的，要么选最差的

如果都达不到保真度阈值，就再试试始终选最优的

最后那种策略在全过程中达到的最大保真度最高，就选用那个策略

@author: Waikikilick
email: 1250125907@qq.com
"""
from mindquantum import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from time import *
import copy

np.random.seed(1)

T = np.pi
dt = np.pi/20
step_max = T/dt
sx = X.matrix() # 定义泡利算符，根据 mindquantum 基本门的 matrix() 功能得到泡利矩阵 
sy = Y.matrix() 
sz = Z.matrix()


action_space = np.mat([[1,0,0], #可以选择的动作范围，各列的每项分别代表着 sigma x, y, z 前面的系数。
                       [2,0,0], #每次执行的动作都是单独的绕 x, y, z 轴一定角度的旋转
                       [0,1,0], # x, y 方向的值可以取负，但 z 方向的只能取正值
                       [0,2,0],
                       [0,0,1],
                       [0,0,2],
                       [-1,0,0],
                       [-2,0,0],
                       [0,-1,0],
                       [0,-2,0]])

theta_num = 6 #除了 0 和 Pi 两个点之外，点的数量
varphi_num = 21#varphi 角度一圈上的点数

theta = np.linspace(0,np.pi,theta_num+2,endpoint=True) 
varphi = np.linspace(0,np.pi*2,varphi_num,endpoint=False) 

def psi_set():
    psi_set = []
    for ii in range(1,theta_num+1):
        for jj in range(varphi_num):
            psi_set.append(np.mat([[np.cos(theta[ii]/2)],[np.sin(theta[ii]/2)*(np.cos(varphi[jj])+np.sin(varphi[jj])*(0+1j))]]))
    # 最后再加上 |0> 和 |1> 两个态
    psi_set.append(np.mat([[1], [0]], dtype=complex)) 
    psi_set.append(np.mat([[0], [1]], dtype=complex))
    return psi_set

target_set = psi_set() # 目标量子态集
init_set = psi_set() #初始量子态集

# 动作选择策略 0: 动作直接选最优的
def step0(psi,target_psi,F):
    
    fid_list = []
    psi_list = []
    action_list = list(range(len(action_space)))
    
    for action in action_list:
        
        H = float(action_space[action,0])*sx/2 + float(action_space[action,1])*sy/2 - float(action_space[action,2])*sz/2
        U = UnivMathGate('U',expm(-1j * H * dt)) # 根据 mindquantum 普适门 来定义 时间演化算符。 ‘U’ 为自定义门名称，后面为自定义的矩阵
        psi_ = U.matrix() * psi 
        fid = (np.abs(psi_.H * target_psi) ** 2).item(0).real 
        psi_list.append(psi_)
        fid_list.append(fid)
        best_action = fid_list.index(max(fid_list))
        best_fid = max(fid_list)
        
    psi_ = psi_list[best_action]
    
    return best_action, best_fid, psi_

# 动作选择策略 1：动作选最优的，或者最差的
def step1(psi,target_psi,F):
    
    fid_list = []
    psi_list = []
    action_list = list(range(len(action_space)))
    
    for action in action_list:
        
        H = float(action_space[action,0])*sx/2 + float(action_space[action,1])*sy/2 - float(action_space[action,2])*sz/2
        U = UnivMathGate('U',expm(-1j * H * dt)) # 根据 mindquantum 普适门 来定义 时间演化算符。 ‘U’ 为自定义门名称，后面为自定义的矩阵
        psi_ = U.matrix() * psi 
        fid = (np.abs(psi_.H * target_psi) ** 2).item(0).real 
        
        psi_list.append(psi_)
        fid_list.append(fid)
    
    if F < max(fid_list):
        best_action = fid_list.index(max(fid_list))
        best_fid = max(fid_list)
    else:
        
        best_action = fid_list.index(min(fid_list))
        best_fid = min(fid_list)
    psi_ = psi_list[best_action]
    # print(best_action)
    return best_action, best_fid, psi_

# 动作选择策略 2：动作选最优的，或者次优的
def step2(psi,target_psi,F):
    
    fid_list = []
    psi_list = []
    action_list = list(range(len(action_space)))
    
    for action in action_list:
        
        H = float(action_space[action,0])*sx/2 + float(action_space[action,1])*sy/2 - float(action_space[action,2])*sz/2
        U = UnivMathGate('U',expm(-1j * H * dt)) # 根据 mindquantum 普适门 来定义 时间演化算符。 ‘U’ 为自定义门名称，后面为自定义的矩阵
        psi_ = U.matrix() * psi 
        fid = (np.abs(psi_.H * target_psi) ** 2).item(0).real 
        
        psi_list.append(psi_)
        fid_list.append(fid)
        
    if F < max(fid_list):
        best_action = fid_list.index(max(fid_list))
        best_fid = max(fid_list)
    else:
        
        fid_list[np.argmax(fid_list)] = 0 # 将最大保真度故意赋值为 0
        best_action = np.argmax(fid_list) # 那么再对保真度列表求一次最大值，就是实际上的次最大值了
        best_fid = fid_list[best_action] 
        
    psi_ = psi_list[best_action]
    
    return best_action, best_fid, psi_

def job(target_set): # 输入为一个128个采样点的目标态集合。
                     # 每个目标态target_psi都由初始态集中的初态制备

    F_list = [] # 用于记录制备所有目标态的平均保真度
    count = 0 # 用于监测程序执行进度，达到128时即完成
     
    for target_psi in target_set:
        
        print(count)
        count += 1
        
        fids_list = [] # 用于记录所有初始态制备本目标态中目标态的保真度
        
        for psi1 in init_set: # 对初始态集合中的每个初始态进行遍历执行
            
            psi = psi1
            F = (np.abs(psi1.H * target_psi) ** 2).item(0).real # 先计算一下初始态的保真度，以便与下一时刻的保真度进行比较，以判断是否陷入局部最优
            
            
            fid_max = F # 每个策略分开执行
            fid_max1 = F
            fid_max2 = F
            fid_max0 = F
            
            # 执行策略 1：选 最佳动作 或 次优动作
            step_n = 0 # 计算控制脉冲数
            while True: 
                action, F, psi_ = step1(psi,target_psi,F) # 采用策略 1 来确定动作、下一时刻的保真度 和 下一时刻量子态
                fid_max1 = max(F,fid_max1) # 记录此策略下能达到的最大保真度，用于判定效果和截取控制脉冲序列。
                psi = psi_ # 迭代量子态
                step_n += 1
                if fid_max1>0.999 or step_n>step_max: # 当保真度大于阈值0.999或总步数超过限制就终止循环
                    break
                
            # 与上面类似的操作，执行的是策略 2：选最佳动作 或 最差动作   
            step_n = 0 
            F = (np.abs(psi1.H * target_psi) ** 2).item(0).real 
            psi = psi1
            while True:
                action, F, psi_ = step2(psi,target_psi,F)
                fid_max2 = max(F,fid_max2)
                psi = psi_
                step_n += 1
                if fid_max2>0.999 or step_n>step_max:
                    break 
            # 与上面类似的操作，执行的是策略 0：只选选最佳动作  
            step_n = 0
            F = (np.abs(psi1.H * target_psi) ** 2).item(0).real 
            psi = psi1
            while True:
                action, F, psi_ = step0(psi,target_psi,F)
                fid_max0 = max(F,fid_max0)
                psi = psi_
                step_n += 1
                if fid_max0>0.999 or step_n>step_max:
                    break 
                
            fid_max = max(fid_max1,fid_max2,fid_max0)  # 能达到最大保真度的策略即为最佳策略
            fids_list.append(fid_max) # 将这个初始态能达到的最大保真度记录下来
        F_list.append(np.mean(fids_list))
    return  F_list  # 返回所有态制备任务的平均保真度
 

if __name__ == '__main__':
    
    time1 = time()
    F_list = job(target_set) # 执行函数 得到记录着目标态态制备任务的各自保真度的列表
    
    print('F_list = ',F_list)
        
    # 对数据进行处理，画出热图
    F_0 = F_list[-2]
    F_1 = F_list[-1]
    
    del F_list[-1]
    del F_list[-1]
    
    F_0_list = []
    F_1_list = []
    
    for _ in range(varphi_num):
        F_0_list.append(F_0)
        
    for _ in range(varphi_num):
        F_1_list.append(F_1)
        
    F_list_plot = F_0_list + F_list + F_1_list
    F_array_plot = np.array(F_list_plot)
    F_array_plot = F_array_plot.reshape((theta_num+2,varphi_num))
    
    plt.figure(figsize=(12,12))
    plt.title('State Preparing Fidelity Heat-map in Superconducting circuits')
    plt.xlabel(r'$\varphi/\pi$')
    plt.xticks(ticks=[0,5,10,15,20],labels=[0,0.5,1,1.5,2])
    plt.ylabel(r'$\theta/\pi$')
    plt.yticks(ticks=[0,1.4,2.8,4.2,5.6,7],labels=[1.0,0.8,0.6,0.4,0.2,0.0])
    plt.imshow(F_array_plot)  
    plt.colorbar(shrink=0.32,aspect=10,label=r'$\bar{F}$',ticks=[0.99880,0.99896,0.99912,0.99928,0.99944,0.99996])
    plt.show() 

    plt.savefig('./src/sj_single_qubit_heat_map.png')
    time2 = time()
    print('time_cost is: ',time2-time1) # 得到程序运行的总时间

# 128测试点
#动作 x,y: 0,1,2,-1,-2; z: 0,1,2

# [0.9992626016568953, 0.999514484401934, 0.9995460533336445, 0.9994487703895798, 0.9994341030357903, 0.9994277361138151, 0.9995065968535353, 0.9995340631758372, 0.9995306864977319, 0.999426204281705, 0.9993963640685807, 0.999411472525704, 0.9995366303277334, 0.9993891174005957, 0.9994904903874451, 0.9994201956923201, 0.9993321384538574, 0.9995540107497178, 0.9994607633414887, 0.9995125756924348, 0.9993758571776079, 0.9992989411366318, 0.9994344492569684, 0.9995358246179075, 0.9995188853013948, 0.9994997090832396, 0.9993183356366728, 0.9994400424435175, 0.9994868779795223, 0.999538155729168, 0.9995312501317175, 0.9993789657781853, 0.9994442339539455, 0.9994548054567861, 0.999477453460133, 0.9994847719313813, 0.9994484890569231, 0.9994418498151304, 0.9995079291848901, 0.9994711079783859, 0.9995770571775666, 0.999485094422466, 0.9994551053468359, 0.9995230980837229, 0.9994664195841043, 0.9994094370995563, 0.9995110420974335, 0.9994362014931573, 0.9995731027656826, 0.9995489943208492, 0.9990514218806035, 0.9995147454366133, 0.9994823498684318, 0.9994760814110144, 0.999577758784197, 0.9990079323385279, 0.9994297317319802, 0.9994418324776184, 0.9994075732558478, 0.9995401661775851, 0.9992999654172481, 0.9995206790708213, 0.9995324703488049, 0.9994551053468359, 0.9995230980837231, 0.9994664195841046, 0.9994094370995564, 0.9995110420974336, 0.9994362014931574, 0.9995731027656828, 0.9995489943208493, 0.9990492450563018, 0.9995147454366134, 0.9994823498684318, 0.9994760814110146, 0.9995777587841973, 0.999007932338528, 0.9994297317319805, 0.9994418324776186, 0.9994075732558481, 0.9995401661775851, 0.9992999654172482, 0.9995206790708215, 0.999532470348805, 0.9993013807849245, 0.9994344492569686, 0.9995358246179076, 0.9995188853013948, 0.9994997090832395, 0.9993183356366728, 0.9994400424435175, 0.9994868779795223, 0.999538155729168, 0.9995312501317175, 0.9993789657781853, 0.9994442339539457, 0.9994548054567862, 0.999477453460133, 0.9994847719313812, 0.9994484890569231, 0.9994418498151303, 0.9995079291848901, 0.999471107978386, 0.9995770571775666, 0.999485094422466, 0.9992626016568953, 0.9995144844019338, 0.9995460533336447, 0.9994487703895798, 0.9994341030357902, 0.9994277361138151, 0.9995065968535355, 0.9995340631758372, 0.999530686497732, 0.999426204281705, 0.9993963640685806, 0.999411472525704, 0.9995366303277334, 0.9993891174005957, 0.999490490387445, 0.9994201956923203, 0.9993321384538573, 0.9995540107497178, 0.9994607633414886, 0.9995125756924349, 0.9993758571776077, 0.998921714734468, 0.9989193984437936]
# 0.9994439365950352
# time_cost is:  80.03747224807739


