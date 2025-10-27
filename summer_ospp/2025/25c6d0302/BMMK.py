import numpy as np
from joblib import Parallel,delayed


def BMMC_cost(angle,edges,k=2):

    E=0
    
    if k==2:
        theta=angle
        for j,k,w in edges:
            e=w/2*(1-np.cos(theta[int(j)]-theta[int(k)]))
            E+=e
            
    elif k==3:
        n=int(len(angle)/2)
        theta=angle[:n]
        phi=angle[n:]
        for j,k,w in edges:
            
            xj=np.array([np.sin(theta[int(j)])*np.cos(phi[int(j)]),np.sin(theta[int(j)])*np.sin(phi[int(j)]),np.cos(theta[int(j)])])
            xk=np.array([np.sin(theta[int(k)])*np.cos(phi[int(k)]),np.sin(theta[int(k)])*np.sin(phi[int(k)]),np.cos(theta[int(k)])])
            e=np.linalg.norm(xj-xk)**2
           
            E+=e/2*w
            

    return -E
        



def gradient_function(para,edges,epsilon,k=2):
    e1=BMMC_cost(para, edges,k)
    
    gra=np.zeros(len(para))
    
    
    for j in range(len(para)):
        p_j=para[j].copy()
        para[j]=para[j]+epsilon   
        e2=BMMC_cost(para, edges,k)
        para[j]=p_j
        gra[j]=(e2-e1)/epsilon    
    
    return gra,e1




def BMMK_opt(para_input):
    # --- 1. 从输入字典中解包参数 ---
    para0=para_input['para0']
    edges=para_input['edges']
    max_iters=para_input['max_iters']
    k=para_input['k']
    
    # --- 2. 设置优化参数 ---
    para_precision=1e-5
    final_precision=1e-2

    
    # --- 3. 设置优化算法参数 ---
    beta1=0.9 # 动量因子
    beta2=0.999 # 二阶动量因子
    alpha=0.02 # 学习率
    epsilon=10**(-8) # 防止除零的小量
    

    # --- 4. 初始化优化算法状态 ---
    m=np.zeros(len(para0)) # 一阶动量
    v=np.zeros(len(para0)) # 二阶动量


    # --- 5. 初始化优化变量 ---
    theta=np.zeros((max_iters+1,len(para0)))

    theta[0]=para0.copy() # 初始化优化变量

    E=np.zeros(max_iters+1)# 创建一个数组来存储每次迭代的能量

    
    g=np.zeros((max_iters+1,len(para0)))# 创建一个数组来存储每次迭代的梯度

    
    g[0],E[0]=gradient_function(theta[0],edges,para_precision,k)  # 计算初始梯度和能量


    # --- 6. 开始优化 ---
    t=0
    for t in range(1,max_iters+1):
        
        m=beta1*m+(1-beta1)*g[t-1] # 更新一阶动量
        v=beta2*v+(1-beta2)*g[t-1]**2 # 更新二阶动量
        
        alphat=alpha*np.sqrt(1-beta2**t)/(1-beta1**t) # 计算学习率
        
        theta_change=alphat*m/(np.sqrt(v)+epsilon)
        theta[t]=theta[t-1]-theta_change

        g[t],E[t]=gradient_function(theta[t],edges,para_precision,k)

        
        if np.abs(E[t]-E[t-1]) < final_precision:
            # 如果能量变化小于最终精度，则停止优化
            break

    res={'energy':-E[t],'theta_opt':theta[t],'ite':t+1}

    return res

def BMMK_optR(para0,edges,max_iters,k=2):
    
 
    R=len(para0)
    para_input=[{'para0':para0[i],'edges':edges,'max_iters':max_iters,'k':k} for i in range(R)]
    
    BMMK_energy=np.zeros(R)
    BMMK_solution=np.zeros((R,len(para0[0])))
    BMMK_ite=np.zeros(R)
    
    
    BMMK_result=Parallel(n_jobs=10)(delayed(BMMK_opt)(para) for para in para_input)
    
    for j in range(R):
        BMMK_energy[j]=BMMK_result[j]['energy']
        BMMK_solution[j]=BMMK_result[j]['theta_opt']
        BMMK_ite[j]=BMMK_result[j]['ite']

   # --- 选出最佳结果 ---
    # 找到能量最大（即割边权重最大）的那次运行的索引
    p=np.where(BMMK_energy==np.max(BMMK_energy))[0][0]
    
    # 提取最佳的能量、解和迭代次数。
    E=BMMK_energy[p]
    theta= BMMK_solution[p]
    ite=BMMK_ite[p]
    
    return E,theta,ite


