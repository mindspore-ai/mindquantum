import numpy as np
from Gradient import GradientInput,adam_iters
from joblib import Parallel,delayed
import time

class AdaptiveQaoaInput:

    def __init__(self,initial_pointR,n,pro_solve,qaoa_variant,sys_para):

        self.R=len(initial_pointR)
        
        self.initial_pointR=initial_pointR

        self.n=n

        self.qaoa_variant=qaoa_variant

        self.pro_solve=pro_solve

        self.sys_para=sys_para




class AdaptiveQaoaOutput:
        
    def __init__(self,energy,bias_energy,itera,bias,others):

        self.energy=energy

        self.bias_energy=bias_energy

        self.itera=itera

        self.bias=bias

        self.others=others
        
        

def adaptive_qaoa(input_para):

    n=input_para.n

    R_qaoa=input_para.R
    
    initial_pointR=input_para.initial_pointR.copy()
   
    # 为每次运行的结果创建存储容器
    best_bias_energy = np.zeros((2))     # 存储最佳偏置能量
    opt_bias_energyR = np.zeros((R_qaoa)) # 存储每次运行的偏置能量
    best_bias = np.zeros(n)              # 存储最佳偏置向量

    ave_iteration = np.zeros(1)          # 存储平均迭代次数
    opt_iterationR = np.zeros((R_qaoa))  # 存储每次运行的迭代次数

    best_energy = np.zeros(1)            # 存储找到的最佳能量
    opt_energyR = np.zeros((R_qaoa))   # 存储每次运行的能量

    best_other_opt = np.empty(1, dtype=object) # 存储其他优化细节

    t1=time.perf_counter()

    # 核心准备步骤：创建任务列表
    adam_input=[GradientInput(
                              initial_pointR[k],
                              n,
                              input_para.qaoa_variant,
                              input_para.pro_solve,
                              input_para.sys_para,
                              ) 
                     for k in range(R_qaoa)]
            
        
    #output
    # 如果需要并行的运行次数大于1
    if R_qaoa != 1:
        # 使用joblib库并行执行所有任务
        gradient_result = Parallel(n_jobs=10)(delayed(adam_iters)(adam) for adam in adam_input)
    else:
        # 如果只运行一次，则直接执行，无需并行
        gradient_result = [adam_iters(adam) for adam in adam_input]
        
    gradient_result=[gradient_result[k] for k in range(R_qaoa)]

   # 将每次运行的结果从gradient_result列表中取出，填入之前准备好的表格中
    for k in range(R_qaoa):
        opt_energyR[k] = gradient_result[k].energy / (-1) # 获取能量值
        opt_iterationR[k] = gradient_result[k].itera    # 获取迭代次数
        opt_bias_energyR[k] = gradient_result[k].bias_energy # 获取偏置能量
   # 找到能量最高的那次运行
    ma = np.where(opt_energyR == np.max(opt_energyR))
    # 获取那次运行的索引
    p_ma = ma[0][0]
    # 使用这个最优索引，从所有结果中精确地提取出最好的一次运行的所有信息
    best_energy=opt_energyR[p_ma]
    ave_iteration=np.mean(opt_iterationR,axis=0)
    best_bias=gradient_result[p_ma].bias_opt
    best_bias_energy[0]=gradient_result[p_ma].bias_energy
    best_bias_energy[1]=np.max(opt_bias_energyR)
    best_other_opt=gradient_result[p_ma].others
    t2=time.perf_counter()
    print('Type:',input_para.qaoa_variant['type'],',energy:',best_energy,',bias_energy:',best_bias_energy,',warm-start energy:',input_para.qaoa_variant['energy_WS'],',iterations:',ave_iteration,',time',t2-t1)
    t1=t2
    # 将所有最优结果打包成一个标准格式的输出对象
    result = AdaptiveQaoaOutput(best_energy,
                                best_bias_energy,
                                ave_iteration,
                                best_bias,
                                best_other_opt)
    return result

