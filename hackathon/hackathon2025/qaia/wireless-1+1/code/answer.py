
import numpy as np
import torch
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm
import random
from mindquantum.algorithm.qaia import LQA,BSB
import time

# 定义优化函数，根据输入变量选择不同的优化路径
def optimized(variables):
    """
    优化函数，根据输入变量选择不同的优化路径并返回最优的相位角和振幅。
    
    参数:
        variables (list): 输入变量列表，包含：
            - variables[0]: theta_0 (目标角度)
            - variables[1]: n_bit_phase (相位量化位数)
            - variables[2]: opt_amp_or_not (是否优化振幅，布尔值)
    
    返回:
        phase_angle (numpy.ndarray): 最优相位角
        amp (numpy.ndarray): 最优振幅
    """
    # 固定随机数种子，确保结果可复现
    np.random.seed(0)
    # 如果不优化振幅
    if variables[2] == False:
        param1=get_param1(variables)
        A1 = YJC0002_more_amp(variables,param1)
        A1.solve()
        A1.plot()
        param2=get_param2(variables)
        A2 = YJC0001_more_amp(variables,param2)
        A2.solve()
        A2.plot()
        if get_score(A1.phase_angle, A1.amp, variables,n_angle = 500)>get_score(A2.phase_angle, A2.amp, variables,n_angle = 500):
            phase_angle = A1.phase_angle
            amp = A1.amp
        else:
            phase_angle = A2.phase_angle
            amp = A2.amp      
    # 如果优化振幅
    if variables[2] == True:
        param=get_param3(variables)
        A = YJC0001_more_amp(variables,param)
        A.solve()
        A.plot()
        phase_angle = A.phase_angle
        amp = A.amp

    return phase_angle, amp
# 定义函数 get_theta_0_add_k，用于计算 theta_0_add_k 的值
def get_theta_0_add_k(x,dim):
    # 将输入角度转换为 [0, 1] 范围
    x/=180
    if dim==2:
        if x<=0.5:
            y = 503.5719566682560639*x**3 + -580.2211270839466124*x**2 + 213.0068173694828033*x + -23.2261957225655351
        elif x>0.5:
            y = -503.5719566682409436*x**3 + 930.4947429207979894*x**2 + -563.2804332063460606*x + 113.1314512312249008
    else:
        if x<=0.5:
            y = 21.9858827334342521*x**3 + -39.9473463683101713*x**2 + 20.4431373889164121*x + -1.2100770418056084
        elif x>0.5:
            y = -79.3573086897350350*x**3 + 140.7531537445944423*x**2 + -82.6029147807481365*x + 18.0049292828227188
    return y
# 定义函数 get_X_bbb_k，用于计算 X_bbb_k 的值
def get_X_bbb_k(x,dim):
    # 将输入角度转换为 [0, 1] 范围
    x/=180
    if dim==2:
        if x<=0.5:
            y = -4073.7362452397364905*x**3 + 4777.2821532591324285*x**2 + -1832.2067281221573012*x + 232.3236762837144624
        elif x>0.5:
            y = 4073.7362452394463617*x**3 + -7443.9265824595549930*x**2 + 4498.8511573227951885*x + -896.3371438189875562
    else:
        if x<=0.5:
            y = -2596.5208752116222968*x**3 + 3100.6438889121664033*x**2 + -1191.7913415673731379*x + 154.2252326882029649
        elif x>0.5:
            y = 3361.3540819411077791*x**3 + -6218.5851501817151075*x**2 + 3794.5324566372164554*x + -756.5194471414134796
    return y
# 定义函数 get_theta_0_add_k_True
def get_theta_0_add_k_True(x,dim):
    # 将输入角度转换为 [0, 1] 范围
    x/=180
    if dim==2:
        if x<=0.5:
            y = 555.2989370749552336*x**3 + -623.3790418292505819*x**2 + 224.5955368990101420*x + -24.2954264772491122
        elif x>0.5:
            y = -555.2989370748728106*x**3 + 1042.5177693954667575*x**2 + -643.7342644652869694*x +132.2200056674486177
    else:
        if x<=0.5:
            y = 233.0187512659841786*x**3 + -285.6899804796076410*x**2 + 113.6466984120207684*x + -12.9643475584361365
        elif x>0.5:
            y = -233.0187512659927052*x**3 + 413.3662733183607543*x**2 + -241.3229912507678421*x + 48.0111216399631786
    return y
# 定义函数 get_theta_0_add_amp_k_True
def get_theta_0_add_amp_k_True(x,dim):
    x/=180
    if dim==2:
        if x<=0.5:
            y = 466.7583870110232169*x**3 + -542.7051773498774310*x**2 + 200.9413421984049251*x + -22.0405482373096788
        elif x>0.5:
            y = -466.7583870109958184*x**3 + 857.5699836831438461*x**2 + -515.8061485316916333*x +102.9540036222356463
    else:
        if x<=0.5:
            y = 230.2643853785793624*x**3 + -297.5193247923270974*x**2 + 123.1095342268179280*x + -14.5476855255997624
        elif x>0.5:
            y = -230.2643853785949659*x**3 + 393.2738313434415431*x**2 + -218.8640407779214740*x + 41.3069092874745607
    return y
# 定义函数 get_p3_k，用于计算 p3_k 的值
def get_p3_k(x,dim):
    x/=180
    if dim==2:
        if x<=0.5:
            y = -3525.4123603232792448*x**3 + 3929.9419831407958554*x**2 + -1398.9883160472197687*x + 163.5031744088124128
        elif x>0.5:
            y = 3525.4123603229527362*x**3 + -6646.2950978284616212*x**2 + 4115.3414307351267780*x + -830.9555188208255458
    else:
        if x<=0.5:
            y = 4706.6469114320334484*x**3 + -5613.3988360525008829*x**2 + 2153.7167281296951842*x + -261.1246831008564868
        elif x>0.5:
            y = -4706.6469114313549653*x**3 + 8506.5418982423598209*x**2 + -5046.8597903200470682*x + 985.8401204082227878
    return y
# 定义函数 get_p4_5_k，用于计算 p4_5_k 的值
def get_p4_5_k(x,dim):
    x/=180
    if dim==2:
        if x<=0.5:
            y = 220.2590356862753822*x**3 + 57.6314525510600149*x**2 + -151.0254141412489730*x + 37.1581574925727622
        elif x>0.5:
            y = -220.2590356863179579*x**3 + 718.4085596099660052*x**2 + -625.0145980197465860*x + 164.0232315886692334
    else:
        if x<=0.5:
            y = 1810.8689283088133379*x**3 + -2047.6818251161300850*x**2 + 744.9862796104183644*x + -85.7112081999688513
        elif x>0.5:
            y = -1810.8689283090543540*x**3 + 3384.9249598107548991*x**2 + -2082.2294143048693513*x + 422.4621746031870657
    return y
# 定义函数 get_X_bbb_k_True
def get_X_bbb_k_True(x,dim):
    x/=180
    if dim==2:
        if x<=0.5:
            y = 1071.5784279488696029*x**3 + -1083.8602589759318562*x**2 + 313.1086079539708749*x + -19.0561741932220272
        elif x>0.5:
            y = -1071.5784279488864286*x**3 + 2130.8750248707074206*x**2 + -1360.1233738487344453*x + 281.7706027336902821
    else:
        if x<=0.5:
            y = -2267.3417689447655903*x**3 + 2032.4980352194975239*x**2 + -538.3567204027611979*x + 44.6826747759475680
        elif x>0.5:
            y = 2267.3417689448015153*x**3 + -4769.5272716148638210*x**2 + 3275.3859567981030523*x + -728.5177793520899741
    return y
# 定义函数 get_theta_0_add_k_True_2
def get_theta_0_add_k_True_2(x,dim):
    x/=180
    if dim==2:
        if x<=0.5:
            y = 101.0139806479716640*x**3 + -91.8529477485871126*x**2 + 25.9620252909809999*x + -0.6756227680959118
        elif x>0.5:
            y = -101.0139806479733551*x**3 + 211.1889941953311904*x**2 + -145.2980717377239444*x +34.4474354222701038
    else:
        if x<=0.5:
            y = 249.1022801172016727*x**3 + -256.9378165111863837*x**2 + 83.3364703880582454*x + -6.7857779795309767
        elif x>0.5:
            y = -249.1022801172353809*x**3 + 490.3690238404823845*x**2 + -316.7676777173301730*x + 68.7151560145506579
    return y
# 定义函数 get_p3_k_2
def get_p3_k_2(x,dim):
    x/=180
    if dim==2:
        if x<=0.5:
            y = -4054.6451451533143882*x**3 + 4694.1889181927035679*x**2 + -1753.1552875423756177*x + 213.4216082480864713
        elif x>0.5:
            y = 4054.6451451532611827*x**3 + -7469.7465172671400069*x**2 + 4528.7128866168504828*x + -900.1899062548877737
    else:
        if x<=0.5:
            y = -1847.9008441489766028*x**3 + 2343.1660507124142896*x**2 + -974.0647790611349137*x + 133.0333647384189533
        elif x>0.5:
            y = 1847.9008441489181678*x**3 + -3200.5364817344188850*x**2 + 1831.4352100831847565*x + -345.7662077592694345
    return y
# 定义函数 get_p4_5_k_2
def get_p4_5_k_2(x,dim):
    x/=180
    if dim==2:
        if x<=0.5:
            y = 6186.4784530539045591*x**3 + -7004.1574900989344314*x**2 + 2544.6209650704090564*x + -290.2614658069988423
        elif x>0.5:
            y = -6186.4784530537071987*x**3 + 11555.2778690624581941*x**2 + -7095.7413440340851594*x + 1436.6804622183522042
    else:
        if x<=0.5:
            y = 3244.5423747402041954*x**3 + -3798.9932856218256347*x**2 + 1446.4504127990746838*x + -170.5856955013913421
        elif x>0.5:
            y = -3244.5423747399149761*x**3 + 5934.6338385982617183*x**2 + -3582.0909657757224522*x + 721.4138064160000567
    return y
# 定义函数 get_param1
def get_param1(variables):
    """
    生成参数集合 param，用于配置波束成形的参数。
    
    参数:
        variables (list): 包含两个元素的列表，分别为 theta_0 和 n_bit_phase。
    
    返回:
        param (dict): 包含波束成形参数的字典。
    """
    theta_0_add_k=get_theta_0_add_k(variables[0],variables[1])
    X_bbb_k=get_X_bbb_k(variables[0],variables[1])
    theta_0_add_amp_k=1 # 波束成形方向增加的幅度因子
    # 构建参数字典
    param = {
    'theta_0_add': 3*theta_0_add_k,  # 波束成形方向增加的小角度
    'theta_0_add_amp': 3*theta_0_add_amp_k,  # 波束成形方向增加的幅度
    'X_aaa':100,    # 参数 X_aaa，固定值
    'X_ccc':20,     # 参数 X_ccc，固定值
    'X_bbb':20*X_bbb_k,    # 参数 X_bbb，与 X_bbb_k 相关
    'n_iter':2000,# 迭代次数
    }

    return param
# 定义函数 get_param2
def get_param2(variables):
    """
    生成参数集合 param，用于配置波束成形的参数。
    
    参数:
        variables (list): 包含两个元素的列表，分别为 theta_0 和 n_bit_phase。
    
    返回:
        param (dict): 包含波束成形参数的字典。
    """
    # 计算相关参数
    theta_0_add_k=get_theta_0_add_k_True_2(variables[0],variables[1])
    theta_0_add_amp_k=1  # 波束成形方向增加的幅度因子
    p3_k=get_p3_k_2(variables[0],variables[1])
    p4_5_k=get_p4_5_k_2(variables[0],variables[1])
    X_bbb_k=1  # 固定值

    param = {
            'theta_0_add': 3*theta_0_add_k,  # 波束成形方向增加的小角度 3 
            'p0':1,   # J0 主峰单角度 
            'p1':0,   # J1 主峰范围角度【-3,3】 b
            'p2':0, # J2 近旁瓣范围角度【-30,-3】【3,30】  c
            'p3':60*p3_k,  # J3 远旁瓣范围角度【-theta,-30】【30,180-theta】    a    
            'p4':8*p4_5_k,   # J4 theta-3 角度
            'p5':8*p4_5_k,   # J5 theta+3 角度  

            'theta_0_add_amp': 3*theta_0_add_amp_k,  # 波束成形方向增加的小角度 3 
            'X_aaa':100,    # aaa=torch.max(10 * torch.log10(loss3 / (loss0+1e-16)) +15, torch.tensor(0.0))    #[[-theta,-30],[30,180-theta]]
            'X_ccc':20,     # ccc=torch.max(10 * torch.log10(loss2 / (loss0+1e-16)) +30, torch.tensor(0.0))    # [[-30, -3], [3, 30]]
            'X_bbb':20*X_bbb_k,     # bbb=torch.max(10 * torch.log10(loss1 / (loss0+1e-16)) +30, torch.tensor(0.0))    #-3,3
            'n_iter':2000,
        }

    return param
# 定义函数 get_param3
def get_param3(variables):
    """
    生成参数集合 param，用于配置波束成形的参数。
    
    参数:
        variables (list): 包含两个元素的列表，分别为 theta_0 和 n_bit_phase。
    
    返回:
        param (dict): 包含波束成形参数的字典。
    """
    # 计算相关参数
    theta_0_add_k=get_theta_0_add_k_True(variables[0],variables[1])
    theta_0_add_amp_k=get_theta_0_add_amp_k_True(variables[0],variables[1])
    p3_k=get_p3_k(variables[0],variables[1])
    p4_5_k=get_p4_5_k(variables[0],variables[1])
    X_bbb_k=get_X_bbb_k_True(variables[0],variables[1])

    param = {
            'theta_0_add': 3*theta_0_add_k,  # 波束成形方向增加的小角度 3 
            'p0':1,   # J0 主峰单角度 
            'p1':0,   # J1 主峰范围角度【-3,3】 b
            'p2':0, # J2 近旁瓣范围角度【-30,-3】【3,30】  c
            'p3':60*p3_k,  # J3 远旁瓣范围角度【-theta,-30】【30,180-theta】    a    
            'p4':8*p4_5_k,   # J4 theta-3 角度
            'p5':8*p4_5_k,   # J5 theta+3 角度  

            'theta_0_add_amp': 3*theta_0_add_amp_k,  # 波束成形方向增加的小角度 3 
            'X_aaa':100,    # aaa=torch.max(10 * torch.log10(loss3 / (loss0+1e-16)) +15, torch.tensor(0.0))    #[[-theta,-30],[30,180-theta]]
            'X_ccc':20,     # ccc=torch.max(10 * torch.log10(loss2 / (loss0+1e-16)) +30, torch.tensor(0.0))    # [[-30, -3], [3, 30]]
            'X_bbb':20*X_bbb_k,     # bbb=torch.max(10 * torch.log10(loss1 / (loss0+1e-16)) +30, torch.tensor(0.0))    #-3,3
            'n_iter':2000,
        }

    return param
# 定义函数 cosd_f
def cosd_f(x):
    """
    计算角度 x 的余弦值，输入角度以度为单位。
    
    参数:
        x (float): 角度值（单位：度）。
    
    返回:
        float: 角度 x 的余弦值。
    """
    return np.cos(x * np.pi / 180)
# 定义函数 sind_f
def sind_f(x):
    """
    计算角度 x 的正弦值，输入角度以度为单位。
    
    参数:
        x (float): 角度值（单位：度）。
    
    返回:
        float: 角度 x 的正弦值。
    """
    return np.sin(x * np.pi / 180)
# 定义函数 get_score
def get_score(phase_angle, amplitude, variables,n_angle = 10):
    """
    计算波束成形的评分函数。
    
    参数:
        phase_angle (numpy.ndarray): 相位角度数组。
        amplitude (numpy.ndarray): 振幅数组。
        variables (list): 包含 theta_0、n_bit_phase 和 opt_amp_or_not。
        n_angle (int, optional): 角度采样点数，默认为 10。
    
    返回:
        float: 评分值。
    """
    n_angle = n_angle

    N = 32
    theta_0 = variables[0]
    n_bit_phase = variables[1]
    opt_amp_or_not = variables[2]

    # 确保相位和振幅是按照赛题要求的取值
    if opt_amp_or_not is True:
        amplitude = amplitude / np.max(amplitude)
    else:
        amplitude = np.ones(N)
    phase_angle = np.angle(np.exp(1.0j * phase_angle)) + np.pi
    phase_angle = np.round(phase_angle / (2 * np.pi) * (2 ** n_bit_phase)) / (2 ** n_bit_phase) * (2 * np.pi)

    efield = get_efield(n_angle, N, theta_0)
    theta_array = np.linspace(0, 180, 180 * n_angle + 1)
    amp_phase = []
    for i in range(N):
        amp_phase.append(amplitude[i] * np.exp(1.0j * phase_angle[i]))
    F = np.einsum('i, ij -> j', np.array(amp_phase), efield)
    FF = np.real(F.conj() * F)
    db_array = 10 * np.log10(FF / np.max(FF))

    x = theta_array - theta_0
    value_list = []
    for i in range(theta_array.shape[0]):
        if abs(x[i]) >= 30:
            value_list.append(db_array[i] + 15)
    a = max(np.max(value_list), 0)

    target = np.max(db_array)
    for i in range(theta_array.shape[0]):
        if db_array[i] == target:
            max_index = i
            break

    theta_up = 180
    theta_down = 0
    theta_min_up = 180
    theta_min_down = 0
    if abs(theta_array[max_index] - theta_0) > 1:
        y = -100000

    else:
        for i in range(1, n_angle*20):
            if db_array[i + max_index] <= -30:
                theta_up = theta_array[i + max_index]
                break

        for i in range(1, n_angle*20):
            if db_array[-i + max_index] <= -30:
                theta_down = theta_array[-i + max_index]
                break

        for i in range(1, n_angle*20):
            if (db_array[i + max_index] < db_array[i - 1 + max_index]) and (
                    db_array[i + max_index] < db_array[i + 1 + max_index]):
                theta_min_up = theta_array[i + max_index]
                break

        for i in range(1, n_angle*20):
            if (db_array[-i + max_index] < db_array[-i - 1 + max_index]) and (
                    db_array[-i + max_index] < db_array[-i + 1 + max_index]):
                theta_min_down = theta_array[-i + max_index]
                break

        if theta_up == 180 or theta_down == 0:
            y = -100000

        elif theta_min_up < theta_up or theta_min_down > theta_down:
            y = -100000

        else:
            W = theta_up - theta_down
            b = max(W - 6, 0)

            value_list_2 = []
            for i in range(theta_array.shape[0]):
                if abs(x[i]) <= 30 and (x[i] >= theta_min_up - theta_0 or x[i] <= theta_min_down - theta_0):
                    value_list_2.append(db_array[i] + 30)
            c = np.max(value_list_2)

            y = 1000 - 100 * a - 80 * b - 20 * c
    return y
# 定义函数 get_efield
def get_efield(n_angle, N, theta_0):
    """
    计算电场分布。
    
    参数:
        n_angle (int): 角度采样点数。
        N (int): 元素数量。
        theta_0 (float): 中心角度。
    
    返回:
        numpy.ndarray: 电场分布矩阵。
    """
    theta = np.linspace(0, 180, 180 * n_angle + 1)
    x = 12 * ((theta - 90) / 90) ** 2
    E_dB = -1.0 * np.where(x < 30, x, 30)
    E_theta = 10 ** (E_dB / 10)
    EF = E_theta ** 0.5

    phase_x = 1j * np.pi * cosd_f(theta)
    AF = np.exp(phase_x[None, :] * np.arange(N)[:, None])

    efield = EF[None, ...] * AF

    return efield
# 定义函数 better_angle1
def better_angle1(phase_angle, amp, variable):
    """
    优化相位角度以提高评分。
    
    参数:
        phase_angle (numpy.ndarray): 初始相位角度数组。
        amp (numpy.ndarray): 振幅数组。
        variable (list): 包含 theta_0、n_bit_phase 和 opt_amp_or_not。
    
    返回:
        numpy.ndarray: 优化后的相位角度数组。
    """
    best_score = get_score(phase_angle, amp, variable)
    best_phase_angle = phase_angle.copy()
    new_phase_angle = phase_angle.copy()

    for epoch in range(100): # 最多迭代 100 次
        data_score=[]
        for i in range(len(phase_angle)):
            # 尝试增加相位角度
            new_phase_angle=phase_angle.copy()
            new_phase_angle[i] = phase_angle[i]+2*np.pi/(2**variable[1])
            new_score=get_score(new_phase_angle, amp, variable)
            data_score.append(new_score)
            # 尝试减少相位角度
            new_phase_angle=phase_angle.copy()
            new_phase_angle[i] = phase_angle[i]-2*np.pi/(2**variable[1])
            new_score=get_score(new_phase_angle, amp, variable)
            data_score.append(new_score)

        index_max=np.argmax(data_score)

        if data_score[index_max]>best_score:
            if index_max%2==0:
                best_phase_angle[index_max//2] = phase_angle[index_max//2]+2*np.pi/(2**variable[1])
            if index_max%2==1:
                best_phase_angle[index_max//2] = phase_angle[index_max//2]-2*np.pi/(2**variable[1])

            best_score = data_score[index_max]
            phase_angle = best_phase_angle.copy()

        else:
            break  # 如果没有改进，则停止迭代

    return best_phase_angle

########################################################################################
class YJC0001_more_amp():
    """
    一个用于波束成形优化的类，支持相位和振幅优化。
    主要功能包括：
    - 相位优化：通过QUBO（二次无约束二进制优化）方法优化相位。
    - 振幅优化：通过梯度下降方法优化振幅。
    - 结果可视化：绘制波束成形结果。
    """
    def __init__(self, variables,param0001):
        """
        初始化函数，设置波束成形优化的参数。

        Args:
            variables (list): 包含优化变量的列表，如波束方向、是否优化振幅等。
            param0001 (dict): 包含额外优化参数的字典。
        """

        self.param0001=param0001

        self.param = {
            'theta_0': variables[0],  # 波束成形方向，以90度为例
            'N': 32,  # 天线阵子总数
            'n_angle': 10,  # 1度中被细分的次数
            'encode_qubit': 2,  # 进行相位编码的比特个数，样例代码中固定为2，实际对应变量 variables[1]

            # bSB参数界面
            'xi': 0.1,  # 模拟分叉算法中调节损失函数的相对大小
            'dt': 0.3,  # 演化步长
            'n_iter': self.param0001['n_iter'],  # 迭代步数

            # 相位损失函数界面 
            'weight': 0.01,  # 调节损失函数中分子和分母的相对大小
            'range_list': [[-30, -self.param0001['theta_0_add']], [self.param0001['theta_0_add'], 30]], # 需要压制的旁瓣范围相对于主瓣波束成形方向的角度表示
            'range_list_weight': [1, 1], # 每个压制的旁瓣范围各自的权重

            'main_range_list': [[-self.param0001['theta_0_add'], -0], [0, self.param0001['theta_0_add']]], # 需要压制的旁瓣范围相对于主瓣波束成形方向的角度表示
            'main_range_list_weight': [1, 1], # 每个压制的旁瓣范围各自的权重

            'more_far_range_list': [[-variables[0], -30], [30, 180-variables[0]]], # 需要压制的旁瓣范围相对于主瓣波束成形方向的角度表示
            'more_far_range_list_weight': [1, 1], # 每个压制的旁瓣范围各自的权重

            # 连续振幅优化
            'opt_amp_or_not': variables[2], # 根据输入控制是否优化振幅
            'lr': 0.001, # 学习率
        }

        self.variable=variables
        self.EF = self._generate_power_pattern()
        self.AF = self._generate_array_factor()
        self.amp = np.ones(self.param['N'])
        self.efield = torch.tensor(self.EF[None, ...] * self.AF)

    def solve(self):
        """
        执行波束成形优化。
        包括相位优化和（可选的）振幅优化。
        """
        np.random.seed(0)
        begin=time.time()
        if self.param['opt_amp_or_not'] is False:
            self.x_final = self.opt_phase_QUBO()
            self.phase = self.encode(self.x_final)
            self.phase_angle = np.angle(self.phase)

            self.phase_angle = better_angle1(self.phase_angle, self.amp, self.variable)

        # 优化振幅
        if self.param['opt_amp_or_not'] is True:

            self.x_final = self.opt_phase_QUBO()
            self.phase = self.encode(self.x_final)
            self.phase_angle = np.angle(self.phase)

            self.phase_angle = better_angle1(self.phase_angle, self.amp, self.variable)
            best_sc=-1000000
            new_sc=get_score(self.phase_angle, self.amp, self.variable)

            while new_sc>best_sc:
                best_sc=new_sc
                best_amp=self.amp.copy()

                if time.time()-begin>82:
                    break

                self.phase_angle = better_angle1(self.phase_angle, self.amp, self.variable)

                self.amp, _ = self.opt_amp(self.amp, self.phase_angle)
                self.amp = np.array(self.amp.clone().detach().numpy())

                new_sc=get_score(self.phase_angle, self.amp, self.variable,n_angle=100)

            self.amp=best_amp.copy()

        
    def opt_phase_QUBO(self):
        """
        使用QUBO方法优化相位。
        构建QUBO矩阵并调用BSB模块进行优化。
        """
        c1 = 0.5 + 0.5j
        c2 = 0.5 - 0.5j
        # 针对32个相位，对每个相位采用2比特编码，进而用64个实数变量描述损失函数，并且构建对应的64*64的J矩阵
        factor_array = torch.cat((self.efield[:, self._get_index(self.param['theta_0'])] * c1, self.efield[:, self._get_index(self.param['theta_0'])] * c2), dim=0)
        J0 = torch.einsum('i, j -> ij ', factor_array.conj(), factor_array)    
        # 添加主瓣范围的约束
        J1 = 0.0
        for i in range(len(self.param['main_range_list_weight'])):
            num = 0
            a_0 = 0.0
            for j in range(round(self._get_index(self.param['theta_0'] + self.param['main_range_list'][i][0])), round(self._get_index(self.param['theta_0'] + self.param['main_range_list'][i][1])), 1):
                num += 1
                factor_array = torch.cat((self.efield[:, j] * c1, self.efield[:, j] * c2), dim=0)
                a_0 += torch.einsum('i, j -> ij', factor_array.conj(), factor_array)
            J1 += self.param['main_range_list_weight'][i] * a_0 / num
        # 添加旁瓣范围的约束
        J2 = 0.0
        for i in range(len(self.param['range_list_weight'])):
            num = 0
            a_0 = 0.0
            for j in range(round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])), round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][1])), 1):
                num += 1
                factor_array = torch.cat((self.efield[:, j] * c1, self.efield[:, j] * c2), dim=0)
                a_0 += torch.einsum('i, j -> ij', factor_array.conj(), factor_array)
            J2 += self.param['range_list_weight'][i] * a_0 / num
        # 添加更远旁瓣范围的约束            
        J3 = 0.0
        for i in range(len(self.param['more_far_range_list_weight'])):
            num = 0
            a_0 = 0.0
            for j in range(round(self._get_index(self.param['theta_0'] + self.param['more_far_range_list'][i][0])), round(self._get_index(self.param['theta_0'] + self.param['more_far_range_list'][i][1])), 1):
                num += 1
                factor_array = torch.cat((self.efield[:, j] * c1, self.efield[:, j] * c2), dim=0)
                a_0 += torch.einsum('i, j -> ij', factor_array.conj(), factor_array)
            J3 += self.param['more_far_range_list_weight'][i] * a_0 / num
        # 添加额外的约束
        factor_array4 = torch.cat((self.efield[:, self._get_index(self.param['theta_0']-self.param0001['theta_0_add'])] * c1, self.efield[:, self._get_index(self.param['theta_0']-self.param0001['theta_0_add'])] * c2), dim=0)
        J4 = torch.einsum('i, j -> ij ', factor_array4.conj(), factor_array4) 

        factor_array5 = torch.cat((self.efield[:, self._get_index(self.param['theta_0']+self.param0001['theta_0_add'])] * c1, self.efield[:, self._get_index(self.param['theta_0']+self.param0001['theta_0_add'])] * c2), dim=0)
        J5 = torch.einsum('i, j -> ij ', factor_array5.conj(), factor_array5)  


        p0=self.param0001['p0']   # J0 主峰单角度 
        p1=self.param0001['p1']   # J1 主峰范围角度【-3,3】 b
        p2=self.param0001['p2']   # J2 近旁瓣范围角度【-30,-3】【3,30】  c
        p3=self.param0001['p3']   # J3 远旁瓣范围角度【-theta,-30】【30,180-theta】    a  
        p4=self.param0001['p4']   # J4 theta-3 角度  b
        p5=self.param0001['p5']   # J5 theta+3 角度  b  

        J = torch.real(p0*J0+p1*J1-p2*J2-p3*J3-p4*J4-p5*J5).numpy() # 由矩阵的构造可知J[i, j] = J*[j, i]，因此取实部不影响计算结果

        # 根据损失函数的形式完成J矩阵构建后调用mindquantum中QAIA的BSB模块进行优化
        solver = BSB(np.array(J, dtype="float64"), batch_size=128)
        solver.update()
        sample = np.sign(solver.x)
        energy = np.sum(J.dot(sample) * sample, axis=0)

        opt_index = np.argmax(energy)
        x_bit = sample[:, opt_index]

        return  x_bit.reshape(self.param['N'], self.param['encode_qubit'], order='F') # 将x_bit的形式统一为 self.param['N'] * self.param['encode_qubit'] 这一二维矩阵的形式

    def opt_amp(self, amp, phase_angle):

        # 优化振幅的损失函数
        def cost_func_for_amp(amp, phase_angle):

            phase = torch.exp(1.0j * phase_angle)

            main_lobe = torch.einsum('i, i -> ', phase * amp, self.efield[:, self._get_index(self.param['theta_0'])])
            loss0 = torch.real(torch.conj(main_lobe) * main_lobe)

            loss1 = 0
            data1_max = []
            index = self._get_index(self.param['theta_0'] - self.param0001['theta_0_add_amp'])  
            efield_slice = self.efield[:, index].unsqueeze(1)
            one_range = torch.einsum('i, ij -> j', phase * amp, efield_slice)
            data1_max.append(torch.real(torch.conj(one_range) * one_range))
            index = self._get_index(self.param['theta_0'] + self.param0001['theta_0_add_amp'])
            efield_slice = self.efield[:, index].unsqueeze(1)
            one_range = torch.einsum('i, ij -> j', phase * amp, efield_slice)
            data1_max.append(torch.real(torch.conj(one_range) * one_range))
            index_max_1 = torch.argmax(torch.stack(data1_max))
            loss1 += data1_max[index_max_1]

            loss2 = 0
            data2_max=[]
            for i in range(len(self.param['range_list_weight'])):
                one_range = torch.einsum('i, ij -> j', phase * amp, self.efield[:, round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])): round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][1]))])
                data2=torch.real(torch.conj(one_range) * one_range)
                index_max=torch.argmax(data2)
                data2_max.append(data2[index_max])
            index_max_2=torch.argmax(torch.stack(data2_max))
            loss2 += self.param['range_list_weight'][index_max_2] * data2_max[index_max_2]

            loss3 = 0
            data3_max=[]
            for i in range(len(self.param['more_far_range_list_weight'])):
                one_range = torch.einsum('i, ij -> j', phase * amp, self.efield[:, round(self._get_index(self.param['theta_0'] + self.param['more_far_range_list'][i][0])): round(self._get_index(self.param['theta_0'] + self.param['more_far_range_list'][i][1]))])
                data3=torch.real(torch.conj(one_range) * one_range)
                index_max=torch.argmax(data3)
                data3_max.append(data3[index_max])
            index_max_3=torch.argmax(torch.stack(data3_max))
            loss3 += self.param['more_far_range_list_weight'][index_max_3] * data3_max[index_max_3]

            aaa=torch.max(10 * torch.log10(loss3 / (loss0+1e-16)) +15, torch.tensor(0.0))    #[[-theta,-30],[30,180-theta]]
            ccc=torch.max(10 * torch.log10(loss2 / (loss0+1e-16)) +30, torch.tensor(0.0))    # [[-30, -3], [3, 30]]
            bbb=torch.max(10 * torch.log10(loss1 / (loss0+1e-16)) +30, torch.tensor(0.0))    #-3,3

            obj = self.param0001['X_aaa']*aaa + self.param0001['X_ccc']*ccc + self.param0001['X_bbb']*bbb

            return obj,aaa,ccc,bbb

        amplitude = torch.tensor(amp.copy())
        amplitude.requires_grad = True
        optimizer = torch.optim.Adam([amplitude], lr=self.param['lr'])

        for iter in range(self.param['n_iter'] + 1):
            optimizer.zero_grad()
            loss,aaa,ccc,bbb = cost_func_for_amp(amplitude, torch.tensor(phase_angle))
            loss.backward()
            optimizer.step()

        return amplitude, loss


    def encode(self, x_bit):

        c0 = 0.5 + 0.5j
        c1 = 0.5 - 0.5j
        N = x_bit.shape[0]
        phase = c0 * x_bit[:, 0] + c1 * x_bit[:, 1]
        phase.reshape(N)
        return phase
    
    # 优化结果画图函数
    def plot(self):
        # 计算画图相关数据
        self.theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        F = torch.einsum('i, ij -> j', torch.tensor(self.amp) * np.exp(1.0j * self.phase_angle), self.efield).numpy()
        self.FF = np.real(F.conj() * F)
        self.y = 10 * np.log10(self.FF / np.max(self.FF))

        # 画图
        plt.figure()
        plt.plot(self.theta, self.y)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$lg|F(\theta)|^2 - lg|F(\theta)|^2_{max}$' + ' (dB)')
        plt.title('Beamforming Outcome')
        plt.savefig(str(self.param['theta_0']) + '_beamforming.jpg')

    # 生成单元阵子的辐射电场强度（随角度变化的函数）
    def _generate_power_pattern(self):
        theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        x = 12 * ((theta - 90) / 90) ** 2
        E_dB = -1.0 * np.where(x < 30, x, 30)
        E_theta = 10 ** (E_dB / 10)
        EF = E_theta ** 0.5
        return EF

    # 生成阵因子A_n
    def _generate_array_factor(self): 
        theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        phase_x = 1j * np.pi * cosd_f(theta)
        AF = np.exp(phase_x[None, :] * np.arange(self.param['N'])[:, None])
        return AF

    # 获得theta角度对应的矩阵指标
    def _get_index(self, angle_value):
        index = round(angle_value * self.param['n_angle'])
        return index


class YJC0002_more_amp():
    """
    与YJC0001_more_amp类似，但使用不同的相位优化方法。
    """
    def __init__(self, variables,param0002):
        """
        初始化函数，设置波束成形优化的参数。
        """
        self.param0002=param0002

                # 在后续优化过程中使用的参数
        self.param = {
            'theta_0': variables[0],  # 波束成形方向，以90度为例
            'N': 32,  # 天线阵子总数
            'n_angle': 10,  # 1度中被细分的次数
            'encode_qubit': 2,  # 进行相位编码的比特个数，样例代码中固定为2，实际对应变量 variables[1]

            # bSB参数界面
            'xi': 0.1,  # 模拟分叉算法中调节损失函数的相对大小
            'dt': 0.3,  # 演化步长
            'n_iter': self.param0002['n_iter'],  # 迭代步数

            # 相位损失函数界面 
            'weight': 0.01,  # 调节损失函数中分子和分母的相对大小
            'range_list': [[-30, -self.param0002['theta_0_add']], [self.param0002['theta_0_add'], 30]], # 需要压制的旁瓣范围相对于主瓣波束成形方向的角度表示
            'range_list_weight': [1, 1], # 每个压制的旁瓣范围各自的权重

            'main_range_list': [[-self.param0002['theta_0_add'], -0], [0, self.param0002['theta_0_add']]], # 需要压制的旁瓣范围相对于主瓣波束成形方向的角度表示
            'main_range_list_weight': [1, 1], # 每个压制的旁瓣范围各自的权重

            'more_far_range_list': [[-variables[0], -30], [30, 180-variables[0]]], # 需要压制的旁瓣范围相对于主瓣波束成形方向的角度表示
            'more_far_range_list_weight': [1, 1], # 每个压制的旁瓣范围各自的权重

            # 连续振幅优化
            'opt_amp_or_not': variables[2], # 根据输入控制是否优化振幅
            'lr': 0.001, # 学习率
        }

        self.variable=variables

        self.EF = self._generate_power_pattern()
        self.AF = self._generate_array_factor()
        self.amp = np.ones(self.param['N'])
        self.efield = torch.tensor(self.EF[None, ...] * self.AF)

    def solve(self):

        begin=time.time()
        if self.param['opt_amp_or_not'] is False:
            self.x_final = self.opt_phase()
            self.phase = self.encode(self.x_final)
            self.phase_angle = np.angle(self.phase)

            self.phase_angle = better_angle1(self.phase_angle, self.amp, self.variable)

        # 优化振幅
        if self.param['opt_amp_or_not'] is True:

            self.x_final = self.opt_phase()
            self.phase = self.encode(self.x_final)
            self.phase_angle = np.angle(self.phase)

            self.phase_angle = better_angle1(self.phase_angle, self.amp, self.variable)

            best_sc=-1000000
            new_sc=get_score(self.phase_angle, self.amp, self.variable)

            while new_sc>best_sc:
                best_sc=new_sc
                best_amp=self.amp.copy()

                if time.time()-begin>82:
                    break

                self.phase_angle = better_angle1(self.phase_angle, self.amp, self.variable)

                self.amp, _ = self.opt_amp(self.amp, self.phase_angle)
                self.amp = np.array(self.amp.clone().detach().numpy())

                new_sc=get_score(self.phase_angle, self.amp, self.variable,n_angle=100)

            self.amp=best_amp.copy()

        
    def opt_phase(self):
        # 优化相位的损失函数
        def cost_func(x_bit):
            '''
            Args: 
                x_bit: SB算法中的变量x
            Returns:
                obj: 损失函数的数值
            '''
            phase = self.encode(x_bit)
            amp = torch.tensor(self.amp.copy())

            main_lobe = torch.einsum('i, i -> ', phase * amp, self.efield[:, self._get_index(self.param['theta_0'])])
            loss0 = torch.real(torch.conj(main_lobe) * main_lobe)

            loss1 = 0
            data1_max = []
            # 获取索引并保持二维结构
            index = self._get_index(self.param['theta_0'] - self.param0002['theta_0_add'])
            # 使用unsqueeze添加一个维度，使形状为 [N, 1]
            efield_slice = self.efield[:, index].unsqueeze(1)
            one_range = torch.einsum('i, ij -> j', phase * amp, efield_slice)
            data1_max.append(torch.real(torch.conj(one_range) * one_range))
            index = self._get_index(self.param['theta_0'] + self.param0002['theta_0_add'])
            efield_slice = self.efield[:, index].unsqueeze(1)
            one_range = torch.einsum('i, ij -> j', phase * amp, efield_slice)
            data1_max.append(torch.real(torch.conj(one_range) * one_range))
            index_max_1 = torch.argmax(torch.stack(data1_max))
            loss1 += data1_max[index_max_1]

            loss2 = 0
            data2_max=[]
            for i in range(len(self.param['range_list_weight'])):
                one_range = torch.einsum('i, ij -> j', phase * amp, self.efield[:, round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])): round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][1]))])
                data2=torch.real(torch.conj(one_range) * one_range)
                index_max=torch.argmax(data2)
                data2_max.append(data2[index_max])

            index_max_2=torch.argmax(torch.stack(data2_max))
            loss2 += self.param['range_list_weight'][index_max_2] * data2_max[index_max_2]

            loss3 = 0
            data3_max=[]
            for i in range(len(self.param['more_far_range_list_weight'])):
                one_range = torch.einsum('i, ij -> j', phase * amp, self.efield[:, round(self._get_index(self.param['theta_0'] + self.param['more_far_range_list'][i][0])): round(self._get_index(self.param['theta_0'] + self.param['more_far_range_list'][i][1]))])

                data3=torch.real(torch.conj(one_range) * one_range)
                index_max=torch.argmax(data3)
                data3_max.append(data3[index_max])
            index_max_3=torch.argmax(torch.stack(data3_max))
            loss3 += self.param['more_far_range_list_weight'][index_max_3] * data3_max[index_max_3]

            aaa=torch.max(10 * torch.log10(loss3 / (loss0+1e-16)) +15, torch.tensor(0.0))    #[[-theta,-30],[30,180-theta]]
            ccc=torch.max(10 * torch.log10(loss2 / (loss0+1e-16)) +30, torch.tensor(0.0))    # [[-30, -3], [3, 30]]
            bbb=torch.max(10 * torch.log10(loss1 / (loss0+1e-16)) +30, torch.tensor(0.0))    #-3,3

            obj = self.param0002['X_aaa']*aaa + self.param0002['X_ccc']*ccc + self.param0002['X_bbb']*bbb

            return obj,aaa,ccc,bbb
    
        # 初始化
        x = 0.01 * (np.random.randn(self.param['N'], self.param['encode_qubit']))
        y = 0.01 * (np.random.randn(self.param['N'], self.param['encode_qubit']))

        for iter in range(self.param['n_iter'] + 1):
            x_torch = torch.tensor(x)
            x_torch.requires_grad = True

            # 计算梯度与更新参数
            x_sign = x_torch - (x_torch - torch.sign(x_torch)).detach() #dSB
            loss,aaa,ccc,bbb = cost_func(x_sign)
            loss.backward()
            x_grad = (x_torch.grad).clone().detach().numpy()
            y += (-(0.5 - iter / self.param['n_iter']) * x - self.param['xi'] * x_grad / np.linalg.norm(x_grad)) * self.param['dt']
            x = x + y * self.param['dt']
            cond = np.abs(x) > 1
            x = np.where(cond, np.sign(x), x)
            y = np.where(cond, np.zeros_like(y), y)

        return np.sign(x)

    def opt_amp(self, amp, phase_angle):

        # 优化振幅的损失函数
        def cost_func_for_amp(amp, phase_angle):

            phase = torch.exp(1.0j * phase_angle)

            main_lobe = torch.einsum('i, i -> ', phase * amp, self.efield[:, self._get_index(self.param['theta_0'])])
            loss0 = torch.real(torch.conj(main_lobe) * main_lobe)

            loss1 = 0
            data1_max = []
            # 获取索引并保持二维结构
            index = self._get_index(self.param['theta_0'] - self.param0002['theta_0_add_amp'])
            # 使用unsqueeze添加一个维度，使形状为 [N, 1]
            efield_slice = self.efield[:, index].unsqueeze(1)
            one_range = torch.einsum('i, ij -> j', phase * amp, efield_slice)
            data1_max.append(torch.real(torch.conj(one_range) * one_range))
            index = self._get_index(self.param['theta_0'] + self.param0002['theta_0_add_amp'])
            efield_slice = self.efield[:, index].unsqueeze(1)
            one_range = torch.einsum('i, ij -> j', phase * amp, efield_slice)
            data1_max.append(torch.real(torch.conj(one_range) * one_range))
            index_max_1 = torch.argmax(torch.stack(data1_max))
            loss1 += data1_max[index_max_1]

            loss2 = 0
            data2_max=[]
            for i in range(len(self.param['range_list_weight'])):
                one_range = torch.einsum('i, ij -> j', phase * amp, self.efield[:, round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][0])): round(self._get_index(self.param['theta_0'] + self.param['range_list'][i][1]))])
                data2=torch.real(torch.conj(one_range) * one_range)
                index_max=torch.argmax(data2)
                data2_max.append(data2[index_max])

            index_max_2=torch.argmax(torch.stack(data2_max))
            loss2 += self.param['range_list_weight'][index_max_2] * data2_max[index_max_2]

            loss3 = 0
            data3_max=[]
            for i in range(len(self.param['more_far_range_list_weight'])):
                one_range = torch.einsum('i, ij -> j', phase * amp, self.efield[:, round(self._get_index(self.param['theta_0'] + self.param['more_far_range_list'][i][0])): round(self._get_index(self.param['theta_0'] + self.param['more_far_range_list'][i][1]))])

                data3=torch.real(torch.conj(one_range) * one_range)
                index_max=torch.argmax(data3)
                data3_max.append(data3[index_max])
            index_max_3=torch.argmax(torch.stack(data3_max))
            loss3 += self.param['more_far_range_list_weight'][index_max_3] * data3_max[index_max_3]

            aaa=torch.max(10 * torch.log10(loss3 / (loss0+1e-16)) +15, torch.tensor(0.0))    #[[-theta,-30],[30,180-theta]]
            ccc=torch.max(10 * torch.log10(loss2 / (loss0+1e-16)) +30, torch.tensor(0.0))    # [[-30, -3], [3, 30]]
            bbb=torch.max(10 * torch.log10(loss1 / (loss0+1e-16)) +30, torch.tensor(0.0))    #-3,3

            obj = self.param0002['X_aaa']*aaa + self.param0002['X_ccc']*ccc + self.param0002['X_bbb']*bbb

            return obj,aaa,ccc,bbb

        amplitude = torch.tensor(amp.copy())
        amplitude.requires_grad = True
        optimizer = torch.optim.Adam([amplitude], lr=self.param['lr'])

        for iter in range(self.param['n_iter'] + 1):
            optimizer.zero_grad()
            loss,aaa,ccc,bbb = cost_func_for_amp(amplitude, torch.tensor(phase_angle))
            loss.backward()
            optimizer.step()

        return amplitude, loss

    def encode(self, x_bit):

        c0 = 0.5 + 0.5j
        c1 = 0.5 - 0.5j
        N = x_bit.shape[0]
        phase = c0 * x_bit[:, 0] + c1 * x_bit[:, 1]
        phase.reshape(N)
        return phase
    
    # 优化结果画图函数
    def plot(self):
        # 计算画图相关数据
        self.theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        F = torch.einsum('i, ij -> j', torch.tensor(self.amp) * np.exp(1.0j * self.phase_angle), self.efield).numpy()
        self.FF = np.real(F.conj() * F)
        self.y = 10 * np.log10(self.FF / np.max(self.FF))

        # 画图
        plt.figure()
        plt.plot(self.theta, self.y)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$lg|F(\theta)|^2 - lg|F(\theta)|^2_{max}$' + ' (dB)')
        plt.title('Beamforming Outcome')
        plt.savefig(str(self.param['theta_0']) + '_beamforming.jpg')

    # 生成单元阵子的辐射电场强度（随角度变化的函数）
    def _generate_power_pattern(self):
        theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        x = 12 * ((theta - 90) / 90) ** 2
        E_dB = -1.0 * np.where(x < 30, x, 30)
        E_theta = 10 ** (E_dB / 10)
        EF = E_theta ** 0.5
        return EF

    # 生成阵因子A_n
    def _generate_array_factor(self):
        theta = np.linspace(0, 180, 180 * self.param['n_angle'] + 1)
        phase_x = 1j * np.pi * cosd_f(theta)
        AF = np.exp(phase_x[None, :] * np.arange(self.param['N'])[:, None])
        return AF

    # 获得theta角度对应的矩阵指标
    def _get_index(self, angle_value):
        index = round(angle_value * self.param['n_angle'])
        return index

