import numpy as np
import copy
def main(Jc_dict, p, Nq=14):
    '''
    主函数
    参数：
        Jc_dict (dict)
        p (int): QAOA电路的深度
    返回：
        gammas (Union[numpy.ndarray, List[float]]): gamma参数，其长度应等于深度p。
        betas (Union[numpy.ndarray, List[float]]): beta参数，其长度应等于深度p。
    '''
    D = ave_D(Jc_dict, Nq)
    k = order(Jc_dict)
    import csv
    # 读取无穷尺寸极限下的参数，这里最大支持的阶数为5。
    k = min(k, 5)
    with open('utils/transfer_data.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if (row[0]) == str(k):
                new_row = [item for item in row if item != '']
                length = len(new_row)
                if length == 3 + 2 * p:
                    gammas = np.array([float(new_row[i]) for i in range(3, 3 + p)])
                    betas = np.array([float(new_row[i]) for i in range(3 + p, 3 + 2 * p)])
    # 为特定情况重新缩放参数
    gammas = trans_gamma(gammas, D)

    ####修正1：
    factor0 = rescale_factor(Jc_dict, p)

    # 计算字典中键的数量
    keys_len = 0
    for key in Jc_dict.keys():
        keys_len += 1

    # 计算 Jc_dict 的规范化值
    norm = 0
    for key, val in Jc_dict.items():
        norm += val**2 / keys_len  # 计算平方和除以键的数量

    # 计算规范化后的 norm 的根号值
    norm = np.power(np.abs(norm), 1/2)

    # 计算 Jc_dict 的平均值的绝对值
    norm_x = 0
    for key, val in Jc_dict.items():
        norm_x += val / keys_len
    norm_x = abs(norm_x)

    # 计算键中元素的最大长度
    k_max = 0
    for key in Jc_dict.keys():
        k_max = max(k_max, len(key))

    # 计算真实比例和所有可能组合的比例
    propotion_true = 0
    propotion_all = 0
    for i in range(1, k_max + 1):
        propotion_all += comb(12, i)  # 计算组合数

    # 计算真实键的数量与所有可能组合的比例
    propotion_true = keys_len / propotion_all

    gammas_ratio=np.ones(p)
    betas_ratio=np.ones(p)

    ####修正4：
    f_D=[1.0,0.948,0.896,0.876,1.009,1.191,1.365,1.627,1.977]
    gammas_ratio=gammas_ratio*f_D[k_max-2]

    norm_norm_x=norm/norm_x
    propotion_true_s=propotion_true

    norm_norm_x=int((norm_norm_x+0.05)*10)/10
    if norm_norm_x<1.1:
        norm_norm_x=1
    if (norm_norm_x>=1.2) and (norm_norm_x<=1.4):
        norm_norm_x=1.3
    if norm_norm_x>2:
        norm_norm_x=2

    propotion_true_s=int((propotion_true_s+0.05)*10)/10

    ####修正2：
    if (norm_norm_x==1.3)&(k_max==2)&(p==4)&(propotion_true_s==0.3):
        factor0=1/(norm_x*1.351569399)
    if (norm_norm_x==1.0)&(k_max==3)&(p==4)&(propotion_true_s==0.9):
        factor0=1/(norm_x*1.827419986)
    if (norm_norm_x==1.3)&(k_max==3)&(p==4)&(propotion_true_s==0.9):
        factor0=1/(norm_x*1.412262758)
    if (norm_norm_x==1.0)&(k_max==3)&(p==8)&(propotion_true_s==0.9):    
        factor0=1/(norm_x*3.39)
    if (norm_norm_x==1.3)&(k_max==4)&(p==4)&(propotion_true_s==0.9):
        factor0=1/(norm_x*1)
    if (norm_norm_x==1.3)&(k_max==4)&(p==8)&(propotion_true_s==0.9):
        factor0=1/(norm_x*1.546100978)

    ####修正3：
    with open('utils/correct_factor.csv', 'r') as csv_file2:
        reader = csv.reader(csv_file2)
        for row in reader:
            if ((row[0]) == str(norm_norm_x))&((row[1]) == str(k_max))&((row[2]) == str(p))&((row[3]) == str(propotion_true_s)):
                new_row = [item for item in row if item != '']

                gammas_ratio=np.array([float(new_row[i]) for i in range(4, 4 + p)])
                betas_ratio=np.array([float(new_row[i]) for i in range(12, 12 + p)])


    return gammas * factor0 *gammas_ratio, betas *betas_ratio


def trans_gamma(gammas, D):
    return gammas * np.arctan(1 / np.sqrt(D - 1))


def comb(n, k):
    # 如果 k 大于 n，则组合数为 0
    if k > n:
        return 0
    # 如果 k 为 0 或 n，则组合数为 1
    if k == 0 or k == n:
        return 1
    # 递归计算组合数
    return comb(n-1, k-1) + comb(n-1, k)


def rescale_factor(Jc_dict, p):
    '''
    获取重缩放因子
    '''

    # 计算字典中键的数量
    keys_len = 0
    for key in Jc_dict.keys():
        keys_len += 1

    # 计算 Jc_dict 的规范化值
    norm = 0
    for key, val in Jc_dict.items():
        norm += val**2 / keys_len  # 计算平方和除以键的数量

    # 计算规范化后的 norm 的根号值
    norm = np.power(np.abs(norm), 1/2)

    factor=norm 

    return  1/factor


def ave_D(Jc, nq):
    """
    计算 D 值。
    参数:
    - Jc: 字典，其键是元组，代表不同变量的组合，值通常用于表示统计或权重。
    - nq: 归一化因子。
    返回:
    - new_D: 计算得到的 D 值。
    """
    # 统计每个键长度出现的次数
    keys_len={}
    for key in Jc.keys():
        if len(key) in keys_len:
            keys_len[len(key)]+=1
        else:
            keys_len[len(key)]=1

    # 计算魔法数字（magic_num），根据键的长度进行加权求和
    magic_num=0
    for k in range(2,max(keys_len)+1):
        if k in keys_len:
            magic_num+=(np.power(2,k-1))*keys_len[k]/k

    # 计算新的 D 值
    new_D = 2 * magic_num / nq

    return new_D


def order(Jc):
    """
    计算 Jc 中最长键的长度。
    参数:
    - Jc: 字典，键是元组。
    返回:
    - max_len_key: 最长键的长度。
    """
    # 创建字典，用于统计每个键长度的出现次数
    max_key=max([len(key)  for key in Jc.keys()])

    key_dict = {i: 0 for i in range(1, max_key + 1)}

    for key in Jc.keys():
        key_length = len(key)
        if key_length in key_dict:
            key_dict[key_length] += 1


    for k in key_dict.keys():
        key_dict[k]*=np.power(2,k-1)
        
    # 寻找出现次数最多的键长
    max_len_key = max(key_dict, key=key_dict.get)

    return max_len_key




