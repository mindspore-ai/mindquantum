import numpy as np


def main(Jc_dict, p, Nq=14):
    '''
        The main function you need to change!!!
    Args:
        Jc_dict (dict): the ising model
        p (int): the depth of qaoa circuit
    Returns:
        gammas (Union[numpy.ndarray, List[float]]): the gamma parameters, the length should be equal to depth p.
        betas (Union[numpy.ndarray, List[float]]): the beta parameters, the length should be equal to depth p.
    '''
    D = ave_D(Jc_dict,Nq)
    k = order(Jc_dict)
    import csv
    # Read the parameters of infinite size limit, and the maximum order is 6 surported here.
    k = 6
    with open('utils/transfer_data.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if (row[0]) == str(k):
                new_row = [item for item in row if item != '']
                length = len(new_row)
                if length == 3+2*p:
                    gammas = np.array([float(new_row[i]) for i in range(3,3+p)] )
                    betas = np.array([float(new_row[i]) for i in range(3+p,3+2*p)])
    fac2 = [0.02, -0.1,0.05,0.15]
    fac4 = [0.1, 0, 0, 0.01]
    fac8 = [0.12, 0.05, 0.05, 0, -0.05, -0.05, -0.05, 0]
    fac16 = [-0.05, -0.05, 0.1, -0.05, 0, 0, 0, 0]

    if p == 4:
        betas = [x + y for x, y in zip(betas, fac4)]
        gammas = [x + y for x, y in zip(gammas, fac2)]
        gammas = np.array(gammas).astype(float)
    elif p == 8:
        betas = [x + y for x, y in zip(betas, fac8)]
        gammas = [x + y for x, y in zip(gammas, fac16)]
        gammas = np.array(gammas).astype(float)
    else:
        betas = betas
    # rescale the parameters for specific case
    gammas = trans_gamma(gammas, D)
    factor = rescale_factor(Jc_dict)

    return gammas*factor, betas

def trans_gamma(gammas, D):
    return gammas*np.arctan(1/np.sqrt(D-1))

def rescale_factor(Jc):
    '''
    Get the rescale factor, a technique from arXiv:2305.15201v1
    '''
    import copy
    Jc_dict=copy.deepcopy(Jc)
    keys_len={}

    for key in Jc_dict.keys():
        if len(key) in keys_len:
            keys_len[len(key)]+=1
        else:
            keys_len[len(key)]=1
    norm=0
    #remove_negative_weights(Jc_dict)
    for key,val in Jc_dict.items():
        norm+= val**2/1.32/(keys_len[len(key)])
    norm = np.sqrt(norm)
    return  1/norm

def remove_negative_weights(Jc_dict):
    '''
    遍历字典中的键，移除权重小于0的键。
    '''
    # 遍历Jc_dict中的键值对
    for key, value in list(Jc_dict.items()):  # 使用list()来复制键值对列表，因为字典在迭代时可能会被修改
        if abs(value) < 4:  # 检查权重是否小于0
            del Jc_dict[key]  # 如果权重小于0，删除该键

def ave_D(Jc,nq):
    return 2*len(Jc)/nq

def order(Jc):
    return max([len(key)  for key in Jc.keys()])
