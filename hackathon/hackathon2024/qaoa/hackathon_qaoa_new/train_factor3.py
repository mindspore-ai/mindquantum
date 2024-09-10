import numpy as np
import json, glob
from train_parameter import *

def get_parameter(factor3, Jc_dict, p, Nq=12):
    k = order(Jc_dict)
    import csv
    # Read the parameters of infinite size limit, and the maximum order is 6 surported here.
    k = min(k,10)
    with open('utils/transfer_data_high_order.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if (row[0]) == str(k):
                new_row = [item for item in row if item != '']
                length = len(new_row)
                if length == 3+2*p:
                    gammas = np.array([float(new_row[i]) for i in range(3,3+p)] )
                    betas = np.array([float(new_row[i]) for i in range(3+p,3+2*p)])
    # rescale the parameters for specific case
    gammas = gammas*factor3
    factor = rescale_factor(Jc_dict)
    return gammas*factor, betas

def single_score(factor3, Jc_dict, Nq=12):    
    hamop = build_ham_high(Jc_dict)
    ham = Hamiltonian(hamop)
    s = 0
    for depth in [4,8]:
        gamma_List, beta_List= get_parameter(factor3, Jc_dict, depth, Nq=Nq)
        circ= qaoa_hubo(Jc_dict, Nq, gamma_List,beta_List ,p=depth)
        sim=Simulator('mqvector',n_qubits=Nq)
        E = sim.get_expectation(ham, circ).real   
        s += -E
    return s

#损失函数
def cost_func(factor3, Jc_dict):
    s = single_score(factor3, Jc_dict)
    print(factor3, s)
    return -s

#返回Jc_dict的能量最小值和对应的factor3
def train_factor3(Jc_dict, Nq=12):
    num_edge = len(Jc_dict) #边的数目
    D = max(2*num_edge/Nq, 1.001)
    factor3_0 = np.arctan(1/np.sqrt(D-1))   #factor3的初始值
    res = minimize(cost_func, factor3_0, args=(Jc_dict), method="BFGS")
	#res = minimize(cost_func, factor3_0, args=(Jc_dict), method="BFGS", tol=1)
    #res = minimize(cost_func, factor3_0, args=(Jc_dict), method="COBYLA")
    return res.fun, res.x

def train_factor3_dir(dir):
    try:
        file_handle = open('factor3_dict.json', 'r')
        factor3_dict = json.load(file_handle)
    except:
        factor3_dict = {}
    pattern = f'{dir}/*.json'  # 匹配所有子目录下的.json文件
    files = glob.glob(pattern, recursive=True)  # 使用glob.glob获取所有匹配的文件路径列表
    num_file = len(files)
    i = 1 #记录搜索文件数
    num_train_file = 1  #记录新增训练文件数
    total_score = 0 #记录总得分
    for file in files:
        k, distribution_index, portion, num_edges, file_i = get_sourse_from_file_name(file)
        the_key = f'{(k, distribution_index, portion, num_edges, file_i)}'
        if the_key not in factor3_dict or factor3_dict[the_key]<0:
            Jc_dict = load_data(f'{file}')
            print(f'(正在训练文件：{file}, 阶数: {k}, 分布: {distribution_index}, 比例: {portion}, 边数: {num_edges}, 文件序号: {file_i}, 进度：{num_train_file}/{i}/{num_file})')
            fun, x = train_factor3(Jc_dict)
            file_score = -fun
            total_score += file_score
            factor3_dict[the_key] = x[0]
            print(f'找到最优factor3: {x[0]}')
            print(f'得分: {file_score}')
            with open('factor3_dict.json', 'w') as file_handle:  #这里存盘有个好处，就是不用一次训练完，可以下次接着训练，自动继续
                json.dump(factor3_dict, file_handle)
            num_train_file += 1
        else:
            print(f"文件先前已训练: {file}")
        i += 1
    print(f'训练完成: 文件数: {num_train_file-1}, 总得分: {total_score}, 平均分: {total_score/(num_train_file-1)}')

#返回Jc_dict的能量最小值和对应的factor3
def train_factor3_with_init(Jc_dict, factor3_0, Nq=12):
    res = minimize(cost_func, factor3_0, args=(Jc_dict), method="COBYLA")
    return res.fun, res.x

def train_factor3_dir_with_init(dir):
    try:
        file_handle = open('factor3_dict.json', 'r')
        factor3_dict = json.load(file_handle)
    except:
        factor3_dict = {}
    pattern = f'{dir}/*.json'  # 匹配所有子目录下的.json文件
    files = glob.glob(pattern, recursive=True)  # 使用glob.glob获取所有匹配的文件路径列表
    num_file = len(files)
    i = 1 #记录搜索文件数
    num_train_file = 1  #记录新增训练文件数
    total_score = 0 #记录总得分
    for file in files:
        k, distribution_index, portion, num_edges, file_i = get_sourse_from_file_name(file)
        the_key = f'{(k, distribution_index, portion, num_edges, file_i)}'
        if k > 6:
            Jc_dict = load_data(f'{file}')
            factor3_0 = factor3_dict[the_key]
            print(f'(正在训练文件：{file}, factor3_0: {factor3_0}, 阶数: {k}, 分布: {distribution_index}, 比例: {portion}, 边数: {num_edges}, 文件序号: {file_i}, 进度：{num_train_file}/{i}/{num_file})')
            fun, x = train_factor3_with_init(Jc_dict, factor3_0)
            file_score = -fun
            total_score += file_score
            factor3_dict[the_key] = x[0]
            print(f'找到最优factor3: {x[0]}')
            print(f'得分: {file_score}')
            with open('factor3_dict.json', 'w') as file_handle:  #这里存盘有个好处，就是不用一次训练完，可以下次接着训练，自动继续
                json.dump(factor3_dict, file_handle)
            num_train_file += 1
        else:
            print(f"文件先前已训练: {file}")
        i += 1
    print(f'训练完成: 文件数: {num_train_file-1}, 总得分: {total_score}, 平均分: {total_score/(num_train_file-1)}')


if __name__ == '__main__':
    # train_factor3_dir('data/train')
    train_factor3_dir_with_init('data/train')
