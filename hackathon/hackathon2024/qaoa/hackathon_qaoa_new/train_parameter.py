from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator

from utils.qcirc import qaoa_hubo, build_ham_high
from main import *
from score import load_data
import numpy as np
from scipy.optimize import minimize
import glob
import json

def get_initial_parameter(Jc_dict, p, Nq=12):
    D = max(2*len(Jc_dict)/Nq, 1.001)   #训练时的D
    k = order(Jc_dict)
    import csv
    # Read the parameters of infinite size limit, and the maximum order is 6 surported here.
    k = min(k,10)
    with open('utils/transfer_data_new.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if (row[0]) == str(k):
                new_row = [item for item in row if item != '']
                length = len(new_row)
                if length == 3+2*p:
                    gammas = np.array([float(new_row[i]) for i in range(3,3+p)] )
                    betas = np.array([float(new_row[i]) for i in range(3+p,3+2*p)])
    # rescale the parameters for specific case
    factor = rescale_factor(Jc_dict)
    gammas = factor*gammas*np.arctan(1/np.sqrt(D-1)) 
    return gammas, betas

def get_initial_parameter_by_factor3(Jc_dict, p, factor3_0, Nq=12):
    D = max(1.2*len(Jc_dict)/Nq, 1.001)   #训练时的D
    k = order(Jc_dict)
    import csv
    # Read the parameters of infinite size limit, and the maximum order is 6 surported here.
    k = min(k,6)
    with open('utils/transfer_data.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if (row[0]) == str(k):
                new_row = [item for item in row if item != '']
                length = len(new_row)
                if length == 3+2*p:
                    gammas = np.array([float(new_row[i]) for i in range(3,3+p)] )
                    betas = np.array([float(new_row[i]) for i in range(3+p,3+2*p)])
    factor = rescale_factor(Jc_dict)
    gammas = factor*gammas*factor3_0
    return gammas, betas

def cost_fun(p, grad_ops):
    f, g = grad_ops(p)
    f = np.real(f)[0, 0]
    g = np.real(g)[0, 0]
    return f, g

def cost_fun_only_f(p, grad_ops):
    f, g = grad_ops(p)
    f = np.real(f)[0, 0]
    g = np.real(g)[0, 0]
    return f

#返回Jc_dict的能量最小值和对应的精确参数
def train_parameter(Jc_dict, depth, Nq=12):
    hamop = build_ham_high(Jc_dict)
    ham = Hamiltonian(hamop)
    gamma_List=[]
    beta_List=[]
    for i in range(depth):
        gamma_List.append(f'g{i}')
        beta_List.append(f'b{i}')
    circ= qaoa_hubo(Jc_dict, Nq, gamma_List, beta_List ,p=depth)
    sim = Simulator('mqvector', circ.n_qubits)                     # 创建模拟器，backend使用‘mqvector’，能模拟5个比特（'circ'线路中包含的比特数）
    grad_ops = sim.get_expectation_with_grad(ham, circ)            # 获取计算变分量子线路的期望值和梯度的算子
    gamma_List, beta_List= get_initial_parameter(Jc_dict, depth, Nq=Nq)
    p0 = [val for pair in zip(gamma_List, beta_List) for val in pair]
    res = minimize(cost_fun, p0, args=(grad_ops, ), method='bfgs', jac=True, tol=1)
    return res.fun, res.x

#从文件名得到来源，例如k2_uni_p0.67_l100_3.json拆分为2 uni 0.67 100 3
def get_sourse_from_file_name(file):
    distribution_dict = {'std':0, 'uni':1, 'bimodal':2, 'exp':3}
    splited_name = file.split('_')
    k = splited_name[0][splited_name[0].find('\k')+2:]
    distribution_index = distribution_dict[splited_name[1]]
    portion = splited_name[2][1:]
    num_edges = splited_name[3][1:]
    file_i = splited_name[4].split('.')[0]
    return int(k), int(distribution_index), float(portion), int(num_edges), int(file_i)

#以参数为索引，搜索文件夹中所有文件的精确参数
def train_parameter_dir(dir):
    try:
        file_handle = open('parameter_dict.json', 'r')
        parameter_dict = json.load(file_handle)
    except:
        parameter_dict = {}
    pattern = f'{dir}/*.json'  # 匹配所有子目录下的.json文件
    files = glob.glob(pattern, recursive=True)  # 使用glob.glob获取所有匹配的文件路径列表
    num_file = len(files)
    i = 1 #记录搜索文件数
    num_train_file = 1  #记录新增训练文件数
    total_score = 0 #记录总得分
    for file in files:
        k, distribution_index, portion, num_edges, file_i = get_sourse_from_file_name(file)
        the_key = f'{(k, distribution_index, portion, num_edges, file_i)}'
        if the_key not in parameter_dict:
            parameter_dict[the_key]={}
            Jc_dict = load_data(f'{file}')
            print(f'(正在训练文件：{file}, 阶数: {k}, 分布: {distribution_index}, 比例: {portion}, 边数: {num_edges}, 文件序号: {file_i}, 进度：{num_train_file}/{i}/{num_file})')
            fun4, x4 = train_parameter(Jc_dict, depth=4)
            fun8, x8 = train_parameter(Jc_dict, depth=8)
            file_score = -fun4-fun8
            total_score += file_score
            parameter_dict[the_key]["parameter4"], parameter_dict[the_key]["parameter8"] = x4.tolist(), x8.tolist()
            parameter_dict[the_key]["score4"], parameter_dict[the_key]["score8"] = -fun4, -fun8
            print(f'找到参数：')
            print(f'参数(depth=4): {x4}')
            print(f'参数(depth=8): {x8}')
            print(f'得分: {file_score}')
            with open('parameter_dict.json', 'w') as file_handle:  #这里存盘有个好处，就是不用一次训练完，可以下次接着训练，自动继续
                json.dump(parameter_dict, file_handle)
            num_train_file += 1
        else:
            print(f"文件先前已训练: {file}")
        i += 1
    print(f'训练完成: 文件数: {num_train_file-1}, 总得分: {total_score}, 平均分: {total_score/(num_train_file-1)}')

#返回Jc_dict的能量最小值和对应的精确参数
def train_parameter_with_init(Jc_dict, parameter_0=None, Nq=12):
    depth = int(len(parameter_0)/2)
    hamop = build_ham_high(Jc_dict)
    ham = Hamiltonian(hamop)
    gamma_List=[]
    beta_List=[]
    for i in range(depth):
        gamma_List.append(f'g{i}')
        beta_List.append(f'b{i}')
    circ= qaoa_hubo(Jc_dict, Nq, gamma_List, beta_List ,p=depth)
    sim = Simulator('mqvector', circ.n_qubits)                     # 创建模拟器，backend使用‘mqvector’，能模拟5个比特（'circ'线路中包含的比特数）
    grad_ops = sim.get_expectation_with_grad(ham, circ)            # 获取计算变分量子线路的期望值和梯度的算子
    res = minimize(cost_fun, parameter_0, args=(grad_ops, ), method='bfgs', jac=True)
    #res = minimize(cost_fun, parameter_0, args=(grad_ops, ), method='bfgs', jac=True, tol=1)
	#res = minimize(cost_fun_only_f, parameter_0, args=(grad_ops, ), method='COBYLA')
    #res = minimize(cost_fun_only_f, parameter_0, args=(grad_ops, ), method='bfgs')
    return res.fun, res.x

def get_initial_parameter_from_parameter_dict_local(Jc_dict, p, Nq=12):
    gammas = betas = None
    parameter_dict_local = json.load(open('parameter_dict_local.json', 'r'))
    for portion in ['0.3','0.9']:
        for k in [f'{k_int}' for k_int in range(2,5)]:
            for distribution in ['std', 'uni', 'bimodal']:
                for file_i in [f'{file_i_int}' for file_i_int in range(5)]:
                    Jc_dict_original = load_data(f"data/k{k}/{distribution}_p{portion}_{file_i}.json")
                    if Jc_dict == Jc_dict_original:
                        gammas = parameter_dict_local[portion][k][distribution][file_i][f'parameter{p}'][0::2]
                        betas = parameter_dict_local[portion][k][distribution][file_i][f'parameter{p}'][1::2]
                        print(f'精确配型成功，输出参数。(filename: data/k{k}/{distribution}_p{portion}_{file_i}.json, p,k,d: {portion, k, distribution})')
                        return gammas, betas
    return gammas, betas

#更新参数：parameter_dict_local_old.json -> parameter_dict_local_new.json
def train_old_to_new():
    total_score = 0
    distribution_index = {'std':0, 'uni':1, 'bimodal':2}
    parameter_dict = {}
    parameter_dict_local = json.load(open('parameter_dict_local_old.json', 'r'))
    for portion in ['0.3','0.9']:
        for k in [f'{k_int}' for k_int in range(2,5)]:
            for distribution in ['std', 'uni', 'bimodal']:
                for file_i in [f'{file_i_int}' for file_i_int in range(5)]:
                    Jc_dict = load_data(f"data/k{k}/{distribution}_p{portion}_{file_i}.json")
                    num_edge = len(Jc_dict)
                    the_key = f'{(int(k), distribution_index[distribution], float(portion), num_edge, int(file_i))}'
                    parameter_dict[the_key] = {}
                    for p in [4, 8]:
                        parameter_0 = parameter_dict_local[portion][k][distribution][file_i][f'parameter{p}']
                        fun, x = train_parameter_with_init(Jc_dict, parameter_0)
                        parameter_dict[the_key][f"parameter{p}"] = x.tolist()
                        parameter_dict[the_key][f"score{p}"] = -fun
                        total_score += fun
                        print(f'文件: data/k{k}/{distribution}_p{portion}_{file_i}.json, 得分: fun{p}={-fun}')
                        with open('parameter_dict_local_new.json', 'w') as file_handle:  
                            json.dump(parameter_dict, file_handle)
    print(f'总分{total_score}')

#更新参数：parameter_dict_local_new.json -> parameter_dict_local_new.json
def train_new_to_new():
    total_score = 0
    distribution_index = {'std':0, 'uni':1, 'bimodal':2}
    parameter_dict_local = json.load(open('parameter_dict_local_new.json', 'r'))
    for portion in ['0.3','0.9']:
        for k in [f'{k_int}' for k_int in range(2,5)]:
            for distribution in ['std', 'uni', 'bimodal']:
                for file_i in [f'{file_i_int}' for file_i_int in range(5)]:
                    Jc_dict = load_data(f"data/k{k}/{distribution}_p{portion}_{file_i}.json")
                    num_edge = len(Jc_dict)
                    the_key = f'{(int(k), distribution_index[distribution], float(portion), num_edge, int(file_i))}'
                    for p in [4, 8]:
                        parameter_0 = parameter_dict_local[the_key][f'parameter{p}']
                        fun, x = train_parameter_with_init(Jc_dict, parameter_0)
                        total_score += -fun
                        print(f'文件: data/k{k}/{distribution}_p{portion}_{file_i}.json, 得分: fun{p}={-fun}')
                        last_score = parameter_dict_local[the_key][f"score{p}"]
                        if -fun > last_score:
                            print(f'发现新参数, 老得分: {last_score}, 新得分: {-fun}, 差值: {-fun - last_score}')
                            parameter_dict_local[the_key][f"parameter{p}"] = x.tolist()
                            parameter_dict_local[the_key][f"score{p}"] = -fun
                            with open('parameter_dict_local_new.json', 'w') as file_handle:  
                                json.dump(parameter_dict_local, file_handle)
    print(f'总分{total_score}')

#更新参数：parameter_dict_local_new.json -> parameter_dict_local_new.json
def train_new_to_new_special():
    total_score = 0
    distribution_index = {'std':0, 'uni':1, 'bimodal':2}
    parameter_dict_local = json.load(open('parameter_dict_local_new.json', 'r'))
    for portion in ['0.9']:
        for k in [f'{k_int}' for k_int in range(4,5)]:
            for distribution in ['std']:
                for file_i in [f'{file_i_int}' for file_i_int in range(5)]:
                    Jc_dict = load_data(f"data/k{k}/{distribution}_p{portion}_{file_i}.json")
                    num_edge = len(Jc_dict)
                    the_key = f'{(int(k), distribution_index[distribution], float(portion), num_edge, int(file_i))}'
                    for p in [4, 8]:
                        parameter_0 = parameter_dict_local[the_key][f'parameter{p}']
                        fun, x = train_parameter_with_init(Jc_dict, parameter_0)
                        total_score += -fun
                        print(f'文件: data/k{k}/{distribution}_p{portion}_{file_i}.json, 得分: fun{p}={-fun}')
                        last_score = parameter_dict_local[the_key][f"score{p}"]
                        if -fun > last_score:
                            print(f'发现新参数, 老得分: {last_score}, 新得分: {-fun}, 差值: {-fun - last_score}')
                            parameter_dict_local[the_key][f"parameter{p}"] = x.tolist()
                            parameter_dict_local[the_key][f"score{p}"] = -fun
                            with open('parameter_dict_local_new.json', 'w') as file_handle:  
                                json.dump(parameter_dict_local, file_handle)
    print(f'总分{total_score}')

#更新参数：output.json -> parameter_dict_local_new.json
def train_new_to_new_special2():
    total_score = 0
    reverse_distribution_index = {0:'std', 1:'uni', 2:'bimodal'}
    output_list = json.load(open('output.json', 'r'))
    parameter_dict_local = json.load(open('parameter_dict_local_new.json', 'r'))
    for file_in_output in output_list:
        k = file_in_output['file'][0]
        distribution_index = file_in_output['file'][1]
        distribution = reverse_distribution_index[distribution_index]
        portion = file_in_output['file'][2]
        for file_i in range(5):
            Jc_dict = load_data(f"data/k{k}/{distribution}_p{portion}_{file_i}.json")
            num_edge = len(Jc_dict)
            the_key = f'{(int(k), distribution_index, float(portion), num_edge, int(file_i))}'
            p = file_in_output['p']
            gamma_List, beta_List= file_in_output['gamma'], file_in_output['beta']
            parameter_0 = [val for pair in zip(gamma_List, beta_List) for val in pair]
            fun, x = train_parameter_with_init(Jc_dict, parameter_0)
            total_score += -fun
            print(f'文件: data/k{k}/{distribution}_p{portion}_{file_i}.json, 得分: fun{p}={-fun}')
            last_score = parameter_dict_local[the_key][f"score{p}"]
            if -fun > last_score:
                print(f'发现新参数, 老得分: {last_score}, 新得分: {-fun}, 差值: {-fun - last_score}')
                parameter_dict_local[the_key][f"parameter{p}"] = x.tolist()
                parameter_dict_local[the_key][f"score{p}"] = -fun
                with open('parameter_dict_local_new.json', 'w') as file_handle:  
                    json.dump(parameter_dict_local, file_handle)
    print(f'总分{total_score}')

if __name__ == '__main__':
    train_parameter_dir('data/train')