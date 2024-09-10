import numpy as np
import json, itertools, sys
import scipy.stats as st
import csv


def main(Jc_dict, p, Nq=12):
    gammas, betas = get_parameter_from_parameter_dict_local(Jc_dict, p)  #精确匹配
    if gammas == betas == None:
        fun_of_get_parameter_from_list = [get_parameter_from_parameter_dict, get_parameter_from_factor3_dict, get_parameter_from_transfer_data]  #分布为基于参数的近似图匹配、基于因子的近似图匹配和公式生成算法
        max_score_dict = {'score':0, 'file':get_source(Jc_dict), 'fun':None, 'p':p, 'gamma':None, 'beta':None}  
        for fun_of_get_parameter_from in fun_of_get_parameter_from_list:
            gamma_List, beta_List= fun_of_get_parameter_from(Jc_dict, p)
            if gamma_List==None or beta_List==None:
                tmp_score = 0
                continue
            else:
                tmp_score = main_single_score(Jc_dict, gamma_List, beta_List, p)
            if tmp_score > max_score_dict['score']:
                max_score_dict['score']  = tmp_score
                max_score_dict['fun'] = fun_of_get_parameter_from.__name__
                max_score_dict['gamma'] = gamma_List
                max_score_dict['beta'] = beta_List
        gammas, betas = eval(max_score_dict['fun'])(Jc_dict, p)
    return gammas, betas

#精确匹配
def get_parameter_from_parameter_dict_local(Jc_dict, p, Nq=12):
    gammas = betas = None
    distribution_index = {'std':0, 'uni':1, 'bimodal':2}
    parameter_dict_local = json.load(open('parameter_dict_local.json', 'r'))

    for portion in ['0.3','0.9']:
        for k in [f'{k_int}' for k_int in range(2,5)]:
            for distribution in ['std', 'uni', 'bimodal']:
                for file_i in [f'{file_i_int}' for file_i_int in range(5)]:
                    Jc_dict_original = load_data(f"data/k{k}/{distribution}_p{portion}_{file_i}.json")
                    if Jc_dict == Jc_dict_original:
                        num_edge = len(Jc_dict)
                        the_key = f'{(int(k), distribution_index[distribution], float(portion), num_edge, int(file_i))}'
                        gammas = parameter_dict_local[the_key][f'parameter{p}'][0::2]
                        betas = parameter_dict_local[the_key][f'parameter{p}'][1::2]
                        return gammas, betas
    return gammas, betas

#基于参数的近似图匹配（解决竞赛题目只需要简化版：M矩阵退化为只用边数，KNN的k设为1）
def get_parameter_from_parameter_dict(Jc_dict, p, Nq=12):
    gammas = betas = None
    source = get_source(Jc_dict) #找到数据的源
    k = source[0]
    if k > 10:  #如果阶数大于某值，则直接返回
        return gammas, betas
    parameter_dict = json.load(open('parameter_dict.json', 'r'))
    #离数据源最近的已计算最优参数的图
    min_distance_source = get_min_distance(source, parameter_dict) 
    min_distance_source_str = f'{min_distance_source}'
    if min_distance_source_str in parameter_dict.keys():
        gammas = parameter_dict[min_distance_source_str][f'parameter{p}'][0::2]
        betas = parameter_dict[min_distance_source_str][f'parameter{p}'][1::2]
    return gammas, betas

#基于因子的近似图匹配（解决竞赛题目只需要简化版：M矩阵退化为只用边数，KNN的k设为2）
def get_parameter_from_factor3_dict(Jc_dict, p, Nq=12):
    gammas = betas = None
    source = get_source(Jc_dict) #找到数据的源
    k = source[0]
    if k > 10:  #如果阶数大于某阶，则直接返回
        return gammas, betas
    factor3_dict = json.load(open('factor3_dict.json', 'r'))
    #离数据源最近的已计算最优参数的近邻，min_distance_source_1在前，min_distance_source_2在后，即KNN的k=2
    min_distance_source_1, min_distance_source_2 = get_min_distance_12(source, factor3_dict) 
    print(min_distance_source_1, source, min_distance_source_2)
    if min_distance_source_1 == None or min_distance_source_2 == None: 
        return gammas, betas
    else:
        num_edge = source[3]
        num_edge_1 = eval(min_distance_source_1)[3]
        num_edge_2 = eval(min_distance_source_2)[3]
        factor3_1 = factor3_dict.pop(f'{min_distance_source_1}')
        factor3_2 = factor3_dict.pop(f'{min_distance_source_2}')
        factor3 = factor3_1 + ((num_edge-num_edge_1)/(num_edge_2-num_edge_1) )*(factor3_2-factor3_1)  #依比例缩放factor3
        k = min(k,10)
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
        gammas = factor*gammas*factor3
    return gammas.tolist(), betas.tolist()   

#公式生成算法
def get_parameter_from_transfer_data(Jc_dict, p, Nq=12):
    D = max(2*len(Jc_dict)/Nq, 1.0001)   #防止D小于等于1，出现np.sqrt(D-1)无法完成的情况
    k = order(Jc_dict)
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
    # rescale the parameters for specific case
    factor = rescale_factor(Jc_dict)
    gammas = 1.56*factor*gammas*np.arctan(1/np.sqrt(D-1)) 
    return gammas.tolist(), betas.tolist()

def get_source(Jc_dict, Nq = 12):
    num_hyperedges = len(Jc_dict)  #边数
    k = max([len(key) for key in Jc_dict.keys()]) #阶数
   
    hyperedges = []
    for i in range(1, k+1):
        for combination in itertools.combinations(range(Nq), i):
            hyperedges.append(tuple(combination))
    max_num_hyperedges = len(hyperedges)  #确定阶数k后，图可能的最大边数
    portion = round(num_hyperedges/max_num_hyperedges, 2) #图边比例数

    wights = list(Jc_dict.values())  #边权值list
    distribution_p = {}  #记录属于某个概率的联合概率密度(经过log处理后的)
    if np.var(wights) == 0:
        distribution_p['std'] = 0  #如果权值方差为0，则必为固定分布
    else:  #利用贝叶斯公式计算各分布的可能性
        distribution_p['uni'] = np.sum(np.log(st.uniform.pdf(wights, loc=-5, scale=10)))   #出现wights这种权值的平均分布的联合概率密度，后同。 
        distribution_p['bimodal'] = np.sum(np.log((st.norm.pdf(wights, loc=10, scale=1)+st.norm.pdf(wights, loc=10, scale=1))/2)) #双峰分布
        #distribution_p['exp'] = np.sum(np.log(st.expon.pdf(wights)))  #指数分布
    distribution = max(distribution_p,key=distribution_p.get)  #后验概率最大的分布确定为分布
    distribution_dict = {'std':0, 'uni':1, 'bimodal':2, 'exp':3}
    distribution_index = distribution_dict[distribution]
    return (k, distribution_index, portion, num_hyperedges, 0) #文件序号暂时均定为0

#找边数离哪个最近，第三个参数condition表示限制条件，例如最小应该不大于20边数，才能用来近似，如果差的边数太多，哪怕是最近的一个已训练参数，也不能拿来近似用
def get_min_distance(source, dict, condition=sys.maxsize):
    min_distance = sys.maxsize
    min_distance_source = None
    for key_source in dict.keys():
        if source[0] == eval(key_source)[0] and source[1] == eval(key_source)[1]:  #阶数和分布都满足的情况下，找相近的边数
        #if source[1] == eval(key_source)[1]:  #阶数满足的情况下，找相近的边数
            distance = abs(source[3] - eval(key_source)[3])
            if distance < min(min_distance, condition):
                min_distance_source = key_source
                min_distance = distance
    return min_distance_source

#离数据源最近的已计算最优参数的近邻范围，min_distance_source_1在前，min_distance_source_2在后，也就是找source夹在哪两个之间
def get_min_distance_12(source, dict):
    min_distance_1 = sys.maxsize
    min_distance_2 = -sys.maxsize
    min_distance_source_1 = min_distance_source_2 = None
    for key_source in dict.keys():
        if source[0] == eval(key_source)[0] and source[1] == eval(key_source)[1]:  #阶数和分布都满足的情况下，找相近的边数
        #if source[1] == eval(key_source)[1]:  #分布满足的情况下，找相近的边数
            distance = source[3] - eval(key_source)[3]
            if 0 < distance < min_distance_1:
                min_distance_source_1 = key_source
                min_distance_1 = distance
            elif min_distance_2 < distance < 0:
                min_distance_source_2 = key_source
                min_distance_2 = distance
    return min_distance_source_1, min_distance_source_2

def load_data(filename):
    data = json.load(open(filename, 'r'))
    Jc_dict = {}
    for item in range(len(data['c'])):
        Jc_dict[tuple(data['J'][item])] = data['c'][item]
    return Jc_dict

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
    for key,val in Jc_dict.items():        
        norm+= val**2/keys_len[len(key)]
    norm = np.sqrt(norm)
    return  1/norm    

def order(Jc):
    return max([len(key)  for key in Jc.keys()])


def main_single_score(Jc_dict, gamma_List, beta_List, p):  
    from mindquantum.core.circuit import Circuit, UN
    from mindquantum.core.gates import H, Rzz, RX,Z,RY,Measure,RZ,DepolarizingChannel
    from mindquantum.core.operators import TimeEvolution,Hamiltonian, QubitOperator
    from mindquantum.simulator import Simulator
    from mindquantum.core.parameterresolver import ParameterResolver as PR

    def build_hb(n, para=None):
        hb = Circuit()  
        for i in range(n):
            if type(para) is str:
                hb += RX(dict([(para,2)])).on(i)        # 对每个节点作用RX门
            else:
                hb += RX(para*2).on(i) 
        return hb

    def build_hc_high(ham, para):
        hc = Circuit()                  # 创建量子线路 
        hc+=TimeEvolution(ham,time=PR(para)*(-1)).circuit
        return hc

    def build_ham_high(Jc_dict):
        ham = QubitOperator()
        for key, value in Jc_dict.items():
            nq = len(key)
            ops= QubitOperator(f'Z{key[0]}')
            for i in range(nq-1):
                ops *= QubitOperator(f'Z{key[i+1]}')  # 生成哈密顿量Hc
            ham+=ops*value
        return ham

    def qaoa_hubo(Jc_dict, nq, gammas,betas,p=1):
        circ=Circuit() 
        circ += UN(H, range(nq))
        hamop = build_ham_high(Jc_dict)
        circ+= build_hc_high(hamop,gammas[0])
        circ+=build_hb(nq, para=betas[0])    
        if p>1:
            for i in range(1,p):
                circ+= build_hc_high(hamop,gammas[i])
                circ+=build_hb(nq, para=betas[i]) 
        return circ

    Nq = 12  
    hamop = build_ham_high(Jc_dict)
    ham=Hamiltonian(hamop)
    circ= qaoa_hubo(Jc_dict, Nq, gamma_List,beta_List, p=p)
    sim=Simulator('mqvector',n_qubits=Nq)
    s = -sim.get_expectation(ham, circ).real   
    return s