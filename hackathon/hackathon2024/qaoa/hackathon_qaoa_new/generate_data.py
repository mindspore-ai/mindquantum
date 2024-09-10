import itertools
import random
import numpy as np
import json 
import os, shutil

Nq=12

#创建目录的函数
def mkdir(path):
    # os.path.exists 函数判断文件夹是否存在
    folder = os.path.exists(path)
    # 判断是否存在文件夹如果不存在则创建为文件夹
    if not folder:
        # os.makedirs 传入一个path路径，生成一个递归的文件夹；如果文件夹存在，就会报错,因此创建文件夹之前，需要使用os.path.exists(path)函数判断文件夹是否存在；
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

#删除目录
def deltree(dir):
    try:
        shutil.rmtree(dir)
    except:
        pass

#重置目录，就是把目录清空
def reset_dir(dir):
    deltree(dir)
    mkdir(dir)

def generate_hyperedges(n, k, portion=0.2):
    '''
    Generate hypergraphs with n nodes and k order, with the hyperedges chosen from all possible hyperedges. 
    Args:
        n (int): the number of nodes.
        k (int): the largest number including in the hyperedges.
        portion (float): the portion of the chosen hyperedges in all of them
    Returns:
        random_hyperedges (List[Tuple[int]]): randomly chosen hyperedges, e.g. [(0,1,2),(2,3),...]
    '''
    hyperedges = []
    for i in range(1, k+1):
        for combination in itertools.combinations(range(n), i):
            hyperedges.append(tuple(combination))

    num_hyperedges = len(hyperedges)
    num_to_select = int(num_hyperedges * portion)
    random_hyperedges = random.sample(hyperedges, num_to_select)
    return random_hyperedges

def set_coef(hyperedges, coef='std'):
    '''
    Set the coefficient for each hyperedge. The coefficents are chosen from 4 distributions: std/uniform/exponential/bimodal. 
        - 'std': the std distribution means the coefficients are all +5
        - 'uni': the uniform distribution means the coefficents chosen uniformly from [-5, +5]
        - 'exp': the exponential distribution means the coefficients takes from p(J)~exp(-0.2*J) with J>0
        - 'bimodal': the bimodal distribution are superposition of two normal distribution N(mu=1,sigma=1) and N(mu=10,sigma=1)
    Args:
        hyperedges (List[Tuple[int]]): a list of the hyperedges.
        mode (string): the distribution name.
    Returns:
        model (dict): the ising model expressed as a dict. For example, H = 1*Z0Z1Z2-0.5Z1 corresponds to a dict like: J_dict={"J": [[0, 1, 2], [1]], "c": [1, -0.5]}
    '''
    model={"J":[],"c":[]}
    for edge in hyperedges:
        model["J"].append(list(edge))
        if coef=='std':
            model["c"].append(5)
        elif coef=='uni':
            model["c"].append(random.uniform(-5, 5))
        elif coef=='exp':
            model["c"].append(random.expovariate(1))
        elif coef=='bimodal':
            s1 = np.random.normal(1, 1)
            s2 = np.random.normal(10, 1)
            model["c"].append(np.random.choice([s1, s2]))
    return model
            
def generate_data(Nq=Nq,kmax=5,p=0.2):
    random.seed(2024)
    np.random.seed(2024)
    for k in range(2,kmax+1):
        for r in range(10):
            hyperedges = generate_hyperedges(Nq, k, portion=p)
            for coef in ['std', 'uni','bimodal']:
                model = set_coef(hyperedges, coef=coef)
                model_str = json.dumps(model)
                with open(f"k{k}/{coef}_p{p}_{r}.json", "w") as f:
                    f.write(model_str)

def generate_data_test(seed=2027, dir='data/_hidden', Nq=Nq,kmin=2, kmax=4,p=0.2,size=5):
    random.seed(seed)
    np.random.seed(seed)  #修改了种子数，不然生成的和本地数据一样了
    i = 0
    distrubutions = ['std', 'uni','bimodal']
    for k in range(kmin,kmax+1):
        for r in range(size):
            hyperedges = generate_hyperedges(Nq, k, portion=p)
            for coef in distrubutions:
                model = set_coef(hyperedges, coef=coef)
                model_str = json.dumps(model)
                with open(f"{dir}/k{k}_{coef}_p{p}_l{len(hyperedges)}_{r}.json", "w") as f:
                    f.write(model_str)
                i += 1
    print(f'依据设置Nq={Nq}, kmin={kmin}, kmax={kmax}, p={p}, size={size}, d={distrubutions}，在{dir}生成{i}个文件')

if __name__ == '__main__':
    #生成训练集'data/train'
    for portion in [p/100 for p in range(10, 100, 20)]:
        generate_data_test(dir='data/train', kmin=7, kmax=10, p=portion, size=1)
