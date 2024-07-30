from pyqubo import Array
from neal import SimulatedAnnealingSampler
import itertools
import random
import numpy as np 

num_spins = 15

def generate_hyperedges(n, k=2, portion=0.2):
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

def set_coef(hyperedges):
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
        s1 = np.random.normal(0, 1)
        model["c"].append(np.random.choice([s1]))
    return model


def Calculate_energy(J_dict):
    '''
    J_dict 是一个字典，里面存储了边的信息和权重信息 例如
    H = 1*Z0Z2-0.5Z1 corresponds to a dict like: J_dict={"J": [[0, 2], [1]], "c": [1, -0.5]}
    '''
    s = Array.create('s',shape=num_spins,vartype='SPIN')
    H = 0.0
    for index,J in enumerate(J_dict['J']):
        H += s[J[0]] * s[J[1]] * J_dict['c'][index] if len(J) == 2 else s[J[0]] * J_dict['c'][index]
    model = H.compile()
    qubo, offset = model.to_qubo()

    # 使用模拟退火求解 QUBO 问题
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(qubo)

    # 获取最优解
    solution = sampleset.first.sample
    return solution,(sampleset.first.energy + offset)
    
def generate_data(count,filename):
    hyperedges = generate_hyperedges(num_spins,k=2,portion=0.5)
    J_dict = set_coef(hyperedges)
    # J_dict = {'J': [[5, 12], [14], [3], [7, 10], [1, 8], [1, 4], [0, 14], [0, 3], [11, 14], [13], [6, 9], [10, 12], [4, 9], [11], [5, 6], [3, 4], [4], [12, 13], [9, 11], [0, 13], [1, 2], [3, 14], [5, 8], [8, 12], [4, 11], [0, 11], [6, 14], [5, 14], [6, 12], [9, 12], [2, 14], [10, 14], [3, 7], [9, 10], [11, 13], [0], [0, 6], [8, 14], [2, 4], [6, 8], [0, 5], [8, 10], [13, 14], [10, 11], [8, 11], [2, 9], [12], [2, 6], [2, 5], [1, 6], [5], [3, 8], [0, 1], [4, 14], [10], [1, 10], [2, 7], [1, 9], [4, 13], [4, 12], [8, 13], [2], [2, 3], [12, 14], [2, 10], [0, 4], [6, 13], [3, 6], [6], [0, 10], [10, 13], [8, 9], [1, 13], [0, 9], [4, 5], [3, 13], [0, 8], [5, 7], [3, 5], [3, 9], [1, 11], [3, 10], [1, 7], [4, 7], [1, 12], [4, 6], [2, 11], [1, 14], [2, 13], [0, 7], [5, 11], [0, 12], [4, 10], [5, 9], [1], [7]], 'c': [-1.3604882295494118, 0.5726016647847184, 1.8484070569718185, -1.718275779901665, 1.0191354347116568, 1.5271976970424135, 1.3902935227267745, 0.12930315567621228, 2.666562102587284, 0.45937862158976567, -0.9865310175953785, -0.05968487873834175, -1.8586490274259382, -2.0876302092040593, 1.7504548842989394, -1.1887309488399045, 0.09829174794191226, -0.8797824914094483, 0.08068773513257421, 0.4140377292591147, 1.6565604353737593, 0.606298886417282, -0.8353618638970972, 1.1490226117142759, 0.7213553222676544, 0.42263310118566216, 0.2544866835126557, -0.5607637103429287, -0.4240364324530401, -0.7045194200573551, 0.5849722046285643, 0.28161874972900125, -1.0354863812253021, -0.8636535544006663, 1.104357342137283, -0.7816488882671289, 0.020405110448233316, 1.9705724186050944, 0.1595899346316177, 2.034565892171489, 0.5896179529064257, -1.3963820914257274, 1.7968997721885291, -0.6022876055888408, 0.32209175833577797, -0.5190521656139688, -0.6055712558208514, 1.4495644163151222, -2.027398616064719, -1.4041270249187974, -0.7980821166929327, -2.0884922791075806, -0.36168897974103403, -0.11728441387421916, -0.6518601652525559, 1.194961262062422, 0.40433057284914997, -0.2313897611237638, -0.07714050541348816, 2.4863877466322686, 0.13540454267513086, 0.8197649658611066, 0.03401916216517339, -0.09962652303139337, -0.3407252802663147, 1.3275187482889677, -0.38212395233390506, 1.7806777292087688, 0.5334855572094418, 0.1269078200946184, -1.2219489188157164, 1.1332205842174519, 0.2676543384509089, 0.4771974066953199, 0.6173114032443024, -0.6523465125771509, 0.6057810388594363, 1.1019188431359606, 0.5603016459242859, -0.09291458001096363, -0.6594269849185179, -0.3451022491257856, -0.2829672559386366, 1.3859834332706127, 0.008641764261595864, 1.8753774479910277, -0.16749707870814673, 0.0005924433260822996, 0.126559050661805, -0.3740150394717516, 0.7334944019434387, 0.8205080195921117, 0.35185709047071345, 0.06620595422064103, 0.27342836257550573, -1.696668135173791]}
    with open(filename,'w') as f:
        for _ in range(count):
            sol,energy = Calculate_energy(J_dict)
            # print(energy)
            f.write(f'{J_dict}\t{sol}\t{energy}\n')

def load_data(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
        return [line.strip('\n').split('\t') for line in lines]

def strtobin(s):
    '''
    s = "{'s[0]': 0, 's[1]': 0, 's[2]': 0, 's[3]': 0, 's[4]': 0, 's[5]': 0, 's[6]': 1, 's[7]': 0, 's[8]': 1}"
    '''
    st1 = eval(s)
    bin2 = [str(i) for i in st1.values()][::-1]
    return ''.join(bin2)

def sample_seed_dataset(filename):
    # 加载数据集
    dataset = load_data(filename)
    # 对能量进行排序，能量按从小到大排序
    dataset = sorted(dataset,key=lambda x:float(x[2]))
    data_seed = []
    for data in dataset:
        configuration = strtobin(data[1])
        energy = float(data[2])
        if len(configuration) == num_spins:
            data_seed.append((configuration,energy))

    return data_seed

def origin_dist(data_seed):
    dist = {} 
    L = len(data_seed)
    for configuation,energy in data_seed:
        if energy in dist.keys():
            dist[energy] += 1 / L
        else:
            dist[energy] = 1 / L

    return dist

def dataseed_std(data_seed):
    return np.std([x[1] for x in data_seed])

def distribution(data_seed):
    # 用字典进行存储
    dist = dict()
    # “温度”
    T = dataseed_std(data_seed)
    # 第一步进行exp指数求和
    exp_sum = sum([np.exp(-energy/T) for configuation,energy in data_seed])
    for data in data_seed:
        p_i = np.exp(-data[1] / T) / exp_sum
        if data[0] in dist:
            dist[data[0]] += p_i
        else:
            dist[data[0]] = p_i
    for key in list(dist.keys()):
        if dist[key] < 1e-3:
            dist.pop(key)
    return dist

def bin_to_config(x):
    '''
    x = '000110001000001'
    '''
    x = x[::-1]
    x_dict = {} 
    str_list = [str(i) for i in range(num_spins)]
    str_list.sort()
    str_list.remove('1')
    str_list.insert(num_spins % 10 + 1,'1')
    for index,val in enumerate(x) :
        x_dict['s'+str([int(str_list[index])])] = int(val)
    return x_dict 


def ConfigToEnergy(J_dict,x):
    '''
    s = "{'s[0]': 0, 's[1]': 0, 's[2]': 0, 's[3]': 0, 's[4]': 0, 's[5]': 0, 's[6]': 1, 's[7]': 0, 's[8]': 1}"
    '''
    s = Array.create('s',shape=num_spins,vartype='SPIN')
    H = 0.0
    for index,J in enumerate(J_dict['J']):
        H += s[J[0]] * s[J[1]] * J_dict['c'][index] if len(J) == 2 else s[J[0]] * J_dict['c'][index]
    model = H.compile()

    decoded_sample = model.decode_sample(x, vartype='SPIN')
    energy = decoded_sample.energy
    return energy
    

if __name__ == "__main__":
    # generate_data(1000,filename='./data/dataset.txt')
    filename = './data/dataset.txt'
    data_seed = sample_seed_dataset(filename)
    print(origin_dist(data_seed[350:850]))
    #print(bin_to_config('000110001001000'))

    dist = distribution(data_seed[350:850])

    print(dist)
    

    



