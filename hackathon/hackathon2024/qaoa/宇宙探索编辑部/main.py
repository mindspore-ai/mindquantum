import numpy as np
import pickle
import networkx as nx

import mindspore
from mindspore import nn
from mindspore import Tensor
import mindspore.ops as ops


def comb(n, k):
    if k > n:
        return 0
    k = min(k, n - k)
    if k == 0:
        return 1
    numerator = 1
    denominator = 1
    for i in range(1, k + 1):
        numerator *= n - i + 1
        denominator *= i
    return numerator // denominator



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
    feature=[]
    # Read the parameters of infinite size limit, and the maximum order is 6 surported here.
    k = min(k,6)
    for p_use in [4,8]:
        with open('utils/transfer_data.csv', 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if (row[0]) == str(k):
                    new_row = [item for item in row if item != '']
                    length = len(new_row)
                    if length == 3+2*p_use:
                        gammas = np.array([float(new_row[i]) for i in range(3,3+p_use)] )
                        betas = np.array([float(new_row[i]) for i in range(3+p_use,3+2*p_use)])
        # rescale the parameters for specific case
        gammas = trans_gamma(gammas, D)
        factor = rescale_factor(Jc_dict)
        gammas = gammas*factor
        
        if p==p_use:
            gamma_guess= gammas.copy()
            beta_guess = betas.copy()
        feature.extend(list(gammas))
        feature.extend(list(betas))
        
    J = list(Jc_dict.keys())
    c = np.array(list(Jc_dict.values()))
    
    n = max(max(edge) for edge in J) + 1
    k = max(len(edge) for edge in J)
    
    def calculate_possible_hyperedges(n, k):
        total = 0
        for i in range(1, k+1):
            total += comb(n, i)
        return total
    
    # 计算边密度
    density=ave_D(Jc_dict,n)/calculate_possible_hyperedges(n, k) 
    feature.append(density)
    
    adj_matrix = np.zeros((15, 15))
    for i_edge, edge in enumerate(J):
        for i in range(len(edge)):
            for j in range(len(edge)):
                adj_matrix[edge[i]][edge[j]] += c[i_edge]
                adj_matrix[edge[j]][edge[i]] += c[i_edge]
                
    # 计算每个节点的聚类系数，返回值是字典
    clustering_coefficients = list(nx.clustering(nx.Graph(adj_matrix)).values())
    clustering_coefficients = np.array(clustering_coefficients)
    
    clustering_coefficients_mean = np.mean(clustering_coefficients)
    clustering_coefficients_std = np.std(clustering_coefficients)
    feature.extend([clustering_coefficients_mean, clustering_coefficients_std])
    clustering_coefficients_normalized = (clustering_coefficients - clustering_coefficients_mean) / (clustering_coefficients_std + 1e-6)
    feature.extend(np.mean(clustering_coefficients_normalized ** np.arange(3, 6)[:, np.newaxis], axis=1))
    
    num_vertices = max(max(edge) for edge in J) + 1
    num_edges = len(J)
    edge_density = num_edges / (num_vertices * (num_vertices - 1) / 2)
    feature.append(edge_density)
    
    degrees = np.array([len(edge) for edge in J])
    degrees_mean = np.mean(degrees)
    degrees_std = np.std(degrees)
    feature.extend([degrees_mean, degrees_std])
    
    degrees_normalized = (degrees - degrees_mean) / (degrees_std + 1e-6)
    feature.extend(np.mean(degrees_normalized ** np.arange(3, 6)[:, np.newaxis], axis=1))



    feature=np.array(feature)
    
    # ----------下面读取模型----------------
    # 读取data.pkl文件

    with open('data.pkl', 'rb') as file:
        data = pickle.load(file)

    # 提取数据
    mean_features = data['mean_features']
    std_features = data['std_features']
    mean_labels = data['mean_labels']
    std_labels = data['std_labels']

    feature = (feature - mean_features)/std_features
    
    
    class SimpleNNWithSkipConnections(nn.Cell):
        def __init__(self, input_dim=36):
            super(SimpleNNWithSkipConnections, self).__init__(auto_prefix=True)
            self.input_dim = input_dim
            self.fc1 = nn.Dense(input_dim, 32)
            self.fc2 = nn.Dense(input_dim + 32, 64)
            self.fc3 = nn.Dense(input_dim + 64, 32)
            self.fc4 = nn.Dense(input_dim + 32, 24)
            self.relu = ops.ReLU()

        def construct(self, x):
            x1 = self.relu(self.fc1(x))
            x2 = self.relu(self.fc2(ops.cat((x, x1), -1)))
            x3 = self.relu(self.fc3(ops.cat((x, x2), -1)))
            x4 = self.fc4(ops.cat((x, x3), -1))
            return x4

    model= SimpleNNWithSkipConnections()
    param_dict = mindspore.load_checkpoint("best_model.ckpt")
    mindspore.load_param_into_net(model, param_dict)
    feature = Tensor(feature, mindspore.float32)
    feature = model(feature)

    feature = feature.asnumpy()
    
    feature = feature*std_labels + mean_labels
    

    if p==4:
        #0~3的数据
        gammas = feature[:4]
        #4~7的数据
        betas = feature[4:8]
    elif p==8:
        #8~16的数据
        gammas = feature[8:16]
        #16~23的数据
        betas = feature[16:24]
    else:
        gammas=0
        betas=0

    if k>5:
        gammas=0
        betas=0

    if (density >0.11 and k>=3) or k>5:
        gammas=gamma_guess
        betas=beta_guess
    else:
        gammas+=gamma_guess
        betas+=beta_guess
    return gammas, betas
        
def trans_gamma(gammas, D):
    return gammas*np.arctan(1/(np.sqrt(D-1)-1e-6))

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

def calculate_possible_hyperedges(n, k):
    total = 0
    for i in range(1, k+1):
        total += comb(n, i)
    return total

def ave_D(Jc,nq):
    ave=0
    weight=np.array(list(Jc.values()))
    weight=np.abs(weight)
    max_85_weight = np.percentile(weight,85)

    for key in Jc.keys():
        ave+=(np.abs(Jc[key])/max_85_weight)
    ave=ave/nq*2
    return ave

def order(Jc):
    return max([len(key)  for key in Jc.keys()])
        