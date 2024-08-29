# Generate the instances of Ising models. 
# The models are expressed as a dict. For example, H = 1*Z0Z1Z2-0.5Z1 corresponds to a dict like: 
#   J_dict={"J": [[0, 1, 2], [1]], "c": [1, -0.5]}
# Models are chosen follow three conditions: 
#  1. 2nd/3rd/4th/5th order; 
#  2. hyperedges chosen with probability p=0.3/0.6/0.9 of all possible hyperedges; 
#  3. the coefficient chosen from std/uniform/bimodal distribution.
# There are 4*3*3=36 types. And we will test 10 random cases of each type.

# Notice: There is a hidden model not included in these types when running on the platform.
import itertools
import random
import numpy as np
import networkx as nx
import json 

Nq=12

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


if __name__ == '__main__':
    generate_data(p=0.3)
    generate_data(p=0.6)
    generate_data(p=0.9)