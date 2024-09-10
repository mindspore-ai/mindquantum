# We will generate the parameters from `parameters.py` of depth 4/8 to test the performance. So make sure your code supports generating the parameters for depth at least 8. 

# Notice: Do not change this file!!!
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator
import json 
from utils.qcirc import qaoa_hubo, build_ham_high
from main import main
import time
import os 


def load_data(filename):
    '''
    Load the data for scoring.
    Args:
        filename (str): the name of ising model.
    Returns:
        Jc_dict (dict): new form of ising model for simplicity like {(0,): 1, (1, 2, 3): -1.}
    '''
    data = json.load(open(filename, 'r'))
    Jc_dict = {}
    for item in range(len(data['c'])):
        Jc_dict[tuple(data['J'][item])] = data['c'][item]
    return Jc_dict
    
def single_score(Jc_dict):    
    hamop = build_ham_high(Jc_dict)
    ham=Hamiltonian(hamop)
    s=0
    for depth in [4,8]:
        gamma_List, beta_List= main(Jc_dict, depth, Nq=Nq)
        circ= qaoa_hubo(Jc_dict, Nq, gamma_List,beta_List ,p=depth)
        sim=Simulator('mqvector',n_qubits=Nq)
        E = sim.get_expectation(ham, circ).real   
        s += -E
    return s
    
def score():
    score=0
    start=time.time()
    files = os.listdir('data/_hidden')
    for file in files:
        Jc_dict = load_data('data/_hidden/'+file)
        score+=single_score(Jc_dict)
    end=time.time()
    for propotion in [0.3,0.9]:
        for k in range(2,5):
            for coef in ['std', 'uni', 'bimodal']:
                for r in range(5):
                    Jc_dict = load_data(f"data/k{k}/{coef}_p{propotion}_{r}.json")
                    score+=single_score(Jc_dict)
                end=time.time()
    print(f'score: {score}')  
    return score

    
            
            
if __name__ == '__main__':
    Nq=12
    score()