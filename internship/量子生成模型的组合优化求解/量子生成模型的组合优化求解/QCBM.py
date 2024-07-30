from mindquantum.core import Circuit
from mindquantum.core import RY,Ryy,H,X,Measure,RZ,Rxx
from mindquantum.simulator import Simulator
from mindquantum.core import Hamiltonian,QubitOperator
import numpy as np
from scipy.optimize import minimize
from dataset_seed import distribution,sample_seed_dataset,num_spins

def real_dist(filename):
    data_seed = sample_seed_dataset(filename)
    dist_dict = distribution(data_seed[350:850])  # 选择500个次优解的种子数据
    dist_list = [0. for _ in range(2**num_spins)]
    for bin2 in dist_dict.keys():
        dist_list[int(bin2,2)] = dist_dict[bin2]
    return np.array(dist_list)

count = 0

def Block(n_qubits):
    '''
    每一块的电路构架
    '''
    global count
    circ = Circuit()
    
    for i in range(n_qubits):
        circ += RY(f'theta_{count}').on(i)
        count += 1
    for i in range(n_qubits-1):
        circ += Ryy(f'theta_{count}').on([i,i+1])
        count += 1
    for i in range(n_qubits-1):
        circ += RY(f'theta_{count}').on(i+1,i)
        count += 1
    return circ 

def Ansatz(n_qubits,p):
    ansatz_circ = Circuit()
    for _ in range(p):
        ansatz_circ += Block(n_qubits)
    return ansatz_circ


def block1(n_qubits):
    global count 
    circ = Circuit()
    for i in range(n_qubits):
        circ += RY(f'theta_{count}').on(i)
        count += 1
    for i in range(n_qubits):
        circ += RZ(f'theta_{count}').on(i)
        count += 1 
    for i in range(n_qubits):
        circ += Rxx(f'theta_{count}').on([(i+1) % n_qubits,i])
        count += 1 
    return circ 

def Ansatz1(n_qubits,p):
    ansatz_circ = Circuit()
    for _ in range(p):
        ansatz_circ += block1(n_qubits)
    return ansatz_circ


def pdf(circ,params,sim):
    sim.reset()
    sim.apply_circuit(circ,params)
    vec = sim.get_qs().reshape(-1, 2**circ.n_qubits)
    P = np.einsum("ab, ab -> b", np.conj(vec), vec).real
    return P 


KL_div = lambda x,target : target* (np.log(target + 1e-18) - np.log(x + 1e-18))  # KL散度


def loss(logits,labels):
    return np.sum(KL_div(logits,labels))

def loss1(params):
    global step 
    step += 1
    sim.reset()
    sim.apply_circuit(circ,params)
    vec = sim.get_qs().reshape(-1, 2**circ.n_qubits)
    P = np.einsum("ab, ab -> b", np.conj(vec), vec).real
    loss2 = loss(P,Q)
    if step % 500 == 0:
        print(f'step:{step},loss:{loss2}')
    return loss2

def train(filename):
    p = 2
    Q = np.array(real_dist(filename))
    circ = Ansatz(num_spins,p)
    sim = Simulator('mqvector',circ.n_qubits)
    params = np.random.randn(len(circ.params_name)) 
    P = pdf(circ,params,sim)
    res = minimize(loss1,params,method='SLSQP')
    params_dict = dict(zip(circ.params_name,res.x))
    for i in range(num_spins):
        circ += Measure(f'q{i}').on(i)
    sim.reset()
    res = sim.sampling(circ,shots=10000,pr=params_dict)
    return res 

if __name__ == "__main__":
    step = 0
    p = 7
    filename = './data/dataset.txt'
    Q = real_dist(filename)
    circ = Ansatz(num_spins,p)
    sim = Simulator('mqvector',circ.n_qubits)
    params = np.random.randn(len(circ.params_name))     
    P = pdf(circ,params,sim)
    res = minimize(loss1,params,method='SLSQP')
    print(res.fun)
    np.save(f'params_p_{p}_ansatz_2.npy',res.x)
    # params_dict = dict(zip(circ.params_name,res.x))
    # for i in range(num_spins):
    #     circ += Measure(f'q{i}').on(i)
    # sim.reset()
    # res = sim.sampling(circ,shots=10000,pr=params_dict)
    # print(res.data)
    
    




