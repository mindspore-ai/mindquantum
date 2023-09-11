import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from qupack.qaoa import QAOA
from mindquantum.core.operators import QubitOperator, count_qubits
from scipy.optimize import minimize
'''
本文件禁止改动!!!
'''

def read_graph(filename):
    '''
    返回图g，以及QUBO问题对应的矩阵
    '''
    g=nx.Graph()
    with open(filename,"r") as file:
        print('Nodes Edges:',file.readline())
        for line in file:
            e1,e2=[int(x) for x in line.split()]
            g.add_edge(e1,e2)
    edges=[]
    for i in g.edges:
        edges.append((int(i[0]),int(i[1]),int(1))) 
    edge_list = np.array(edges)
    n=len(g.nodes)
    G = csr_matrix((-1 * edge_list[:, 2], (edge_list[:, 0], edge_list[:, 1])), shape=(n, n)) 
    G = (G + G.T)/2 
    return g,G

def calc_subqubo(sub_index, x, J, h=None,C=0.0 ):
    x = np.sign(x-0.5)
    fix_index=[i for i in range(len(x)) if i not in sub_index ]
    J_sub=(J[:,sub_index])[sub_index]
    
    temp=0
    if h is not None:
        h_sub=h[sub_index]
    else:
        h_sub=np.array([0]*len(sub_index))
    C_sub=float(C)
    # h'
    for i in range(len(h_sub)):
        Jarr=J.toarray()[sub_index[i],fix_index]
        h_sub[i]=h_sub[i]+2*Jarr.dot(x[fix_index])  
    # C'
    if h is not None:
        C_sub+=(h[fix_index]).dot(x[fix_index])
    Jarr=(J[:,fix_index])[fix_index]
    C_sub=C_sub+(Jarr.dot(x[fix_index])).dot(x[fix_index]) 
    return J_sub,h_sub,C_sub

class CallingCounter(object):
    def __init__ (self, func):
        self.func = func
        self.count = 0

    def __call__ (self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

def check_degenerate(J_dict, h):
    nodes=()
    for x in list(J_dict.keys()):
        nodes = nodes+x
    for i in range(len(h)):
        if h[i] == 0 and i not in nodes:
            h[i] += ((i%2)-0.5)*2
    
def build_ham(J_dict,h):
    n_qubits = len(h)
    ham = QubitOperator()
    for node,Jij in J_dict.items():
        if node[0]<node[1]:
            ham += QubitOperator('Z{} Z{}'.format(*node), -2*Jij)
    for i in range(n_qubits):
        ham+= QubitOperator('Z{}'.format(i), -h[i])
    return ham

@CallingCounter
def QAOAsolver(J, h, C, depth = 2, tol=1e-4,info=True):
    n_qubits = len(h)
    J_dict=dict(J.todok())
    ham=build_ham(J_dict,h)
    if count_qubits(ham)<n_qubits: # if some spin are degenerate, randomly specify one term for h
        check_degenerate(J_dict, h) 
        ham=build_ham(J_dict,h) 
    sim = QAOA(n_qubits, depth, ham)
    # 优化门参数gamma和beta使目标哈密顿量的期望值最小化
    init_p = np.random.uniform(size=depth*2)
    def func(x):
        expectation, gamma_grad, beta_grad = sim.get_expectation_with_grad(x[:depth], x[depth:])
        return expectation, gamma_grad + beta_grad
    global Nfeval
    Nfeval = 1
    def callbackF(Xi):
        global Nfeval
        print('{0:4d} Parameters: {1: 3.6f}   {2: 3.6f}  {3: 3.6f}   {4: 3.6f} Expectation: {5: 3.6f}'.format(Nfeval, Xi[0], Xi[1],Xi[2], Xi[3], C-func(Xi)[0]))
        Nfeval += 1
    if info==False:
        callbackF = None  
    res = minimize(func, init_p, method='bfgs', tol=tol,jac=True, callback=callbackF)

    expectation2 = sim.get_expectation(res.x[:depth], res.x[depth:])
    #打印最终得到的max—cut值
    sim.evolution(res.x[:depth], res.x[depth:])
    sol_vec=sim.get_qs()
    ind = np.argmax(np.power(np.abs(sol_vec),2))
    #cost_all=-(sim.get_expectation(ham)).real
    str_sol=str(bin(ind).replace('0b',''))
    str_sol=str_sol.rjust(n_qubits,'0')   
    sol=[1-int(x) for x in str_sol][::-1]
    return np.array(sol)


def solve_QAOA(J_sub,h_sub,C_sub,sub_index,x,depth=2,tol=1e-4):
    '''
    利用QupackQAOA求解子问题
    可调节QAOA求解器depth深度以及tol
    '''
    subqubo=calc_qubo_x(J_sub,x[sub_index],h=h_sub,C=C_sub)
    xarr=np.array(x[sub_index]-0.5)
    xarr=xarr.reshape((-1,1))
    sol2= QAOAsolver(J_sub, h_sub,C_sub,depth=depth, tol=tol, info=False)
    ind=np.argmax(subqubo)
    x[sub_index]=sol2[:]
    return x



def calc_qubo_x(J,x,h=None,C=0):
    #对应x为+1->|0>态，-1->|1>态
    x = np.sign(x-0.5)
    Jterm =  np.sum(J.dot(x) * x, axis=0)
    hterm=0
    if h is not None:
        hterm += h.dot(x)
    qubo_term=Jterm+hterm+C
    #print('J:',Jterm,'h:',hterm,'C:', C)
    return qubo_term


def calc_cut_x(G, x):
    x = np.sign(x-0.5)
    energy = np.sum(G.dot(x) * x, axis=0)
    ret = (energy/2 - 0.5 * G.sum())
    return ret


def init_solution(n): 
    return np.random.randint(2,size=n)


    