import os
import numpy as np
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, Rzz, RX,Z,RY,Measure,RZ,DepolarizingChannel
from mindquantum.core.operators import TimeEvolution,Hamiltonian, QubitOperator
from mindquantum.core import MeasureResult
from mindquantum.simulator import Simulator

import networkx as nx
from scipy.optimize import minimize
from mindquantum.core.parameterresolver import ParameterResolver as PR
import time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import re
import random
import json
from main import *

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




    