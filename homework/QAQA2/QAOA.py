from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, ZZ, RX,Measure
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator
import networkx as nx
import mindspore.nn as nn
import mindspore as ms
import matplotlib.pyplot as mp
from matplotlib import pyplot as plt
import json
from utilities import *

def qaoa(G:Graph,Shots:int=1000,p:int=1,const=0):
    '''
    standard qaoa for max cut
    --------------------------
    G : Graph 

    shots : number of circuit shots

    n_layers : number of QAOA layers

    const : constant in max cut objective function

    Return cut value and solution
    '''

    def build_hc(g, para):
        hc = Circuit()                  # 创建量子线路
        for i in g.e:
            hc += ZZ(para).on(i[:2])        # 对图中的每条边作用ZZ门
        hc.barrier()                    # 添加Barrier以方便展示线路
        return hc

    def build_hb(g, para):
        hb = Circuit()                  # 创建量子线路
        for i in range(len(g.v)):
            hb += RX(para).on(i)        # 对每个节点作用RX门
        hb.barrier()                    # 添加Barrier以方便展示线路
        return hb

    def build_ansatz(g, p):                    # g是max-cut问题的图，p是ansatz线路的层数
        circ = Circuit()                       # 创建量子线路
        for i in range(p):
            circ += build_hc(g, f'g{i}')       # 添加Uc对应的线路，参数记为g0、g1、g2...
            circ += build_hb(g, f'b{i}')       # 添加Ub对应的线路，参数记为b0、b1、b2...
        return circ

    def build_ham(g):
        ham = QubitOperator()
        for i in g.e:
            ham += QubitOperator(f'Z{i[0]} Z{i[1]}')  # 生成哈密顿量Hc
        return ham


    n_wires = G.n_v
    edges = G.e
    replace_dict = {G.v[i]:i for i in range(n_wires)}
    print(replace_dict)
    print(edges)
    #将子图中的边重新定义为0到n
    sim = Simulator('projectq', n_wires)  

    ham = Hamiltonian(build_ham(G))              # 生成哈密顿量
    init_state_circ = UN(H, n_wires)             # 生成均匀叠加态，即对所有量子比特作用H门
    ansatz = build_ansatz(G, p)                  # 生成ansatz线路
    circ = init_state_circ + ansatz              # 将初始化线路与ansatz线路组合成一个线路
   
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    grad_ops = sim.get_expectation_with_grad(ham, circ)            # 获取计算变分量子线路的期望值和梯度的算子
    net = MQAnsatzOnlyLayer(grad_ops)                              # 生成待训练的神经网络
    opti = nn.Adam(net.trainable_params(), learning_rate=0.01)     # 设置针对网络中所有可训练参数、学习率为0.05的Adam优化器
    train_net = nn.TrainOneStepCell(net, opti) 
    print(G.e)
    for i in range(100):
        cut = (len(G.e) - train_net()) / 2      # 将神经网络训练一步并计算得到的结果（切割边数）。注意：每当'train_net()'运行一次，神经网络就训练了一步
        if i%10 == 0:
            print("train step:", i, ", cut:", cut)  # 每训练10步，打印当前训练步数和当前得到的切割边数
    pr = dict(zip(ansatz.params_name, net.weight.asnumpy())) # 获取线路参数
    print(pr)
    print(net.weight.asnumpy())
    params = []
    for i in range(2):
        params.append([])
        for j in range(p):
             params[i].append(net.weight.asnumpy()[j*2+i])
    print(params)

    def circuit2(g, p, params,shots):                    # g是max-cut问题的图，p是ansatz线路的层数
        circ2 = Circuit() 
        n_wires = g.n_v
        circ2 += UN(H, n_wires)                   # 创建量子线路
        for i in range(p):
            circ2 += build_hc(g, params[0][i])       # 添加Uc对应的线路，参数记为g0、g1、g2...
            circ2 += build_hb(g, params[1][i])       # 添加Ub对应的线路，参数记为b0、b1、b2...
        for j in range(n_wires):
            circ2 += Measure('').on(j)
        print(circ2)
        sim.reset()
        result = sim.sampling(circ2, shots=1000)
        return result

    samples = circuit2(G, p=1, params=params,shots=Shots)
    # print(type(counts))
    counts = samples.bit_string_data
    sol = None
    obj = 10e6
    for bitstring in counts.keys():
        #print(bitstring)
        obj_temp = 0
        for edge in edges:
            obj_temp += 0.5 * edge[2] * (2*( bitstring[ edge[0] ]==bitstring[ edge[1] ] )-1)
        if obj_temp < obj:
            obj = obj_temp
            sol = bitstring
    return const - obj, sol




 