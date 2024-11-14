'''
该模块实现了文章arXiv:1711.11240v1中的quantum neuron单元和下面的quantum hopfield network
'''

import numpy as np
from numpy import pi,cos,sin,sqrt
from mindquantum.core import Circuit,X,H,RY,Y,RZ,Measure
from mindquantum.core.circuit import dagger
from mindquantum.simulator import Simulator  # pylint: disable=unused-import
import matplotlib.pyplot as plt


def load_w():
    '''
    根据文章中给的吸引子加载网络的权重
    '''
    mat = np.array([[0,1,1],[1,0,0],[0,1,1]])
    mat1 = np.array([[1,0,1],[0,1,0],[0,1,0]])
    data = np.array([mat,mat1])
    w = w_matrix(data)
    return w


def mat2vec(mat):
    '''
    Convert matrix to vector
    Args:
        mat: the input matrix,the matrix dimension is (n,m,m) where n is the numbers of dataset
    Returns:
        the vector the shape is (n,m*m)
    '''
    return mat.flatten().tolist()


def w_matrix(data):
    '''
    Input is three dimension
    utilize the hebb rule to construct W matrix
    '''
    n_data = len(data)
    dim = len(data[0]) ** 2
    w = np.zeros((dim,dim))
    for i in range(n_data):
        vec = mat2vec(data[i])
        w += np.outer(vec,vec)
    diag_w = np.diag(np.diag(w))
    w = w - diag_w
    w /= n_data
    return w


def input_rotation(w,gamma,bias,update_qubit:int,num_update:int):
    '''
    Args:
        w:the weight matrix
        gamma: the scaling factor
        update_qubit:which qubit needs to update
        bias: the vector of bias
    Return:
        the circuit of input rotation
    '''
    assert len(w) == len(bias)
    n = len(w)
    circ = Circuit()
    for ctrl_qubit in range(n):
        if w[update_qubit][ctrl_qubit] == 0:
            continue
        circ += RY(4*gamma*w[update_qubit][ctrl_qubit]).on(n+num_update,ctrl_qubit)
    circ += RY(2*bias[update_qubit]).on(n+num_update)
    return circ


def quantum_neuron(w,gamma,bias,update_qubit:int,num_update:int):
    '''
    实现quantum neuron单元接口
    其中w为权重矩阵，gamma为放缩因子，bias为偏置，update_qubit为需要更新的量子比特,num_update为第几次更新
    '''
    n = len(w) # control qubit ,we need total n+1 qubits
    rotationcirc = input_rotation(w,gamma,bias,update_qubit,num_update)
    rotationcirc_dag = dagger(rotationcirc)
    circ = Circuit()  # initial circuit instance
    circ += rotationcirc
    circ += Y.on(update_qubit,n+num_update)
    circ += RZ(-pi/2).on(n+num_update)
    circ += rotationcirc_dag
    circ += Measure().on(n+num_update)
    return circ


def initial_state(vec):
    '''
    convert the vector to the circuit
    '''
    circ = Circuit()
    for i,x in enumerate(vec):
        if x == '+':
            circ += H.on(i)
        elif x == '0':
            continue
        elif x == '1':
            circ += X.on(i)
    return circ


def q(theta,k):
    '''
    实现非线性激活函数
    '''
    return np.arctan(np.tan(0.7*theta)**(2**k))


def ry(theta):
    '''
    Ry量子门的矩阵形式
    '''
    return np.array([[cos(theta/2),-sin(theta/2)],[sin(theta/2),cos(theta/2)]])


def ry_activate(theta,psi):
    '''
    psi is the state which needs to activate
    '''
    act_psi = ry(2*theta) @ psi
    return act_psi


def first_update(vec1,w,gamma,bias,update_qubit=0,num_update=0):
    '''
    第一次更新
    '''
    circ = Circuit()
    circ += initial_state(vec1)
    circ += quantum_neuron(w,gamma,bias,update_qubit,num_update)

    psi = vec1[update_qubit]
    if psi == '+':
        psi = 1/sqrt(2) * np.array([1,1])
    elif psi == '1':
        psi = np.array([0,1])
    elif psi == '0':
        psi = np.array([1,0])

    theta0 = 4*gamma*0.5 + 2*bias[update_qubit]
    theta1 = 4*gamma*1 + 2*bias[update_qubit]

    act_psi = 1/sqrt(2)*ry_activate(q(theta0,1),psi) +1/sqrt(2)* ry_activate(q(theta1,1),psi)
    if abs(act_psi[0]) >= abs(act_psi[1]):
        vec1[update_qubit] = '1'
    else:
        vec1[update_qubit] = '0'
    return vec1


def second_update(vec2,w,gamma,bias,update_qubit=1,num_update=1):
    '''
    第二次更新
    '''
    circ = Circuit()
    circ += initial_state(vec2)
    circ += quantum_neuron(w,gamma,bias,update_qubit,num_update)
    psi = vec2[update_qubit]

    if psi == '+':
        psi = 1/sqrt(2) * np.array([1,1])
    elif psi == '1':
        psi = np.array([0,1])
    elif psi == '0':
        psi = np.array([1,0])
    theta0 = 4*gamma*1.5 + 2*bias[update_qubit]
    theta1 = 4*gamma*2 + 2*bias[update_qubit]
    act_psi = 1/sqrt(2)*ry_activate(q(theta0,1),psi) +1/sqrt(2)* ry_activate(q(theta1,1),psi)
    if abs(act_psi[0]) <= abs(act_psi[1]):
        vec2[update_qubit] = '1'
    else:
        vec2[update_qubit] = '0'
    return vec2


def third_update(vec3,w,gamma,bias,update_qubit=2,num_update=2):
    '''
    第三次更新
    '''
    circ = Circuit()
    circ += initial_state(vec3)
    circ += quantum_neuron(w,gamma,bias,update_qubit,num_update)
    psi = vec3[update_qubit]

    if psi == '+':
        psi = 1/sqrt(2) * np.array([1,1])
    elif psi == '1':
        psi = np.array([0,1])
    elif psi == '0':
        psi = np.array([1,0])
    theta0 = 4*gamma*2.5 + 2*bias[update_qubit]
    act_psi = ry_activate(q(theta0,1),psi)
    if abs(act_psi[0]) <= abs(act_psi[1]):
        vec3[update_qubit] = '1'
    else:
        vec3[update_qubit] = '0'
    return vec3


def vec2mat(vec):
    '''
    convert vector to matrix
    '''
    vec = np.array(vec)
    dim = len(vec)
    data = np.reshape(vec,(int(sqrt(dim)),int(sqrt(dim))))
    return data


def str2int(vec):
    '''
    字符串转int数据类型
    '''
    new_vec = []
    for i in vec:
        if i == '0':
            new_vec.append(0)
        elif i == '+':
            new_vec.append(0.5)
        elif i=='1':
            new_vec.append(1)
    return new_vec

def plot(vec):
    '''
    画某个子图
    '''
    new_vec = str2int(vec)
    mat = vec2mat(new_vec)
    plt.imshow(mat)
    plt.axis('off')
    plt.colorbar()
    # plt.savefig(f'../fig/result/update_{update}.png')
    plt.show()


def total_plot(total_state):
    '''
    画出总的更新图
    '''
    total_state1 = []
    for i in total_state:
        total_state1.append(str2int(i))
    fig, axs = plt.subplots(1,4, figsize=(20,6))  # 1 row, 4 columns of plots
    titles = ['initial input', 'after 1 update', 'after 2 updates', 'after 3 updates']
    cmap = plt.cm.viridis  # pylint: disable=no-member
    norm = plt.Normalize(0, 1)
    cbar = axs[0].imshow(vec2mat(total_state1[0]), cmap=cmap, norm=norm)
    axs[0].set_title(titles[0])
    for i in range(1,len(total_state1)):
        updated_matrix = vec2mat(total_state1[i])
        axs[i].imshow(updated_matrix, cmap=cmap, norm=norm)
        axs[i].set_title(titles[i])

    fig.colorbar(cbar, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    for ax in axs:
        ax.axis('off')
    # plt.tight_layout()
    plt.savefig('../fig/result/total_update1.png')
    plt.show()


def main():
    '''
    把所有的都集成到main函数上
    '''
    inital_state = ['+','+','+','1','0','0','0','1','1']
    w = load_w()
    bias = np.array([np.pi/4] * 9 )
    gamma = 0.7 /(w.max()*len(w) + bias.max())
    total_state = []
    total_state.append(inital_state.copy())
    for i in range(3):
        if i == 0:
            update_state = first_update(inital_state,w,gamma,bias)
            plot(update_state)
            total_state.append(update_state.copy())
        elif i == 1:
            update_state = second_update(update_state,w,gamma,bias)
            plot(update_state)
            total_state.append(update_state.copy())
        elif i == 2:
            update_state = third_update(update_state,w,gamma,bias)
            plot(update_state)
            total_state.append(update_state.copy())
    total_plot(total_state)


if __name__ == '__main__':
    main()
    