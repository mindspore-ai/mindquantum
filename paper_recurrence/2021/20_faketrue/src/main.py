# -*- coding: utf-8 -*-
from mindquantum.core import Circuit, Hamiltonian, X, UN, H, RX, RY, RZ, QubitOperator, PhaseShift
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator
import mindspore.nn as nn
import numpy as np
import mindspore as ms
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 初始化一个3*3的零矩阵
k = np.zeros((3, 3), dtype=int)
# 在关联矩阵中添加覆盖
k[0][0] = 1
k[0][2] = 1
k[1][1] = 1
k[1][2] = 1
k[2][1] = 1
k[2][2] = 1
print(k)


#根据关联矩阵k，计算系数Jij
def get_Jij(i, j, k):
    Jij = 0
    for l in range(k.shape[0]):
        Jij += k[l][i] * k[l][j] * 0.5
    return Jij


#计算第l行的klj之和
def get_klj(l, k):
    klj = 0
    for j in range(k.shape[1]):
        klj += k[l][j]
    return klj


#根据关联矩阵k，计算系数hi
def get_hi(i, k):
    hi = 0
    for l in range(k.shape[0]):
        hi += k[l][i] * (-1 + 0.5 * get_klj(l, k))
    return hi


# 搭建UC(gamma)对应的量子线路
def build_UC(k, g, para):
    # 创建量子线路
    UC = Circuit()
    for j in range(k.shape[1]):
        for i in range(j):
            Jij = get_Jij(i, j, k)
            # UCij = Phase(i, 2 * gamma * Jij) + Phase(j, 2 * gamma * Jij) + Phase(i, j, 4 * gamma * Jij)
            # i < j
            if (g == ''):
                UC += PhaseShift({para: 2 * Jij}).on(i)
                UC += PhaseShift({para: 2 * Jij}).on(j)
                UC += PhaseShift({para: -4 * Jij}).on(j, i)
            else:
                UC += PhaseShift(2 * Jij * g).on(i)
                UC += PhaseShift(2 * Jij * g).on(j)
                UC += PhaseShift(-4 * Jij * g).on(j, i)
    # 添加Barrier以方便展示线路
    UC.barrier()
    return UC


# 搭建UB(β)对应的量子线路：
def build_UB(k, b, para):
    # 创建量子线路
    UB = Circuit()
    for i in range(k.shape[1]):
        if (b == ''):
            # 对每个节点作用RX门
            UB += RX({para: 2}).on(i)
        else:
            UB += RX(2 * b).on(i)
    # 添加Barrier以方便展示线路
    UB.barrier()
    return UB


# 构建哈密顿量HC
def build_HC(k):
    # 生成哈密顿量HC
    HC = QubitOperator()
    # HC = Sigma(Jij * Zi * Zj) + Sigma(hi * Zi)
    for j in range(k.shape[1]):
        for i in range(j):
            HC += get_Jij(i, j, k) * QubitOperator(f'Z{i} Z{j}')
    for i in range(k.shape[1]):
        HC += get_hi(i, k) * QubitOperator(f'Z{i}')
    return HC


# 搭建多层的训练网络
# p是ansatz线路的层数
def build_ansatz(k, p, g, b):
    # 创建量子线路
    circ = Circuit()
    for i in range(p):
        # 添加UC对应的线路，参数记为g0、g1、g2...
        circ += build_UC(k, g, f'g{i}')
        # 添加UB对应的线路，参数记为b0、b1、b2...
        circ += build_UB(k, b, f'b{i}')

    return circ


# pylint: disable=W0104
# QAOA量子线路的层数p
p = 1
# 生成初态（均匀叠加态），即对所有量子比特作用H门
init_state_circ = UN(H, k.shape[1])
# 生成哈密顿量
ham = Hamiltonian(build_HC(k))
# 构建ansatz线路
ansatz = build_ansatz(k, p, '', '')
# 组合成完整线路
circ = init_state_circ + ansatz
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
# 生成一个基于mqvector后端的模拟器，并设置模拟器的比特数为量子线路的比特数
sim = Simulator('mqvector', circ.n_qubits)
# 获取模拟器基于当前量子态的量子线路演化以及期望、梯度求解算子
grad_ops = sim.get_expectation_with_grad(ham, circ)

plt.figure(figsize=(15, 10))

# 第一张图
step = 60
beta_max = np.pi / 2
beta_step = beta_max / step
gamma_max = np.pi
gamma_step = gamma_max / step
beta = np.arange(0, beta_max, beta_step)
gamma = np.arange(0, gamma_max, gamma_step)
costs = np.zeros((beta.size, gamma.size), dtype=np.float64)

for i in range(beta.size):
    for j in range(gamma.size):
        bg = np.array([gamma[j], beta[i]])
        cost, grad = grad_ops(bg)
        costs[i][j] = cost[0][0].real

plt.subplot(231)
plt.title("variational", fontsize=20)
plt.xlim((0, np.pi / 2))
plt.ylim((0, np.pi))
sns.heatmap(data=costs, vmin=-1, vmax=1, xticklabels=15,
            yticklabels=15).invert_yaxis()

# 第五张图
plt.subplot(235)
plt.title("fe_gamma", fontsize=20)
plt.ylim((0, 100))
plt.xlim((0, np.pi))
plt.xticks([0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
           ['0', 'pi/4', 'pi/2', 'pi*3/4'],
           fontsize=15)

data = np.zeros((200, 10, 4), dtype=np.float64)
for j in range(10):
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    net = MQAnsatzOnlyLayer(grad_ops)
    opti = nn.Adam(net.trainable_params(), learning_rate=0.5)
    train_net = nn.TrainOneStepCell(net, opti)
    for i in range(200):
        loss = train_net().asnumpy()
        data[i][j][0] = i
        data[i][j][1] = net.weight.asnumpy()[0]
        data[i][j][2] = net.weight.asnumpy()[1]
        data[i][j][3] = loss

for i in range(10):
    fe = data[:, i]
    fe = pd.DataFrame(fe, columns=['fe', 'gamma', 'beta', 'cost'])
    plt.plot(fe['gamma'], fe['fe'])

# 第四张图
plt.subplot(234)
plt.title("gamma_cost, pi/8(blue), pi*3/8(orange)", fontsize=20)
plt.xlabel('gamma')
plt.ylabel('Cost')
plt.ylim((-1.2, 1.2))
plt.xticks([0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
           ['0', 'pi/4', 'pi/2', 'pi*3/4'],
           fontsize=15)

costs = np.zeros((gamma.size, 2), dtype=np.float64)
for i in range(gamma.size):
    ansatz = build_ansatz(k, p, '', np.pi / 8)
    circ = init_state_circ + ansatz
    sim = Simulator('mqvector', circ.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    g = np.array([gamma[i]])
    cost, grad = grad_ops(g)
    costs[i][0] = gamma[i]
    costs[i][1] = cost[0][0].real

costs = pd.DataFrame(costs, columns=['gamma', 'cost'])
plt.plot(costs['gamma'], costs['cost'])

costs = np.zeros((gamma.size, 2), dtype=np.float64)
for i in range(gamma.size):
    ansatz = build_ansatz(k, p, '', np.pi * 3 / 8)
    circ = init_state_circ + ansatz
    sim = Simulator('mqvector', circ.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    g = np.array([gamma[i]])
    cost, grad = grad_ops(g)
    costs[i][0] = gamma[i]
    costs[i][1] = cost[0][0].real

costs = pd.DataFrame(costs, columns=['gamma', 'cost'])
plt.plot(costs['gamma'], costs['cost'])

# 第六张图
plt.subplot(236)
plt.title("fe_cost", fontsize=20)
plt.xlim((0, 50))
plt.ylim((-1.2, 1.2))

costs = np.zeros((200, 1), dtype=np.float64)
costs = pd.DataFrame(costs, columns=['cost'])
for i in range(10):
    fe = data[:, i]
    fe = pd.DataFrame(fe, columns=['fe', 'gamma', 'beta', 'cost'])
    costs['cost'] += fe['cost']
    plt.plot(fe['fe'], fe['cost'])

costs['cost'] /= 10
plt.plot(fe['fe'], costs['cost'], color='k')

# 第三张图
plt.subplot(233)
plt.title("fe_beta", fontsize=20)
plt.xlim((0, 100))
plt.ylim((0, np.pi / 2))
plt.yticks([0, np.pi / 8, np.pi / 4, np.pi * 3 / 8],
           ['0', 'pi/8', 'pi/4', 'pi*3/8'],
           fontsize=15)

for i in range(10):
    fe = data[:, i]
    fe = pd.DataFrame(fe, columns=['fe', 'gamma', 'beta', 'cost'])
    plt.plot(fe['fe'], fe['beta'])

#第二张图
plt.subplot(232)
plt.title("noise-free", fontsize=20)
plt.xlim((0, np.pi / 2))
plt.ylim((0, np.pi))

costs = np.zeros((beta.size, gamma.size), dtype=np.float64)
for i in range(beta.size):
    for j in range(gamma.size):
        ansatz = build_ansatz(k, p, gamma[j], beta[i])
        circ = init_state_circ + ansatz
        sim = Simulator('mqvector', circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(ham, circ)
        gb = np.array([])
        cost, grad = grad_ops(gb)
        costs[i][j] = cost[0][0].real

sns.heatmap(data=costs, vmin=-1, vmax=1, xticklabels=15,
            yticklabels=15).invert_yaxis()

# pylint: disable=W0104
# QAOA量子线路的层数p
p = 4
# 生成初态（均匀叠加态），即对所有量子比特作用H门
init_state_circ = UN(H, k.shape[1])
# 生成哈密顿量
ham = Hamiltonian(build_HC(k))
# ansatz是求解该问题的量子线路
ansatz = build_ansatz(k, p, '', '')
# 将初始化线路与ansatz线路组合成一个线路
circ = init_state_circ + ansatz

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
# 生成一个基于mqvector后端的模拟器，并设置模拟器的比特数为量子线路的比特数
sim = Simulator('mqvector', circ.n_qubits)
# 获取模拟器基于当前量子态的量子线路演化以及期望、梯度求解算子
grad_ops = sim.get_expectation_with_grad(ham, circ)
# 生成待训练的神经网络
net = MQAnsatzOnlyLayer(grad_ops)
# 设置针对网络中所有可训练参数、学习率为0.05的Adam优化器
opti = nn.Adam(net.trainable_params(), learning_rate=0.2)
# 对神经网络进行一步训练，TrainOneStepCell的返回值是loss
train_net = nn.TrainOneStepCell(net, opti)
# 训练200次
for i in range(200):
    train_net()

# 将最优参数提取出来并存储为字典类型，与之前线路中命名的参数一一对应
pr = dict(zip(ansatz.params_name, net.weight.asnumpy()))
# 为线路中所有比特添加测量门
circ.measure_all()
# 将最优参数代入量子线路，通过对量子线路进行1000次采样，画出最终量子态在计算基矢下的概率分布
sim.sampling(circ, pr=pr, shots=1000)