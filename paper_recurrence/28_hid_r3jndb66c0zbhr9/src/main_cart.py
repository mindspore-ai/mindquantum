# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import gym
import os
import math
import time
import argparse
import numpy as np
from mindspore import context, Tensor
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms                            # 导入mindspore库并简写为ms
from collections import deque
import random
from mindspore.common.initializer import Normal
from mindquantum.core import Circuit                 # 导入Circuit模块，用于搭建量子线路
from mindquantum.core import UN                      # 导入UN模块
from mindquantum.core import H, X, RZ                # 导入量子门H, X, RZ
from mindquantum.algorithm import HardwareEfficientAnsatz      # 导入HardwareEfficientAnsatz
from mindquantum.core import RX, RY
from mindquantum.core import QubitOperator                     # 导入QubitOperator模块，用于构造泡利算符
from mindquantum.core import Hamiltonian                       # 导入Hamiltonian模块，用于构建哈密顿量
from mindquantum.framework import MQLayer                      # 导入MQLayer
from mindquantum.simulator import Simulator
from mindspore.nn import SoftmaxCrossEntropyWithLogits                         # 导入SoftmaxCrossEntropyWithLogits模块，用于定义损失函数
from mindspore.nn import Adam, Accuracy                                        # 导入Adam模块和Accuracy模块，分别用于定义优化参数，评估预测准确率
from mindspore import Model                                                    # 导入Model模块，用于建立模型
from mindspore.dataset import NumpySlicesDataset                               # 导入NumpySlicesDataset模块，用于创建模型可以识别的数据集
from mindspore.train.callback import Callback, LossMonitor                     # 导入Callback模块和LossMonitor模块，分别用于定义回调函数和监控损失
from mindspore import dtype as mstype
from scipy import stats
import matplotlib.pyplot as plt
from IPython.display import Image, display


# %%
class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


# %%
class Actor(nn.Cell):
    def __init__(self, num_class=10, num_channel=1):
        super(Actor, self).__init__()
        self.encoder = self.__get_encoder()
        self.ansatz = self.__get_ansatz()
        self.circuit = self.encoder + self.ansatz
        self.observable = self.__get_observable()
        self.sim = Simulator('projectq',self.circuit.n_qubits)
        self.quantum_net = self.__get_quantum_net()
        self.softmax = ops.Softmax()
    def construct(self, x):
        x = self.quantum_net(x)
        x = self.softmax(x)
        return x
    def __get_encoder(self):
        encoder = Circuit()                                  # 初始化量子线路
        encoder += UN(H, 4)                                  # H门作用在每1位量子比特
        for i in range(4):                                   # i = 0, 1, 2, 3
            encoder += RY(f'alpha{i}').on(i)                 # RZ(alpha_i)门作用在第i位量子比特
        encoder = encoder.no_grad()                          # Encoder作为整个量子神经网络的第一层，不用对编码线路中的梯度求导数，因此加入no_grad()
        encoder.summary()                                    # 总结Encoder
        return encoder
    def __get_ansatz(self):
        ansatz = HardwareEfficientAnsatz(4, single_rot_gate_seq=[RX,RY,RZ], entangle_gate=X, depth=3).circuit     # 通过HardwareEfficientAnsatz搭建Ansatz
        ansatz += X.on(2,0)
        ansatz += X.on(3,1)
        ansatz.summary()                                                                                    # 总结Ansatz
        # print(ansatz)
        return ansatz
    def __get_observable(self):
        hams = [Hamiltonian(QubitOperator(f'Y{i}')) for i in [2, 3]]   # 分别对第2位和第3位量子比特执行泡利Z算符测量，且将系数都设为1，构建对应的哈密顿量
        print(hams)
        return hams
    def __get_quantum_net(self):
        ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        ms.set_seed(1)  
        grad_ops = self.sim.get_expectation_with_grad(self.observable,
                                                self.circuit,
                                                None,
                                                None,
                                                self.encoder.params_name,
                                                self.ansatz.params_name,
                                                parallel_worker=1)
        QuantumNet = MQLayer(grad_ops)          # 搭建量子神经网络
        return QuantumNet
    def select_action(self, state):
        x = self.quantum_net(state)
        x = self.softmax(x)
        if np.random.rand() <= x[0][0]:
            action = 0
        else:
            action = 1
        return action


# %%
class Critic(nn.Cell):
    def __init__(self):
        super(Critic, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Dense(4, 64)
        self.fc2 = nn.Dense(64, 256)
        self.fc3 = nn.Dense(256, 1)

    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# %%
class MyActorLoss(nn.LossBase):
    """定义损失"""
    def __init__(self, reduction="mean"):
        super(MyActorLoss, self).__init__(reduction)
        self.abs = ops.Abs()

    def construct(self, base, batch_action, old_p, advantage):
        prob = base[:,0]*(1-batch_action)+base[:,1]*batch_action
        log_prob = ops.log(prob)
        old_prob = old_p[:,0]*(1-batch_action)+old_p[:,1]*batch_action
        old_log_prob = ops.log(old_prob)
        ratio = ops.exp(log_prob - old_log_prob)
        L1 = ratio * advantage
        L2 = ratio.clip(0.9, 1.1) * advantage
        loss = 0-ops.minimum(L1, L2)
        return self.get_loss(loss)


# %%
class MyWithLossActor(nn.Cell):
    """定义损失网络"""
    def __init__(self, backbone, loss_fn):
        super(MyWithLossActor, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, batch_action, old_p, advantage):
        out = self.backbone(data)
        return self.loss_fn(out, batch_action, old_p, advantage)

    def backbone_network(self):
        return self.backbone

class MyWithLossCritic(nn.Cell):
    """定义损失网络"""
    def __init__(self, backbone,  loss_fn):
        super(MyWithLossCritic, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, label):
        out = self.backbone(data)
        return self.loss_fn(out, label)

    def backbone_network(self):
        return self.backbone

class MyTrainStep(nn.TrainOneStepCell):
    """定义训练流程"""
    def __init__(self, network, optimizer):
        """参数初始化"""
        super(MyTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, data, label):
        """构建训练过程"""
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads)
    

class MyActorTrainStep(nn.TrainOneStepCell):
    """定义训练流程"""
    def __init__(self, network, optimizer):
        """参数初始化"""
        super(MyActorTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, data, batch_action, old_p, advantage):
        """构建训练过程"""
        weights = self.weights
        loss = self.network(data, batch_action, old_p, advantage)
        grads = self.grad(self.network, weights)(data, batch_action, old_p, advantage)
        # print("grads: ",grads)
        return loss, self.optimizer(grads)


# %%

parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])

args = parser.parse_known_args()[0]
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
def train(N):
    starttime = time.time()
    env = gym.make('CartPole-v0')
    actor = Actor()
    old_actor = Actor()
    optim = Adam(actor.quantum_net.trainable_params(), learning_rate=1e-3)
    critic = Critic()
    value_optim = Adam(critic.trainable_params(), learning_rate=1e-4)
    gamma = 0.98
    lambd = 0.95
    # epsilon = 0.01
    memory = Memory(200)

    batch_size = 256

    loss_func_a = MyActorLoss()
    actor_with_criterion = MyWithLossActor(actor, loss_func_a)
    train_actor = MyActorTrainStep(actor_with_criterion, optim)

    loss_func_c = nn.loss.MSELoss()
    critic_with_criterion = MyWithLossCritic(critic, loss_func_c)
    # 构建训练网络
    train_critic = MyTrainStep(critic_with_criterion, value_optim)




    EPOCH=N
    re = []
    for epoch in range(EPOCH):
        state = env.reset()
        episode_reward = 0
        # 将每个参数规范到-π到π中间
        state = np.array([state[0]*np.pi/4.8, np.tanh(state[1])*np.pi, state[2]*np.pi/(4.1887903e-01), np.tanh(state[3])*np.pi])
        for _ in range(200):
            state_tensor = Tensor([state])
            action = actor.select_action(state_tensor)
            next_state, reward, done, _ = env.step(action)
            next_state = np.array([next_state[0]*np.pi/4.8, np.tanh(next_state[1])*np.pi, next_state[2]*np.pi/(4.1887903e-01), np.tanh(next_state[3])*np.pi])
            episode_reward += reward
            memory.add((state, next_state, action, reward, (done+1)%2))
            state = next_state
            if done:
                break

        old_actor.quantum_net.weight = actor.quantum_net.weight

        for _ in range(1):
            experiences = memory.sample(batch_size, True)
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*experiences)

            batch_state = Tensor([state.astype(np.float32) for state in  batch_state])
            batch_next_state = Tensor([next_state.astype(np.float32) for next_state in  batch_next_state])
            batch_action = Tensor([action for action in batch_action])
            batch_reward = Tensor([[reward] for reward in batch_reward])
            batch_done = Tensor([[done] for done in batch_done])

            old_p = old_actor(batch_state)
            value_target = batch_reward + batch_done * gamma * critic(batch_next_state)
            td = value_target - critic(batch_state)
            zeros = ops.Zeros()
            advantage = zeros((len(td)), mstype.float32)
            for i in range(len(td)):
                temp = 0
                for j in  range(len(td)-1, i, -1):
                    temp = (temp + td[j])*lambd*gamma
                temp += td[i]
                advantage[i]=temp

            train_actor(batch_state, batch_action, old_p, advantage)
            train_critic(batch_state, value_target)

        loss_val = critic_with_criterion(batch_state, value_target)
        print("epoch",epoch," loss: ",loss_val)


        memory.clear()
        re.append(episode_reward)
        if (epoch+1) % 1 == 0:
            print('Epoch:{}/{}, episode reward is {}'.format(epoch+1, EPOCH, episode_reward))
            use = time.time()-starttime
            print("Have used ",use," s, / ",use*EPOCH/(epoch+1),"s")
    usetime = time.time() - starttime
    print("Time",usetime)

    plt.plot(re)
    plt.ylim(0,200)
    plt.xlabel("Epoch")
    plt.ylabel("Score")

    plt.savefig('result_cartp2.jpg')
    return re


# %%



