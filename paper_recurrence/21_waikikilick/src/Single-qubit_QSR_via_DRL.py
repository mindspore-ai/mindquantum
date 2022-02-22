# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 16:58:09 2022

基于 Mindspore 搭建一个 DQN 算法。利用该算法，可设计优化控制脉冲，将量子比特一个从任意态制备到指定的 |0> 态。

@author: Waikikilick  1250125907@qq.com
"""

import copy 
import random
from time import *
import numpy as np 
from mindquantum import *
from collections import deque
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mindspore import dataset as ds
from mindspore.common.initializer import Normal
from mindspore import Model, Tensor, context, nn, set_seed
context.set_context(mode=context.GRAPH_MODE, device_target="CPU") # 设置静态图模式，CPU 支持此模式

set_seed(1) # 设置宏观随机数种子

sx = X.matrix() # 定义泡利算符，根据 mindquantum 基本门的 matrix() 功能来得到矩阵形式
sz = Z.matrix()
dt = np.pi/10

action_space = [0,1,2,3]

# 定义训练网络：
class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__() # 网络： 两个隐藏层，分别有 32/32 个神经元；输出层 4 个神经元。
        self.fc1 = nn.Dense(4, 32, Normal(0.02), Normal(0.02), True, "relu")
        self.fc2 = nn.Dense(32, 32, Normal(0.02), Normal(0.02), True, "relu")
        self.fc3 = nn.Dense(32, 4, Normal(0.02), Normal(0.02), True, "relu") 
        
    def construct(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q_values = self.fc3(x)
        return q_values
    
net = LinearNet()

# 定义智能体，所有动作选择、网络训练等任务都由其完成。
class Agent(nn.Cell):
    def __init__(self, 
            n_actions=4,
            n_features=4,
            learning_rate=0.0001,
            reward_decay=0.9,
            e_greedy=0.99,
            replace_target_iter=250,
            memory_size=2000,
            batch_size=32,
            e_greedy_increment=None):
        
        self.n_actions = n_actions # 允许的分立动作数
        self.n_features = n_features # 用实向量描述量子态所需维度
        self.lr = learning_rate # 神经网络的学习率
        self.gamma = reward_decay # 奖励折扣因子
        self.epsilon_max = e_greedy # 执行动态贪心算法进行动作选择时，采用网络来预测动作的最大概率 epsilon
        self.epsilon_increment = e_greedy_increment # 动态贪心算法 epsilon 的每次增幅
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max
        self.replace_target_iter = replace_target_iter # 目标网络更新间隔：每训练主网络 replace_target_iter 次，就更新一次目标网络。
        self.batch_size = batch_size # 训练批数据尺寸
        self.learn_step_counter = 0 # 记录训练次数，每 replace_target_iter 次学习, 更新 target 网络参数一次
        self.memory_size = memory_size # 记忆库容量上限
        self.memory = deque(maxlen=self.memory_size) # 容量有限的记忆库
        self.memory_counter = 0 # 记忆库存储次数。用于初期判断是否可以开始对主网络进行训练
        self.main_net = Model(net, nn.loss.MSELoss(), nn.RMSProp(params=net.trainable_params(), learning_rate=self.lr)) # 主网络
        self.target_net = self.main_net # 目标网络
                
    def store_transition(self, s, a, r, s_): # 保存 经验单元
        self.memory.append([s, a, r, s_]) 
        self.memory_counter += 1

    def choose_action(self, state, tanxin): # tanxin 的值代表着 预测动作的 策略选择
      # tanxin = 0 意味着完全随机取动作
      # tanxin = 0.5 意味着执行动态贪心策略
      # tanxin = 1 意味着完全靠网络选择动作
        
        if tanxin == 0: # tanxin = 0, 意味着完全随机选动作
            action = np.random.randint(0, self.n_actions)
        elif tanxin == 1: # tanxin = 1, 意味着全靠网络预测动作, 通常在测试阶段用 # 其他值的话就意味着动作选择概率处于动态调整中, 比如 tanxin = 0.5
            action = np.argmax(self.main_net.predict(Tensor([state])))
        else:
            if np.random.uniform() < self.epsilon:	
                action = np.argmax(self.main_net.predict(Tensor([state])))
            else:
                action = np.random.randint(0, self.n_actions)
            
        return action        
    
    # 采集数据。
    def get_samples(self, s_batch, Q): 
        for i in range(len(s_batch)):
            yield s_batch[i], Q[i]  
            
    # 根据采集到的数据，生成训练数据
    def get_train_data(self, s_batch, Q):
        train_data = ds.GeneratorDataset(list(self.get_samples(s_batch, Q)),column_names=['state','Q'])
        train_data = train_data.batch(self.batch_size) # 进行批处理
        return train_data
    
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0: # 每 replace_target_iter 次学习, 更新 target 网络参数一次
            self.target_net = self.main_net # target 网络参数从 主网络完全复制而来
                    
        if self.memory_counter < self.batch_size: # 只有当 记忆库中的数据量 比 批大小 多的时候才开始学习
            return # 如果 if 成立, 那么 return 后面的 且与 if 同级的代码不再执行
        batch_memory = random.sample(self.memory, self.batch_size) # 从 记忆库 中随机选出 批大小 数量的样本
        s_batch = np.array([replay[0] for replay in batch_memory]).astype(np.float32) # 多行数组
        Q = self.main_net.predict(Tensor(s_batch)) # 主网络预测 Q 值
        
        next_s_batch = np.array([replay[3] for replay in batch_memory]).astype(np.float32)
        Q_next = self.target_net.predict(Tensor(next_s_batch)) # 目标网络预测下一时刻状态的 Q 值
        
        # 使用公式更新训练集中的 Q 值   
        for i, replay in enumerate(batch_memory):
            _, a, reward, _ = replay
            Q[int(i)][int(a)] =   reward + self.gamma * max(Q_next[i]) # Q 值计算法则
        
        train_data = self.get_train_data(s_batch, Q)
        self.main_net.train(1, train_data, dataset_sink_mode=False) # 训练一次网络
        
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

def _matrix(coeff):
    return expm(-1j*(coeff*sz+sx)*dt)

def _diff_matrix(coeff):
    return -1j*dt*_matrix(coeff)@sz

Evo = gene_univ_parameterized_gate('U', _matrix, _diff_matrix) # 给定动作的时间演化算符
circ = Circuit()+Evo('coeff').on(0) # 将时间演化算符作为量子门编到线路中
ham = Hamiltonian(QubitOperator("")) # 定义一个空的哈密顿量，用于后面求解两个量子态之间的保真度

# 模拟量子系统作为交互环境
class env(): 
    def  __init__(self, 
        action_space = [0,1], # 允许的动作，默认两个分立值，只是默认值，真正值由调用时输入
        ): 
        self.action_space = action_space
        self.n_actions = len(self.action_space)
        self.n_features = 4  # 描述状态所用的长度
        self.target_psi =  np.array([1,0])  # 最终的目标态为 |0>
        self.training_set, self.validation_set, self.testing_set = self.psi_set() 
        
    # 生成训练集、验证集和测试集
    def psi_set(self):
        theta_num = 6 # 除了 0 和 Pi 两个点之外，点的数量
        varphi_num = 21 # varphi 角度一圈上的点数
        # 总点数为 theta_num * varphi_num + 2(布洛赫球两极) # 6 * 21 + 2 = 128
        
        theta = np.linspace(0,np.pi,theta_num+2,endpoint=True) 
        varphi = np.linspace(0,np.pi*2,varphi_num,endpoint=False)  
        
        psi_set = []
        for ii in range(1,theta_num+1):
            for jj in range(varphi_num):
                psi_set.append(np.array([np.cos(theta[ii]/2),np.sin(theta[ii]/2)*(np.cos(varphi[jj])+np.sin(varphi[jj])*(0+1j))]).astype(np.complex64))
                
        psi_set.append(np.array([1,0]).astype(np.complex64)) # 最后再加上 |0> 和 |1> 两个态
        psi_set.append(np.array([0,1]).astype(np.complex64))
        
        random.shuffle(psi_set) # 打乱点集
        
        # 数据集切割
        training_set = psi_set[0:32]
        validation_set = psi_set[32:64]
        testing_set = psi_set[64:128]
        
        return training_set, validation_set, testing_set
        
    
    def reset(self, sim, init_psi): # 在一个新的回合开始时，归位到开始选中的那个点上
        sim.set_qs(init_psi)
        init_state = np.array([init_psi[0].real, init_psi[1].real, init_psi[0].imag, init_psi[1].imag])
        # np.array([1实，2实，1虚，2虚])
        
        return sim, init_state
    
    
    def step(self, sim, sim_target, action, nstep):
        fid_ops = sim.get_expectation_with_grad(ham, # 单位算符，除了占位，啥用也没有。因为要计算的是两个量子态之间的保真度
                                                circ, # 右线路，由时间演化算符所定义。用于根据当前时刻量子态计算下一时刻量子态
                                                Circuit(), # 左线路，空的。保持目标态不变
                                                sim_target, # 目标量子态
                                                encoder_params_name=['coeff'], # 调入动作值
                                                parallel_worker=1) # 并行核
        fid_list, _ = fid_ops(np.array([[action]])) # action 整数
        fid = np.abs(fid_list[0,0])**2   
        sim.apply_circuit(circ, pr=action)
        psi_ = sim.get_qs()        
        state_ = np.array([psi_[0].real, psi_[1].real, psi_[0].imag, psi_[1].imag])
        
        err = 10e-4
        rwd = fid
        done = (((1 - fid) < err) or nstep >= 2 * np.pi / dt) # 用于判断智能体是否停止当前任务
        
        return sim, state_, rwd, done, fid 


# 训练过程
def training(ep_max): 
    training_set = env.training_set
    validation_set = env.validation_set
    
    sim = Simulator('projectq', 1) 
    sim_target = Simulator('projectq', 1)
    sim_target.set_qs(env.target_psi) # 设置目标量子态
    print('--------------------------')
    print('训练中...')
    for i in range(ep_max):
        if i % 5 == 0:
            print('当前训练回合为：', i)
        training_init_psi = random.choice(training_set)
        fid_max = 0
        
        sim, observation = env.reset(sim, training_init_psi)
        nstep = 0
        
        while True:
            action = agent.choose_action(observation, 0.5) 
            sim, observation_, reward, done, fid = env.step(sim, sim_target, action, nstep)  
            nstep += 1
            fid_max = max(fid_max, fid)
            agent.store_transition(observation, action, reward, observation_)
            agent.learn()
            observation = observation_
                
            if done:
                break
            
        if i % 1 == 0: # 每 x 个回合用验证集验证一下效果，动作全靠网络，保存最大保真度和最大奖励
            validation_fid_list = []
            validation_reward_tot_list = []
            for validation_init_psi in validation_set:
                validation_fid_max = 0
                validation_reward_tot = 0
                
                sim, observation = env.reset(sim, validation_init_psi)
                nstep = 0
                
                while True:
                    action = agent.choose_action(observation, 1) 
                    sim, observation_, reward, done, fid = env.step(sim, sim_target, action, nstep) 
                    nstep += 1
                    validation_fid_max = max(validation_fid_max, fid)
                    validation_reward_tot = validation_reward_tot + reward * (agent.gamma ** nstep)
                    observation = observation_
                    
                    if done:
                        break
                validation_fid_list.append(validation_fid_max)                
                validation_reward_tot_list.append(validation_reward_tot)
            
            validation_reward_history.append(np.mean(validation_reward_tot_list))
            validation_fid_history.append(np.mean(validation_fid_list))
            
            print('本回合验证集平均保真度: ', np.mean(validation_fid_list))
            # print('本回合验证集平均总奖励: ', np.mean(validation_reward_tot_list))

def testing(): # 测试 测试集中的点 得到 保真度 分布
    print('\n测试中, 请稍等...')
    
    testing_set = env.testing_set
    fid_list = []
    
    sim = Simulator('projectq', 1) 
    sim_target = Simulator('projectq', 1)
    sim_target.set_qs(env.target_psi) # 设置目标量子态
    
    for test_init_psi in testing_set:
        fid_max = 0
        sim, observation = env.reset(sim, test_init_psi)
        nstep = 0 
        
        while True:
            action = agent.choose_action(observation, 1)  
            sim, observation_, reward, done, fid = env.step(sim, sim_target, action, nstep)  
            nstep += 1
            fid_max = max(fid_max, fid)
            observation = observation_
                
            if done:
                break
            
        fid_list.append(fid_max)
        
    return fid_list

if __name__ == "__main__":
    env = env(action_space = list(range(4)) # 离散动作空间
             )
    
    agent = Agent(env.n_actions, env.n_features,
              learning_rate = 0.01,
              reward_decay = 0.9, 
              e_greedy = 0.95,
              replace_target_iter = 200,
              memory_size = 20000,
              e_greedy_increment = 0.001)
            
    validation_reward_history = []
    validation_fid_history = []
        
    begin_training = time()

    # 训练模块 此模块可反复利用
    training(ep_max = 100)   
   
    end_training = time()
    
    training_time = end_training - begin_training
    
    print('\n训练过程共用时：',training_time,"s")  
        
    print('各验证回合回合奖励记录为: ', validation_reward_history)
    print('各验证回合最大保真度记录为: ', validation_fid_history)

    # 绘出训练过程中的表现
    x = np.arange(len(validation_reward_history))
    fig, ax = plt.subplots()
    ax.plot(x,np.array(validation_reward_history)/7,'--',label='validation reward history')
    ax.plot(x,validation_fid_history,':',label='validation fidelity history')
    legend = ax.legend(loc='lower center', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('#00FFCC')
    plt.show()
    
    # 测试
    testing_fid_list = testing()
    print('测试集平均保真度为：', np.mean(testing_fid_list))
