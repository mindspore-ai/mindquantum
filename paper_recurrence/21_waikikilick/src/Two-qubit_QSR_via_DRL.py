# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 16:58:09 2022

基于 Mindspore 搭建一个 DQN 算法。利用该算法，可设计优化控制脉冲，将双量子比特一个从任意态制备到制备到指定的 (|00>+|11>)/sqrt(2) 态。

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

# 定义训练网络：
class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__() # 网络： 三个隐藏层，分别有 256/256/128 个神经元；输出层 25 个神经元。
        self.fc1 = nn.Dense(8, 256, Normal(0.02), Normal(0.02), True, "relu")
        self.fc2 = nn.Dense(256, 256, Normal(0.02), Normal(0.02), True, "relu")
        self.fc3 = nn.Dense(256, 128, Normal(0.02), Normal(0.02), True, "relu") 
        self.fc4 = nn.Dense(128, 25, Normal(0.02), Normal(0.02), True, "relu") 
        
    def construct(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        q_values = self.fc4(x)
        return q_values
        
net = LinearNet()

# 定义智能体，所有动作选择、网络训练等任务都由其完成。
class Agent(nn.Cell):
    def __init__(self, 
            n_actions=25,
            n_features=8,
            learning_rate=0.0001,
            reward_decay=0.9,
            e_greedy=0.95,
            replace_target_iter=200,
            memory_size=4096,
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
    
    # 根据训练样本生成训练数据。
    def get_samples(self, s_batch, Q): 
        for i in range(len(s_batch)):
            yield s_batch[i], Q[i]  
            
    # 对训练数据进行处理
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

#---------------------------------------------------------------------------------------------------
# 模拟量子系统作为交互环境
class env(): 
    def  __init__(self, 
        dt = np.pi / 2
        ): 
        self.action_space = np.array(  [[1,1],
                                        [1,2],
                                        [1,3],
                                        [1,4],
                                        [1,5],
                                        [2,1],
                                        [2,2],
                                        [2,3],
                                        [2,4],
                                        [2,5],
                                        [3,1],
                                        [3,2],
                                        [3,3],
                                        [3,4],
                                        [3,5],
                                        [4,1],
                                        [4,2],
                                        [4,3],
                                        [4,4],
                                        [4,5],
                                        [5,1],
                                        [5,2],
                                        [5,3],
                                        [5,4],
                                        [5,5]] )
        self.n_actions = len(self.action_space)
        self.n_features = 8 #描述状态所用的长度
        self.target_psi =  np.mat([[1], [0], [0], [1]], dtype=np.complex64)/np.sqrt(2) #最终的目标态
        self.h_1 = 1
        self.h_2 = 1
        self.I = I.matrix()
        self.s_x = X.matrix() # 定义泡利算符，根据 mindquantum 基本门的 matrix() 功能得到泡利矩阵 
        self.s_z = Z.matrix() 
        self.dt = dt
        self.training_set, self.validation_set, self.testing_set = self.psi_set()
        
    # 生成训练集、验证集和测试集
    def psi_set(self):
        alpha_num = 4
        
        theta = [np.pi/8,np.pi/4,3*np.pi/8]
        theta_1 = theta
        theta_2 = theta
        theta_3 = theta
        
        alpha = np.linspace(0,np.pi*2,alpha_num,endpoint=False)
        alpha_1 = alpha
        alpha_2 = alpha
        alpha_3 = alpha
        alpha_4 = alpha
        
        psi_set = []#np.matrix([[0,0,0,0]],dtype=complex) #第一行用来占位，否则无法和其他行并在一起，在最后要注意去掉这一行
        for ii in range(3): #theta_1
            for jj in range(3): #theta_2
                for kk in range(3): #theta_3
                    for mm in range(alpha_num): #alpha_1
                        for nn in range(alpha_num): #alpha_2
                            for oo in range(alpha_num): #alpha_3
                                for pp in range(alpha_num): #alpha_4
                                    
                                    a_1_mo = np.cos(theta_1[ii])
                                    a_2_mo = np.sin(theta_1[ii])*np.cos(theta_2[jj])
                                    a_3_mo = np.sin(theta_1[ii])*np.sin(theta_2[jj])*np.cos(theta_3[kk])
                                    a_4_mo = np.sin(theta_1[ii])*np.sin(theta_2[jj])*np.sin(theta_3[kk])
                                    
                                    a_1_real = a_1_mo*np.cos(alpha_1[mm])
                                    a_1_imag = a_1_mo*np.sin(alpha_1[mm])
                                    a_2_real = a_2_mo*np.cos(alpha_2[nn])
                                    a_2_imag = a_2_mo*np.sin(alpha_2[nn])
                                    a_3_real = a_3_mo*np.cos(alpha_3[oo])
                                    a_3_imag = a_3_mo*np.sin(alpha_3[oo])
                                    a_4_real = a_4_mo*np.cos(alpha_4[pp])
                                    a_4_imag = a_4_mo*np.sin(alpha_4[pp])
                                    
                                    a_1_complex = a_1_real + a_1_imag*1j
                                    a_2_complex = a_2_real + a_2_imag*1j
                                    a_3_complex = a_3_real + a_3_imag*1j
                                    a_4_complex = a_4_real + a_4_imag*1j
                                    
                                    a_complex = np.mat([[ a_1_complex], [a_2_complex], [a_3_complex], [a_4_complex]])
                                    # psi_set = np.row_stack((psi_set,a_complex))
                                    psi_set.append(a_complex)
                                    
        # psi_set = np.array(np.delete(psi_set,0,axis=0)) # 删除矩阵的第一行
        random.shuffle(psi_set) #打乱顺序
    
        training_set = psi_set[0:256]
        validation_set = psi_set[256:512]
        testing_set = psi_set[512:]
        
        return training_set, validation_set, testing_set
        
    
    def reset(self, init_psi): # 在一个新的回合开始时，归位到开始选中的那个点上
        # psi: np.mat([[1.+2.j]
                    # [3.+4.j]
                    # [5.+6.j]
                    # [7.+8.j]])
                
        init_state = (np.array(init_psi.real.tolist() + init_psi.imag.tolist()).T).squeeze() # 从 复矩阵列向量表示 变为 实数组行 形式 
        # state: np.array([1. 3. 5. 7. 2. 4. 6. 8.])
        
        return init_state
    
    
    def step(self, state, action, nstep):
        
        psi = np.mat((np.array([state[:4]]).astype(np.complex64) + np.array([state[4:]]).astype(np.complex64)*1j).T)  # 从 实数组行 变回 复矩阵向量 形式
        # psi: np.mat([[1.+2.j]
                    # [3.+4.j]
                    # [5.+6.j]
                    # [7.+8.j]])
        
        J_1, J_2 =  self.action_space[action,0], self.action_space[action,1]  # control field strength
        J_12 = J_1 * J_2 /2
        
        H =  (J_1*np.kron(self.s_z, self.I) + J_2*np.kron(self.I, self.s_z) + \
                        J_12/2*np.kron((self.s_z-self.I), (self.s_z-self.I)) + \
                        self.h_1*np.kron(self.s_x,self.I) + self.h_2*np.kron(self.I,self.s_x))/2
        U = UnivMathGate('U', expm(-1j * H * self.dt)) # 根据 mindquantum 普适门 来定义 时间演化算符。 ‘U’ 为自定义门名称，后面为自定义的矩阵
        
        psi = U.matrix() * psi  # next state

        err = 10e-4
        fid = (np.abs(psi.H * self.target_psi) ** 2).item(0).real  
        rwd = fid
        
        done = (((1 - fid) < err) or nstep >= 20 * np.pi / self.dt)  

        #再将量子态的 psi 形式恢复到 state 形式。 # 因为网络输入不能为复数，否则无法用寻常基于梯度的算法进行反向传播
    
        state = (np.array(psi.real.tolist() + psi.imag.tolist()).T).squeeze() # 实数组形式

        return state, rwd, done, fid      

# 训练过程
def training(ep_max): 
    print('训练中...')
    training_set = env.training_set
    validation_set = env.validation_set
    
    for i in range(ep_max):
        training_init_psi = random.choice(training_set)
        fid_max = 0
        observation = env.reset(training_init_psi)
        nstep = 0
        
        while True:
            action = agent.choose_action(observation, 0.5) 
            observation_, reward, done, fid = env.step(observation, action, nstep)  
            nstep += 1
            fid_max = max(fid_max, fid)
            agent.store_transition(observation, action, reward, observation_)
            agent.learn() 
            observation = observation_
                
            if done:
                break
            
        if i % 20 == 0: # 每 x 个回合用验证集验证一下效果，动作全靠网络，保存最大保真度和最大奖励
            validation_fid_list = []
            validation_reward_tot_list = []
            
            for validation_init_psi in validation_set:
                validation_fid_max = 0
                validation_reward_tot = 0
                
                observation = env.reset(validation_init_psi)
                nstep = 0
                while True:
                    action = agent.choose_action(observation, 1) 
                    observation_, reward, done, fid = env.step(observation, action, nstep)  
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
    
            print('第{}个回合验证集平均保真度为: '.format(i), np.mean(validation_fid_list))

def testing(): # 测试 测试集中的点 得到 保真度 分布
    print('\n测试中, 请稍等...')
    
    testing_set = env.testing_set
    fid_list = []
    
    for test_init_psi in testing_set:
        fid_max = 0
        observation = env.reset(test_init_psi)
        nstep = 0
        
        while True:
            action = agent.choose_action(observation, 1)  
            observation_, reward, done, fid = env.step(observation, action, nstep)  
            nstep += 1
            fid_max = max(fid_max, fid)
            observation = observation_
                
            if done:
                break
            
        fid_list.append(fid_max)
        
    return fid_list
                 
    
if __name__ == "__main__":
    dt = np.pi/2
    env = env(dt = dt)
    
    agent = Agent(n_actions = env.n_actions, 
                  n_features = env.n_features,
                  learning_rate = 0.001,
                  reward_decay = 0.9, 
                  e_greedy = 0.95,
                  replace_target_iter = 200,
                  memory_size = 40000,
                  e_greedy_increment = 0.0001)
            
    validation_reward_history = []
    validation_fid_history = []
        
    begin_training = time()

    # 训练模块 此模块可反复利用
    training(ep_max = 1000)   
   
    end_training = time()
    
    training_time = end_training - begin_training
    
    print('\n训练过程共用时：',training_time,"s")  
        
    print('各验证回合回合奖励记录为: ', validation_reward_history)
    print('各验证回合最大保真度记录为: ', validation_fid_history)

    # 绘出训练过程中的表现
    x = np.arange(len(validation_reward_history))
    fig, ax = plt.subplots()
    ax.plot(x,np.array(validation_reward_history)/8,'--',label='validation reward history')
    ax.plot(x,validation_fid_history,':',label='validation fidelity history')
    legend = ax.legend(loc='lower center', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('#00FFCC')
    plt.show()
    
    # 测试
    testing_fid_list = testing()
    print('测试集平均保真度为：', np.mean(testing_fid_list))
    
 