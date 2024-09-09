# 量子组合优化赛道-QAOA组合优化的初参设置问题
## 背景
量子近似优化算法（quantum approximate optimization algorithm, QAOA）是可以在近期含噪中等规模量子（noisy intermediate-scale quantum, NISQ）设备上运行同时有广泛应用前景的量子算法，其目的是近似地求解组合优化问题。
QAOA是一种变分算法，需要在初始输入的线路参数的基础上根据目标函数不断调整参数，直到目标函数达到极小值。因此初始参数的选取非常重要，直接影响到算法最终的收敛效果和时间开销。如何设置QAOA线路中的初始参数使得初始状态下目标函数尽可能小，本赛题旨在探索该问题的可行方案。

## 提交内容
请完善get_initial_paras()函数的内容，使得输入不同的Ising问题和所需要的层数p，输出对应QAOA线路的最优初始参数$γ=(γ_1,⋯,γ_p ),β=(β_1,⋯,β_p)$。函数主体内请依据输入问题的特点实现确定性算法，禁止使用目标函数对参数进行迭代优化。

## 评分规则
对于给定的公开数据集和隐藏测试集案例D，计算每个案例d的初始参数$γ_d,β_d $下的ising问题的能量期望值$C_d$，并对所有案例求和取负号后作为总分数：

$Score =-∑_{(d∈D )} C_d (γ_d,β_d ) $

## 数据集
评分所使用的Ising问题包括公开和隐藏数据集：
- 公开数据集：存放在data文件夹中，包括以下几种特征：1. 从二阶到5阶；2. 从顶点间所有可能的超边中按照0.3/0.6/0.9的比例随机选取作为ising模型的超边；3. Ising模型中每一条超边上的系数从固定分布/均匀分布/双峰分布中产生。
- 隐藏数据集：存放在data/_hidden文件夹中，在评分时计入总分数。

## 赛题要求
- get_initial_paras()函数主体设计需确保整体代码运行时间在30min以内
- 所使用库可通过pip或conda安装，不使用收费库

## 参考资料
【1】Mindquantum的QAOA教学文档：https://www.mindspore.cn/mindquantum/docs/zh-CN/master/case_library/quantum_approximate_optimization_algorithm.html

【2】Sureshbabu S H, Herman D, Shaydulin R, et al. Parameter setting in quantum approximate optimization of weighted problems[J]. Quantum, 2024, 8: 1231.

【3】Shaydulin R, Lotshaw P C, Larson J, et al. Parameter transfer for quantum approximate optimization of weighted maxcut[J]. ACM Transactions on Quantum Computing, 2023, 4(3): 1-15.
