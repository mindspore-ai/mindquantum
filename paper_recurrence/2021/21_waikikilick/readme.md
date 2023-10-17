# 论文名字 Deep reinforcement learning for universal quantum state preparation via dynamic pulse control

## 项目介绍

此论文提出，采用深度强度学习中的 DQN (deep Q network) 算法，可在满足给定实际物理的硬件限制条件下，设计最佳时序控制脉冲，实现 "量子态重置" 任务。本项目可作为经典深度强化学习与量子系统相交互产生最优控制的典型案例。 main.ipynb 中对算法和模型及代码实现进行了详细介绍。

## 主要结果

本项目依托 MindSpore 机器学习框架和 MindQuantum  量子计算框架复现了此论文最核心的两项结果：通过充足的训练，可设计最佳的时序脉冲以实现在半导体双量子点系统下，单和双量子比特任意态的重置任务。此项目展示了 MindSpore 和 MindQuantum 相协同，可以很好地完成量子系统优化控制设计任务。例证了 MindQuantum  在含时控制下的量子态模拟中具有很大的适用空间。

## 创新点

1. 将原代码迁移到了 MindSpore 和 MindQuantum 框架下，例证了该框架在量子信息处理领域中的广泛适用性； 

2. 增加了数据处理过程和数据图片显示模块，便于在 notebook 中进行展示；

3. 为增加代码的可读性，本项目将论文原代码进行了较大优化调整，并为重要代码添加了详细的注释，以期在社区实现更好的开源效果。

作者：Waikikilick
邮箱地址：1250125907@qq.com