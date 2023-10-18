# -*- coding: utf-8 -*-
"""
Configuration.
"""

rand_generator = 'C'  # 随机比特发生器 模式['C' 经典| 'Q' 量子]
sim_backend = 'mqvector'  # 模拟器

# QAOA
qaoa_level = 1  # 线路层数
qaoa_step = 100  # 训练步数
qaoa_lr = 0.05  #学习率

# RQAOA
rqaoa_nc = 8  # the cutoff value for variable elimination
