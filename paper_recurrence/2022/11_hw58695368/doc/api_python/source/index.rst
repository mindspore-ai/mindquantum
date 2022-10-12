.. api_python documentation master file, created by
   sphinx-quickstart on Wed Oct 12 22:59:10 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

赛题十一：利用MindQuantum实现多基矢编码的变分量子算法
========================================================

`昇腾AI创新大赛2022-昇思赛道 <https://www.hiascend.com/zh/developer/contests/details/48c53c2c697c482ba464111aaabb47ce>`_

赛题十一：利用MindQuantum实现多基矢编码的变分量子算法

论文：

`Variational Quantum Optimization with Multi-Basis Encodings <https://arxiv.org/pdf/2106.13304.pdf>`_

复现要求：

利用MindQuantum实现Multi-Basis Encoding的变分量子算法来解决maxcut问题， 复现fig2中的变分量子线路，实现小与10个节点的maxcut图分割。

@NPark-NoEvaa

API参考
--------------------------------------------------------

.. toctree::
   :maxdepth: 1

   src.dataset
   src.ansatz_mpo
   src.mbe_loss
   src.maxcut
   src.layer

样例：
--------------------------------------------------------

1.单问题求解

>>> from src.dataset import *
>>> from src.maxcut import maxcut
>>> n, problem = build_dataset1()                               # 获取问题1
>>> opti_args = dict(method='bfgs', jac=True)                   # 优化器参数
>>> _, _, res = maxcut(n, 7, problem, grad=True, **opti_args)   # 求解问题
>>> d1_result = res[:n]                                         # 问题1求解结果
>>> d1_score = score(problem, res)                              # 结果评分

2.双问题并行求解

>>> from src.dataset import *
>>> from src.maxcut import maxcut
>>> p1 = build_dataset1()                                       # 获取问题1
>>> p2 = build_dataset2()                                       # 获取问题2
>>> n, problem, od = build_dataset_parallel(*p1, *p2)           # 合并为并行问题
>>> opti_args = dict(method='bfgs', jac=True)                   # 优化器参数
>>> _, _, res = maxcut(n, 7, problem, grad=True, **opti_args)   # 求解问题
>>> pb = [p1, p2]
>>> p1_result = res[:pb[od[0]][0]]                              # 问题1求解结果
>>> p1_score = score(pb[od[0]][1], res[:m])                     # 问题1结果评分
>>> p2_result = res[m:m+pb[od[1]][0]]                           # 问题2求解结果
>>> p2_score = score(pb[od[1]][1], res[m:])                     # 问题2结果评分

3.MindSpore支持

>>> import numpy as np
>>> import mindspore as ms
>>> import mindspore.nn as nn
>>> import matplotlib.pyplot as plt
>>> from src.mbe_loss import MBELoss
>>> from src.layer import MBELayer
>>> from src.dataset import *
>>> ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
>>> n, problem = build_dataset2()                               # 获取问题
>>> loss = MBELoss(n, 7)                                        # 生成损失函数
>>> loss.set_graph(problem)                                     # 问题绑定
>>> ms.set_seed(1202)                                           # 设置随机数种子
>>> net = MBELayer(loss)                                        # 训练网络
>>> opti = nn.Adam(net.trainable_params(), learning_rate=0.05)  # 优化器
>>> train_net = nn.TrainOneStepCell(net, opti)                  # 单步训练器
>>> for i in range(50):                                         # 训练
        c = train_net().asnumpy()[0]
        w = net.weight.asnumpy()
        r = np.concatenate(loss.measure(w))
        s = score(problem, np.sign(r))
        print("train step:", i, ", loss:", c, "score:", s)
