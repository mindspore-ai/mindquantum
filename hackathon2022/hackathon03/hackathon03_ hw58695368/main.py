# -*- coding: utf-8 -*-
"""
Just execute this file.

@NoEvaa
"""
import numpy as np
from solution import Hackathon03

train_x = np.load('train_x.npy', allow_pickle=True)
train_y = np.load('train_y.npy', allow_pickle=True)
model = Hackathon03()
model.train(train_x[:20], train_y[:20], train_x, train_y,
            acc_tol=1, iter_info=True, tol=1e-5)
model.save('s_J49U26A03')

# 由于Hackathon03的predict未开发完整(懒), 将使用`predict.py`生成`test_y.npy`
