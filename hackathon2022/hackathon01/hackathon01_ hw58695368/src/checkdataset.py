# -*- coding: utf-8 -*-
"""
Author: NoEvaa
Description: 数据集质量检查.
FilePath: /src/checkdataset.py
"""

import numpy as np

def encode(x): # 编码: 01矩阵(list 4*4) -> 二进制(str)
    s = ''
    for j in x:
        for k in j:
            for m in k:
                s += str(int(m))
    return s
    return int(s, 2)

def decode(n): # 解码 二进制(str) -> 01矩阵(list 4*4)
    s = bin(n)[2:]
    i = []
    j = []
    for k in range(16-len(s)):
        j.append([0])
        if len(j) == 4:
            i.append(j)
            j = []
    for k in s:
        j.append([int(k)])
        if len(j) == 4:
            i.append((j))
            j = []
    return i
    return np.ndarray(i, dtype=np.float32)

def calcu_acc(x, y, c): # 通过 非重复项预测结果 计算 模型在训练集上的Acc
    if len(x) != len(y):
        raise
    acc = 0
    for i in range(len(y)):
        s = int(encode(x[i]), 2)
        t = (s, y[i])
        acc += c.get(t, 0)
        c[t] = 0
    return acc / 5000

if __name__ == '__main__':
    # 训练集检查
    origin_test_data = np.load("train.npy", allow_pickle=True)[0]
    origin_test_x = origin_test_data['train_x']
    origin_test_y = origin_test_data['train_y']

    c = {} # 非重复项统计
    for i in range(5000):
        s = int(encode(origin_test_x[i]), 2)
        t = (s, origin_test_y[i])
        c[t] = c.get(t, 0) + 1

    l1 = sorted(list(c.keys())) # 非重复项

    r = [] # 矛盾项
    for i in range(len(l1)-1):
        if l1[i][0] == l1[i+1][0]:
            r.append(l1[i][0])

    m = 0 # 最小矛盾数
    for i in r:
        m += min(c[(i, True)], c[(i, False)])

    o = [len(l1), len(r)]
    print('训练集统计结果')
    print(' 非重复数据:', o[0])              # 182
    print(' 矛盾数据:', o[1])                # 31
    print(' 有效数据:', o[0] - 2 * o[1])     # 120
    print(' 最大精确度', 1 - m / 5000)       # 0.9184

    '''
    import pickle

    # 生成非重复项测试集并保存
    tx, ty = [], []
    for i in l1:
        tx.append(decode(i[0]))
        ty.append(i[1])
    td = {
          'test_x':np.array(tx, dtype=np.float32),
          'test_y':np.array(ty)
          }

    with open ("test.pkl", 'wb') as f:
        pickle.dump(td, f)

    # 保存非重复项统计结果
    with open ("train_statistics.pkl", 'wb') as f:
        pickle.dump(c, f)
    '''
'''
# 测试集检查
origin_test_data = np.load("test.npy", allow_pickle=True)[0]
origin_test_x = origin_test_data['test_x']
origin_test_y = origin_test_data['test_y']
ll = []
for i in range(800):
    x = origin_test_x[i]
    s = encode(origin_test_x[i])
    ll.append((int(s, 2), origin_test_y[i]))
ll = list(set(ll))
ll = sorted(ll)

#>>> ll
# Out: [(0, False)]
'''
