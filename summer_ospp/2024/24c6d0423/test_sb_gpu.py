import warnings
warnings.filterwarnings('ignore')

from mindquantum.algorithm.qaia import ASB, BSB, DSB, SimCIM, NMFA, BSB_INT8, BSB_HALF, DSB_INT8, DSB_HALF

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import time

# TEST
# From : mindquantum/tests/st/test_algorithm/test_qaia/test_qaia.py
# from pathlib import Path
# from mindquantum.utils.fdopen import fdopen
def test_bSB():
    """
    Description: Test BSB
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    x = 0.01 * (np.random.rand(N, 1) - 0.5)
    y = 0.01 * (np.random.rand(N, 1) - 0.5)
    solver = BSB(G, n_iter=1)
    solver.x = x.copy()
    solver.y = y.copy()
    solver.update()
    y += (-(1 - solver.p[0]) * x + solver.xi * G @ x) * solver.dt
    x += y * solver.dt
    x = np.where(np.abs(x) > 1, np.sign(x), x)
    assert np.allclose(x, solver.x)

def test_dSB():
    """
    Description: Test DSB
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    x = 0.01 * (np.random.rand(N, 1) - 0.5)
    y = 0.01 * (np.random.rand(N, 1) - 0.5)
    solver = DSB(G, n_iter=1)
    solver.x = x.copy()
    solver.y = y.copy()
    solver.update()
    y += (-(1 - solver.p[0]) * x + solver.xi * G @ np.sign(x)) * solver.dt
    x += y * solver.dt
    x = np.where(np.abs(x) > 1, np.sign(x), x)
    assert np.allclose(x, solver.x)

def read_gset(filename, negate=True):
    # 读取图表
    graph = pd.read_csv(filename, sep=' ')
    # 节点的数量
    n_v = int(graph.columns[0])
    # 边的数量
    n_e = int(graph.columns[1])

    # 如果节点和边不匹配，会抛出错误
    assert n_e == graph.shape[0], 'The number of edges is not matched'

    # 将读取的数据转换为一个COO矩阵（Coordinate List Format），并返回一个稀疏矩阵
    G = coo_matrix((np.concatenate([graph.iloc[:, -1], graph.iloc[:, -1]]),
                        (np.concatenate([graph.iloc[:, 0]-1, graph.iloc[:, 1]-1]),
                         np.concatenate([graph.iloc[:, 1]-1, graph.iloc[:, 0]-1]))
                    ), shape=(n_v, n_v))
    if negate:
        G = -G

    return G

def show_res(s):
    cut_value_list = np.array(s.calc_cut())
    mean_cut = np.mean(cut_value_list)
    max_cut = np.max(cut_value_list)
    energy_list = np.array(s.calc_energy())
    mean_energy = np.mean(energy_list)
    max_energy = np.max(energy_list)
    print(f'cut mean_val:', mean_cut)
    # print(f'cut max_val:', max_cut)
    print(f'energy mean_val:', mean_energy)
    # print(f'energy max_val:', max_energy)


G = read_gset("./G22.txt")
# G = read_gset("./G39.txt")
# G = read_gset("./G81.txt")

s = BSB_INT8(G, batch_size=100, n_iter=1000)
s.h = np.random.random(s.x.shape)
h = s.h
print("------------ BSB_INT8 -------------")
stime = time.time()
s.update()
dtime = time.time()
print(f"run time: {dtime-stime:.3f}s")
show_res(s)

s = BSB_HALF(G, batch_size=100, n_iter=1000)
s.h = h
print("------------ BSB_HALF ------------")
stime = time.time()
s.update()
dtime = time.time()
print(f"run time: {dtime-stime:.3f}s")
show_res(s)

s = BSB(G, batch_size=100, n_iter=1000)
s.h = h
print("------------ BSB cpu  ------------")
stime = time.time()
s.update()
dtime = time.time()
print(f"run time: {dtime-stime:.3f}s")
show_res(s)

s = DSB_INT8(G, batch_size=100, n_iter=1000)
print("------------ DSB_INT8 -------------")
stime = time.time()
s.update()
dtime = time.time()
print(f"run time: {dtime-stime:.3f}s")
show_res(s)

s = DSB_HALF(G, batch_size=100, n_iter=1000)
print("------------ DSB_HALF ------------")
stime = time.time()
s.update()
dtime = time.time()
print(f"run time: {dtime-stime:.3f}s")
show_res(s)

s = DSB(G, batch_size=100, n_iter=1000)
print("------------ DSB cpu ------------")
stime = time.time()
s.update()
dtime = time.time()
print(f"run time: {dtime-stime:.3f}s")
show_res(s)