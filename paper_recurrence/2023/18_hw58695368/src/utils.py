# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
# 二值随机数发生器
import random
from src.config import rand_generator, sim_backend
def rand_bin(m=None):
    """
    Binary random number generator.
    """
    m = m or rand_generator
    m = m.upper()
    if m == 'C':
        return random.randint(0, 1)
    elif m == 'Q':
        return _rand_bin_qm()
    else:
        return None

from mindquantum.core.gates import H
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit
def _rand_bin_qm():
    circ = Circuit()
    circ += H.on(0)
    circ.measure_all()
    sim = Simulator(sim_backend, 1)
    res = sim.apply_circuit(circ).data
    return int(tuple(res.keys())[0])

# 图结构生成器
from src.mygraph import MyGraph
def generate_graph(node_num):
    """
    Generate the graph structure.
    Random weights w∈{+1, -1}.

    Modified from [1].
    [1] https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/2022/11_%20xhliang05/main.ipynb

    Args:
        node_num (int): Number of nodes.
    """
    edges = []
    # 添加环边
    for node in range(node_num-1): 
        edges.append([node, node+1, 2*rand_bin()-1])
    edges.append([node_num-1, 0, 2*rand_bin()-1])
     # 添加内边
    if node_num % 2 == 0: # 如果顶点数为偶数，就采用 3-regular 结构
        for node in range(int(node_num/2)):
            edges.append([node, int(node+node_num/2), 2*rand_bin()-1])
    else:                 # 如果顶点为奇数，只留一个顶点有两条边，其他为三条边
        for node in range(int((node_num-1)/2)): 
            edges.append([node, int(node+(node_num-1)/2), 2*rand_bin()-1])
    return to_graph(edges)
def to_graph(edges):
    """Convert edges to graph."""
    g = MyGraph()
    for e in edges:
        g.add_edge_as(e[0], e[1], J=e[2])
    return g

# 枚举法求解MaxCut问题
import networkx as nx
def maxcut_enum(g, weight='J'):
    """Solving MaxCut problem with enum."""
    nodes = list(g.nodes)
    n = len(nodes)
    return _maxcut_enum(g, weight, nodes, (0, []), n, (n + 1) // 2, [])
def _maxcut_enum(g, weight, nodes, max_cut, u, c, s):
    if c > 0:
        for v in range(u):
            s_ = [*s, nodes[v]]
            max_cut = max(max_cut, (nx.cut_size(g, s_, weight=weight), s_), key=lambda x:x[0])
            max_cut = max(max_cut, _maxcut_enum(g, weight, 
                                                nodes, max_cut, v, c-1, s_), key=lambda x:x[0])
    return max_cut
