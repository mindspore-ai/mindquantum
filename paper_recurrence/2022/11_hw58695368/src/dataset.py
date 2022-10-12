# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
import numpy as np
def score(problem, res):
    """Scoring."""
    s = 0
    for p in problem:
        s += (1 - res[p[0]] * res[p[1]]) * p[2] / 2
    return s
def build_dataset_parallel(n1, problem1, n2, problem2):
    """
    Build dataset for MBE MaxCut Solver in parallel.

    Args:
        n1 (int): Nodes of graph.
        problem1 (list): Edge of graph.
        n2 (int): Nodes of graph.
        problem2 (list): Edge of graph.
    """
    if n1 >= n2:
        n = n1
        problem = np.array(problem2) + [n, n, 0]
        problem = np.concatenate((problem1, problem))
        od = [0, 1]
    else:
        n = n2
        problem = np.array(problem1) + [n, n, 0]
        problem = np.concatenate((problem2, problem))
        od = [1, 0]
    return n * 2, problem.tolist(), od
# problem (list): [[node1, node2, weight], ...]
def build_dataset1():
    """
    Build dataset.
    n = 8.
    maxcut: 10.
    method: {0, 1}, {2, 3, 4, 5, 6, 7}.
    """
    n = 8
    problem = [[0, 1, 1],
               [0, 2, 1],
               [0, 3, 1],
               [0, 4, 1],
               [0, 5, 1],
               [0, 6, 1],
               [1, 3, 1],
               [1, 4, 1],
               [1, 5, 1],
               [1, 6, 1],
               [1, 7, 1],
              ]
    return n, problem
def build_dataset2():
    """
    Build dataset.
    n = 8.
    Bigraph: G = <V1,E,V2>
             V1 = {0, 2, 3, 6}
             V2 = {1, 4, 5, 7}
    maxcut: 16.
    """
    n = 8
    problem = [[0, 1, 1],
               [0, 4, 1],
               [0, 5, 1],
               [0, 7, 1],
               [2, 1, 1],
               [2, 4, 1],
               [2, 5, 1],
               [2, 7, 1],
               [3, 1, 1],
               [3, 4, 1],
               [3, 5, 1],
               [3, 7, 1],
               [6, 1, 1],
               [6, 4, 1],
               [6, 5, 1],
               [6, 7, 1],
              ]
    return n, problem
def build_dataset3():
    """
    Build dataset.
    n = 10.
    3-regular graph.
    maxcut: 12.
    """
    n = 10
    problem = [[0, 2, 1],
               [0, 5, 1],
               [0, 6, 1],
               [1, 6, 1],
               [1, 7, 1],
               [1, 8, 1],
               [2, 5, 1],
               [2, 6, 1],
               [3, 4, 1],
               [3, 8, 1],
               [3, 9, 1],
               [4, 5, 1],
               [4, 9, 1],
               [7, 8, 1],
               [7, 9, 1],
              ]
    return n, problem
