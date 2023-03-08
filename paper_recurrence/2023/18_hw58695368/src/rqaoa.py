# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
import pickle
from src.eliminate_variable import eliminate_variable
from src.qaoa import qaoa, get_partition
from src.utils import maxcut_enum
from src.config import rqaoa_nc

def rqaoa(g, nc=None, level=None, iter_show=True):
    """
    See FIG. 2 in article [1].

    [1] Obstacles to State Preparation and Variational Optimization from Symmetry Protection

    Args:
        g (Graph): The graph structure. Every element of graph
            is a edge that constructed by two nodes and one weight.
        nc (int): The cutoff value for variable elimination. See [1].
        level (int): The depth of QAOA circuit.
        iter_show (bool): Default: True.
    """
    nc = nc or rqaoa_nc
    Xi = []
    print('RQAOA started!')
    while len(g.nodes) > nc:
        g, xi = _eliminate_variable(g, None, iter_show)
        Xi.append(xi)
    res = rqaoa_translate(g, Xi, iter_show)
    print('--------------------------')
    print(f'Result of RQAOA: {res}')
    print('--------------------------')
    print('RQAOA ended!')
    return res

def _eliminate_variable(g, level=None, iter_show=True):
    if iter_show:
        print('--------------------------')
        print(f'nodes_num: {len(g.nodes)}')
        print(f'edges_num: {len(g.edges)}')
    g, xi = eliminate_variable(g, level)
    if iter_show:
        print(f'>>> eliminated variable: {xi[0]}')
        print(f'>>> correlated variable: {xi[1]}')
        print(f'>>> σ: {xi[2]}')
    return g, xi

def rqaoa_recursion(g, nc, Xi=[], level=None, iter_show=True, checkpoint=None):
    """
    See `src.rqaoa.rqaoa` for more information.

    You can save the checkpoint during execution.

    Args:
        g (Graph): The graph structure. Every element of graph
            is a edge that constructed by two nodes and one weight.
        nc (int): The cutoff value for variable elimination. See [1].
        Xi (list): The restricted set.
        level (int): The depth of QAOA circuit.
        iter_show (bool): Default: True.
        checkpoint (str): Checkpoint save path.
    """
    if len(g.nodes) <= nc:
        return g, Xi
    g, xi = _eliminate_variable(g, level, iter_show)
    Xi.append(xi)
    if checkpoint is not None:
        cp = {'g': g, 'Xi': Xi}
        f = open(f'{checkpoint}_to_n{len(g.nodes)}.pkl', 'wb')
        pickle.dump(cp, f)
        f.close()
    return rqaoa_recursion(g, nc, Xi, level, iter_show, checkpoint)

def rqaoa_translate(g, Xi, iter_show=True):
    """
    Any maximum x' of C' can directly be translated to
    a corresponding maximum of C over the restricted set.

    See [1] for more information.

    [1] Obstacles to State Preparation and Variational Optimization from Symmetry Protection

    Args:
        g (Graph): The graph structure. Every element of graph
            is a edge that constructed by two nodes and one weight.
        Xi (list): The restricted set.
        iter_show (bool): Default: True.
    """
    #res = _maxcut_qaoa(g)
    res = _maxcut_enum(g)
    if iter_show:
        print('--------------------------')
        print(f'nodes_num: {len(g.nodes)}')
        print(f'edges_num: {len(g.edges)}')
        print(f'Result of MaxCut: {res}')
    for xi in Xi[::-1]:
        res[xi[0]] = res[xi[1]] * xi[2]
    return res

def _maxcut_qaoa(g):
    """Solving MaxCut problem with QAOA."""
    circ, pr = qaoa(g, 'J', 8, 200)
    return get_partition(g, circ, pr)[0]
def _maxcut_enum(g):
    """Solving MaxCut problem with enum."""
    _, n = maxcut_enum(g, 'J')
    res = dict()
    for i in g.nodes:
        res[i] = -1 if i in n else 1
    return res
