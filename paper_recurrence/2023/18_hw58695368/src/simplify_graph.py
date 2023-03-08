# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
from src.mygraph import MyGraph
from src.utils import rand_bin

def simplify_graph(g, f, sigma):
    """
    Simplify the graph with Eqs. (C3), (C4), (C5), (C6) in article [1].

    [1] Obstacles to State Preparation and Variational Optimization from Symmetry Protection
    """
    f = set(f)
    # Pick v∈f arbitrarily
    v = tuple(f)[rand_bin()]
    # (C3) (C4)
    g_new = MyGraph()
    for e in g.edges:
        if v in e:
            e = set(e)
            if e != f:
                 # (C6)
                g_new.add_edge_as(*tuple(e^f), J=sigma*g.edges[e]['J'])
            continue
        # (C5)
        g_new.add_edge_as(*e, J=g.edges[e]['J'])
    g_new.clean_edge('J')
    g_new.add_nodes_from(set(g.nodes)^{v}^set(g_new.nodes)) # Fix: noEdge node loss
    return g_new, v


def __main():
    Es = [(0, 1, 2),
          (1, 2, 2),
          (3, 2, 2),
          (3, 4, 2),
          (0, 2, 2),
          (0, 2, -1),
          (4, 3, -2),]
    g = MyGraph()
    for e in Es:
        g.add_edge_as(e[0], e[1], J=e[2])
    g.clean_edge('J')
    print(g.nodes())
    print(g.edges(data=True))

    f = {0, 1}
    sigma = -1
    gn, v = simplify_graph(g, f, sigma)
    print(v)
    print(gn.nodes())
    print(gn.edges(data=True))

if __name__ == '__main__':
    __main()
