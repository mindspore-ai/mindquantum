# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
import numpy as np
from src.simplify_graph import simplify_graph
from src.qaoa import qaoa, correlate_edges

def eliminate_variable(g, level=None):
    """
    [1] Obstacles to State Preparation and Variational Optimization from Symmetry Protection

    Returns:
        G', (v, f\{v}, σ)
    """
    circ, pr = qaoa(g, 'J', level)
    Me, E = correlate_edges(g, circ, pr)
    i = np.abs(Me).argmax()
    f = set(E[i])
    sigma = int(np.sign(Me[i].real))
    g_new, v = simplify_graph(g, f, sigma)
    return g_new, (v, list(f^set([v]))[0], sigma)
