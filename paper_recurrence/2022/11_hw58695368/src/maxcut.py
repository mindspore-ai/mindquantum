# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
import numpy as np
from scipy.optimize import minimize
from src.mbe_loss import MBELoss
def func(w, loss, grad=False, show_iter_val=False):
    """Loss function."""
    f, g = loss.get_loss(w, grad)
    if show_iter_val:
        print('loss:', f)
    if grad:
        return np.real(np.squeeze(f)), np.real(np.squeeze(g))
    return np.real(np.squeeze(f))
def maxcut(n, depth, problem, weight=None,
           grad=False, show_iter_val=False, **kwargs):
    """
    Solving the maxcut problem.

    Args:
        n (int): Nodes of graph.
        depth (int): Depth of circuit.
        problem (list): Graph list [[node1, node2, weight], ...].
        weight (np.ndarray): Initialization parameters.
        grad (bool): Loss function returns gradient or not.
        show_iter_val (bool): Print iteration process or not.
        kwargs (dict): Args of `minimize`.

    Returns:
        Object of `MBELoss`.
        Solution of maxcut problem
    """
    loss = MBELoss(n, depth)
    loss.set_graph(problem)
    if weight is None:
        weight = np.random.rand(len(loss.circ.params_name)) * np.pi
    res = minimize(func, weight,
                   args=(loss, grad, show_iter_val),
                   **kwargs
                  )
    print(res)
    r = np.concatenate(loss.measure(res.x))
    return loss, res.x, np.sign(r)
