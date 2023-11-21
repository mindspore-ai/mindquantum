"""Time evolution of quantum state."""

import numpy as np
import mindspore.ops as ops


def cmatmul(m1, m2):
    """Matrix multiply for complex number."""
    re = ops.matmul(m1[0], m2[0]) - ops.matmul(m1[1], m2[1])
    im = ops.matmul(m1[0], m2[1]) + ops.matmul(m1[1], m2[0])
    return re, im


def ctranspose(m, axis=None):
    """Transpose for complex number."""
    return m[0].transpose(axis), m[1].transpose(axis)


def creshape(m, shape):
    """Reshape for complex number."""
    return m[0].reshape(shape), m[1].reshape(shape)


def get_permute_index(n_qudit, obj_qudit, ctrl_qudit=[]):
    """Get the permute index."""
    mid_qudit = [i for i in range(n_qudit) if i not in obj_qudit + ctrl_qudit]
    idx1 = obj_qudit + mid_qudit + ctrl_qudit
    idx2 = np.argsort(idx1).tolist()
    return idx1, idx2


def state_evolution(mat, qs, obj_qudits, ctrl_qudits=[], ctrl_states=[]):
    """State revolution after act specific qudit gate."""
    n_qudits = qs[0].dim()
    idx1, idx2 = get_permute_index(n_qudits, obj_qudits, ctrl_qudits)
    qs2 = ctranspose(qs, idx1)
    select_idx = [...] + ctrl_states
    qs2_part = (qs2[0][select_idx], qs2[1][select_idx])
    sh = qs2[0][select_idx].shape
    qs2_part = creshape(cmatmul(mat, creshape(
        qs2_part, (mat[0].shape[0], -1))), sh)
    qs2[0][select_idx] = qs2_part[0]
    qs2[1][select_idx] = qs2_part[1]
    qs2 = ctranspose(qs2, idx2)
    return qs2
