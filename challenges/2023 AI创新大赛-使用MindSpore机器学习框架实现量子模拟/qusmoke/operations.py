"""Operations for complex number."""

from typing import List

import numpy as np
import mindspore.ops as ops


def cmatmul(m1, m2):
    """Matrix multiply for complex number.
    """
    re = ops.matmul(m1[0], m2[0]) - ops.matmul(m1[1], m2[1])
    im = ops.matmul(m1[0], m2[1]) + ops.matmul(m1[1], m2[0])
    return re, im

def ctranspose(m, axis=None):
    """Transpose for complex number.
    """
    return m[0].transpose(axis), m[1].transpose(axis)

def creshape(m, shape):
    """Reshape for complex number.
    """
    return m[0].reshape(shape), m[1].reshape(shape)

def cslice(m, index):
    """Slice  for complex number.
    """
    return m[0][index], m[1][index]

def cdot(v1, v2):
    """Dot for complex number.
    """
    return (v1[0] * v2[0] - v1[1] * v2[1]).sum(), (v1[0] * v2[1] + v1[1] * v2[0]).sum()

def cconjugate(m):
    """Conjugate for complex number.
    """
    return m[0], -m[1]

def cravel(m):
    """Ravel(flatten) for complex number.
    """
    return m[0].ravel(), m[1].ravel()

def cstack(m1, m2, axis=0):
    """Stack for complex number.
    """
    return ops.stack([m1[0], m2[0]], axis), ops.stack([m1[1], m2[1]], axis)


def get_transpose_index(n_qubit, obj_qubit, ctrl_qubit=None):
    assert obj_qubit < n_qubit, "The `obj_qubit` should less than `n_qubit`."

    if ctrl_qubit is None:
        idx1 = np.roll(range(n_qubit), obj_qubit).tolist()
        idx2 = np.roll(range(n_qubit), -obj_qubit).tolist()
        return idx1, idx2
    elif isinstance(ctrl_qubit, int):
        obj_q = n_qubit - obj_qubit - 1
        ctrl_q = n_qubit - ctrl_qubit - 1
        idx1 = [obj_q]
        for i in range(n_qubit):
            if i not in [obj_q, ctrl_q]:
                idx1.append(i)
        idx1.append(ctrl_q)
        idx2 = np.argsort(idx1).tolist()
        return idx1, idx2
    elif isinstance(ctrl_qubit, list):
        obj_q = n_qubit - obj_qubit - 1
        ctrl_q = [n_qubit - c - 1 for c in ctrl_qubit]
        idx1 = [obj_q]
        for i in range(n_qubit):
            if i not in [obj_q] + ctrl_q:
                idx1.append(i)
        idx1.extend(ctrl_q)
        idx2 = np.argsort(idx1).tolist()
        return idx1, idx2


def get_transpose_index_for_zz(n_qubit, obj_qubit1, obj_qubit2):
    assert max(obj_qubit1, obj_qubit2) < n_qubit, "The `obj_qubit` should less than `n_qubit`."

    obj_q1 = n_qubit - obj_qubit1 - 1
    obj_q2 = n_qubit - obj_qubit2 - 1
    idx1 = []
    for i in range(n_qubit):
        if i not in [obj_q1, obj_q2]:
            idx1.append(i)
    idx1.extend([obj_q1, obj_q2])
    idx2 = np.argsort(idx1).tolist()
    return idx1, idx2
