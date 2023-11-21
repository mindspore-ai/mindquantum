
from typing import List, Tuple

import numpy as np
import mindspore.ops as ops
from mindspore import Tensor

from quditop.global_var import DTYPE


def ket(i, dim):
    """Get the numerical column vector.

    Args:
        i: Value of ket.
        dim: Dimension of qudits.
    """
    vec = ops.zeros((dim, 1), dtype=DTYPE)
    vec[i, 0] = 1
    return vec


def bra(i, dim):
    """Get the numerical row vector.

    Args:
        i: Value of ket.
        dim: Dimension of qudits.
    """
    vec = ops.zeros((1, dim), dtype=DTYPE)
    vec[0, i] = 1
    return vec


def check_unitary(u: Tensor, atol=1e-8):
    """Check if input matrix is unitary."""
    u = np.array(u)
    if u.ndim != 2:
        raise ValueError(
            f"Input matrix must be 2-D matrix, but got shape {u.shape}.")
    v = np.matmul(u, u.conj().T)
    flag = np.isclose(v.real, np.eye(len(u)), atol=atol).all()
    if flag and np.iscomplexobj(v):
        flag = flag and np.isclose(v.imag, np.zeros_like(u), atol=atol).all()
    return flag


def get_complex_tuple(mat, shape=None):
    """Convert the input matrix to a tuple which represents the complex matrix.

    Args:
        mat: Input matrix, which can be real or complex, and data type can be numpy.ndarray or
            mindspore.Tensor.
        shape: If not None, reshape the `mat` shape.
    """
    if isinstance(mat, (Tuple, List)):
        assert len(mat) == 2, "The input is real and imaginary part respectively."
        re, im = mat
        if isinstance(re, np.ndarray):
            re = Tensor(re, dtype=DTYPE)
            im = Tensor(im, dtype=DTYPE)
        elif isinstance(re, Tensor):
            re = re.copy()
            im = im.copy()
    elif isinstance(mat, np.ndarray):
        if np.iscomplexobj(mat):
            re = Tensor(mat.real, dtype=DTYPE)
            im = Tensor(mat.imag, dtype=DTYPE)
        else:
            re = Tensor(mat, dtype=DTYPE)
            im = ops.zeros_like(re)
    elif isinstance(mat, Tensor):
        if ops.is_complex(mat):
            re = mat.real.clone().detach().type(DTYPE)
            im = mat.imag.clone().detach().type(DTYPE)
        else:
            re = mat.copy().astype(DTYPE)
            im = ops.zeros_like(re)
    else:
        raise TypeError(f"The type of input `mat` should be numpy.array, Tensor or Tuple[re, im] where "
                        f"re and im are both numpy.array or tensor.tensor, but got type {type(mat)}.")
    if shape:
        re = re.reshape(shape)
        im = im.reshape(shape)
    return re, im
