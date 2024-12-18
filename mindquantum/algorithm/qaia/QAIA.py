# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""The base class of QAIA."""
# pylint: disable=invalid-name
import numpy as np
from scipy import sparse as sp
from mindquantum.utils.type_value_check import _check_int_type, _check_value_should_not_less


class QAIA:
    r"""
    The base class of QAIA.

    This class contains the basic and common functions of all the algorithms.

    Note:
        For memory efficiency, the input array 'x' is not copied and will be modified
        in-place during optimization. If you need to preserve the original data,
        please pass a copy using `x.copy()`.

    Args:
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N x 1).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, J, h=None, x=None, n_iter=1000, batch_size=1):
        """Construct a QAIA algorithm."""
        if not isinstance(J, (np.ndarray, sp.spmatrix)):
            raise TypeError(f"J requires numpy.array or scipy sparse matrix, but get {type(J)}")
        if len(J.shape) != 2 or J.shape[0] != J.shape[1]:
            raise ValueError(f"J must be a square matrix, but got shape {J.shape}")

        if h is not None:
            if not isinstance(h, np.ndarray):
                raise TypeError(f"h requires numpy.array, but get {type(h)}")
            if h.shape != (J.shape[0],) and h.shape != (J.shape[0], 1):
                raise ValueError(f"h must have shape ({J.shape[0]},) or ({J.shape[0]}, 1), but got {h.shape}")
            if len(h.shape) == 1:
                h = h[:, np.newaxis]

        if x is not None:
            if not isinstance(x, np.ndarray):
                raise TypeError(f"x requires numpy.array, but get {type(x)}")
            if len(x.shape) != 2:
                raise ValueError(f"x must be a 2D array, but got shape {x.shape}")
            if x.shape[0] != J.shape[0] or x.shape[1] != batch_size:
                raise ValueError(f"x must have shape ({J.shape[0]}, {batch_size}), but got {x.shape}")

        _check_int_type("n_iter", n_iter)
        _check_value_should_not_less("n_iter", 1, n_iter)
        _check_int_type("batch_size", batch_size)
        _check_value_should_not_less("batch_size", 1, batch_size)

        self.J = J
        self.h = h
        self.x = x
        # The number of spins
        self.N = self.J.shape[0]
        self.n_iter = n_iter
        self.batch_size = batch_size

    def initialize(self):
        """Randomly initialize spin values."""
        if self.x is None:
            self.x = 0.02 * (np.random.rand(self.N, self.batch_size) - 0.5)

    def calc_cut(self, x=None):
        r"""
        Calculate cut value.

        Args:
            x (numpy.array): The spin value with shape (N x batch_size).
                If ``None``, the initial spin will be used. Default: ``None``.
        """
        if x is None:
            sign = np.sign(self.x)
        else:
            sign = np.sign(x)

        return 0.25 * np.sum(self.J.dot(sign) * sign, axis=0) - 0.25 * self.J.sum()

    def calc_energy(self, x=None):
        r"""
        Calculate energy.

        Args:
            x (numpy.array): The spin value with shape (N x batch_size).
                If ``None``, the initial spin will be used. Default: ``None``.
        """
        if x is None:
            sign = np.sign(self.x)
        else:
            sign = np.sign(x)

        if self.h is None:
            return -0.5 * np.sum(self.J.dot(sign) * sign, axis=0)
        return -0.5 * np.sum(self.J.dot(sign) * sign, axis=0, keepdims=True) - self.h.T.dot(sign)


class OverflowException(Exception):
    r"""
    Custom exception class for handling overflow errors in numerical calculations.

    Args:
        message: Exception message string, defaults to "Overflow error".
    """

    def __init__(self, message="Overflow error"):
        self.message = message
        super().__init__(self.message)
