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
"""Noisy Mean-field Annealing algorithm."""
# pylint: disable=invalid-name
import numpy as np
from scipy.sparse import csr_matrix

from mindquantum.utils.type_value_check import (
    _check_number_type,
    _check_value_should_between_close_set,
    _check_value_should_not_less,
)
from .QAIA import QAIA


class NMFA(QAIA):
    r"""
    Noisy Mean-field Annealing algorithm.

    Reference: `Emulating the coherent Ising machine with a mean-field
    algorithm <https://arxiv.org/abs/1806.08422>`_.

    Note:
        For memory efficiency, the input array 'x' is not copied and will be modified
        in-place during optimization. If you need to preserve the original data,
        please pass a copy using `x.copy()`.

    Args:
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N, ).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            zeros. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        alpha (float): Momentum factor. Default: ``0.15``.
        sigma (float): The standard deviation of noise. Default: ``0.15``.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.qaia import NMFA
        >>> J = np.array([[0, -1], [-1, 0]])
        >>> solver = NMFA(J, batch_size=5)
        >>> solver.update()
        >>> print(solver.calc_cut())
        [1. 1. 1. 1. 1.]
        >>> print(solver.calc_energy())
        [-1. -1. -1. -1. -1.]
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        J,
        h=None,
        x=None,
        n_iter=1000,
        batch_size=1,
        alpha=0.15,
        sigma=0.15,
    ):
        """Construct NMFA algorithm."""
        _check_number_type("alpha", alpha)
        _check_value_should_between_close_set("alpha", 0, 1, alpha)

        _check_number_type("sigma", sigma)
        _check_value_should_not_less("sigma", 0, sigma)

        super().__init__(J, h, x, n_iter, batch_size)
        self.J = csr_matrix(self.J)
        if self.h is None:
            self.J_norm = np.sqrt(np.asarray(csr_matrix.power(self.J, 2).sum(axis=1)))
        else:
            self.J_norm = np.sqrt(np.asarray(csr_matrix.power(self.J, 2).sum(axis=1)) + np.sum(self.h**2))

        self.alpha = alpha
        self.sigma = sigma
        self.initialize()

    def initialize(self):
        """Initialize spin values."""
        # initialize x to zeros
        if self.x is None:
            self.x = np.zeros((self.N, self.batch_size))

        if self.x.shape[0] != self.N:
            raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")
        # inverse temperature
        self.beta = 1 / self.n_iter

    def update(self):
        """Dynamical evolution."""
        for _ in range(self.n_iter):
            if self.h is None:
                phi = self.J.dot(self.x) / self.J_norm + np.random.normal(0, self.sigma, size=self.x.shape)
            else:
                phi = (self.J.dot(self.x) + self.h) / self.J_norm + np.random.normal(0, self.sigma, size=self.x.shape)

            x_hat = np.tanh(phi * self.beta)

            self.x = self.alpha * x_hat + (1 - self.alpha) * self.x

            self.beta += 1 / self.n_iter
