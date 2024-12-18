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
"""Simulated Coherent Ising Machine."""
# pylint: disable=invalid-name
import numpy as np
from scipy.sparse import csr_matrix

from mindquantum.utils.type_value_check import (
    _check_number_type,
    _check_value_should_not_less,
    _check_value_should_between_close_set,
)
from .QAIA import QAIA


class SimCIM(QAIA):
    r"""
    Simulated Coherent Ising Machine.

    Reference: `Annealing by simulating the coherent Ising
    machine <https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-7-10288&id=408024>`_.

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
        dt (float): The step size. Default: ``1``.
        momentum (float): momentum factor. Default: ``0.9``.
        sigma (float): The standard deviation of noise. Default: ``0.03``.
        pt (float): Pump parameter. Default: ``6.5``.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.qaia import SimCIM
        >>> J = np.array([[0, -1], [-1, 0]])
        >>> solver = SimCIM(J, batch_size=5)
        >>> solver.update()
        >>> print(solver.calc_cut())
        [1. 1. 1. 0. 0.]
        >>> print(solver.calc_energy())
        [-1. -1. -1.  1.  1.]
    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes
    def __init__(
        self,
        J,
        h=None,
        x=None,
        n_iter=1000,
        batch_size=1,
        dt=0.01,
        momentum=0.9,
        sigma=0.03,
        pt=6.5,
    ):
        """Construct SimCIM algorithm."""
        _check_number_type("dt", dt)
        _check_value_should_not_less("dt", 0, dt)

        _check_number_type("momentum", momentum)
        _check_value_should_between_close_set("momentum", 0, 1, momentum)

        _check_number_type("sigma", sigma)
        _check_value_should_not_less("sigma", 0, sigma)

        _check_number_type("pt", pt)
        _check_value_should_not_less("pt", 0, pt)

        super().__init__(J, h, x, n_iter, batch_size)
        self.J = csr_matrix(self.J)
        self.dt = dt
        self.momentum = momentum
        self.sigma = sigma
        self.pt = pt
        self.initialize()

    def initialize(self):
        """Initialize spin."""
        # Initialization of spin value
        if self.x is None:
            self.x = np.zeros((self.N, self.batch_size))
        # gradient
        self.dx = np.zeros_like(self.x)
        if self.x.shape[0] != self.N:
            raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")
        # pump-loss factor
        self.p_list = (np.tanh(np.linspace(-3, 3, self.n_iter)) - 1) * self.pt

    # pylint: disable=attribute-defined-outside-init
    def update(self):
        """Dynamical evolution."""
        for _, p in zip(range(self.n_iter), self.p_list):
            if self.h is None:
                newdc = self.x * p + (
                    self.J.dot(self.x) * self.dt + np.random.normal(size=(self.N, self.batch_size)) * self.sigma
                )
            else:
                newdc = self.x * p + (
                    (self.J.dot(self.x) + self.h) * self.dt
                    + np.random.normal(size=(self.N, self.batch_size)) * self.sigma
                )
            # gradient + momentum
            self.dx = self.dx * self.momentum + newdc * (1 - self.momentum)
            ind = (np.abs(self.x + self.dx) < 1.0).astype(np.int64)
            self.x += self.dx * ind
