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

from .QAIA import QAIA


class SimCIM(QAIA):
    r"""
    Simulated Coherent Ising Machine.

    Reference: `Annealing by simulating the coherent Ising
    machine <https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-7-10288&id=408024>`_.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        momentum (float): momentum factor. Default: ``0.9``.
        sigma (float): The standard deviation of noise. Default: ``0.03``.
        pt (float): Pump parameter. Default: ``6.5``.
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
            ind = (np.abs(self.x + self.dx) < 1.0).astype(np.int)
            self.x += self.dx * ind
