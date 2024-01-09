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
""".Local quantum annealing algorithm."""
# pylint: disable=invalid-name
import numpy as np
from scipy.sparse import csr_matrix

from .QAIA import QAIA


class LQA(QAIA):
    r"""
    Local quantum annealing algorithm.

    Reference: `Quadratic Unconstrained Binary Optimization via Quantum-Inspired
    Annealing <https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.18.034016>`_.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        gamma (float): The coupling strength. Default: ``0.1``.
        momentum (float): Momentum factor. Default: ``0.99``.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        J,
        h=None,
        x=None,
        n_iter=1000,
        batch_size=1,
        gamma=0.1,
        dt=1.0,
        momentum=0.99,
    ):
        """Construct LQA algorithm."""
        super().__init__(J, h, x, n_iter, batch_size)
        self.J = csr_matrix(self.J)
        self.gamma = gamma
        self.dt = dt
        self.momentum = momentum

        self.initialize()

    def initialize(self):
        """Initialize spin values."""
        if self.x is None:
            self.x = 0.2 * (np.random.rand(self.N, self.batch_size) - 0.5)

        if self.x.shape[0] != self.N:
            raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")

    def update(self, beta1=0.9, beta2=0.999, epsilon=10e-8):
        """
        Dynamical evolution with Adam.

        Args:
            beta1 (float): Beta1 parameter. Default: ``0.9``.
            beta2 (float): Beta2 parameter. Default: ``0.999``.
            epsilon (float): Epsilon parameter. Default: ``10e-8``.
        """
        m_dx = 0
        v_dx = 0

        for i in range(1, self.n_iter):
            t = i / self.n_iter
            tmp = np.pi / 2 * np.tanh(self.x)
            z = np.sin(tmp)
            y = np.cos(tmp)
            if self.h is None:
                dx = np.pi / 2 * (-t * self.gamma * self.J.dot(z) * y + (1 - t) * z) * (1 - np.tanh(self.x) ** 2)
            else:
                dx = (
                    np.pi
                    / 2
                    * (-t * self.gamma * (self.J.dot(z) + self.h) * y + (1 - t) * z)
                    * (1 - np.tanh(self.x) ** 2)
                )

            # momentum beta1
            m_dx = beta1 * m_dx + (1 - beta1) * dx
            # rms beta2
            v_dx = beta2 * v_dx + (1 - beta2) * dx**2
            # bias correction
            m_dx_corr = m_dx / (1 - beta1**i)
            v_dx_corr = v_dx / (1 - beta2**i)

            self.x = self.x - self.dt * m_dx_corr / (np.sqrt(v_dx_corr) + epsilon)
