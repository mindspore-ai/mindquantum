# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Simulated bifurcation (SB) algorithms and its variants."""
# pylint: disable=invalid-name
import math
import numpy as np
from .SB import SB

try:
    import torch

    assert torch.cuda.is_available()
    _INSTALL_TORCH = True
except (ImportError, AssertionError):
    _INSTALL_TORCH = False


class TSB(SB):
    r"""
    Ternary Simulated Bifurcation algorithm.

    Reference: `Quantized Simulated Bifurcation for the Ising Model <https://ieeexplore.ieee.org/document/10231308>`_.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        strategy(str): ternary quantization method, include linear, exponential, logarithmic. Default: ``linear``.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        J,
        h=None,
        x=None,
        n_iter=1000,
        batch_size=1,
        backend="cpu-float32",
        dt=1,
        xi=None,
        strategy="linear",
    ):
        """Construct TSB algorithm."""
        valid_backends = {'cpu-float32', 'gpu-float32'}
        if not isinstance(backend, str):
            raise TypeError(f"backend requires a string, but get {type(backend)}")
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")
        super().__init__(J, h, x, n_iter, batch_size, dt, xi, backend)

        self.strategy = strategy
        self.time_strategy = {
            "linear": self.linear_time,
            "exponential": self.exponential_time,
            "logarithmic": self.logarithmic_time,
        }

        super().initialize()

    def update(self):
        """Dynamical evolution based on Modified explicit symplectic Euler method."""
        time_factor = self.time_strategy[self.strategy]
        if self.backend == "cpu-float32":
            for i in range(self.n_iter):
                tri_x = np.where(
                    np.abs(self.x) <= time_factor(i, self.n_iter),
                    np.array(0.0),
                    np.sign(self.x),
                )

                if self.h is None:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * self.J.dot(tri_x)) * self.dt
                else:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * (self.J.dot(tri_x) + self.h)) * self.dt

                self.x += self.dt * self.y * self.delta
                cond = np.abs(self.x) > 1
                self.x = np.where(cond, np.sign(self.x), self.x)
                self.y = np.where(cond, np.zeros_like(self.y), self.y)

        elif self.backend == "gpu-float32":
            for i in range(self.n_iter):
                tri_x = torch.where(
                    torch.abs(self.x) <= time_factor(i, self.n_iter),
                    torch.tensor(0.0, device="cuda"),
                    torch.sign(self.x),
                )

                if self.h is None:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * torch.sparse.mm(self.J, tri_x)) * self.dt
                else:
                    self.y += (
                        -(self.delta - self.p[i]) * self.x + self.xi * (torch.sparse.mm(self.J, tri_x) + self.h)
                    ) * self.dt

                self.x += self.dt * self.y * self.delta
                cond = torch.abs(self.x) > 1
                self.x = torch.where(cond, torch.sign(self.x), self.x)
                self.y = torch.where(cond, torch.zeros_like(self.y), self.y)

    def linear_time(self, i, n_iter):
        return 0.7 * i / n_iter

    def exponential_time(self, i, n_iter):
        return 0.7 * (2 ** (i / n_iter - 1))

    def logarithmic_time(self, i, n_iter):
        return 0.7 * math.log2(i / n_iter + 1)


class USB(SB):
    r"""
    Uniformly Quantized Simulated Bifurcation.

    Reference: `Quantized Simulated Bifurcation for the Ising Model <https://ieeexplore.ieee.org/document/10231308>`_.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        threshold(int): dynamic threshold. Default: 10.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        J,
        h=None,
        x=None,
        n_iter=1000,
        batch_size=1,
        backend="cpu-float32",
        dt=1,
        xi=None,
        threshold=10,
    ):
        """Construct USB algorithm."""
        valid_backends = {'cpu-float32', 'gpu-float32'}
        if not isinstance(backend, str):
            raise TypeError(f"backend requires a string, but get {type(backend)}")
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")
        super().__init__(J, h, x, n_iter, batch_size, dt, xi, backend)
        self.threshold = threshold

        super().initialize()

    def update(self):
        """Dynamical evolution based on Modified explicit symplectic Euler method."""
        if self.backend == "cpu-float32":
            for i in range(self.n_iter):
                p_x = np.round((2 * self.threshold + 1) * np.abs(self.x) / 2)
                u_x = np.where(
                    p_x <= (self.threshold - 1),
                    np.sign(self.x) * 2 * p_x / (2 * self.threshold + 1),
                    np.sign(self.x),
                )

                if self.h is None:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * self.J.dot(u_x)) * self.dt
                else:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * (self.J.dot(u_x) + self.h)) * self.dt

                self.x += self.dt * self.y * self.delta
                cond = np.abs(self.x) > 1
                self.x = np.where(cond, np.sign(self.x), self.x)
                self.y = np.where(cond, np.zeros_like(self.y), self.y)

        elif self.backend == "gpu-float32":
            for i in range(self.n_iter):
                p_x = torch.round((2 * self.threshold + 1) * torch.abs(self.x) / 2)
                u_x = torch.where(
                    p_x <= (self.threshold - 1),
                    torch.sign(self.x) * 2 * p_x / (2 * self.threshold + 1),
                    torch.sign(self.x),
                )

                if self.h is None:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * torch.sparse.mm(self.J, u_x)) * self.dt
                else:
                    self.y += (
                        -(self.delta - self.p[i]) * self.x + self.xi * (torch.sparse.mm(self.J, u_x) + self.h)
                    ) * self.dt

                self.x += self.dt * self.y * self.delta
                cond = torch.abs(self.x) > 1
                self.x = torch.where(cond, torch.sign(self.x), self.x)
                self.y = torch.where(cond, torch.zeros_like(self.y), self.y)


class LSB(SB):
    r"""
    Logarithmic Quantized Simulated Bifurcation.

    Reference: `Quantized Simulated Bifurcation for the Ising Model <https://ieeexplore.ieee.org/document/10231308>`_.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        threshold(int): dynamic threshold. Default: 8.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        J,
        h=None,
        x=None,
        n_iter=1000,
        batch_size=1,
        backend="cpu-float32",
        dt=1,
        xi=None,
        threshold=8,
    ):
        """Construct LSB algorithm."""
        valid_backends = {'cpu-float32', 'gpu-float32'}
        if not isinstance(backend, str):
            raise TypeError(f"backend requires a string, but get {type(backend)}")
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")
        super().__init__(J, h, x, n_iter, batch_size, dt, xi, backend)
        self.threshold = threshold

        super().initialize()

    def update(self):
        """Dynamical evolution based on Modified explicit symplectic Euler method."""
        if self.backend == "cpu-float32":
            for i in range(self.n_iter):
                l_x = np.ceil(np.log2(np.abs(self.x))).astype(int)

                active = l_x > -int(self.threshold)
                scale = np.where(active, np.ldexp(np.ones_like(self.x), -l_x), np.zeros_like(self.x))
                v = np.sign(self.x) * scale
                sparse_term = self.J.dot(v)

                if self.h is None:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * sparse_term) * self.dt
                else:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * (sparse_term + self.h)) * self.dt

                self.x += self.dt * self.y * self.delta
                cond = np.abs(self.x) > 1
                self.x = np.where(cond, np.sign(self.x), self.x)
                self.y = np.where(cond, np.zeros_like(self.y), self.y)

        elif self.backend == "gpu-float32":
            for i in range(self.n_iter):
                l_x = torch.ceil(torch.log2(torch.abs(self.x)))
                active = l_x > -int(self.threshold)
                scale = torch.where(
                    active,
                    torch.ldexp(torch.ones_like(self.x), -l_x),
                    torch.zeros_like(self.x),
                )
                v = torch.sign(self.x) * scale
                sparse_term = torch.sparse.mm(self.J, v)

                if self.h is None:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * sparse_term) * self.dt
                else:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * (sparse_term + self.h)) * self.dt

                self.x += self.dt * self.y * self.delta
                cond = torch.abs(self.x) > 1
                self.x = torch.where(cond, torch.sign(self.x), self.x)
                self.y = torch.where(cond, torch.zeros_like(self.y), self.y)
