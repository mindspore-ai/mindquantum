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
"""Simulated bifurcation (SB) algorithms and its variants."""
# pylint: disable=invalid-name
import warnings
import numpy as np
from scipy.sparse import csr_matrix

from .QAIA import QAIA, OverflowException


class SB(QAIA):
    r"""
    The base class of SB.

    This class is the base class for SB. It contains the initialization of
    spin values and momentum.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        J,
        h=None,
        x=None,
        n_iter=1000,
        batch_size=1,
        dt=1,
        xi=None,
    ):
        """Construct SB algorithm."""
        super().__init__(J, h, x, n_iter, batch_size)
        self.J = csr_matrix(self.J)
        # positive detuning frequency
        self.delta = 1
        self.dt = dt
        # pumping amplitude
        self.p = np.linspace(0, 1, self.n_iter)
        self.xi = xi
        if self.xi is None:
            self.xi = 0.5 * np.sqrt(self.N - 1) / np.sqrt(csr_matrix.power(self.J, 2).sum())
        self.x = x

        self.initialize()

    def initialize(self):
        """Initialize spin values and momentum."""
        if self.x is None:
            self.x = 0.02 * (np.random.rand(self.N, self.batch_size) - 0.5)

        if self.x.shape[0] != self.N:
            raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")

        self.y = 0.02 * (np.random.rand(self.N, self.batch_size) - 0.5)


class ASB(SB):  # noqa: N801
    r"""
    Adiabatic SB algorithm.

    Reference: `Combinatorial optimization by simulating adiabatic bifurcations in nonlinear
    Hamiltonian systems <https://www.science.org/doi/10.1126/sciadv.aav2372>`_.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        M (int): The number of update without mean-field terms. Default: ``2``.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        J,
        h=None,
        x=None,
        n_iter=1000,
        batch_size=1,
        dt=1,
        xi=None,
        M=2,
    ):
        """Construct ASB algorithm."""
        super().__init__(J, h, x, n_iter, batch_size, dt, xi)
        # positive Kerr coefficient
        self.K = 1
        self.M = M
        # Time step for updating without mean-field terms
        self.dm = self.dt / self.M

    def update(self):
        """Dynamical evolution based on Modified explicit symplectic Euler method."""
        # iterate on the number of MVMs
        for i in range(self.n_iter):
            for _ in range(self.M):
                self.x += self.dm * self.y * self.delta
                self.y -= (self.K * self.x**3 + (self.delta - self.p[i]) * self.x) * self.dm
            if self.h is None:
                self.y += self.xi * self.dt * self.J.dot(self.x)
            else:
                self.y += self.xi * self.dt * (self.J.dot(self.x) + self.h)

            if np.isnan(self.x).any():
                raise OverflowException("Value is too large to handle due to large dt or xi.")


class BSB(SB):  # noqa: N801
    r"""
    Ballistic SB algorithm.

    Reference: `High-performance combinatorial optimization based on classical
    mechanics <https://www.science.org/doi/10.1126/sciadv.abe7953>`_.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        device (str): The device to use for computation ('cpu' or 'gpu'). Default: ``'cpu'``.
        precision (str): Precision type when using GPU ('float32', 'float16', 'int8'). Default: ``'float32'``.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, J, h=None, x=None, n_iter=1000, batch_size=1, dt=1, xi=None, device='cpu', precision='float32'):
        """Construct BSB algorithm."""
        super().__init__(J, h, x, n_iter, batch_size, dt, xi)

        self.device = device.lower()
        if self.device == 'gpu':
            try:
                from mindquantum import _qaia_sb
            except ImportError:
                warnings.warn(
                    "Unable to import bSB GPU backend. This could be due to your environment "
                    "not satisfying the requirements. Will use CPU backend instead.",
                    stacklevel=2,
                )
                self.device = 'cpu'
            except RuntimeError as err:
                warnings.warn(f"Disable bSB GPU backend due to: {err}. Will use CPU backend instead.", stacklevel=2)
                self.device = 'cpu'

        self.precision = precision.lower()
        if self.precision not in ['float32', 'float16', 'int8']:
            raise ValueError("precision must be one of 'float32', 'float16', 'int8'")

        if self.precision != 'float32' and device.lower() == 'cpu':
            raise NotImplementedError(f"BSB with {precision} precision only supports GPU computation")

        self.initialize()

        if self.device == 'gpu':
            _qaia_sb.cuda_init(self.J.shape[0], self.batch_size)
            if self.precision == 'float32':
                self._update_func = _qaia_sb.bsb_update
            elif self.precision == 'float16':
                self._update_func = _qaia_sb.bsb_update_half if self.h is None else _qaia_sb.bsb_update_h_half
            else:  # int8
                self._update_func = _qaia_sb.bsb_update_int8 if self.h is None else _qaia_sb.bsb_update_h_int8
            self._is_gpu = True
        else:
            self._update_func = self._cpu_update
            self._is_gpu = False

    # pylint: disable=unused-argument
    def _cpu_update(self, J, x, h, batch_size, xi, delta, dt, n_iter):
        """CPU implementation of update."""
        for i in range(n_iter):
            if h is None:
                self.y += (-(delta - self.p[i]) * x + xi * J.dot(x)) * dt
            else:
                self.y += (-(delta - self.p[i]) * x + xi * (J.dot(x) + h)) * dt
            x += dt * self.y * delta

            cond = np.abs(x) > 1
            x = np.where(cond, np.sign(x), x)
            self.y = np.where(cond, np.zeros_like(x), self.y)

    # pylint: disable=attribute-defined-outside-init
    def update(self):
        """Dynamical evolution based on Modified explicit symplectic Euler method."""
        self._update_func(self.J, self.x, self.h, self.batch_size, self.xi, self.delta, self.dt, self.n_iter)


class DSB(SB):  # noqa: N801
    r"""
    Discrete SB algorithm.

    Reference: `High-performance combinatorial optimization based on classical
    mechanics <https://www.science.org/doi/10.1126/sciadv.abe7953>`_.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        device (str): The device to use for computation ('cpu' or 'gpu'). Default: ``'cpu'``.
        precision (str): Precision type when using GPU ('float32', 'float16', 'int8'). Default: ``'float32'``.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, J, h=None, x=None, n_iter=1000, batch_size=1, dt=1, xi=None, device='cpu', precision='float32'):
        """Construct DSB algorithm."""
        super().__init__(J, h, x, n_iter, batch_size, dt, xi)

        self.device = device.lower()
        if self.device == 'gpu':
            try:
                from mindquantum import _qaia_sb
            except ImportError:
                warnings.warn(
                    "Unable to import dSB GPU backend. This could be due to your environment "
                    "not satisfying the requirements. Will use CPU backend instead.",
                    stacklevel=2,
                )
                self.device = 'cpu'
            except RuntimeError as err:
                warnings.warn(f"Disable dSB GPU backend due to: {err}. Will use CPU backend instead.", stacklevel=2)
                self.device = 'cpu'

        self.precision = precision.lower()
        if self.precision not in ['float32', 'float16', 'int8']:
            raise ValueError("precision must be one of 'float32', 'float16', 'int8'")

        if self.precision != 'float32' and device.lower() == 'cpu':
            raise NotImplementedError(f"DSB with {precision} precision only supports GPU computation")

        self.initialize()
        if self.device == 'gpu':
            _qaia_sb.cuda_init(self.J.shape[0], self.batch_size)
            if self.precision == 'float32':
                self._update_func = _qaia_sb.dsb_update
            elif self.precision == 'float16':
                self._update_func = _qaia_sb.dsb_update_half if self.h is None else _qaia_sb.dsb_update_h_half
            else:  # int8
                self._update_func = _qaia_sb.dsb_update_int8 if self.h is None else _qaia_sb.dsb_update_h_int8
            self._is_gpu = True
        else:
            self._update_func = self._cpu_update
            self._is_gpu = False

    # pylint: disable=unused-argument
    def _cpu_update(self, J, x, h, batch_size, xi, delta, dt, n_iter):
        """CPU implementation of update."""
        for i in range(n_iter):
            if h is None:
                self.y += (-(delta - self.p[i]) * x + xi * J.dot(np.sign(x))) * dt
            else:
                self.y += (-(delta - self.p[i]) * x + xi * (J.dot(np.sign(x)) + h)) * dt
            x += dt * self.y * delta

            cond = np.abs(x) > 1
            x = np.where(cond, np.sign(x), x)
            self.y = np.where(cond, np.zeros_like(x), self.y)

    # pylint: disable=attribute-defined-outside-init
    def update(self):
        """Dynamical evolution based on Modified explicit symplectic Euler method."""
        self._update_func(self.J, self.x, self.h, self.batch_size, self.xi, self.delta, self.dt, self.n_iter)
