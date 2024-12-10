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
import numpy as np
from scipy.sparse import csr_matrix

from .QAIA import QAIA, OverflowException
from mindquantum.utils.type_value_check import _check_number_type, _check_value_should_not_less, _check_int_type

try:
    from mindquantum import _qaia_sb

    GPU_AVAILABLE = True
except ImportError as err:
    GPU_DISABLE_REASON = "Unable to import SB GPU backend. This backend requires CUDA 11 or higher."
    GPU_AVAILABLE = False
except RuntimeError as err:
    GPU_DISABLE_REASON = f"Disable SB GPU backend due to: {err}."
    GPU_AVAILABLE = False


class SB(QAIA):
    r"""
    The base class of SB.

    This class is the base class for SB. It contains the initialization of
    spin values and momentum.

    Note:
        For memory efficiency, the input array 'x' is not copied and will be modified
        in-place during optimization. If you need to preserve the original data,
        please pass a copy using `x.copy()`.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`.
            Will be modified during optimization. Default: ``None``.
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
        _check_number_type("dt", dt)
        _check_value_should_not_less("dt", 0, dt)

        if xi is not None:
            _check_number_type("xi", xi)
            _check_value_should_not_less("xi", 0, xi)

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

    Note:
        For memory efficiency, the input array 'x' is not copied and will be modified
        in-place during optimization. If you need to preserve the original data,
        please pass a copy using `x.copy()`.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`.
            Will be modified during optimization. Default: ``None``.
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
        _check_int_type("M", M)
        _check_value_should_not_less("M", 1, M)
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

    Note:
        For memory efficiency, the input array 'x' is not copied and will be modified
        in-place during optimization. If you need to preserve the original data,
        please pass a copy using `x.copy()`.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`.
            Will be modified during optimization. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        backend (str): Computation backend and precision to use: 'cpu-float32',
            'gpu-float16', or 'gpu-int8'. Default: ``'cpu-float32'``.
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
        backend='cpu-float32',
    ):
        """Construct BSB algorithm."""
        super().__init__(J, h, x, n_iter, batch_size, dt, xi)

        valid_backends = {'cpu-float32', 'gpu-float16', 'gpu-int8'}
        if not isinstance(backend, str):
            raise TypeError(f"backend requires a string, but get {type(backend)}")
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")
        if backend.startswith('gpu'):
            if not GPU_AVAILABLE:
                raise RuntimeError(f"GPU backend '{backend}' is not available: {GPU_DISABLE_REASON}")
            _qaia_sb.cuda_init(self.J.shape[0], self.batch_size)
        self.backend = backend

    def update(self):
        """Dynamical evolution based on Modified explicit symplectic Euler method."""
        if self.backend == 'gpu-float16':
            if self.h is not None:
                _qaia_sb.bsb_update_h_half(
                    self.J, self.x, self.h, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
            else:
                _qaia_sb.bsb_update_half(
                    self.J, self.x, self.h, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
        elif self.backend == 'gpu-int8':
            if self.h is not None:
                _qaia_sb.bsb_update_h_int8(
                    self.J, self.x, self.h, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
            else:
                _qaia_sb.bsb_update_int8(
                    self.J, self.x, self.h, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
        else:  # cpu-float32
            for i in range(self.n_iter):
                if self.h is None:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * self.J.dot(self.x)) * self.dt
                else:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * (self.J.dot(self.x) + self.h)) * self.dt
                self.x += self.dt * self.y * self.delta

                cond = np.abs(self.x) > 1
                self.x = np.where(cond, np.sign(self.x), self.x)
                self.y = np.where(cond, np.zeros_like(self.x), self.y)


class DSB(SB):  # noqa: N801
    r"""
    Discrete SB algorithm.

    Reference: `High-performance combinatorial optimization based on classical
    mechanics <https://www.science.org/doi/10.1126/sciadv.abe7953>`_.

    Note:
        For memory efficiency, the input array 'x' is not copied and will be modified
        in-place during optimization. If you need to preserve the original data,
        please pass a copy using `x.copy()`.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`.
            Will be modified during optimization. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        backend (str): Computation backend and precision to use: 'cpu-float32',
            'gpu-float16', or 'gpu-int8'. Default: ``'cpu-float32'``.
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
        backend='cpu-float32',
    ):
        """Construct DSB algorithm."""
        super().__init__(J, h, x, n_iter, batch_size, dt, xi)
        valid_backends = {'cpu-float32', 'gpu-float16', 'gpu-int8'}
        if not isinstance(backend, str):
            raise TypeError(f"backend requires a string, but get {type(backend)}")
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")
        if backend.startswith('gpu'):
            if not GPU_AVAILABLE:
                raise RuntimeError(f"GPU backend '{backend}' is not available: {GPU_DISABLE_REASON}")
            _qaia_sb.cuda_init(self.J.shape[0], self.batch_size)
        self.backend = backend

    def update(self):
        """Dynamical evolution based on Modified explicit symplectic Euler method."""
        if self.backend == 'gpu-float16':
            if self.h is not None:
                _qaia_sb.dsb_update_h_half(
                    self.J, self.x, self.h, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
            else:
                _qaia_sb.dsb_update_half(
                    self.J, self.x, self.h, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
        elif self.backend == 'gpu-int8':
            if self.h is not None:
                _qaia_sb.dsb_update_h_int8(
                    self.J, self.x, self.h, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
            else:
                _qaia_sb.dsb_update_int8(
                    self.J, self.x, self.h, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
        else:  # cpu-float32
            for i in range(self.n_iter):
                if self.h is None:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * self.J.dot(np.sign(self.x))) * self.dt
                else:
                    self.y += (
                        -(self.delta - self.p[i]) * self.x + self.xi * (self.J.dot(np.sign(self.x)) + self.h)
                    ) * self.dt
                self.x += self.dt * self.y * self.delta

                cond = np.abs(self.x) > 1
                self.x = np.where(cond, np.sign(self.x), self.x)
                self.y = np.where(cond, np.zeros_like(self.x), self.y)
