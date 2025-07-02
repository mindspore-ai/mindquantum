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

from mindquantum.utils.type_value_check import _check_number_type, _check_value_should_not_less, _check_int_type
from .QAIA import QAIA

try:
    import torch

    assert torch.cuda.is_available()
    _INSTALL_TORCH = True
except (ImportError, AssertionError):
    _INSTALL_TORCH = False

try:
    import torch
    import torch_npu

    assert torch_npu.npu.is_available()
    _INSTALL_TORCH_NPU = True
except (ImportError, AssertionError):
    _INSTALL_TORCH_NPU = False

try:
    from mindquantum import _qaia_sb

    GPU_AVAILABLE = True
except ImportError as err:
    GPU_DISABLE_REASON = "Unable to import SB GPU backend. This backend requires CUDA 11 or higher."
    GPU_AVAILABLE = False

    # Temporarily disable GPU backend due to PyPI package size limitations
    GPU_DISABLE_REASON = (
        "GPU backend is temporarily disabled in 0.10.0 version due to PyPI package size limitations. "
        "To use GPU backend, please install mindquantum 0.10.0 from https://www.mindspore.cn/versions"
    )
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
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N, ).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values uniformly distributed in [-0.01, 0.01]. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        backend (str): Computation backend and precision to use: 'cpu-float32',
            'gpu-float32','npu-float32'. Default: ``'cpu-float32'``.
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
        """Construct SB algorithm."""
        _check_number_type("dt", dt)
        _check_value_should_not_less("dt", 0, dt)

        if xi is not None:
            _check_number_type("xi", xi)
            _check_value_should_not_less("xi", 0, xi)

        super().__init__(J, h, x, n_iter, batch_size, backend)
        if self.backend == "cpu-float32":
            self.J = csr_matrix(self.J)

        # positive detuning frequency
        self.delta = 1
        self.dt = dt
        # pumping amplitude
        self.p = np.linspace(0, 1, self.n_iter)
        self.xi = xi

        if self.xi is None:
            if self.backend == "cpu-float32":
                self.xi = 0.5 * np.sqrt(self.N - 1) / np.sqrt(csr_matrix.power(self.J, 2).sum())
            elif self.backend == "gpu-float32":
                if h is not None:
                    self.xi = (
                        0.5
                        * np.sqrt(self.N - 1)
                        / torch.sqrt((self.J.to_dense() ** 2).sum() + 2 * ((self.h / 2) ** 2).sum())
                    )
                else:
                    self.xi = 0.5 * np.sqrt(self.N - 1) / torch.sqrt((self.J.to_dense() ** 2).sum())
            elif self.backend == "npu-float32":
                if h is not None:
                    self.xi = (
                        0.5
                        * np.sqrt(self.N - 1)
                        / np.sqrt(
                            csr_matrix.power(csr_matrix(self.J.cpu().numpy()), 2).sum() + 2 * ((self.h / 2) ** 2).sum()
                        )
                    )
                else:
                    self.xi = (
                        0.5 * np.sqrt(self.N - 1) / np.sqrt(csr_matrix.power(csr_matrix(self.J.cpu().numpy()), 2).sum())
                    )
        self.x = x

        self.initialize()

    def initialize(self):
        """Initialize spin values and momentum."""
        if self.backend == "cpu-float32":
            if self.x is None:
                self.x = 0.02 * (np.random.rand(self.N, self.batch_size) - 0.5)
            if self.x.shape[0] != self.N:
                raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")
            self.y = 0.02 * (np.random.rand(self.N, self.batch_size) - 0.5)

        elif self.backend == "gpu-float32":
            if self.x is None:
                self.x = 0.02 * (torch.rand(self.N, self.batch_size, device="cuda") - 0.5)
            else:
                if isinstance(self.x, np.ndarray):
                    self.x = torch.from_numpy(self.x).float().to("cuda")
            if self.x.shape[0] != self.N:
                raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")
            self.y = 0.02 * (torch.rand(self.N, self.batch_size, device="cuda") - 0.5)

        elif self.backend == "npu-float32":
            if self.x is None:
                self.x = 0.02 * (torch.rand(self.N, self.batch_size).npu() - 0.5)
            else:
                if isinstance(self.x, np.ndarray):
                    self.x = torch.from_numpy(self.x).float().to("npu")
            if self.x.shape[0] != self.N:
                raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")
            self.y = 0.02 * (torch.rand(self.N, self.batch_size).npu() - 0.5)


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
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N, ).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values uniformly distributed in [-0.01, 0.01]. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        M (int): The number of update without mean-field terms. Default: ``2``.
        backend (str): Computation backend and precision to use: 'cpu-float32',
            'gpu-float32','npu-float32'. Default: ``'cpu-float32'``.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.qaia import ASB
        >>> J = np.array([[0, -1], [-1, 0]])
        >>> solver = ASB(J, batch_size=5)
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
        dt=1,
        xi=None,
        M=2,
        backend='cpu-float32',
    ):
        """Construct ASB algorithm."""
        _check_int_type("M", M)
        _check_value_should_not_less("M", 1, M)
        super().__init__(J, h, x, n_iter, batch_size, dt, xi, backend)
        # positive Kerr coefficient
        self.K = 1
        self.M = M
        # Time step for updating without mean-field terms
        self.dm = self.dt / self.M

    def update(self):
        """Dynamical evolution based on Modified explicit symplectic Euler method."""
        # iterate on the number of MVMs
        if self.backend == "cpu-float32":
            for i in range(self.n_iter):
                for _ in range(self.M):
                    self.x += self.dm * self.y * self.delta
                    self.y -= (self.K * self.x**3 + (self.delta - self.p[i]) * self.x) * self.dm
                if self.h is None:
                    self.y += self.xi * self.dt * self.J.dot(self.x)
                else:
                    self.y += self.xi * self.dt * (self.J.dot(self.x) + self.h)

                if np.isnan(self.x).any():
                    raise ValueError(
                        f"NaNs detected in tensor 'x'. "
                        f"This often indicates numerical instability. "
                        f"Consider adjusting parameters like dt={self.dt} or xi={self.xi}."
                    )

        elif self.backend == "gpu-float32":
            for i in range(self.n_iter):
                for _ in range(self.M):
                    self.x += self.dm * self.y * self.delta
                    self.y -= (self.K * self.x**3 + (self.delta - self.p[i]) * self.x) * self.dm
                if self.h is None:
                    self.y += self.xi * self.dt * torch.sparse.mm(self.J, self.x)
                else:
                    self.y += self.xi * self.dt * (torch.sparse.mm(self.J, self.x) + self.h)

                if torch.isnan(self.x).any():
                    raise ValueError(
                        f"NaNs detected in tensor 'x'. "
                        f"This often indicates numerical instability. "
                        f"Consider adjusting parameters like dt={self.dt} or xi={self.xi}."
                    )

        elif self.backend == "npu-float32":
            for i in range(self.n_iter):
                for _ in range(self.M):
                    self.x += self.dm * self.y * self.delta
                    self.y -= (self.K * self.x**3 + (self.delta - self.p[i]) * self.x) * self.dm
                if self.h is None:
                    self.y += self.xi * self.dt * torch.sparse.mm(self.J, self.x)
                else:
                    self.y += self.xi * self.dt * (torch.sparse.mm(self.J, self.x) + self.h)

                if torch.isnan(self.x).any():
                    raise ValueError(
                        f"NaNs detected in tensor 'x'. "
                        f"This often indicates numerical instability. "
                        f"Consider adjusting parameters like dt={self.dt} or xi={self.xi}."
                    )


class BSB(SB):  # noqa: N801
    r"""
    Ballistic SB algorithm.

    Reference: `High-performance combinatorial optimization based on classical
    mechanics <https://www.science.org/doi/10.1126/sciadv.abe7953>`_.

    Note:
        For memory efficiency, the input array 'x' is not copied and will be modified
        in-place during optimization. If you need to preserve the original data,
        please pass a copy using `x.copy()`.

        When using backend='gpu-int8', be aware that it may not perform well on dense graphs
        or graphs with continuous coefficients. Please try adjusting parameters or consider
        using 'cpu-float32' or 'gpu-float16' in these cases.

    Args:
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N, ).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values uniformly distributed in [-0.01, 0.01]. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        backend (str): Computation backend and precision to use: 'cpu-float32','gpu-float32',
            'gpu-float16', 'gpu-int8','npu-float32'. Default: ``'cpu-float32'``.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.qaia import BSB
        >>> J = np.array([[0, -1], [-1, 0]])
        >>> solver = BSB(J, batch_size=5, backend='cpu-float32')
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
        dt=1,
        xi=None,
        backend='cpu-float32',
    ):
        """Construct BSB algorithm."""
        valid_backends = {'cpu-float32', 'gpu-float32', 'gpu-float16', 'gpu-int8', 'npu-float32'}
        if not isinstance(backend, str):
            raise TypeError(f"backend requires a string, but get {type(backend)}")
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")

        if backend in ['cpu-float32', 'gpu-float32', 'npu-float32']:
            super().__init__(J, h, x, n_iter, batch_size, dt, xi, backend)
        elif backend in ['gpu-float16', 'gpu-int8']:
            super().__init__(J, h, x, n_iter, batch_size, dt, xi)
            if not GPU_AVAILABLE:
                raise RuntimeError(f"GPU backend '{backend}' is not available: {GPU_DISABLE_REASON}")
            _qaia_sb.cuda_init(self.J.shape[0], self.batch_size)
            self.backend = backend

    def update(self):
        """Dynamical evolution based on Modified explicit symplectic Euler method."""
        if self.backend == 'gpu-float16':
            if self.h is not None:
                h_broadcast = np.repeat(self.h, self.batch_size).reshape(self.J.shape[0], self.batch_size)
                _qaia_sb.bsb_update_h_half(
                    self.J, self.x, h_broadcast, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
            else:
                _qaia_sb.bsb_update_half(
                    self.J, self.x, self.h, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
        elif self.backend == 'gpu-int8':
            if self.h is not None:
                h_broadcast = np.repeat(self.h, self.batch_size).reshape(self.J.shape[0], self.batch_size)
                _qaia_sb.bsb_update_h_int8(
                    self.J, self.x, h_broadcast, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
            else:
                _qaia_sb.bsb_update_int8(
                    self.J, self.x, self.h, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
        elif self.backend == 'cpu-float32':
            for i in range(self.n_iter):
                if self.h is None:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * self.J.dot(self.x)) * self.dt
                else:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * (self.J.dot(self.x) + self.h)) * self.dt
                self.x += self.dt * self.y * self.delta

                cond = np.abs(self.x) > 1
                self.x = np.where(cond, np.sign(self.x), self.x)
                self.y = np.where(cond, np.zeros_like(self.x), self.y)
        elif self.backend == 'gpu-float32':
            for i in range(self.n_iter):
                if self.h is None:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * torch.sparse.mm(self.J, self.x)) * self.dt
                else:
                    self.y += (
                        -(self.delta - self.p[i]) * self.x + self.xi * (torch.sparse.mm(self.J, self.x) + self.h)
                    ) * self.dt
                self.x += self.dt * self.y * self.delta

                cond = torch.abs(self.x) > 1
                self.x = torch.where(cond, torch.sign(self.x), self.x)
                self.y = torch.where(cond, torch.zeros_like(self.x), self.y)
        elif self.backend == 'npu-float32':
            for i in range(self.n_iter):
                if self.h is None:
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * torch.sparse.mm(self.J, self.x)) * self.dt
                else:
                    self.y += (
                        -(self.delta - self.p[i]) * self.x + self.xi * (torch.sparse.mm(self.J, self.x) + self.h)
                    ) * self.dt
                self.x += self.dt * self.y * self.delta

                cond = torch.abs(self.x) > 1
                self.x = torch.where(cond, torch.sign(self.x), self.x)
                self.y = torch.where(cond, torch.zeros_like(self.x), self.y)


class DSB(SB):  # noqa: N801
    r"""
    Discrete SB algorithm.

    Reference: `High-performance combinatorial optimization based on classical
    mechanics <https://www.science.org/doi/10.1126/sciadv.abe7953>`_.

    Note:
        For memory efficiency, the input array 'x' is not copied and will be modified
        in-place during optimization. If you need to preserve the original data,
        please pass a copy using `x.copy()`.

        When using backend='gpu-int8', be aware that it may not perform well on dense graphs
        or graphs with continuous coefficients. Please try adjusting parameters or consider
        using 'cpu-float32' or 'gpu-float16' in these cases.

    Args:
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N, ).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values uniformly distributed in [-0.01, 0.01]. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        backend (str): Computation backend and precision to use: 'cpu-float32','gpu-float32',
            'gpu-float16', 'gpu-int8','npu-float32'. Default: ``'cpu-float32'``.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.qaia import DSB
        >>> J = np.array([[0, -1], [-1, 0]])
        >>> solver = DSB(J, batch_size=5, backend='cpu-float32')
        >>> solver.update()
        >>> print(solver.calc_cut())
        [0. 1. 1. 1. 1.]
        >>> print(solver.calc_energy())
        [ 1. -1. -1. -1. -1.]
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
        valid_backends = {'cpu-float32', 'gpu-float32', 'gpu-float16', 'gpu-int8', 'npu-float32'}
        if not isinstance(backend, str):
            raise TypeError(f"backend requires a string, but get {type(backend)}")
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")

        if backend in ['cpu-float32', 'gpu-float32', 'npu-float32']:
            super().__init__(J, h, x, n_iter, batch_size, dt, xi, backend)
        elif backend in ['gpu-float16', 'gpu-int8']:
            super().__init__(J, h, x, n_iter, batch_size, dt, xi)
            if not GPU_AVAILABLE:
                raise RuntimeError(f"GPU backend '{backend}' is not available: {GPU_DISABLE_REASON}")
            _qaia_sb.cuda_init(self.J.shape[0], self.batch_size)
            self.backend = backend

    def update(self):
        """Dynamical evolution based on Modified explicit symplectic Euler method."""
        if self.backend == 'gpu-float16':
            if self.h is not None:
                h_broadcast = np.repeat(self.h, self.batch_size).reshape(self.J.shape[0], self.batch_size)
                _qaia_sb.dsb_update_h_half(
                    self.J, self.x, h_broadcast, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
            else:
                _qaia_sb.dsb_update_half(
                    self.J, self.x, self.h, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
        elif self.backend == 'gpu-int8':
            if self.h is not None:
                h_broadcast = np.repeat(self.h, self.batch_size).reshape(self.J.shape[0], self.batch_size)
                _qaia_sb.dsb_update_h_int8(
                    self.J, self.x, h_broadcast, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
            else:
                _qaia_sb.dsb_update_int8(
                    self.J, self.x, self.h, self.batch_size, self.xi, self.delta, self.dt, self.n_iter
                )
        elif self.backend == 'cpu-float32':
            if self.h is None:
                for i in range(self.n_iter):
                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * self.J.dot(np.sign(self.x))) * self.dt
                    self.x += self.dt * self.y * self.delta

                    cond = np.abs(self.x) > 1
                    self.x = np.where(cond, np.sign(self.x), self.x)
                    self.y = np.where(cond, np.zeros_like(self.x), self.y)
            else:
                for i in range(self.n_iter):
                    self.y += (
                        -(self.delta - self.p[i]) * self.x + self.xi * (self.J.dot(np.sign(self.x)) + self.h)
                    ) * self.dt
                    self.x += self.dt * self.y * self.delta

                    cond = np.abs(self.x) > 1
                    self.x = np.where(cond, np.sign(self.x), self.x)
                    self.y = np.where(cond, np.zeros_like(self.x), self.y)
        elif self.backend == 'gpu-float32':
            if self.h is None:
                for i in range(self.n_iter):
                    self.y += (
                        -(self.delta - self.p[i]) * self.x + self.xi * torch.sparse.mm(self.J, torch.sign(self.x))
                    ) * self.dt

                    self.x += self.dt * self.y * self.delta

                    cond = torch.abs(self.x) > 1
                    self.x = torch.where(cond, torch.sign(self.x), self.x)
                    self.y = torch.where(cond, torch.zeros_like(self.y), self.y)
            else:
                for i in range(self.n_iter):
                    self.y += (
                        -(self.delta - self.p[i]) * self.x
                        + self.xi * (torch.sparse.mm(self.J, torch.sign(self.x)) + self.h)
                    ) * self.dt

                    self.x += self.dt * self.y * self.delta

                    cond = torch.abs(self.x) > 1
                    self.x = torch.where(cond, torch.sign(self.x), self.x)
                    self.y = torch.where(cond, torch.zeros_like(self.y), self.y)
        elif self.backend == 'npu-float32':
            if self.h is None:
                for i in range(self.n_iter):
                    self.y += (
                        -(self.delta - self.p[i]) * self.x + self.xi * torch.sparse.mm(self.J, torch.sign(self.x))
                    ) * self.dt

                    self.x += self.dt * self.y * self.delta

                    cond = torch.abs(self.x) > 1
                    self.x = torch.where(cond, torch.sign(self.x), self.x)
                    self.y = torch.where(cond, torch.zeros_like(self.y), self.y)
            else:
                for i in range(self.n_iter):
                    self.y += (
                        -(self.delta - self.p[i]) * self.x
                        + self.xi * (torch.sparse.mm(self.J, torch.sign(self.x)) + self.h)
                    ) * self.dt

                    self.x += self.dt * self.y * self.delta

                    cond = torch.abs(self.x) > 1
                    self.x = torch.where(cond, torch.sign(self.x), self.x)
                    self.y = torch.where(cond, torch.zeros_like(self.y), self.y)
