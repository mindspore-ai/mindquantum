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
"""Coherent Ising Machine with chaotic amplitude control algorithm."""
# pylint: disable=invalid-name
import numpy as np
from scipy.sparse import csr_matrix

from mindquantum.utils.type_value_check import _check_number_type, _check_value_should_not_less
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


class CAC(QAIA):
    r"""
    Coherent Ising Machine with chaotic amplitude control algorithm.

    Reference: `Coherent Ising machines with optical error correction
    circuits <https://onlinelibrary.wiley.com/doi/full/10.1002/qute.202100077>`_.

    Note:
        For memory efficiency, the input array 'x' is not copied and will be modified
        in-place during optimization. If you need to preserve the original data,
        please pass a copy using `x.copy()`.

    Args:
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N, ).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values drawn from normal distribution N(0, 10^(-4)). Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``0.075``.
        backend (str): Computation backend and precision to use: 'cpu-float32',
            'gpu-float32','npu-float32'. Default: ``'cpu-float32'``.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.qaia import CAC
        >>> J = np.array([[0, -1], [-1, 0]])
        >>> solver = CAC(J, batch_size=5)
        >>> solver.update()
        >>> print(solver.calc_cut())
        [1. 1. 1. 1. 1.]
        >>> print(solver.calc_energy())
        [-1. -1. -1. -1. -1.]
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(self, J, h=None, x=None, n_iter=1000, batch_size=1, dt=0.075, backend='cpu-float32'):
        """Construct CAC algorithm."""
        _check_number_type("dt", dt)
        _check_value_should_not_less("dt", 0, dt)
        super().__init__(J, h, x, n_iter, batch_size, backend)
        if self.backend == "cpu-float32":
            self.J = csr_matrix(self.J)

        self.N = self.J.shape[0]
        self.dt = dt
        # The number of first iterations
        self.Tr = int(0.9 * self.n_iter)
        # The number of additional iterations
        self.Tp = self.n_iter - self.Tr
        # pumping parameters
        self.p = np.hstack([np.linspace(-0.5, 1, self.Tr), np.ones(self.Tp)])
        # target amplitude
        self.alpha = np.hstack([np.linspace(1, 3, self.Tr), 3.0 * np.ones(self.Tp)])
        # coupling strength
        self.xi = None
        if self.backend == "cpu-float32":
            self.xi = np.sqrt(2 * self.N / np.sum(self.J**2))
        if self.backend == "gpu-float32":
            self.xi = torch.sqrt(2 * self.N / torch.sum(self.J.to_dense() ** 2))
        if self.backend == "npu-float32":
            self.xi = torch.sqrt(
                2 * self.N / torch.tensor(csr_matrix.power(csr_matrix(self.J.cpu().numpy()), 2).sum())
            ).npu()
        # rate of change of error variables
        self.beta = 0.3
        self.initialize()

    def initialize(self):
        """Initialize spin values and error variables."""
        if self.backend == "cpu-float32":
            if self.x is None:
                self.x = np.random.normal(0, 10 ** (-4), size=(self.N, self.batch_size))

            if self.x.shape[0] != self.N:
                raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")

            self.e = np.ones((self.N, self.batch_size))

        elif self.backend == "gpu-float32":
            if self.x is None:
                self.x = torch.normal(0, 10 ** (-4), size=(self.N, self.batch_size)).to("cuda")
            else:
                if isinstance(self.x, np.ndarray):
                    self.x = torch.from_numpy(self.x).float().to("cuda")

            if self.x.shape[0] != self.N:
                raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")

            self.e = torch.ones(self.N, self.batch_size, device="cuda")

        elif self.backend == "npu-float32":
            if self.x is None:
                self.x = torch.normal(0, 10 ** (-4), size=(self.N, self.batch_size)).to("npu")
            else:
                if isinstance(self.x, np.ndarray):
                    self.x = torch.from_numpy(self.x).float().to("npu")

            if self.x.shape[0] != self.N:
                raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")

            self.e = torch.ones(self.N, self.batch_size).npu()

    # pylint: disable=attribute-defined-outside-init
    def update(self):
        """Dynamical evolution."""
        if self.backend == "cpu-float32":
            if self.h is None:
                for i in range(self.n_iter):
                    self.x = (
                        self.x
                        + (-self.x**3 + (self.p[i] - 1) * self.x + self.xi * self.e * (self.J @ self.x)) * self.dt
                    )
                    self.e = self.e + (-self.beta * self.e * (self.x**2 - self.alpha[i])) * self.dt
                    cond = np.abs(self.x) > (1.5 * np.sqrt(self.alpha[i]))
                    self.x = np.where(cond, 1.5 * np.sign(self.x) * np.sqrt(self.alpha[i]), self.x)
            else:
                for i in range(self.n_iter):
                    self.x = (
                        self.x
                        + (-self.x**3 + (self.p[i] - 1) * self.x + self.xi * self.e * (self.J @ self.x + self.h))
                        * self.dt
                    )
                    self.e = self.e + (-self.beta * self.e * (self.x**2 - self.alpha[i])) * self.dt
                    cond = np.abs(self.x) > (1.5 * np.sqrt(self.alpha[i]))
                    self.x = np.where(cond, 1.5 * np.sign(self.x) * np.sqrt(self.alpha[i]), self.x)

        elif self.backend == "gpu-float32":
            if self.h is None:
                for i in range(self.n_iter):
                    self.x = (
                        self.x
                        + (-self.x**3 + (self.p[i] - 1) * self.x + self.xi * self.e * torch.sparse.mm(self.J, self.x))
                        * self.dt
                    )
                    self.e = self.e + (-self.beta * self.e * (self.x**2 - self.alpha[i])) * self.dt
                    cond = torch.abs(self.x) > (1.5 * torch.sqrt(torch.tensor(self.alpha[i])))
                    self.x = torch.where(
                        cond, 1.5 * torch.sign(self.x) * torch.sqrt(torch.tensor(self.alpha[i])), self.x
                    )
            else:
                for i in range(self.n_iter):
                    self.x = (
                        self.x
                        + (
                            -self.x**3
                            + (self.p[i] - 1) * self.x
                            + self.xi * self.e * (torch.sparse.mm(self.J, self.x) + self.h)
                        )
                        * self.dt
                    )
                    self.e = self.e + (-self.beta * self.e * (self.x**2 - self.alpha[i])) * self.dt
                    cond = torch.abs(self.x) > (1.5 * torch.sqrt(torch.tensor(self.alpha[i])))
                    self.x = torch.where(
                        cond, 1.5 * torch.sign(self.x) * torch.sqrt(torch.tensor(self.alpha[i])), self.x
                    )

        elif self.backend == "npu-float32":
            if self.h is None:
                for i in range(self.n_iter):
                    self.x = (
                        self.x
                        + (-self.x**3 + (self.p[i] - 1) * self.x + self.xi * self.e * torch.sparse.mm(self.J, self.x))
                        * self.dt
                    )
                    self.e = self.e + (-self.beta * self.e * (self.x**2 - self.alpha[i])) * self.dt

                    cond = torch.abs(self.x) > (1.5 * torch.sqrt(torch.tensor(self.alpha[i])).npu())
                    self.x = torch.where(
                        cond, 1.5 * torch.sign(self.x) * torch.sqrt(torch.tensor(self.alpha[i])).npu(), self.x
                    )
            else:
                for i in range(self.n_iter):
                    self.x = (
                        self.x
                        + (
                            -self.x**3
                            + (self.p[i] - 1) * self.x
                            + self.xi * self.e * (torch.sparse.mm(self.J, self.x) + self.h)
                        )
                        * self.dt
                    )
                    self.e = self.e + (-self.beta * self.e * (self.x**2 - self.alpha[i])) * self.dt
                    cond = torch.abs(self.x) > (1.5 * torch.sqrt(torch.tensor(self.alpha[i])).npu())
                    self.x = torch.where(
                        cond, 1.5 * torch.sign(self.x) * torch.sqrt(torch.tensor(self.alpha[i])).npu(), self.x
                    )
