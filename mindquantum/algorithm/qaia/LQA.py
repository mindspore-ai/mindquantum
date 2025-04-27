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

from mindquantum.utils.type_value_check import (
    _check_number_type,
    _check_value_should_not_less,
    _check_value_should_between_close_set,
)
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


class LQA(QAIA):
    r"""
    Local quantum annealing algorithm.

    Reference: `Quadratic Unconstrained Binary Optimization via Quantum-Inspired
    Annealing <https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.18.034016>`_.

    Note:
        For memory efficiency, the input array 'x' is not copied and will be modified
        in-place during optimization. If you need to preserve the original data,
        please pass a copy using `x.copy()`.

    Args:
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N, ).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values uniformly distributed in [-0.1, 0.1]. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        gamma (float): The coupling strength. Default: ``0.1``.
        momentum (float): Momentum factor. Default: ``0.99``.
        backend (str): Computation backend and precision to use: 'cpu-float32',
            'gpu-float32','npu-float32'. Default: ``'cpu-float32'``.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.qaia import LQA
        >>> J = np.array([[0, -1], [-1, 0]])
        >>> solver = LQA(J, batch_size=5)
        >>> solver.update()
        >>> print(solver.calc_cut())
        [1. 1. 1. 1. 1.]
        >>> print(solver.calc_energy())
        [-1. -1. -1. -1. -1.]
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self, J, h=None, x=None, n_iter=1000, batch_size=1, gamma=0.1, dt=1.0, momentum=0.99, backend='cpu-float32'
    ):
        """Construct LQA algorithm."""
        _check_number_type("gamma", gamma)
        _check_value_should_not_less("gamma", 0, gamma)

        _check_number_type("dt", dt)
        _check_value_should_not_less("dt", 0, dt)

        _check_number_type("momentum", momentum)
        _check_value_should_between_close_set("momentum", 0, 1, momentum)

        super().__init__(J, h, x, n_iter, batch_size, backend)
        if self.backend == "cpu-float32":
            self.J = csr_matrix(self.J)

        self.gamma = gamma
        self.dt = dt
        self.momentum = momentum
        self.backend = backend

        self.initialize()

    def initialize(self):
        """Initialize spin values."""
        if self.x is None:
            if self.backend == "cpu-float32":
                self.x = 0.2 * (np.random.rand(self.N, self.batch_size) - 0.5)
            elif self.backend == "gpu-float32":
                self.x = 0.02 * (torch.rand(self.N, self.batch_size, device="cuda") - 0.5)
            elif self.backend == "npu-float32":
                self.x = 0.02 * (torch.rand(self.N, self.batch_size).npu() - 0.5)
        else:
            if self.backend == "gpu-float32":
                if isinstance(self.x, np.ndarray):
                    self.x = torch.from_numpy(self.x).float().to("cuda")
            elif self.backend == "npu-float32":
                if isinstance(self.x, np.ndarray):
                    self.x = torch.from_numpy(self.x).float().to("npu")

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

        if self.backend == "cpu-float32":
            if self.h is None:
                for i in range(1, self.n_iter):
                    t = i / self.n_iter
                    tanh_x = np.tanh(self.x)
                    tmp = np.pi / 2 * tanh_x
                    z = np.sin(tmp)
                    y = np.cos(tmp)
                    dx = np.pi / 2 * (-t * self.gamma * self.J.dot(z) * y + (1 - t) * z) * (1 - tanh_x**2)

                    # momentum beta1
                    m_dx = beta1 * m_dx + (1 - beta1) * dx
                    # rms beta2
                    v_dx = beta2 * v_dx + (1 - beta2) * dx**2
                    # bias correction
                    m_dx_corr = m_dx / (1 - beta1**i)
                    v_dx_corr = v_dx / (1 - beta2**i)

                    self.x = self.x - self.dt * m_dx_corr / (np.sqrt(v_dx_corr) + epsilon)
            else:
                for i in range(1, self.n_iter):
                    t = i / self.n_iter
                    tanh_x = np.tanh(self.x)
                    tmp = np.pi / 2 * tanh_x
                    z = np.sin(tmp)
                    y = np.cos(tmp)
                    dx = np.pi / 2 * (-t * self.gamma * (self.J.dot(z) + self.h) * y + (1 - t) * z) * (1 - tanh_x**2)

                    # momentum beta1
                    m_dx = beta1 * m_dx + (1 - beta1) * dx
                    # rms beta2
                    v_dx = beta2 * v_dx + (1 - beta2) * dx**2
                    # bias correction
                    m_dx_corr = m_dx / (1 - beta1**i)
                    v_dx_corr = v_dx / (1 - beta2**i)

                    self.x = self.x - self.dt * m_dx_corr / (np.sqrt(v_dx_corr) + epsilon)

        elif self.backend == "gpu-float32":
            if self.h is None:
                for i in range(1, self.n_iter):
                    t = i / self.n_iter
                    tanh_x = torch.tanh(self.x)
                    tmp = torch.pi / 2 * tanh_x
                    z = torch.sin(tmp)
                    y = torch.cos(tmp)

                    dx = (
                        torch.pi
                        / 2
                        * (-t * self.gamma * torch.sparse.mm(self.J, z) * y + (1 - t) * z)
                        * (1 - tanh_x**2)
                    )

                    # momentum beta1
                    m_dx = beta1 * m_dx + (1 - beta1) * dx
                    # rms beta2
                    v_dx = beta2 * v_dx + (1 - beta2) * dx**2
                    # bias correction
                    m_dx_corr = m_dx / (1 - beta1**i)
                    v_dx_corr = v_dx / (1 - beta2**i)

                    self.x = self.x - self.dt * m_dx_corr / (torch.sqrt(v_dx_corr) + epsilon)
            else:
                for i in range(1, self.n_iter):
                    t = i / self.n_iter
                    tanh_x = torch.tanh(self.x)
                    tmp = torch.pi / 2 * tanh_x
                    z = torch.sin(tmp)
                    y = torch.cos(tmp)

                    dx = (
                        np.pi
                        / 2
                        * (-t * self.gamma * (torch.sparse.mm(self.J, z) + self.h) * y + (1 - t) * z)
                        * (1 - tanh_x**2)
                    )

                    # momentum beta1
                    m_dx = beta1 * m_dx + (1 - beta1) * dx
                    # rms beta2
                    v_dx = beta2 * v_dx + (1 - beta2) * dx**2
                    # bias correction
                    m_dx_corr = m_dx / (1 - beta1**i)
                    v_dx_corr = v_dx / (1 - beta2**i)

                    self.x = self.x - self.dt * m_dx_corr / (torch.sqrt(v_dx_corr) + epsilon)

        elif self.backend == "npu-float32":
            if self.h is None:
                for i in range(1, self.n_iter):
                    t = i / self.n_iter
                    tanh_x = torch.tanh(self.x).npu()
                    tmp = torch.pi / 2 * tanh_x
                    z = torch.sin(tmp).npu()
                    y = torch.cos(tmp).npu()

                    dx = np.pi / 2 * (-t * self.gamma * torch.sparse.mm(self.J, z) * y + (1 - t) * z) * (1 - tanh_x**2)

                    # momentum beta1
                    m_dx = beta1 * m_dx + (1 - beta1) * dx
                    # rms beta2
                    v_dx = beta2 * v_dx + (1 - beta2) * dx**2
                    # bias correction
                    m_dx_corr = m_dx / (1 - beta1**i)
                    v_dx_corr = v_dx / (1 - beta2**i)

                    self.x = self.x - self.dt * m_dx_corr / (torch.sqrt(v_dx_corr) + epsilon)
            else:
                for i in range(1, self.n_iter):
                    t = i / self.n_iter
                    tanh_x = torch.tanh(self.x).npu()
                    tmp = torch.pi / 2 * tanh_x
                    z = torch.sin(tmp).npu()
                    y = torch.cos(tmp).npu()

                    dx = (
                        np.pi
                        / 2
                        * (-t * self.gamma * (torch.sparse.mm(self.J, z) + self.h) * y + (1 - t) * z)
                        * (1 - tanh_x**2)
                    )

                    # momentum beta1
                    m_dx = beta1 * m_dx + (1 - beta1) * dx
                    # rms beta2
                    v_dx = beta2 * v_dx + (1 - beta2) * dx**2
                    # bias correction
                    m_dx_corr = m_dx / (1 - beta1**i)
                    v_dx_corr = v_dx / (1 - beta2**i)

                    self.x = self.x - self.dt * m_dx_corr / (torch.sqrt(v_dx_corr) + epsilon)
