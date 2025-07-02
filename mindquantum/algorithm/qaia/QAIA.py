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
"""The base class of QAIA."""
# pylint: disable=invalid-name
import numpy as np
from scipy import sparse as sp
from mindquantum.utils.type_value_check import _check_int_type, _check_value_should_not_less

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


class QAIA:
    r"""
    The base class of QAIA.

    This class contains the basic and common functions of all the algorithms.

    Note:
        For memory efficiency, the input array 'x' is not copied and will be modified
        in-place during optimization. If you need to preserve the original data,
        please pass a copy using `x.copy()`.

    Args:
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N x 1).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values uniformly distributed in [-0.01, 0.01]. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        backend (str): Computation backend and precision to use: 'cpu-float32','gpu-float32',
            'gpu-float16', 'gpu-int8','npu-float32'. Default: ``'cpu-float32'``.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, J, h=None, x=None, n_iter=1000, batch_size=1, backend='cpu-float32'):
        """Construct a QAIA algorithm."""
        valid_backends = {'cpu-float32', 'gpu-float32', 'gpu-float16', 'gpu-int8', 'npu-float32'}
        if not isinstance(backend, str):
            raise TypeError(f"backend requires a string, but get {type(backend)}")
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")
        if backend == "gpu-float32" and not _INSTALL_TORCH:
            raise ImportError("Please install pytorch before using qaia gpu backend, ensure environment has any GPU.")
        if backend == "npu-float32" and not _INSTALL_TORCH_NPU:
            raise ImportError(
                "Please install torch_npu before using qaia npu backend, ensure environment has any Ascend NPU."
            )

        if not isinstance(J, (np.ndarray, sp.spmatrix)):
            raise TypeError(f"J requires numpy.array or scipy sparse matrix, but get {type(J)}")
        if len(J.shape) != 2 or J.shape[0] != J.shape[1]:
            raise ValueError(f"J must be a square matrix, but got shape {J.shape}")
        if isinstance(J, np.ndarray):
            if not np.allclose(J, J.T):
                raise ValueError("J must be a symmetric matrix.")
            if not np.all(np.diag(J) == 0):
                raise ValueError("The diagonal elements of J are not all 0, recommend transferring them to h.")
        if isinstance(J, sp.spmatrix):
            if (J != J.T).nnz != 0:
                raise ValueError("J must be a symmetric matrix.")
            if not np.all(J.diagonal() == 0):
                raise ValueError("The diagonal elements of J are not all 0, recommend transferring them to h.")

        if h is not None:
            if not isinstance(h, np.ndarray):
                raise TypeError(f"h requires numpy.array, but get {type(h)}")
            if h.shape != (J.shape[0],) and h.shape != (J.shape[0], 1):
                raise ValueError(f"h must have shape ({J.shape[0]},) or ({J.shape[0]}, 1), but got {h.shape}")
            if len(h.shape) == 1:
                h = h[:, np.newaxis]

        if x is not None:
            if not isinstance(x, np.ndarray):
                raise TypeError(f"x requires numpy.array, but get {type(x)}")
            if len(x.shape) != 2:
                raise ValueError(f"x must be a 2D array, but got shape {x.shape}")
            if x.shape[0] != J.shape[0] or x.shape[1] != batch_size:
                raise ValueError(f"x must have shape ({J.shape[0]}, {batch_size}), but got {x.shape}")

        _check_int_type("n_iter", n_iter)
        _check_value_should_not_less("n_iter", 1, n_iter)
        _check_int_type("batch_size", batch_size)
        _check_value_should_not_less("batch_size", 1, batch_size)

        if backend == "gpu-float32" and _INSTALL_TORCH:
            J = torch.tensor(J.toarray()).float()
            if J.layout != torch.sparse_csr:
                J = J.to("cuda")
                J = J.to_sparse_csr()
            if h is not None:
                h = torch.from_numpy(h).float().to("cuda")

        if backend == "npu-float32" and _INSTALL_TORCH_NPU:
            J = torch.tensor(J.toarray()).float().to("npu")
            if h is not None:
                h = torch.from_numpy(h).float().to("npu")

        self.J = J
        self.h = h
        self.x = x
        # The number of spins
        self.N = self.J.shape[0]
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.backend = backend

    def initialize(self):
        """Randomly initialize spin values."""
        if self.x is None:
            if self.backend == "cpu-float32":
                self.x = 0.02 * (np.random.rand(self.N, self.batch_size) - 0.5)
            elif self.backend == "gpu-float32":
                self.x = 0.02 * (torch.rand(self.N, self.batch_size, device="cuda") - 0.5)
            elif self.backend == "npu-float32":
                self.x = 0.02 * (torch.rand(self.N, self.batch_size).npu() - 0.5)
        else:
            if self.backend == "gpu-float32":
                self.x = torch.from_numpy(self.x).float().to("cuda")
            elif self.backend == "npu-float32":
                self.x = torch.from_numpy(self.x).float().to("npu")

    def calc_cut(self, x=None):
        r"""
        Calculate cut value.

        Args:
            x (numpy.array): The spin value with shape (N x batch_size).
                If ``None``, the initial spin will be used. Default: ``None``.
        """
        if self.backend in ["cpu-float32", 'gpu-float16', 'gpu-int8']:
            if x is None:
                sign = np.sign(self.x)
            else:
                sign = np.sign(x)
            return 0.25 * np.sum(self.J.dot(sign) * sign, axis=0) - 0.25 * self.J.sum()

        if self.backend == "gpu-float32":
            if x is None:
                sign = torch.sign(self.x)
            else:
                sign = torch.sign(x)
            return 0.25 * torch.sum(torch.sparse.mm(self.J, sign) * sign, dim=0) - 0.25 * self.J.sum()

        if self.backend == "npu-float32":
            if x is None:
                sign = torch.sign(self.x).npu()
            else:
                sign = torch.sign(x).npu()
            return 0.25 * torch.sum(torch.sparse.mm(self.J, sign) * sign, dim=0) - 0.25 * self.J.sum()

        raise ValueError("invalid backend")

    def calc_energy(self, x=None):
        r"""
        Calculate energy.

        Args:
            x (numpy.array): The spin value with shape (N x batch_size).
                If ``None``, the initial spin will be used. Default: ``None``.
        """
        if self.backend in ["cpu-float32", 'gpu-float16', 'gpu-int8']:
            if x is None:
                sign = np.sign(self.x)
            else:
                sign = np.sign(x)

            if self.h is None:
                return -0.5 * np.sum(self.J.dot(sign) * sign, axis=0)
            return -0.5 * np.sum(self.J.dot(sign) * sign, axis=0, keepdims=True) - self.h.T @ sign

        if self.backend == "gpu-float32":
            if x is None:
                sign = torch.sign(self.x)
            else:
                sign = torch.sign(x)

            if self.h is None:
                return -0.5 * torch.sum(torch.sparse.mm(self.J, sign) * sign, dim=0)
            return -0.5 * torch.sum(torch.sparse.mm(self.J, sign) * sign, dim=0, keepdim=True) - self.h.T @ sign

        if self.backend == "npu-float32":
            if x is None:
                sign = torch.sign(self.x).npu()
            else:
                sign = torch.sign(x).npu()

            if self.h is None:
                return -0.5 * torch.sum(torch.sparse.mm(self.J, sign) * sign, dim=0)
            return -0.5 * torch.sum(torch.sparse.mm(self.J, sign) * sign, dim=0, keepdim=True) - self.h.T @ sign

        raise ValueError("invalid backend")


class OverflowException(Exception):
    r"""
    Custom exception class for handling overflow errors in numerical calculations.

    Args:
        message: Exception message string, defaults to "Overflow error".
    """

    def __init__(self, message="Overflow error"):
        self.message = message
        super().__init__(self.message)
