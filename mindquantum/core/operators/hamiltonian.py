# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Hamiltonian module."""
# pylint: disable=too-many-instance-attributes,no-member
from enum import Enum
import warnings

import numpy as np
import scipy.sparse as sp

import mindquantum as mq
from mindquantum import mqbackend as mb


class HowTo(Enum):
    """Hamiltonian type."""  # Need to improve that...

    ORIGIN = 0
    BACKEND = 1
    FRONTEND = 2


class Hamiltonian:
    """
    A QubitOperator hamiltonian wrapper.

    Args:
        hamiltonian (Union[QubitOperator, scipy.sparse]): The pauli qubit operator or a sparse matrix.
        dtype (mindquantum.dtype): data type of hamiltonian. Default: ``None``.

    Examples:
        >>> from mindquantum.core.operators import QubitOperator, Hamiltonian
        >>> ham = Hamiltonian(QubitOperator('Z0 Y1', 0.3))
        >>> ham
        3/10 [Z0 Y1]
    """

    def __init__(self, hamiltonian, dtype=None):
        """Initialize a Hamiltonian object."""
        # pylint: disable=import-outside-toplevel
        from .qubit_operator import QubitOperator as HiQOperator
        from .utils import count_qubits

        support_type = (HiQOperator, sp.csr_matrix)
        if not isinstance(hamiltonian, support_type):
            raise TypeError(f"Require a QubitOperator or a csr_matrix, but get {type(hamiltonian)}!")
        if isinstance(hamiltonian, sp.csr_matrix):
            if dtype is None:
                dtype = mq.to_mq_type(hamiltonian.dtype)
            else:
                if dtype != mq.to_mq_type(hamiltonian.dtype):
                    hamiltonian = hamiltonian.astype(mq.to_np_type(dtype))
            self.ham_dtype = dtype
            if len(hamiltonian.shape) != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
                raise ValueError(
                    f"Hamiltonian requires a two dimension square csr_matrix, but get shape {hamiltonian.shape}"
                )
            if np.log2(hamiltonian.shape[0]) % 1 != 0:
                raise ValueError(f"size of hamiltonian sparse matrix should be power of 2, but get {hamiltonian.shape}")
            self._hamiltonian = None
            self._ham_termlist = None
            self._sparse_matrix = hamiltonian
            self.how_to = HowTo.FRONTEND
            self.n_qubits = int(np.log2(self._sparse_matrix.shape[0]))
        else:
            hamiltonian: HiQOperator
            if dtype is None:
                dtype = hamiltonian.dtype
            else:
                if dtype != hamiltonian.dtype:
                    hamiltonian = hamiltonian.astype(dtype)
            self.ham_dtype = dtype
            self._hamiltonian = hamiltonian
            self._sparse_matrix = None
            self.how_to = HowTo.ORIGIN
            self.n_qubits = count_qubits(hamiltonian)
            has_warned = False
            self._ham_termlist = []
            for i, j in self._hamiltonian.terms.items():
                if not j.is_const():
                    raise ValueError("Hamiltonian cannot be parameterized.")
                if abs(j.const.imag) > 1e-8 and not has_warned:
                    warnings.warn(
                        f"Hamiltonian coefficients must be real numbers. Imaginary part will be discarded.",
                        UserWarning,
                        stacklevel=2,
                    )
                    has_warned = True
                self._ham_termlist.append((i, j.const.real))

        self.ham_cpp = None
        self.herm_ham_cpp = None

    def __str__(self):
        """Return a string representation of the object."""
        if self.how_to == HowTo.FRONTEND:
            return self._sparse_matrix.__str__()
        return self._hamiltonian.__str__()

    def __repr__(self):
        """Return a string representation of the object."""
        if self.how_to == HowTo.FRONTEND:
            return self._sparse_matrix.__str__()
        return self._hamiltonian.__repr__()

    def __getstate__(self):
        """Create a dictionary that will be pickled."""
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if self.ham_cpp:
            state['ham_cpp'] = True
        if self.herm_ham_cpp:
            state['herm_ham_cpp'] = True
        return state

    def __setstate__(self, state):
        """Restore instance state from the unpickled state."""
        self.__dict__.update(state)
        # Add the missing 'ham_cpp' and 'herm_ham_cpp' entries
        if self.ham_cpp:
            self.ham_cpp = self.get_cpp_obj(hermitian=False)
        if self.herm_ham_cpp:
            self.herm_ham_cpp = self.get_cpp_obj(hermitian=True)

    def sparse(self, n_qubits=1):
        """
        Calculate the sparse matrix of this hamiltonian in pqc operator.

        Args:
            n_qubits (int): The total qubit of this hamiltonian, only need when mode is
                'frontend'. Default: ``1``.
        """
        if self.how_to != HowTo.ORIGIN:
            raise ValueError('Already a sparse hamiltonian.')
        if n_qubits < self.n_qubits:
            raise ValueError(f"Can not sparse a {self.n_qubits} qubits hamiltonian to {n_qubits} hamiltonian.")
        self.n_qubits = n_qubits
        self.how_to = HowTo.BACKEND
        return self

    @property
    def dtype(self):
        """Get hamiltonian data type."""
        return self.ham_dtype

    @property
    def hamiltonian(self):
        """
        Get the QubitOperator of the Hamiltonian.

        Raises:
            ValueError: If this Hamiltonian was constructed from a sparse matrix, cannot get QubitOperator.
        """
        if self.how_to == HowTo.FRONTEND:
            raise ValueError(
                "Cannot get QubitOperator: "
                "this Hamiltonian was constructed from a sparse matrix. "
                "To work with QubitOperator, please construct the Hamiltonian using QubitOperator instead."
            )
        return self._hamiltonian

    @property
    def ham_termlist(self):
        """
        Get the Pauli term list of the Hamiltonian.

        Raises:
            ValueError: If this Hamiltonian was constructed from a sparse matrix, cannot get Pauli term list.
        """
        if self.how_to == HowTo.FRONTEND:
            raise ValueError(
                "Cannot get Pauli term list: "
                "this Hamiltonian was constructed from a sparse matrix. "
                "To work with Pauli term list, please construct the Hamiltonian using QubitOperator instead."
            )
        return self._ham_termlist

    @property
    def sparse_matrix(self):
        """
        Get the sparse matrix representation of the Hamiltonian.

        Returns:
            scipy.sparse.csr_matrix, the sparse matrix representation of Hamiltonian.
        """
        if self.how_to != HowTo.FRONTEND:
            self._sparse_matrix = self._hamiltonian.matrix()
            return self._sparse_matrix
        return self._sparse_matrix

    def astype(self, dtype):
        """
        Convert hamiltonian to other type.

        Args:
            dtype (mindquantum.dtype): the new type of hamiltonian.
        """
        if self.how_to == HowTo.FRONTEND:
            return Hamiltonian(self._sparse_matrix, dtype)
        return Hamiltonian(self._hamiltonian, dtype)

    def get_cpp_obj(self, hermitian=False):
        """
        Get the underlying C++ object.

        Args:
            hermitian (bool): Whether to get the cpp object of this hamiltonian in hermitian version.
        """
        if mq.is_double_precision(self.dtype):
            backend_module = mb.double
        else:
            backend_module = mb.float
        if not hermitian:
            if self.ham_cpp is None:
                if self.how_to == HowTo.ORIGIN:
                    ham = backend_module.hamiltonian(self._ham_termlist)
                elif self.how_to == HowTo.BACKEND:
                    ham = backend_module.hamiltonian(self._ham_termlist, self.n_qubits)
                else:
                    dim = self._sparse_matrix.shape[0]
                    nnz = self._sparse_matrix.nnz
                    csr_mat = backend_module.csr_hd_matrix(
                        dim, nnz, self._sparse_matrix.indptr, self._sparse_matrix.indices, self._sparse_matrix.data
                    )
                    ham = backend_module.hamiltonian(csr_mat, self.n_qubits)
                self.ham_cpp = ham
            return self.ham_cpp
        if self.how_to in (HowTo.BACKEND, HowTo.ORIGIN):
            return self.get_cpp_obj()
        if self.herm_ham_cpp is None:
            if self.how_to != HowTo.FRONTEND:
                raise ValueError("Hamiltonian is not a sparse matrix, cannot get hermitian cpp object.")
            herm_sparse_mat = self._sparse_matrix.conjugate().T.tocsr()
            dim = herm_sparse_mat.shape[0]
            nnz = herm_sparse_mat.nnz
            csr_mat = backend_module.csr_hd_matrix(
                dim, nnz, herm_sparse_mat.indptr, herm_sparse_mat.indices, herm_sparse_mat.data
            )
            self.herm_ham_cpp = backend_module.hamiltonian(csr_mat, self.n_qubits)
        return self.herm_ham_cpp


__all__ = ['Hamiltonian']
