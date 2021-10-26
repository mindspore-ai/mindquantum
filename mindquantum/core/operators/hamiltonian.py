# -*- coding: utf-8 -*-
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

import scipy.sparse as sp
import numpy as np
from projectq.ops import QubitOperator as pq_operator
from openfermion.ops import QubitOperator as of_operator
from mindquantum import mqbackend as mb

MODE = {'origin': 0, 'backend': 1, 'frontend': 2}
EDOM = {0: 'origin', 1: 'backend', 2: 'frontend'}


class Hamiltonian:
    """
    A QubitOperator hamiltonian wrapper.

    Args:
        hamiltonian (QubitOperator): The pauli word qubit operator.

    Examples:
        >>> from mindquantum.core.operators import QubitOperator
        >>> from mindquantum import Hamiltonian
        >>> ham = Hamiltonian(QubitOperator('Z0 Y1', 0.3))
    """
    def __init__(self, hamiltonian):
        from mindquantum.core.operators import QubitOperator as hiq_operator
        from mindquantum.core.operators.utils import count_qubits
        support_type = (pq_operator, of_operator, hiq_operator, sp.csr_matrix)
        if not isinstance(hamiltonian, support_type):
            raise TypeError("Require a QubitOperator or a csr_matrix, but get {}!".format(type(hamiltonian)))
        if isinstance(hamiltonian, sp.csr_matrix):
            if len(hamiltonian.shape) != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
                raise ValueError(
                    f"Hamiltonian requires a two dimension square csr_matrix, but get shape {hamiltonian.shape}")
            if np.log2(hamiltonian.shape[0]) % 1 != 0:
                raise ValueError(f"size of hamiltonian sparse matrix should be power of 2, but get {hamiltonian.shape}")
            self.hamiltonian = hiq_operator('')
            self.sparse_mat = hamiltonian
            self.how_to = MODE['frontend']
            self.n_qubits = int(np.log2(self.sparse_mat.shape[0]))
        else:
            self.hamiltonian = hamiltonian
            self.sparse_mat = sp.csr_matrix(np.eye(2, dtype=np.complex64))
            self.how_to = MODE['origin']
            self.n_qubits = count_qubits(hamiltonian)
        self.ham_termlist = [(i, j) for i, j in self.hamiltonian.terms.items()]

    def __str__(self):
        return self.hamiltonian.__str__()

    def __repr__(self):
        return self.hamiltonian.__repr__()

    def sparse(self, n_qubits=1):
        """
        Calculate the sparse matrix of this hamiltonian in pqc operator

        Args:
            n_qubits (int): The total qubit of this hamiltonian, only need when mode is
                'frontend'. Default: 1.
        """
        if EDOM[self.how_to] != 'origin':
            raise ValueError('Already a sparse hamiltonian.')
        if n_qubits < self.n_qubits:
            raise ValueError(f"Can not sparse a {self.n_qubits} qubits hamiltonian to {n_qubits} hamiltonian.")
        self.n_qubits = n_qubits
        self.how_to = MODE['backend']
        return self

    def get_cpp_obj(self, hermitian=False):
        """
        get_cpp_obj

        Args:
            hermitian (bool): Whether to get the cpp object of this hamiltonian in hermitian version.
        """
        if not hermitian:
            if not hasattr(self, 'ham_cpp'):
                if self.how_to == MODE['origin']:
                    ham = mb.hamiltonian(self.ham_termlist)
                elif self.how_to == MODE['backend']:
                    ham = mb.hamiltonian(self.ham_termlist, self.n_qubits)
                else:
                    dim = self.sparse_mat.shape[0]
                    nnz = self.sparse_mat.nnz
                    csr_mat = mb.csr_hd_matrix(dim, nnz, self.sparse_mat.indptr, self.sparse_mat.indices,
                                               self.sparse_mat.data)
                    ham = mb.hamiltonian(csr_mat, self.n_qubits)
                self.ham_cpp = ham
            return self.ham_cpp
        if self.how_to == MODE['backend'] or self.how_to == MODE['origin']:
            return self.get_cpp_obj()
        if not hasattr(self, 'herm_ham_cpp'):
            herm_sparse_mat = self.sparse_mat.conjugate().T.tocsr()
            dim = herm_sparse_mat.shape[0]
            nnz = herm_sparse_mat.nnz
            csr_mat = mb.csr_hd_matrix(dim, nnz, herm_sparse_mat.indptr, herm_sparse_mat.indices, herm_sparse_mat.data)
            self.herm_ham_cpp = mb.hamiltonian(csr_mat, self.n_qubits)
        return self.herm_ham_cpp


__all__ = ['Hamiltonian']
