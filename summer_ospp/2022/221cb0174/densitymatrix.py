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

# This code is part of Mindquantum.
# Density matrix / mixed state module.
# ============================================================================

from ..circuit import Circuit
import numpy as np
# import mindspore as ms
# import mindspore.ops as ops


class DensityMatrix():
    """
    The density matrix class
    """

    def __init__(self, data):
        """Initialize a density matrix object.

        Args:
            data (np.ndarray or list or Circuit):

        """
        if isinstance(data, Circuit):
            # obtain the matrix representation of the input circuit
            circ_array = data.matrix()
            n_qubit = circ_array.shape[0]
            # the initial state is assumed to be |00,...,0>
            init_state = np.zeros(n_qubit, dtype=complex)
            init_state[0] = 1. + 0.j
            # The mindspore Tensor MatMul, and outer product do not support complex data type yet.
            # So I will leave it here temporarily.
            # circ_array = ms.Tensor(circ_array)
            # init_state = ms.Tensor(init_state)
            # matmul = ops.MatMul()
            # ket = matmul(circ_array, init_state)
            # conj = ops.Conj()
            # bra = conj(ket)
            ket = circ_array @ init_state
            bra = np.conj(ket)
            self._matrix = np.outer(ket, bra)
        elif isinstance(data, (list, np.ndarray)):
            data = np.asarray(data, dtype=complex)
            if data.ndim == 2:
               self._matrix = data
            if data.ndim == 1:
               self._matrix = np.outer(data, np.conj(data))

    @property
    def matrix(self):
        """Return the matrix representation"""
        return self._matrix

    def evolve(self, operator, qargs=None):
        # evolve the density matrix by the input operator
        if hasattr(operator, "matrix"):
            op_array = operator.matrix()
            assert op_array.shape == self._matrix.shape, "apply operator error, shape mismatch"
            if qargs is None:
                self._matrix = np.dot(op_array, self._matrix).dot(op_array.T.conj())
            else:
                # only apply gate to specific qubits, reducing complexity compared to the above
                # which requires tensor dot along specific axis
                raise NotImplementedError
        else:
            print("make sure the input operation has attribute matrix")
            assert False

    def conjugate(self):
        """Return the conjugate of the density matrix."""
        return DensityMatrix(np.conj(self._matrix))

    def trace(self):
        """Return the trace of the density matrix."""
        return np.trace(self._matrix)

    def purity(self):
        """Return the purity of the mixed state."""
        return np.trace(np.dot(self._matrix, self._matrix))


