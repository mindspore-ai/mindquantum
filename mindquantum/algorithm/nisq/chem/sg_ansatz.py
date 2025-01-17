# Copyright 2024 Huawei Technologies Co., Ltd
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
"""SG ansatz."""

import random

import numpy as np

from mindquantum.algorithm.nisq._ansatz import Ansatz
from mindquantum.core.circuit import add_prefix, add_suffix
from mindquantum.core.gates import RX, RY, RZ
from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_value_should_not_less,
)


class SGAnsatz(Ansatz):
    """
    SG ansatz for 1D quantum systems.

    The SG ansatz consists of multiple variational quantum circuit blocks, each of which
    is a parametrized quantum circuit applied to several adjacent qubits. With such a structure,
    the SG ansatz naturally adapts to quantum many-body problems.

    Specifically, for 1D quantum systems, the SG ansatz can efficiently generate any matrix product states
    with a fixed bond dimension. For 2D systems, the SG ansatz can generate string-bond states.

    For more detail, please refers `A sequentially generated variational quantum circuit with polynomial complexity
    <https://arxiv.org/abs/2305.12856>`_.

    Args:
        nqubits (int): Number of qubits in the ansatz.
        k (int): log(R) + 1, where R is the bond dimension of a MPS state.
        nlayers (int): Number of layers in each block. Default: ``1``.
        prefix (str): The prefix of parameters. Default: ``''``.
        suffix (str): The suffix of parameters. Default: ``''``.

    Examples:
        >>> from mindquantum.core.algorithm import SGAnsatz
        >>> sg = SGAnsatz(4, 2, 1)
        >>> sg.circuit
              ┏━━━━━━━━━━━┓ ┏━━━━━━━━━━━━┓
        q0: ──┨ RX(a1_00) ┠─┨ RZ(b1_000) ┠────────■─────────────────────────────────────────────────────────────────────
              ┗━━━━━━━━━━━┛ ┗━━━━━━━━━━━━┛        ┃
              ┏━━━━━━━━━━━━┓               ┏━━━━━━┻━━━━━┓ ┏━━━━━━━━━━━━┓
        q1: ──┨ RY(b1_001) ┠───────────────┨ RX(b2_000) ┠─┨ RY(b1_101) ┠────────■───────────────────────────────────────
              ┗━━━━━━━━━━━━┛               ┗━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━┛        ┃
              ┏━━━━━━━━━━━━┓                                             ┏━━━━━━┻━━━━━┓ ┏━━━━━━━━━━━━┓
        q2: ──┨ RY(b1_102) ┠─────────────────────────────────────────────┨ RX(b2_101) ┠─┨ RX(b1_202) ┠────────■─────────
              ┗━━━━━━━━━━━━┛                                             ┗━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━┛        ┃
              ┏━━━━━━━━━━━━┓                                                                           ┏━━━━━━┻━━━━━┓
        q3: ──┨ RY(b1_203) ┠───────────────────────────────────────────────────────────────────────────┨ RZ(b2_202) ┠───
              ┗━━━━━━━━━━━━┛                                                                           ┗━━━━━━━━━━━━┛
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        nqubits,
        k,
        nlayers=1,
        prefix: str = '',
        suffix: str = '',
    ):
        """Initialize a SGAnsatz object."""
        _check_int_type('nlayers', nlayers)
        _check_int_type('k', k)
        _check_value_should_not_less('nlayers', 1, nlayers)
        _check_input_type('prefix', str, prefix)
        _check_input_type('suffix', str, suffix)
        self.prefix = prefix
        self.suffix = suffix
        self.nlayers = nlayers
        self.nqubits = nqubits
        self.k = k
        super().__init__('SGAnsatz', nqubits)

    def _implement(self):
        """Implement of SG ansatz."""
        self._circuit = self._random_block()
        if self.prefix:
            self._circuit = add_prefix(self._circuit, self.prefix)
        if self.suffix:
            self._circuit = add_suffix(self._circuit, self.suffix)

    def _random_block(self):
        """Generate a random quantum circuit block."""
        if self._circuit:
            raise TypeError("There already exists a circuit ansatz!")
        # 0:X 1:Y, 2:Z
        rand_list = [0, 1, 2]

        # Construct the 1-st block on k-1 sites
        for j in range(self.nlayers):
            # firstly apply H gate on each sites
            for i in range(self.k - 1):
                flg = random.choice(rand_list)
                if flg == 0:
                    self._circuit += RX(f'a1_{j}{i}').on(i)
                elif flg == 1:
                    self._circuit += RY(f'a1_{j}{i}').on(i)
                else:
                    self._circuit += RZ(f'a1_{j}{i}').on(i)

            if self.k != 2:
                for i in range(self.k - 2):
                    flg = random.choice(rand_list)
                    if flg == 0:
                        self._circuit += RX(f'a2_{j}{i}').on(i + 1, i)
                    elif flg == 1:
                        self._circuit += RY(f'a2_{j}{i}').on(i + 1, i)
                    else:
                        self._circuit += RZ(f'a2_{j}{i}').on(i + 1, i)

        # Construct the N-k+1 (k-1)-local block
        for d in range(self.nqubits - self.k + 1):
            for j in range(self.nlayers):
                for i in range(d, d + self.k):
                    flg = random.choice(rand_list)
                    if flg == 0:
                        self._circuit += RX(f'b1_{d}{j}{i}').on(i)
                    elif flg == 1:
                        self._circuit += RY(f'b1_{d}{j}{i}').on(i)
                    else:
                        self._circuit += RZ(f'b1_{d}{j}{i}').on(i)

                for i in range(d, d + self.k - 1):
                    flg = random.choice(rand_list)
                    if flg == 0:
                        self._circuit += RX(f'b2_{d}{j}{i}').on(i + 1, i)
                    elif flg == 1:
                        self._circuit += RY(f'b2_{d}{j}{i}').on(i + 1, i)
                    else:
                        self._circuit += RZ(f'b2_{d}{j}{i}').on(i + 1, i)

        return self._circuit


class SGAnsatz2D(Ansatz):
    """
    SG ansatz for 2D quantum systems.

    The SG ansatz consists of multiple variational quantum circuit blocks, each of which
    is a parametrized quantum circuit applied to several adjacent qubits. With such a structure,
    the SG ansatz naturally adapts to quantum many-body problems.

    Specifically, for 1D quantum systems, the SG ansatz can efficiently generate any matrix product states
    with a fixed bond dimension. For 2D systems, the SG ansatz can generate string-bond states.

    For more detail, please refers `A sequentially generated variational quantum circuit with polynomial complexity
    <https://arxiv.org/abs/2305.12856>`_.

    Args:
        nqubits (int): Number of qubits in the ansatz.
        k (int): log(R) + 1, where R is the fixed bond dimension.
        line_set (list, optional): A list set of qubits' lines to generate a specific type of string-bond state.
            If None, will be generated automatically as a 1×N grid where N equals to nqubits. Default: ``None``.
        nlayers (int): Number of layers in each block. Default: ``1``.
        prefix (str): The prefix of parameters. Default: ``''``.
        suffix (str): The suffix of parameters. Default: ``''``.

    Examples:
        >>> from mindquantum.algorithm import SGAnsatz2D
        >>> # Method 1: Create from 2D grid (recommended)
        >>> sg = SGAnsatz2D.from_grid(nrow=2, ncol=3, k=2)
        >>> print(len(sg.circuit))  # Number of quantum gates in the ansatz
        32

        >>> # Method 2: Create with custom line set
        >>> line_set = SGAnsatz2D.generate_line_set(2, 3)  # [[0,3,4,1,2,5], [0,1,2,5,4,3]]
        >>> sg = SGAnsatz2D(nqubits=6, k=2, line_set=line_set)
    """

    @classmethod
    def from_grid(cls, nrow, ncol, k, nlayers=1, prefix='', suffix=''):
        """Create SGAnsatz2D from a 2D grid configuration.

        This is the recommended way to create a SGAnsatz2D instance for 2D quantum systems.
        It automatically generates the appropriate line set based on the grid dimensions.

        Args:
            nrow (int): Number of rows in the 2D grid
            ncol (int): Number of columns in the 2D grid
            k (int): log(R) + 1, where R is the fixed bond dimension
            nlayers (int): Number of layers in each block. Default: 1
            prefix (str): The prefix of parameters. Default: ''
            suffix (str): The suffix of parameters. Default: ''

        Returns:
            SGAnsatz2D: A new instance configured for the specified 2D grid

        Examples:
            >>> from mindquantum.algorithm import SGAnsatz2D
            >>> sg = SGAnsatz2D.from_grid(nrow=2, ncol=3, k=2)
            >>> print(len(sg.circuit))  # Number of quantum gates in the ansatz
            32
        """
        nqubits = nrow * ncol
        line_set = cls.generate_line_set(nrow, ncol)
        return cls(nqubits, k, line_set, nlayers, prefix, suffix)

    def __init__(self, nqubits, k, line_set=None, nlayers=1, prefix='', suffix=''):
        """Initialize directly with number of qubits and custom line_set."""
        _check_int_type('nlayers', nlayers)
        _check_int_type('nqubits', nqubits)
        _check_int_type('k', k)
        _check_value_should_not_less('nlayers', 1, nlayers)
        _check_input_type('prefix', str, prefix)
        _check_input_type('suffix', str, suffix)
        self.prefix = prefix
        self.suffix = suffix
        self.nlayers = nlayers
        self.k = k
        if line_set is None:
            line_set = self.generate_line_set(1, nqubits)
        _check_input_type('line_set', list, line_set)
        for line in line_set:
            _check_input_type('line', list, line)
            for qubit in line:
                _check_int_type('qubit', qubit)
            if len(line) != nqubits:
                raise ValueError(f"The length of line {line} is not equal to nqubits {nqubits}!")
        self.line_set = line_set
        self.nqubits = nqubits
        super().__init__('SGAnsatz2D', nqubits)

    def _implement(self):
        """Implement of SG ansatz."""
        for chain_idx in range(len(self.line_set)):
            self._circuit = self._random_block(chain_idx, self.line_set[chain_idx])
        if self.prefix:
            self._circuit = add_prefix(self._circuit, self.prefix)
        if self.suffix:
            self._circuit = add_suffix(self._circuit, self.suffix)

    @classmethod
    def generate_line_set(cls, nrow, ncol):
        """Generate snake-like traversal patterns for 2D quantum systems.

        This method generates two different traversal paths for a 2D quantum system:
        1. Column-wise snake pattern: traverses each column alternating between up and down
        2. Row-wise snake pattern: traverses each row alternating between left and right

        Args:
            nrow (int): Number of rows in the 2D grid
            ncol (int): Number of columns in the 2D grid

        Returns:
            list: A list containing two traversal paths, where each path is a list of qubit indices.
                  The first path is column-wise, the second is row-wise.

        Examples:
            >>> # For a 2x3 grid with qubits numbered as:
            >>> # 0 1 2
            >>> # 3 4 5
            >>> line_set = SGAnsatz2D.generate_line_set(2, 3)
            >>> print(line_set)
            >>> # Output: [[0,3,4,1,2,5], [0,1,2,5,4,3]]
        """
        a = np.arange(0, nrow * ncol, 1)
        b = a.reshape(nrow, ncol)
        b = b.tolist()
        line1 = []
        for i in range(ncol):
            if i % 2 == 0:
                for j in range(nrow):
                    line1.append(b[j][i])
            else:
                for j in range(nrow - 1, -1, -1):
                    line1.append(b[j][i])

        line2 = []
        for i in range(nrow):
            if i % 2 == 0:
                for j in range(ncol):
                    line2.append(b[i][j])
            else:
                for j in range(ncol - 1, -1, -1):
                    line2.append(b[i][j])

        line_set = []
        line_set.append(line1)
        line_set.append(line2)
        return line_set

    def _random_block(self, idx, chain):
        """Generate a random quantum circuit block."""
        rand_list = [0, 1, 2]

        # Construct the 1-st block on k-1 sites
        for j in range(self.nlayers):
            # firstly apply H gate on each sites
            for i in range(self.k - 1):
                flg = random.choice(rand_list)
                if flg == 0:
                    self._circuit += RX(f'a1_{idx}{j}{i}').on(chain[i])
                elif flg == 1:
                    self._circuit += RY(f'a1_{idx}{j}{i}').on(chain[i])
                else:
                    self._circuit += RZ(f'a1_{idx}{j}{i}').on(chain[i])

            if self.k != 2:
                for i in range(self.k - 2):
                    flg = random.choice(rand_list)
                    if flg == 0:
                        self._circuit += RX(f'a2_{idx}{j}{i}').on(chain[i + 1], chain[i])
                    elif flg == 1:
                        self._circuit += RY(f'a2_{idx}{j}{i}').on(chain[i + 1], chain[i])
                    else:
                        self._circuit += RZ(f'a2_{idx}{j}{i}').on(chain[i + 1], chain[i])

        # Construct the N-k+1 (k-1)-local block
        for d in range(len(chain) - self.k + 1):
            for j in range(self.nlayers):
                for i in range(d, d + self.k):
                    flg = random.choice(rand_list)
                    if flg == 0:
                        self._circuit += RX(f'b1_{idx}{d}{j}{i}').on(chain[i])
                    elif flg == 1:
                        self._circuit += RY(f'b1_{idx}{d}{j}{i}').on(chain[i])
                    else:
                        self._circuit += RZ(f'b1_{idx}{d}{j}{i}').on(chain[i])

                for i in range(d, d + self.k - 1):
                    flg = random.choice(rand_list)
                    if flg == 0:
                        self._circuit += RX(f'b2_{idx}{d}{j}{i}').on(chain[i + 1], chain[i])
                    elif flg == 1:
                        self._circuit += RY(f'b2_{idx}{d}{j}{i}').on(chain[i + 1], chain[i])
                    else:
                        self._circuit += RZ(f'b2_{idx}{d}{j}{i}').on(chain[i + 1], chain[i])

        return self._circuit
