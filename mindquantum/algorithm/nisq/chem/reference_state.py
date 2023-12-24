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

# Author : Xiaoxiao Xiao <xxxwwzi@163.com>
#          Zhendong Li <zhendongli@bnu.edu.cn>
# Affiliation: College of Chemistry, Beijing Normal University
# Publication: Xiao, X., Zhao, H., Ren, J., Fang, W. H., & Li, Z. (2023). arXiv:2307.03563.
"""Get a reference state preparation circuit."""
import sys
from typing import Iterable, Optional, Union

# pylint: disable=wrong-import-position
if sys.version_info < (3, 8):
    from typing_extensions import Literal, get_args
else:
    from typing import Literal, get_args

import numpy as np

from mindquantum.core.circuit import UN, Circuit
from mindquantum.core.gates import BarrierGate, H, X
from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_value_should_not_less,
)

AVA_REF = Literal['HF', 'Neel', 'Bell', 'AllH']  # pylint: disable=invalid-name


# pylint: disable=too-many-branches
def get_reference_circuit(
    n_qubits: int,
    n_ele_alpha: Optional[int] = None,
    n_ele_beta: Optional[int] = None,
    ref: Union[AVA_REF, Iterable] = 'AllH',
):
    """
    Get preparation circuit for reference circuit according to different methods.

    The available methods are:

    +--------+------------------------------------------------------------------------------+
    | Method |   Description                                                                |
    +========+==============================================================================+
    | HF     |   Generate the Hartree-Fock (HF) reference state, where electrons occupy     |
    |        |   the lowest energy orbitals, suitable for Hartree-Fock (HF) orbitals.       |
    +--------+------------------------------------------------------------------------------+
    | Neel   |   Generate the Neel reference state, where electrons with different spins    |
    |        |   occupy adjacent orbitals, suitable for orthonormalized atomic orbitals.    |
    +--------+------------------------------------------------------------------------------+
    | Bell   |   Generate the tensor product of the Bell state as the reference state.      |
    +--------+------------------------------------------------------------------------------+
    | AllH   |   Generate the uniform superposition state as the reference state.           |
    +--------+------------------------------------------------------------------------------+

    Args:
        n_qubits (int): The total qubit number.
        n_ele_alpha (int): The number of alpha electrons. Default: ``None``.
        n_ele_beta (int): The number of beta electrons. Default: ``None``.
        ref (Union[str, Iterable]): The method to construct reference state. If it is a string, it should
            be one of ``'HF'``, ``'Neel'``, ``'Bell'``, ``'AllH'``. Otherwise, it should be a iterator
            of int, and we place a :class:`~.core.gates.RX` gate on each qubit. Default: ``'AllH'``.

    Examples:
        >>> from mindquantum.algorithm.nisq import get_reference_circuit
        >>> get_reference_circuit(4, 1, 1, 'HF')
              ┏━━━┓
        q0: ──┨╺╋╸┠─▓───
              ┗━━━┛ ▓
              ┏━━━┓ ▓
        q1: ──┨╺╋╸┠─▓───
              ┗━━━┛ ▓
                    ▓
        q2: ────────▓───
                    ▓
                    ▓
        q3: ────────▓───
    """
    ava_ref = get_args(AVA_REF)
    _check_int_type('n_qubits', n_qubits)
    _check_value_should_not_less('n_qubits', 0, n_qubits)
    if n_ele_alpha is not None:
        _check_int_type('n_ele_alpha', n_ele_alpha)
        _check_value_should_not_less('n_ele_alpha', 0, n_ele_alpha)
    if n_ele_beta is not None:
        _check_int_type('n_ele_beta', n_ele_beta)
        _check_value_should_not_less('n_ele_beta', 0, n_ele_beta)

    full_barrier = Circuit() + BarrierGate().on(list(range(n_qubits)))
    if isinstance(ref, str):
        if ref not in ava_ref:
            raise ValueError(f'ref should be one of {ava_ref}, but get {ref}')
        if ref == 'HF':
            _check_input_type('n_ele_alpha', int, n_ele_alpha)
            _check_input_type('n_ele_beta', int, n_ele_beta)
            if n_ele_alpha + n_ele_beta > n_qubits:
                raise ValueError(
                    f"Total electrons ({n_ele_alpha} + {n_ele_beta}) can not be greater than n_qubits ({n_qubits})"
                )
            x_poi = np.arange(0, 2 * n_ele_alpha, 2).tolist() + np.arange(1, 2 * n_ele_beta + 1, 2).tolist()
            return UN(X, x_poi) + full_barrier
        if ref == 'Neel':
            _check_input_type('n_ele_alpha', int, n_ele_alpha)
            _check_input_type('n_ele_beta', int, n_ele_beta)
            if 2 * (n_ele_alpha + n_ele_beta) > n_qubits:
                raise ValueError(
                    f"The position of the largest occupied qubit (2 * ({n_ele_alpha} + {n_ele_beta})"
                    f" exceeds the total number of qubits ({n_qubits})."
                )
            temp = []
            if n_ele_alpha != 0:
                n_b = 0
                for n_a in range(n_ele_alpha):
                    temp += [1, 0]
                    if n_b < n_ele_beta:
                        temp += [0, 1]
                        n_b += 1
            else:
                n_a = 0
                for n_b in range(n_ele_beta):
                    temp += [1, 0]
                    if n_a < n_ele_alpha:
                        temp += [0, 1]
                        n_a += 1
            x_poi = [idx for idx, i in enumerate(temp) if i == 1]
            return UN(X, x_poi) + full_barrier
        if ref == 'Bell':
            even = (np.array(range(n_qubits // 2)) * 2).tolist()
            odd = (np.array(range(n_qubits // 2)) * 2 + 1).tolist()
            return UN(X, range(n_qubits)) + UN(H, even) + UN(X, odd, even) + full_barrier
        if ref == 'AllH':
            return UN(H, range(n_qubits)) + full_barrier
    return UN(X, ref) + full_barrier
