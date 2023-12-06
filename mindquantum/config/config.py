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
"""Global config for mindquantum."""
import numbers

import numpy as np

from mindquantum.utils.type_value_check import _check_input_type

__all__ = ['context']

_GLOBAL_MAT_VALUE = {
    'X': np.array([[0, 1], [1, 0]]),
    'Y': np.array([[0, -1j], [1j, 0]]),
    'Z': np.array([[1, 0], [0, -1]]),
    'I': np.array([[1, 0], [0, 1]]),
    'H': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    'S': np.array([[1, 0], [0, 1j]]),
    'T': np.array([[1, 0], [0, (1 + 1j) / np.sqrt(2)]]),
    'SX': np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2,
    'ISWAP': np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]),
    'SWAP': np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
    'CNOT': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
}

_GLOBAL_CONFIG = {
    'PRECISION': 1e-10,
}


class context:  # pylint: disable=invalid-name # noqa: N801
    """
    Set context for running environment.

    See the below table for detail:

    +------------------------+-----------------------------+
    |Configuration Parameters|Description                  |
    +========================+=============================+
    |precision               |Set the atol number precision|
    +------------------------+-----------------------------+

    Note:
        For every parameter, a setter or a getter method is implemented.
    """

    @staticmethod
    def set_precision(atol):
        """
        Set the number precision for mindquantum.

        For example, `is_two_number_close` will use this precision to determine whether
        two number is close to each other.

        Examples:
            >>> from mindquantum import context
            >>> context.set_precision(1e-3)
            >>> context.get_precision()
            0.001
        """
        _check_input_type('atol', numbers.Real, atol)
        _GLOBAL_CONFIG['PRECISION'] = atol

    @staticmethod
    def get_precision():
        """
        Get the number precision.

        Returns:
            float, the number precision.
        """
        return _GLOBAL_CONFIG['PRECISION']
