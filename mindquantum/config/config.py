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

"""Global config for MindQuantum."""

import numbers

import numpy as np

from mindquantum.utils.type_value_check import _check_input_type

__all__ = ['set_context', 'get_context']

_GLOBAL_MAT_VALUE = {
    'X': np.array([[0, 1], [1, 0]]),
    'Y': np.array([[0, -1j], [1j, 0]]),
    'Z': np.array([[1, 0], [0, -1]]),
    'I': np.array([[1, 0], [0, 1]]),
    'H': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    'S': np.array([[1, 0], [0, 1j]]),
    'T': np.array([[1, 0], [0, (1 + 1j) / np.sqrt(2)]]),
    'ISWAP': np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]),
    'SWAP': np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
    'CNOT': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
}

_GLOBAL_CONFIG = {
    'DTYPE': 'double',
    'PRECISION': 1e-10,
    'DEVICE_TARGET': 'CPU',
}


class _Context:
    """Set context for running environment."""

    @staticmethod
    def set_device_target(device_target: str):
        """Set the target device to run, support "GPU", and "CPU"."""
        _check_input_type('device_target', str, device_target)
        if device_target not in ['CPU', 'GPU']:
            raise ValueError(f"device_target should be 'CPU' or 'GPU', but get {device_target}")
        _GLOBAL_CONFIG['DEVICE_TARGET'] = device_target

    @staticmethod
    def get_device_target() -> str:
        """Get running device target."""
        return _GLOBAL_CONFIG.get('DEVICE_TARGET')

    @staticmethod
    def set_dtype(dtype: str):
        """
        Set the simulation backend data precision of mindquantum.

        Simulator like mqvector or mqvector_gpu will use this precision.

        Args:
            dtype (str): data type precision of mindquantum framework, should be
                'float' or 'double'.
        """
        _check_input_type('dtype', str, dtype)
        if dtype not in ['float', 'double']:
            raise ValueError(f"dtype should be 'float' or 'double', but get {dtype}")
        _GLOBAL_CONFIG['DTYPE'] = dtype

    @staticmethod
    def get_dtype() -> str:
        """
        Get the data type precision of mindquantum framework.

        Returns:
            str, precision name of mindquantum.
        """
        return _GLOBAL_CONFIG.get('DTYPE')

    @staticmethod
    def set_precision(atol):
        """
        Set the number precision for mindquantum.

        For example, `is_two_number_close` will use this precision to determine whether two number is close to each
        other.
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


def set_context(**kwargs):
    """
    Set context for running environment.

    Context should be configured before running your program. If there is no configuration,
    it will be automatically set according to the device target by default.

    See the below table for detail:

    +------------------------+-------------------------------+
    |Configuration Parameters|Description                    |
    +========================+===============================+
    |device_target           |Set the target device to run   |
    +------------------------+-------------------------------+
    |dtype                   |Set simulator backend data type|
    +------------------------+-------------------------------+
    |precision               |Set the atol number precision  |
    +------------------------+-------------------------------+

    Note:
        For every parameter, a setter or a getter method is implemented.
    """
    for key, value in kwargs.items():
        if key.lower() == "dtype":
            _Context.set_dtype(value)
            continue
        if key.lower() == "precision":
            _Context.set_precision(value)
            continue
        if key.lower() == "device_target":
            _Context.set_device_target(value)
            continue
        raise ValueError(
            f"For 'set_context', the keyword argument {key} is not recognized! For detailed "
            "usage of 'set_context', please refer to the Mindquantum official website."
        )


def get_context(attr_key: str):
    """
    Get context attribute value according to the input key.

    If some attributes are not set, they will be automatically obtained.

    Args:
        attr_key (str): The key of the attribute.

    Returns:
        Object, The value of given attribute key.

    Raises:
        ValueError: If input key is not an attribute in context.

    Examples:
        >>> import mindquantum as mq
        >>> mq.get_context("device_target")
        >>> mq.get_context("dtype")
    """
    if attr_key.upper() in _GLOBAL_CONFIG:
        return _GLOBAL_CONFIG[attr_key.upper()]
    raise ValueError(
        f"For 'get_context', the argument {attr_key} is not recognized! For detailed "
        f"usage of 'get_context', please refer to the Mindquantum official website."
    )
