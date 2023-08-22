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
"""MindSpore Quantum dtype module."""
import numpy as np

from mindquantum._math import dtype as dtype_
from mindquantum.utils.type_value_check import _check_mq_type

__dtype__ = [
    'float32',
    'float64',
    'complex64',
    'complex128',
]

float32 = dtype_.float32
float64 = dtype_.float64
complex64 = dtype_.complex64
complex128 = dtype_.complex128
mq_number_type = [float32, float64, complex64, complex128]
mq_real_number_type = [float32, float64]
mq_complex_number_type = [complex64, complex128]

str_dtype_map = {
    str(float32): float32,
    str(float64): float64,
    str(complex64): complex64,
    str(complex128): complex128,
}


def to_mq_type(dtype):
    """
    Convert type to mindquantum type.

    Args:
        dtype (Union[mindquantum.dtype, mindspore.dtype, numpy.dtype]): The data type supported by
            mindquantum or mindspore or numpy.

    Examples:
        >>> from mindquantum import to_mq_type
        >>> import numpy as np
        >>> to_mq_type(np.complex128)
        mindquantum.complex128
    """
    type_mapper = {
        float32: float32,
        float64: float64,
        complex128: complex128,
        complex64: complex64,
        float: float64,
        complex: complex128,
        np.double: float64,
        np.float32: float32,
        np.float64: float64,
        np.complex64: complex64,
        np.complex128: complex128,
        np.dtype(np.float32): float32,
        np.dtype(np.float64): float64,
        np.dtype(np.complex64): complex64,
        np.dtype(np.complex128): complex128,
    }
    try:
        import mindspore as ms  # pylint: disable=import-outside-toplevel

        type_mapper[ms.float32] = float32
        type_mapper[ms.float64] = float64
        type_mapper[ms.complex64] = complex64
        type_mapper[ms.complex128] = complex128
    except ImportError:
        pass
    if dtype not in type_mapper:
        raise TypeError(f"type error: {dtype}, now we support {list(type_mapper.keys())}")
    return type_mapper[dtype]


def to_real_type(dtype):
    """
    Convert type to real type while keeping precision.

    Args:
        dtype (mindquantum.dtype): The data type supported by mindquantum.

    Examples:
        >>> from mindquantum import to_real_type, complex128
        >>> to_real_type(complex128)
        mindquantum.float64
    """
    _check_mq_type(dtype)
    return {
        float32: float32,
        float64: float64,
        complex128: float64,
        complex64: float32,
    }[dtype]


def to_complex_type(dtype):
    """
    Convert type to complex type while keeping precision.

    Args:
        dtype (mindquantum.dtype): The data type supported by mindquantum.

    Examples:
        >>> from mindquantum import to_complex_type, float32
        >>> to_complex_type(float32)
        mindquantum.complex64
    """
    _check_mq_type(dtype)
    return {
        float32: complex64,
        float64: complex128,
        complex64: complex64,
        complex128: complex128,
    }[dtype]


def to_double_precision(dtype):
    """
    Convert type to double precision.

    Args:
        dtype (mindquantum.dtype): The data type supported by mindquantum.

    Examples:
        >>> from mindquantum import to_double_precision, float32
        >>> to_double_precision(float32)
        mindquantum.float64
    """
    _check_mq_type(dtype)
    return {
        float32: float64,
        float64: float64,
        complex128: complex128,
        complex64: complex128,
    }[dtype]


def to_single_precision(dtype):
    """
    Convert type to single precision.

    Args:
        dtype (mindquantum.dtype): The data type supported by mindquantum.

    Examples:
        >>> from mindquantum import to_single_precision, complex128
        >>> to_single_precision(complex128)
        mindquantum.complex64
    """
    _check_mq_type(dtype)
    return {
        float32: float32,
        float64: float32,
        complex128: complex64,
        complex64: complex64,
    }[dtype]


def to_precision_like(dtype_src, dtype_des):
    """
    Convert dtype_src to same precision as dtype_des.

    Args:
        dtype_src (mindquantum.dtype): The data type supported by mindquantum.
        dtype_des (mindquantum.dtype): The data type supported by mindquantum.

    Examples:
        >>> from mindquantum import to_precision_like, float32, complex128
        >>> to_precision_like(float32, complex128)
        >>> mindquantum.float64
    """
    if is_double_precision(dtype_des):
        return to_double_precision(dtype_src)
    if is_single_precision(dtype_des):
        return to_single_precision(dtype_src)
    raise TypeError(f"Unknown dtype_des {dtype_des}")


def to_np_type(dtype):
    """
    Convert type to numpy data type.

    Args:
        dtype (mindquantum.dtype): The data type supported by mindquantum.

    Examples:
        >>> from mindquantum import to_np_type, complex128
        >>> to_np_type(complex128)
        numpy.complex128
    """
    return {
        complex128: np.complex128,
        complex64: np.complex64,
        float32: np.float32,
        float64: np.float64,
    }[dtype]


def is_double_precision(dtype) -> bool:
    """
    Check whether a type is double precision or not.

    Args:
        dtype (mindquantum.dtype): The data type supported by mindquantum.

    Examples:
        >>> from mindquantum import is_double_precision, complex128
        >>> is_double_precision(complex128)
        True
    """
    _check_mq_type(dtype)
    return dtype in [complex128, float64]


def is_single_precision(dtype) -> bool:
    """
    Check whether a type is single precision or not.

    Args:
        dtype (mindquantum.dtype): The data type supported by mindquantum.

    Examples:
        >>> from mindquantum import is_single_precision, complex128
        >>> is_single_precision(complex128)
        False
    """
    _check_mq_type(dtype)
    return dtype in [complex64, float32]


def is_same_precision(dtype1, dtype2) -> bool:
    """
    Check whether two type is same precision or not.

    Args:
        dtype1 (mindquantum.dtype): The data type supported by mindquantum.
        dtype2 (mindquantum.dtype): The data type supported by mindquantum.

    Examples:
        >>> from mindquantum import is_same_precision, complex128, float64
        >>> is_same_precision(complex128, float64)
        True
    """
    return (is_double_precision(dtype1) and is_double_precision(dtype2)) or (
        is_single_precision(dtype1) and is_single_precision(dtype2)
    )


def precision_str(dtype) -> str:
    """
    Get precision string.

    Args:
        dtype (mindquantum.dtype): The data type supported by mindquantum.

    Examples:
        >>> from mindquantum import precision_str, complex128
        >>> precision_str(complex128)
        'double precision'
    """
    if is_single_precision(dtype):
        return "single precision"
    return "double precision"
