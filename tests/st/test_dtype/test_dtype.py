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
"""Test mindquantum dtype."""

import mindquantum as mq
from mindquantum.simulator import Simulator
import numpy as np
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("error_dtype", ['int', 'float', 'mq.int', 'mq.flaot', 'mq.double', 'mq.complex', 'mq.int32',
                                         'mq.float31', 'mq.float63', 'mq.complex63', 'mq.complex127', 'np.float32'])
def test_error_dtype(error_dtype):
    """
    Description: test error dtype.
    Expectation: raise error
    """
    with pytest.raises(ValueError):
        Simulator('mqvector', 2, dtype=error_dtype)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dtype():
    """
    Description: test mindquantum dtype.
    Expectation: success
    """
    assert mq.dtype.is_double_precision(mq.float64)
    assert mq.dtype.is_double_precision(mq.complex128)
    assert not mq.dtype.is_double_precision(mq.float32)
    assert not mq.dtype.is_double_precision(mq.complex64)

    assert mq.dtype.is_single_precision(mq.float32)
    assert mq.dtype.is_single_precision(mq.complex64)
    assert not mq.dtype.is_single_precision(mq.float64)
    assert not mq.dtype.is_single_precision(mq.complex128)

    assert mq.dtype.is_same_precision(mq.float32, mq.complex64)
    assert mq.dtype.is_same_precision(mq.float64, mq.complex128)
    assert not mq.dtype.is_same_precision(mq.float32, mq.complex128)

    assert mq.dtype.precision_str(mq.float32) == 'single precision'
    assert mq.dtype.precision_str(mq.float64) == 'double precision'
    assert not mq.dtype.precision_str(mq.complex64) == 'double precision'

    assert mq.dtype.to_real_type(mq.complex64) == mq.float32
    assert mq.dtype.to_real_type(mq.complex128) == mq.float64
    assert mq.dtype.to_real_type(mq.float64) == mq.float64

    assert mq.dtype.to_complex_type(mq.float32) == mq.complex64
    assert mq.dtype.to_complex_type(mq.float64) == mq.complex128
    assert mq.dtype.to_complex_type(mq.complex128) == mq.complex128

    assert mq.dtype.to_single_precision(mq.float64) == mq.float32
    assert mq.dtype.to_single_precision(mq.complex128) == mq.complex64
    assert mq.dtype.to_single_precision(mq.float32) == mq.float32

    assert mq.dtype.to_double_precision(mq.float32) == mq.float64
    assert mq.dtype.to_double_precision(mq.complex64) == mq.complex128
    assert mq.dtype.to_double_precision(mq.float32) == mq.float64

    assert mq.dtype.to_precision_like(mq.float32, mq.complex128) == mq.float64
    assert mq.dtype.to_precision_like(mq.complex128, mq.float32) == mq.complex64

    assert mq.dtype.to_mq_type(np.float32) == mq.float32
    assert mq.dtype.to_mq_type(np.float64) == mq.float64
    assert mq.dtype.to_mq_type(np.complex64) == mq.complex64
    assert mq.dtype.to_mq_type(np.complex128) == mq.complex128

    assert mq.dtype.to_np_type(mq.float32) == np.float32
    assert mq.dtype.to_np_type(mq.complex128) == np.complex128
