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
"""Test iqp_encoding"""

import numpy as np
import pytest

from mindquantum.algorithm.nisq import IQPEncoding
from mindquantum.config import set_context


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_general_iqp_encoding(dtype):
    """
    Description: Test general_iqp_encoding
    Expectation:
    """
    set_context(dtype=dtype, device_target="CPU")
    iqp = IQPEncoding(2)
    data = np.array([0, 0])
    state = iqp.circuit.get_qs(pr=iqp.data_preparation(data))
    state_exp = 1 / 2 * np.array([1, 1, 1, 1])
    assert np.allclose(state, state_exp)
