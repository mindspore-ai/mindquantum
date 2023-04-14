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
"""General W State"""

import numpy as np
import pytest

import mindquantum as mq
from mindquantum.algorithm.library import general_w_state


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dtype', [mq.complex128, mq.complex64])
def test_general_w_state(dtype):
    """
    Description: Test general_w_state
    Expectation:
    """
    state = general_w_state(range(3)).get_qs(dtype=dtype)
    qs_exp = 1 / np.sqrt(3) * np.array([0, 1, 1, 0, 1, 0, 0, 0])
    assert np.allclose(state, qs_exp)
