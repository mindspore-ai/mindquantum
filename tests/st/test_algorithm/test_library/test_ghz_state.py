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
"""General GHZ State"""

import numpy as np
import pytest

from mindquantum.algorithm.library import general_ghz_state
from mindquantum.config import set_context


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_general_ghz_state(dtype):
    """
    Description: Test if three qubit general_ghz_state correct or not.
    Expectation: success.
    """
    set_context(dtype=dtype)
    state = general_ghz_state(range(3)).get_qs()

    state_exp = 1 / np.sqrt(2) * np.array([1, 0, 0, 0, 0, 0, 0, 1])
    assert np.allclose(state, state_exp)
