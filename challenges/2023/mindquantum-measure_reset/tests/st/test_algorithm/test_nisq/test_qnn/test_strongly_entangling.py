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
"""Test strongly entangling"""
import numpy as np
import pytest

import mindquantum as mq
from mindquantum.algorithm.nisq import StronglyEntangling
from mindquantum.core.gates import X


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [mq.complex128, mq.complex64])
def test_strongly_entangling_ansatz(dtype):
    """
    Description: Test strongly_entangling_ansatz
    Expectation: success
    """
    strongly_entangling = StronglyEntangling(3, 2, X)
    circ = strongly_entangling.circuit
    assert len(circ) == 12
    assert len(circ.params_name) == 18
    assert circ[-1] == X.on(1, 2)
    qs_exp = np.array(
        [
            0.79707728 + 0.01013428j,
            -0.11230256 + 0.0640159j,
            -0.02478449 - 0.09007194j,
            0.19063718 + 0.21396115j,
            0.05847713 + 0.11265412j,
            -0.00747775 + 0.16495718j,
            0.25717824 + 0.29613843j,
            0.22900849 + 0.08570448j,
        ]
    )
    assert np.allclose(qs_exp, circ.get_qs(pr=np.linspace(0, 1, 18), dtype=dtype))
