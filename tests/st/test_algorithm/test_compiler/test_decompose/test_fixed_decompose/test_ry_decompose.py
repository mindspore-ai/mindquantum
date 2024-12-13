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
'''test decompose rule'''
import numpy as np
import pytest

from mindquantum.algorithm.compiler.decompose import cry_decompose, cnry_decompose
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RY


def circuit_equal_test(gate, decompose_circ):
    """
    require two circuits are equal.
    """
    orig_circ = Circuit() + gate
    assert np.allclose(orig_circ.matrix(), decompose_circ.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_cry():
    """
    Description: Test cry decompose
    Expectation: success
    """
    cry = RY(1.23).on(1, 0)
    for solution in cry_decompose(cry):
        circuit_equal_test(cry, solution)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_cnry():
    """
    Description: Test cnry decompose
    Expectation: success
    """
    cnry = RY(1.23).on(2, [0, 1])
    for solution in cnry_decompose(cnry):
        circuit_equal_test(cnry, solution)
