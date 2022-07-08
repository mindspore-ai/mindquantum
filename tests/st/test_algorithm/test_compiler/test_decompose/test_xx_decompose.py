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

# pylint: disable=invalid-name

'''test decompose rule'''
import numpy as np

from mindquantum.algorithm.compiler.decompose import cxx_decompose, xx_decompose
from mindquantum.core import XX, Circuit


def circuit_equal_test(gate, decompose_circ):
    """
    require two circuits are equal.
    """
    orig_circ = Circuit() + gate
    assert np.allclose(orig_circ.matrix(), decompose_circ.matrix())


def test_xx():
    """
    Description: Test xx decompose
    Expectation: success
    """
    xx = XX(1).on([0, 1])
    for solution in xx_decompose(xx):
        circuit_equal_test(xx, solution)


def test_cxx():
    """
    Description: Test cxx decompose
    Expectation: success
    """
    cxx = XX(2).on([0, 1], [2, 3])
    for solution in cxx_decompose(cxx):
        circuit_equal_test(cxx, solution)
