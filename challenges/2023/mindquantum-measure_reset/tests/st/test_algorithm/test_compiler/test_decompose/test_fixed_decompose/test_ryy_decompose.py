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
import pytest

from mindquantum.algorithm.compiler.decompose import cryy_decompose, ryy_decompose
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import Ryy


def circuit_equal_test(gate, decompose_circ):
    """
    require two circuits are equal.
    """
    orig_circ = Circuit() + gate
    assert np.allclose(orig_circ.matrix(), decompose_circ.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_ryy():
    """
    Description: Test ryy decompose
    Expectation: success
    """
    ryy = Ryy(1).on([0, 1])
    for solution in ryy_decompose(ryy):
        circuit_equal_test(ryy, solution)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_cryy():
    """
    Description: Test cryy decompose
    Expectation: success
    """
    cryy = Ryy(2).on([0, 1], [2, 3])
    for solution in cryy_decompose(cryy):
        circuit_equal_test(cryy, solution)
