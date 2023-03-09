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

from mindquantum.algorithm.compiler.decompose import cswap_decompose, swap_decompose
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import SWAP


def circuit_equal_test(gate, decompose_circ):
    """
    require two circuits are equal.
    """
    orig_circ = Circuit() + gate
    assert np.allclose(orig_circ.matrix(), decompose_circ.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_swap():
    """
    Description: Test swap decompose
    Expectation: success
    """
    swap = SWAP.on([0, 1])
    for solution in swap_decompose(swap):
        circuit_equal_test(swap, solution)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_cswap():
    """
    Description: Test cswap decompose
    Expectation: success
    """
    cswap = SWAP.on([1, 2], 0)
    for solution in cswap_decompose(cswap):
        circuit_equal_test(cswap, solution)
