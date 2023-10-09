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
"""Test module_circuit."""
from mindquantum.core import gates as G
from mindquantum.core.circuit import UN, SwapParts


def test_un():
    """
    Description: Test UN
    Expectation:
    """
    circ = UN(G.X, [3, 4, 5], [0, 1, 2])
    assert circ[-1] == G.X.on(5, 2)


def test_swappart():
    """
    Description: Test SwapPart
    Expectation:
    """
    circ = SwapParts([1, 2, 3], [4, 5, 6], 0)
    assert circ[-1] == G.SWAP([3, 6], 0)
