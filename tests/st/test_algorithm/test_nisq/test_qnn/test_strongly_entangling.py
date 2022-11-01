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

from mindquantum.algorithm.nisq import StronglyEntangling
from mindquantum.core.gates import X

def test_strongly_entangling_ansatz():
    """
    Description: Test strongly_entangling_ansatz
    Expectation: success
    """
    strongly_entangling = StronglyEntangling(3, 2, X)
    circ = strongly_entangling.circuit
    assert len(circ) == 12
    assert len(circ.params_name) == 18
    assert circ[-1] == X.on(1, 2)
    assert len(circ.__str__()) == 1541
