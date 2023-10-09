# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Test folding circuit."""
import numpy as np

from mindquantum.algorithm.error_mitigation import fold_at_random
from mindquantum.utils import random_circuit


def test_folding_circuit():
    """
    Description: Test folding circuit
    Expectation: success
    """
    np.random.seed(42)
    circ = random_circuit(4, 20)
    locally_folding = fold_at_random(circ, 2.3)
    globally_folding = fold_at_random(circ, 2.3, 'globally')
    assert locally_folding[30].hermitian() == circ[12]
    assert locally_folding[30].hermitian() == globally_folding[25]
    matrix_1 = circ.matrix()
    matrix_2 = locally_folding.matrix()
    matrix_3 = globally_folding.matrix()
    assert np.allclose(matrix_1, matrix_2, atol=1e-4)
    assert np.allclose(matrix_1, matrix_3, atol=1e-4)
