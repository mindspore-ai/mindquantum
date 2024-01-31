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
"""Test randomized benchmarking."""
import numpy as np
import pytest

from mindquantum.algorithm.error_mitigation import (
    generate_double_qubits_rb_circ,
    generate_single_qubit_rb_circ,
)


@pytest.mark.parametrize("method", [generate_single_qubit_rb_circ, generate_double_qubits_rb_circ])
def test_random_benchmarking_circuit(method):
    """
    Description: Test test_random_benchmarking_circuit
    Expectation: success
    """
    length = np.random.randint(0, 100, size=100)
    for i in length:
        circ = method(i, np.random.randint(1 << 20))
        assert np.allclose(np.abs(circ.get_qs())[0], 1, atol=1e-6)
