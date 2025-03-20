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
"""Test gate."""

from mindquantum.utils import random_circuit
from mindquantum.simulator import Simulator
import numpy as np


def test_measure_result_reverse_endian():
    """
    Description: Test reverse_endian method of MeasureResult
    Expectation: success.
    """
    circ = random_circuit(10, 10)
    circ.measure_all()

    sim = Simulator("mqvector", 10)
    res = sim.sampling(circ, shots=100)

    original_keys = res._keys.copy()
    original_data = res.data.copy()
    reversed_res = res.reverse_endian()

    assert reversed_res._keys == original_keys[::-1]

    for bit_string, count in original_data.items():
        reversed_string = bit_string[::-1]
        assert reversed_res.data[reversed_string] == count

    assert reversed_res.shots == res.shots
    assert np.array_equal(reversed_res._samples, np.fliplr(res._samples))
