# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# wITHOUT wARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test set/get functionality of MQChemSimulator."""

import numpy as np

from mindquantum.simulator.mqchem import MQChemSimulator


def test_mqchem_simulator_set_get_state():
    """
    Description: Setting a custom CI state and retrieving it should match.
    Expectation: success.
    """
    # Use n_electrons matching mask bit-count in new_state
    sim = MQChemSimulator(n_qubits=4, n_electrons=1)
    # Define a custom CI state with two determinants
    new_state = [(0b10, 0.5), (0b01, -0.5)]
    sim.set_qs(new_state)
    # Retrieve full state vector
    qs = sim.get_qs()
    expected = np.zeros(1 << sim.n_qubits, dtype=complex)
    for mask, amp in new_state:
        expected[mask] = amp
    np.testing.assert_allclose(qs, expected)
