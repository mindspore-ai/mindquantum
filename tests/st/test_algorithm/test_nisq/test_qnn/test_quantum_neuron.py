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
"""Test quantum neuron"""

from mindquantum.simulator import Simulator
from mindquantum.algorithm import QuantumNeuron
from mindquantum import Circuit, H
import numpy as np
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_quantum_neuron():
    """
    Test the basic functionality and output states of quantum neuron
    Description: Test quantum neuron
    Expectation: success
    """
    # Prepare input state
    circ = Circuit()
    circ += H.on(0)
    circ += H.on(1)

    # Initialize simulator
    sim = Simulator('mqvector', 4, seed=2)
    sim.apply_circuit(circ)

    # Create quantum neuron
    qn = QuantumNeuron(weight=[1, 1], input_qubits=[0, 1], output_qubit=3, ancilla_qubit=2)

    # Execute quantum neuron circuit until success
    while True:
        result = sim.apply_circuit(qn.circuit)
        if next(iter(result.data))[0] == '1':
            sim.apply_circuit(qn.recovery_circuit)
        else:
            break

    # Get final quantum state
    final_state = sim.get_qs()

    # Verify key state amplitudes
    expected_amplitudes = {
        1: -0.22606781j,  # |0001⟩
        2: -0.22606781j,  # |0010⟩
        3: 0.11161816j,  # |0011⟩
        9: -0.54833173j,  # |1001⟩
        10: -0.54833173j,  # |1010⟩
        11: 0.53290967j,  # |1011⟩
    }

    # Verify non-zero amplitudes
    for idx, expected_amp in expected_amplitudes.items():
        assert np.abs(final_state[idx] - expected_amp) < 1e-6

    # Verify other positions should be close to zero
    zero_indices = set(range(16)) - set(expected_amplitudes.keys())
    for idx in zero_indices:
        assert np.abs(final_state[idx]) < 1e-6


def test_quantum_neuron_invalid_inputs():
    """
    Test input validation for quantum neuron
    Description: Test quantum neuron invalid inputs
    Expectation: success
    """
    # Test invalid weight type
    with pytest.raises(TypeError):
        QuantumNeuron(weight="invalid")

    # Test mismatch between weights and input qubits
    with pytest.raises(ValueError):
        QuantumNeuron(weight=[1, 1], input_qubits=[0])

    # Test invalid gamma parameter
    with pytest.raises(TypeError):
        QuantumNeuron(weight=[1], gamma="invalid")

    # Test invalid bias parameter
    with pytest.raises(TypeError):
        QuantumNeuron(weight=[1], bias="invalid")
