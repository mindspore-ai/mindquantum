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
"""Test zero noise extrapolation."""

import numpy as np

from mindquantum.algorithm.error_mitigation import zne
from mindquantum.core.circuit import Circuit
from mindquantum.core.circuit.channel_adder import BitFlipAdder
from mindquantum.core.gates import RX, RY, RZ
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator
from mindquantum.simulator.noise import NoiseBackend


def execute(circ: Circuit, noise_level: float, ham: Hamiltonian, seed=42):
    """Simulator executor."""
    return Simulator(NoiseBackend('mqmatrix', circ.n_qubits, BitFlipAdder(noise_level), seed=seed)).get_expectation(
        ham, circ
    )


def rb_circ(n_gate):
    """Generate random benchmark circuit."""
    circ = Circuit()
    for _ in range(n_gate):
        circ += np.random.choice([RX, RY, RZ])(np.random.uniform(-1, 1)).on(0)
    return circ


def test_zne():
    """
    Description: Test zne
    Expectation: success
    """
    np.random.seed(42)
    circ = rb_circ(50)
    ham = Hamiltonian(QubitOperator('Z0'))
    noise_level = 0.001
    true_value = execute(circ, 0.0, ham)
    noisy_value = execute(circ, noise_level, ham)
    zne_value = zne(circ, execute, scaling=np.linspace(1, 3, 3), args=(noise_level, ham))
    error_1 = abs((true_value - noisy_value) / true_value)
    error_2 = abs((true_value - zne_value) / true_value)
    assert np.allclose(error_1, 0.047095940433263844)
    assert np.allclose(error_2, 0.014593005275941016)
