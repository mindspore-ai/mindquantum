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
"""Test initialization of MQ Chemistry Simulator."""

import numpy as np

from mindquantum.simulator.mqchem import MQChemSimulator


def test_mqchem_simulator_init_default():
    """
    Description: MQChemSimulator should initialize to Hartree-Fock CI state.
    Expectation: success.
    """
    sim = MQChemSimulator(n_qubits=4, n_electrons=2)
    qs = sim.get_qs()
    # Hartree-Fock occupies lowest two spin-orbitals: mask = 0b11
    expected = np.zeros(1 << sim.n_qubits, dtype=complex)
    expected[0b11] = 1.0
    np.testing.assert_allclose(qs, expected)


def test_mqchem_simulator_init_dtype():
    """
    Description: MQChemSimulator supports float and double precisions.
    Expectation: success.
    """
    sim_d = MQChemSimulator(n_qubits=3, n_electrons=1, dtype="double")
    qs_d = sim_d.get_qs()
    expected_d = np.zeros(1 << sim_d.n_qubits, dtype=complex)
    expected_d[0b1] = 1.0
    np.testing.assert_allclose(qs_d, expected_d)

    sim_f = MQChemSimulator(n_qubits=3, n_electrons=1, dtype="float")
    qs_f = sim_f.get_qs()
    expected_f = np.zeros(1 << sim_f.n_qubits, dtype=complex)
    expected_f[0b1] = 1.0
    np.testing.assert_allclose(qs_f, expected_f, rtol=1e-6)
