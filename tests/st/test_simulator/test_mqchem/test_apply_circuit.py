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
# pylint: disable=invalid-name
"""Test apply_circuit method of MQChemSimulator."""

import numpy as np

from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import FermionOperator
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.simulator.mqchem import MQChemSimulator, UCCExcitationGate


def test_apply_circuit_with_ucc_excitation_gate():
    """
    Description: MQChemSimulator.apply_circuit should apply UCCExcitationGate within a Circuit.
    Expectation: success.
    """
    sim = MQChemSimulator(n_qubits=4, n_electrons=2)
    G = FermionOperator('2^ 0', 'theta')
    gate = UCCExcitationGate(G)
    circ = Circuit([gate])
    theta = 0.9
    sim.apply_circuit(circ, pr=ParameterResolver({'theta': theta}))
    qs = sim.get_qs()
    mask_hf = 0b0011
    mask_exc = 0b0110
    np.testing.assert_allclose(qs[mask_hf], np.cos(theta))
    np.testing.assert_allclose(np.abs(qs[mask_exc]), np.sin(theta))


def test_apply_circuit_with_double_excitation_gate():
    """
    Description: MQChemSimulator.apply_circuit should correctly apply a double excitation gate.
    Expectation: success.
    """
    sim = MQChemSimulator(n_qubits=4, n_electrons=2)
    # G = a_2^ a_3^ a_0 a_1 - h.c.
    term = FermionOperator('2^ 3^ 0 1', 'theta')
    gate = UCCExcitationGate(term)
    circ = Circuit([gate])
    theta = 0.5
    sim.apply_circuit(circ, pr=ParameterResolver({'theta': theta}))
    qs = sim.get_qs()

    # Initial Hartree-Fock state |1100> (mask 0b0011 = 3)
    mask_hf = 0b0011
    # Double-excited state |0011> (mask 0b1100 = 12)
    mask_exc = 0b1100

    expected_hf_amp = np.cos(theta)
    expected_exc_amp_abs = np.sin(theta)
    np.testing.assert_allclose(qs[mask_hf], expected_hf_amp)
    np.testing.assert_allclose(np.abs(qs[mask_exc]), expected_exc_amp_abs)
