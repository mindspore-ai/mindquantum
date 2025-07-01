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
"""Test applying a single UCC excitation gate."""

import numpy as np

from mindquantum.core.operators import FermionOperator
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.simulator.mqchem import MQChemSimulator, UCCExcitationGate


def test_apply_single_ucc_gate():
    """
    Description: Apply a single excitation gate to the HF state yields expected superposition.
    Expectation: success.
    """
    sim = MQChemSimulator(n_qubits=4, n_electrons=2)
    G = FermionOperator('2^ 0', 'theta')
    gate = UCCExcitationGate(G)
    theta = 0.7
    pr = ParameterResolver({'theta': theta})
    sim.apply_circuit([gate], pr=pr)
    qs = sim.get_qs()
    mask_hf = 0b0011
    mask_exc = 0b0110
    np.testing.assert_allclose(qs[mask_hf], np.cos(theta))
    np.testing.assert_allclose(np.abs(qs[mask_exc]), np.sin(theta))


def test_apply_gate():
    """
    Description: Apply a single excitation gate via apply_gate to the HF state yields expected superposition.
    Expectation: success.
    """
    sim = MQChemSimulator(n_qubits=4, n_electrons=2)
    G = FermionOperator('2^ 0', 'theta')
    gate = UCCExcitationGate(G)
    theta = 0.9
    pr = ParameterResolver({'theta': theta})
    sim.apply_gate(gate, pr)
    qs = sim.get_qs()
    mask_hf = 0b0011
    mask_exc = 0b0110
    np.testing.assert_allclose(qs[mask_hf], np.cos(theta))
    np.testing.assert_allclose(np.abs(qs[mask_exc]), np.sin(theta))


def test_apply_gate_with_constant():
    """
    Description: Apply a single excitation gate via apply_gate to the HF state yields expected superposition.
    Expectation: success.
    """
    sim = MQChemSimulator(n_qubits=4, n_electrons=2)
    theta = 0.9
    G = FermionOperator('2^ 0', theta)
    gate = UCCExcitationGate(G)
    sim.apply_gate(gate)
    qs = sim.get_qs()
    mask_hf = 0b0011
    mask_exc = 0b0110
    np.testing.assert_allclose(qs[mask_hf], np.cos(theta))
    np.testing.assert_allclose(np.abs(qs[mask_exc]), np.sin(theta))
