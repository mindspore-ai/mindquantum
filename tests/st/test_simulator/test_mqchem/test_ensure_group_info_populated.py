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
"""Test edge cases for CppExcitationOperator::EnsureGroupInfoPopulated."""

import numpy as np
import pytest

from mindquantum.core.operators import FermionOperator
from mindquantum.simulator.mqchem import MQChemSimulator, UCCExcitationGate

def test_single_excitation_with_no_electrons():
    """
    Description: test single excitation with n_electrons = 0
    Expectation: success.
    """
    sim = MQChemSimulator(n_qubits=4, n_electrons=0)
    initial_qs = sim.get_qs().copy()
    gate = UCCExcitationGate(FermionOperator('2^ 0', 1.0))
    sim.apply_gate(gate)
    final_qs = sim.get_qs()
    np.testing.assert_allclose(initial_qs, final_qs)


def test_single_excitation_with_full_electrons():
    """
    Description: test single excitation with n_electrons = n_qubits
    Expectation: success.
    """
    sim = MQChemSimulator(n_qubits=4, n_electrons=4)
    initial_qs = sim.get_qs().copy()
    gate = UCCExcitationGate(FermionOperator('2^ 0', 1.0))
    sim.apply_gate(gate)
    final_qs = sim.get_qs()
    np.testing.assert_allclose(initial_qs, final_qs)


def test_double_excitation_with_one_electron():
    """
    Description: test double excitation with n_electrons = 1
    Expectation: success.
    """
    sim = MQChemSimulator(n_qubits=4, n_electrons=1)
    initial_qs = sim.get_qs().copy()
    gate = UCCExcitationGate(FermionOperator('3^ 2^ 1 0', 1.0))
    sim.apply_gate(gate)
    final_qs = sim.get_qs()
    np.testing.assert_allclose(initial_qs, final_qs)


def test_double_excitation_with_n_minus_one_electrons():
    """
    Description: test double excitation with n_electrons = n_qubits - 1
    Expectation: success.
    """
    sim = MQChemSimulator(n_qubits=4, n_electrons=3)
    initial_qs = sim.get_qs().copy()
    gate = UCCExcitationGate(FermionOperator('3^ 2^ 1 0', 1.0))
    sim.apply_gate(gate)
    final_qs = sim.get_qs()
    np.testing.assert_allclose(initial_qs, final_qs)


def test_hermitian_operator_as_generator():
    """
    Description: test hermitian operator as generator
    Expectation: raises ValueError due to duplicate indices.
    """
    with pytest.raises(ValueError):
        UCCExcitationGate(FermionOperator('1^ 1', 1.0))
