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
"""Test VQE gradients for MQChemSimulator."""

import numpy as np
import pytest

from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import FermionOperator, normal_ordered
from mindquantum.simulator.mqchem import MQChemSimulator, UCCExcitationGate, CIHamiltonian


def test_vqe_gradients_simple():
    """
    Description: test vqe gradients simple
    Expectation: success.
    """
    # Simple CI Hamiltonian: number operators for orbital 0 and 1
    # Build the CI Hamiltonian H = a_0^ a_0 + a_1^ a_1
    ham = CIHamiltonian(normal_ordered(FermionOperator('0^ 0') + FermionOperator('1^ 1')).real)
    circ = Circuit()
    # Construct the UCC excitation term G' = a_0^ a_2 - a_2^ a_0
    term = FermionOperator('0^ 2', 'theta')
    circ += UCCExcitationGate(term)
    sim = MQChemSimulator(4, 2)
    grad_ops = sim.get_expectation_with_grad(ham, circ)

    theta0 = 0.5
    delta = 1e-6
    f0, g0 = grad_ops([theta0])
    f_plus, _ = grad_ops([theta0 + delta])
    f_minus, _ = grad_ops([theta0 - delta])
    finite_diff = (f_plus - f_minus) / (2 * delta)
    assert pytest.approx(finite_diff, rel=1e-4) == g0[0]


def test_vqe_gradient_nonunit_coefficient():
    """
    Description: test vqe gradient nonunit coefficient
    Expectation: success.
    """
    # Test that a non-unit coefficient on FermionOperator scales gradient properly
    ham = CIHamiltonian(normal_ordered(FermionOperator('0^ 0') + FermionOperator('1^ 1')).real)
    coeff = 1.234
    circ = Circuit()
    circ += UCCExcitationGate(FermionOperator('0^ 2', {'theta': coeff}))
    sim = MQChemSimulator(4, 2)
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    theta0 = 0.5
    delta = 1e-6
    f0, g0 = grad_ops([theta0])
    f_plus, _ = grad_ops([theta0 + delta])
    f_minus, _ = grad_ops([theta0 - delta])
    finite_diff = (f_plus - f_minus) / (2 * delta)
    assert pytest.approx(finite_diff, rel=1e-4) == g0[0]


def test_vqe_multi_param_gradients():
    """
    Description: test vqe multi param gradients
    Expectation: success.
    """
    ham_ferm_ops = FermionOperator('0^ 0') + FermionOperator('1^ 1')
    ham = CIHamiltonian(normal_ordered(ham_ferm_ops).real)
    circ = Circuit()
    circ += UCCExcitationGate(FermionOperator('0^ 2', 'theta_1'))
    circ += UCCExcitationGate(FermionOperator('1^ 3', 'theta_2'))
    sim = MQChemSimulator(n_qubits=4, n_electrons=2)
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    theta1_val = 0.5
    theta2_val = 0.8
    params = np.array([theta1_val, theta2_val])
    g1_exact = -np.sin(2 * theta1_val)
    g2_exact = -np.sin(2 * theta2_val)
    _, g_computed = grad_ops(params)
    assert np.allclose(g_computed[0].real, g1_exact, atol=1e-6)
    assert np.allclose(g_computed[1].real, g2_exact, atol=1e-6)


def test_vqe_double_excitation_gradients():
    """
    Description: test vqe double excitation gradients
    Expectation: success.
    """
    ham_ferm_ops = FermionOperator('2^ 2') + FermionOperator('3^ 3')
    ham = CIHamiltonian(normal_ordered(ham_ferm_ops).real)
    circ = Circuit()
    term = FermionOperator('2^ 3^ 0 1', 'theta')
    circ += UCCExcitationGate(term)
    sim = MQChemSimulator(n_qubits=4, n_electrons=2)
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    theta_val = 0.5
    params = np.array([theta_val])
    g_exact = 2 * np.sin(2 * theta_val)
    _, g_computed = grad_ops(params)
    assert np.allclose(g_computed[0].real, g_exact, atol=1e-6)
