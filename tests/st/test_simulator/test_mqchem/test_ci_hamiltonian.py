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
"""Test CIHamiltonian expectation values in CI basis."""

import numpy as np
import pytest

from mindquantum.core.operators import FermionOperator
from mindquantum.simulator.mqchem import MQChemSimulator, CIHamiltonian


@pytest.fixture(name="sys_info")
def fixture_sys_info():
    """System info fixture."""
    n_qubits = 4
    n_electrons = 2
    # Hartree-Fock state for (4, 2) is |0011>, mask = 3
    hf_mask = (1 << n_electrons) - 1
    return {"n_qubits": n_qubits, "n_electrons": n_electrons, "hf_mask": hf_mask}


def test_ci_hamiltonian_identity_term(sys_info):
    """
    Description: Test Hamiltonian with only an identity term.
    Expectation: success.
    """
    sim = MQChemSimulator(n_qubits=sys_info["n_qubits"], n_electrons=sys_info["n_electrons"])
    ham_fo = FermionOperator("", 2.5)
    ham = CIHamiltonian(ham_fo)
    exp_val = sim.get_expectation(ham)
    assert exp_val == pytest.approx(2.5)


def test_ci_hamiltonian_diagonal_term_on_hf(sys_info):
    """
    Description: Test Hamiltonian with only a diagonal (number operator) term on HF state.
    Expectation: success.
    """
    sim = MQChemSimulator(n_qubits=sys_info["n_qubits"], n_electrons=sys_info["n_electrons"])
    # H = 1.5 * a_1^ a_1 on HF state |0011>
    ham_fo = FermionOperator("1^ 1", 1.5)
    ham = CIHamiltonian(ham_fo)
    exp_val = sim.get_expectation(ham)
    assert exp_val == pytest.approx(1.5)


def test_ci_hamiltonian_single_excitation_term(sys_info):
    """
    Description: Test Hamiltonian with a single excitation term on a superposition state.
    Expectation: success.
    """
    sim = MQChemSimulator(n_qubits=sys_info["n_qubits"], n_electrons=sys_info["n_electrons"])
    # Prepare state |psi> = 1/sqrt(2) * (|0011> + |0110>)
    s_mask = 6
    amp = 1 / np.sqrt(2)
    qs_rep = [(sys_info["hf_mask"], amp), (s_mask, amp)]
    sim.set_qs(qs_rep)

    # H = a_2^ a_0 + a_0^ a_2
    term = FermionOperator("2^ 0")
    ham_fo = term + term.hermitian()
    ham = CIHamiltonian(ham_fo)
    exp_val = sim.get_expectation(ham)
    assert exp_val == pytest.approx(-1.0)


def test_ci_hamiltonian_double_excitation_term(sys_info):
    """
    Description: Test Hamiltonian with a double excitation term on a superposition state.
    Expectation: success.
    """
    sim = MQChemSimulator(n_qubits=sys_info["n_qubits"], n_electrons=sys_info["n_electrons"])
    # Prepare state |psi> = 1/sqrt(2) * (|0011> + |1100>)
    d_mask = 12
    amp = 1 / np.sqrt(2)
    qs_rep = [(sys_info["hf_mask"], amp), (d_mask, amp)]
    sim.set_qs(qs_rep)

    # H = a_3^a_2^a_1a_0 + h.c.
    term = FermionOperator("3^ 2^ 1 0")
    ham_fo = term + term.hermitian()
    ham = CIHamiltonian(ham_fo)
    exp_val = sim.get_expectation(ham)
    assert exp_val == pytest.approx(-1.0)


def test_ci_hamiltonian_mixed_terms_on_superposition(sys_info):
    """
    Description: Test a mixed Hamiltonian on a superposition of HF and double-excited state.
    Expectation: success.
    """
    sim = MQChemSimulator(n_qubits=sys_info["n_qubits"], n_electrons=sys_info["n_electrons"])
    # Prepare state |psi> = 1/sqrt(2) * (|0011> + |1100>)
    d_mask = 12
    amp = 1 / np.sqrt(2)
    qs_rep = [(sys_info["hf_mask"], amp), (d_mask, amp)]
    sim.set_qs(qs_rep)

    h_ident = FermionOperator("", 4.6038)
    h_diag = FermionOperator("0^ 0", -2.2818) + FermionOperator("1^ 1", -2.2818)
    h_double = FermionOperator("3^ 2^ 1 0", 0.1) + FermionOperator("0^ 1^ 2 3", 0.1)
    ham_fo = h_ident + h_diag + h_double

    ham = CIHamiltonian(ham_fo)
    exp_val = sim.get_expectation(ham)

    # Expected: 4.6038 (ident) - 2.2818 (diag) - 0.1 (double) = 2.222
    assert exp_val == pytest.approx(2.222)
