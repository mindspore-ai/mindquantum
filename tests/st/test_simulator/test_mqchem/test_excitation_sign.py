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
"""Test the sign of single and double excitation operators."""

import numpy as np
import pytest

from mindquantum.core.operators import FermionOperator
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.simulator.mqchem import MQChemSimulator, UCCExcitationGate


@pytest.mark.parametrize(
    "description, term_str, n_qubits, n_electrons, expected_phase",
    [
        ("Single excitation with positive phase", '2^ 1', 4, 2, 1.0),
        ("Single excitation with negative phase", '2^ 0', 4, 2, -1.0),
        ("Double excitation with negative phase (normal-ordered)", '3^ 2^ 1 0', 4, 2, -1.0),
        ("Double excitation with positive phase (non-normal-ordered)", '2^ 3^ 0 1', 4, 2, -1.0),
    ],
)
def test_excitation_sign(description, term_str, n_qubits, n_electrons, expected_phase):
    """
    Description: Test that the sign (phase) of the excited state amplitude is correct after applying a UCC gate,
                 which depends on the Jordan-Wigner string.
    Expectation: success.

    For a generator G = T - T.dagger, the evolution exp(theta*G) on a state |psi_i>
    that is annihilated by T.dagger results in:
    cos(theta)|psi_i> + phase*sin(theta)|psi_f>
    where T|psi_i> = phase*|psi_f>.
    """
    sim = MQChemSimulator(n_qubits=n_qubits, n_electrons=n_electrons)

    # The generator G = T - T.dagger, where T is the FermionOperator term.
    # The UCCExcitationGate implements exp(theta * G).
    term = FermionOperator(term_str, 'theta')
    gate = UCCExcitationGate(term)

    theta = 0.4
    # apply_gate resolves the parameter 'theta' to the value 0.4
    sim.apply_gate(gate, pr=ParameterResolver({'theta': theta}))

    qs = sim.get_qs(False)  # Get dense vector

    # Get the masks for initial (Hartree-Fock) and final (excited) states
    hf_mask = (1 << n_electrons) - 1

    # Manually determine the excited state mask by applying the excitation
    # defined by term_str to the Hartree-Fock mask.
    p_mask, q_mask = 0, 0
    for part in term_str.split(' '):
        if '^' in part:
            p_mask |= 1 << int(part.strip('^'))
        else:
            q_mask |= 1 << int(part)
    exc_mask = (hf_mask & ~q_mask) | p_mask

    # Check the amplitude of the initial state
    np.testing.assert_allclose(qs[hf_mask], np.cos(theta), atol=1e-7, err_msg=f"{description}: HF amplitude mismatch")

    # Check the amplitude of the excited state (this is the key test for the sign)
    expected_exc_amp = expected_phase * np.sin(theta)
    np.testing.assert_allclose(
        qs[exc_mask], expected_exc_amp, atol=1e-7, err_msg=f"{description}: Excited state amplitude/sign mismatch"
    )
