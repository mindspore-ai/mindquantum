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
"""Test UCCExcitationGate functionality."""
# pylint: disable=invalid-name

import pytest

from mindquantum.core.operators import FermionOperator
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.simulator.mqchem import UCCExcitationGate


def test_ucc_excitation_gate_data():
    """
    Description: UCCExcitationGate should convert FermionOperator to correct binding data.
    Expectation: success.
    """
    G = FermionOperator('2^ 0', 'theta')
    gate = UCCExcitationGate(G)
    pr = ParameterResolver({'theta': 0.5})
    data = gate.term_data
    assert len(data) == 2
    ops_sets = {frozenset((idx, bool(is_creation)) for idx, is_creation in ops) for ops, coeff in data}
    expected_ops_sets = {
        frozenset({(2, True), (0, False)}),
        frozenset({(0, True), (2, False)}),
    }
    assert ops_sets == expected_ops_sets
    coeffs_abs = sorted(abs(coeff) for _, coeff in data)
    assert coeffs_abs == [pytest.approx(1.0), pytest.approx(1.0)]
