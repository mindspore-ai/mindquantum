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
# pylint: disable=import-outside-toplevel
"""Test Quantum Shannon decomposition"""
import pytest
from scipy.stats import unitary_group

from mindquantum.algorithm.compiler import decompose
from mindquantum.algorithm.compiler.decompose import utils
from mindquantum.core import gates

rand_unitary = unitary_group.rvs


def assert_equivalent_unitary(u, v):
    """Assert two unitary equal."""
    assert decompose.utils.is_equiv_unitary(u, v)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_qs_decomposition():
    """
    Feature: Quantum Shannon decomposition
    Description: Test Quantum Shannon decomposition for arbitrary n-qubit gate.
    Expectation: success.
    """

    n = 4
    u = rand_unitary(2**n, random_state=123)
    g = gates.UnivMathGate('U', u).on(range(n))
    circ = decompose.qs_decompose(g)
    assert_equivalent_unitary(u, utils.circuit_to_unitary(circ))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_cu_decomposition():
    """
    Feature: demultiplex pair-unitary Multiplexor
    Description: Test arbitrary-dimension controlled-unitary gate decomposition.
    Expectation: success.
    """
    cqs = [0, 2, 4, 5]  # arbitrary order is OK
    tqs = [1, 6]
    m = len(cqs)
    n = len(tqs)
    u = rand_unitary(2**n, random_state=123)
    circ = decompose.cu_decompose(gates.UnivMathGate('U', u).on(tqs, cqs))
    assert_equivalent_unitary(
        utils.tensor_slots(utils.controlled_unitary_matrix(u, m), max(cqs + tqs) + 1, cqs + tqs),
        utils.circuit_to_unitary(circ),
    )
