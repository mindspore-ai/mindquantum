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

"""Test basic universal gate decomposition rules"""

import numpy as np
import pytest
from scipy.stats import unitary_group

from mindquantum.algorithm.compiler import decompose
from mindquantum.core import gates
from mindquantum.core.circuit import Circuit

rand_unitary = unitary_group.rvs


def assert_equivalent_unitary(u, v):
    """Assert two unitary equal."""
    try:
        import cirq

        cirq.testing.assert_allclose_up_to_global_phase(u, v, atol=1e-5)
    except ModuleNotFoundError:
        assert decompose.utils.is_equiv_unitary(u, v)
    except AttributeError:
        assert decompose.utils.is_equiv_unitary(u, v)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_euler_decomposition():
    """
    Feature: single-qubit Euler decomposition
    Description: Test Euler decomposition for single-qubit unitary gate.
    Expectation: success.
    """

    # ZYZ basis
    u = unitary_group.rvs(2, random_state=123)
    g = gates.UnivMathGate('U', u).on(0)
    circ_original = Circuit() + g
    circ_decomposed = decompose.euler_decompose(g)
    assert_equivalent_unitary(circ_original.matrix(), circ_decomposed.matrix())

    # U3 basis
    circ_decomposed = decompose.euler_decompose(g, basis='u3')
    assert_equivalent_unitary(circ_original.matrix(), circ_decomposed.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_tensor_product_decomposition():
    """
    Feature: two-qubit tensor-product decomposition
    Description: Test 2-qubit tensor-product decomposition.
    Expectation: success.
    """

    g = gates.UnivMathGate('XY', np.kron(gates.X.matrix(), gates.Y.matrix())).on([0, 1])
    circ_original = (Circuit() + g).reverse_qubits()
    circ_decomposed = decompose.tensor_product_decompose(g)
    assert_equivalent_unitary(circ_original.matrix(), circ_decomposed.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_abc_decomposition():
    """
    Feature: two-qubit controlled-unitary gate decomposition
    Description: Test ABC decomposition for 2-qubit controlled gate.
    Expectation: success.
    """

    # special case: CRz(pi)
    g = gates.RZ(np.pi).on(0, 1)
    circ_original = Circuit() + g
    circ_decomposed = decompose.abc_decompose(g)
    assert_equivalent_unitary(circ_original.matrix(), circ_decomposed.matrix())

    # arbitrary CU gate
    u = rand_unitary(2, random_state=123)
    g = gates.UnivMathGate('U', u).on(1, 0)
    circ_original = Circuit() + g
    circ_decomposed = decompose.abc_decompose(g)
    assert_equivalent_unitary(circ_original.matrix(), circ_decomposed.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_kak_decomposition():
    """
    Feature: arbitrary two-qubit gate decomposition
    Description: Test KAK decomposition for arbitrary 2-qubit gate.
    Expectation: success.
    """

    g = gates.UnivMathGate('U', rand_unitary(4, random_state=123)).on([0, 1])
    circ_original = Circuit() + g
    circ_decomposed = decompose.kak_decompose(g)
    assert_equivalent_unitary(circ_original.matrix(), circ_decomposed.matrix())
