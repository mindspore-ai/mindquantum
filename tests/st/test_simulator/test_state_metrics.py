# Copyright 2026 Huawei Technologies Co., Ltd
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
"""State comparison helpers of the simulator package.

``inner_product`` compares two simulators and ``fidelity`` compares two raw
quantum states.  Both functions are shared by the VQE and QAOA workflows, so
their validation contract is checked here: every malformed state must be
rejected with an explicit error instead of returning a meaningless number.
"""

import numpy as np
import pytest

import mindquantum as mq
from mindquantum import Circuit
from mindquantum.simulator import Simulator, fidelity, inner_product

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]


def _simulator(backend='mqvector', n_qubits=2, dtype=mq.complex128):
    return Simulator(backend, n_qubits, dtype=dtype, seed=13)


def test_inner_product_of_two_simulators():
    """
    Feature: ``inner_product`` between two simulators.

    Description: Compare the ground state with a superposition and a state rotated on two qubits.

    Expectation: The value matches the analytic overlap.
    """
    bra = _simulator()
    ket = _simulator()
    np.testing.assert_allclose(inner_product(bra, bra), 1.0)
    ket.apply_circuit(Circuit().h(0))
    np.testing.assert_allclose(inner_product(bra, ket), 1 / np.sqrt(2), atol=1e-12)
    ket.apply_circuit(Circuit().x(1, 0))
    np.testing.assert_allclose(inner_product(bra, ket), 1 / np.sqrt(2), atol=1e-12)


def test_inner_product_validation():
    """
    Feature: Validation of ``inner_product``.

    Description: Compare simulators with different width, backend and precision.

    Expectation: Each mismatch raises the documented exception.
    """
    simulator = _simulator()
    with pytest.raises(ValueError, match='same quantum state'):
        inner_product(simulator, _simulator('mqvector', 3))
    with pytest.raises(ValueError, match='backend of two simulator should be same'):
        inner_product(simulator, _simulator('mqmatrix'))
    with pytest.raises(TypeError, match='Data type of two simulator are different'):
        inner_product(simulator, _simulator('mqvector', dtype=mq.complex64))


def test_fidelity_accepts_vectors_and_density_matrices():
    """
    Feature: ``fidelity`` between pure and mixed states.

    Description: Compute the fidelity of a ket with itself, of a density matrix with itself and of orthogonal states.

    Expectation: The values match the analytic results.
    """
    ket = np.array([1.0, 0.0], dtype=complex)
    np.testing.assert_allclose(fidelity(ket, ket), 1.0)
    density = np.diag([0.25, 0.75]).astype(complex)
    np.testing.assert_allclose(fidelity(density, density), 1.0)
    np.testing.assert_allclose(fidelity(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0, atol=1e-12)
    np.testing.assert_allclose(fidelity(ket, np.diag([1.0, 0.0])), 1.0)


def test_fidelity_validation():
    """
    Feature: Validation of ``fidelity``.

    Description: Provide states with mismatched sizes, a non power of two size, an unnormalized vector,
        a non-Hermitian matrix, a negative population, a non-unit trace and a 3D array.

    Expectation: Every malformed state raises a ``ValueError`` describing the problem.
    """
    with pytest.raises(ValueError, match='size of two quantum state not match'):
        fidelity(np.ones(2), np.ones(4))
    with pytest.raises(ValueError, match='not power of 2'):
        fidelity(np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match='state vector must be normalized'):
        fidelity(np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match='density matrix must be hermitian'):
        fidelity(np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match='diagonal terms in density matrix cannot be negative'):
        fidelity(np.diag([-0.5, 1.5]).astype(complex), np.diag([-0.5, 1.5]).astype(complex))
    with pytest.raises(ValueError, match='trace of density matrix must equal to 1'):
        fidelity(np.diag([2.0, 0.0]).astype(complex), np.diag([2.0, 0.0]).astype(complex))
    with pytest.raises(ValueError, match='requires a vector or matrix'):
        fidelity(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))
