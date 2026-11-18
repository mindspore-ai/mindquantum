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
"""Functional tests for the general ballistic simulated bifurcation solver.

``GBSB`` is a new solver of MindQuantum 0.13 (``mindquantum.algorithm.qaia.GBSB``).
It controls every individual bifurcation through the nonlinear parameter ``A``
and keeps the spin in the unit box, unlike the plain ``BSB`` schedule.
"""

import numpy as np
import pytest

from mindquantum.algorithm.qaia import GBSB

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]

_COUPLING = np.array([[0.0, -1.0, 0.5], [-1.0, 0.0, 0.5], [0.5, 0.5, 0.0]])


def test_gbsb_state_stays_in_the_unit_box():
    """
    Feature: Bounded dynamics of ``GBSB``.

    Description: Run the solver and inspect the spin, the momentum and the individual pressure.

    Expectation: ``|x| <= 1``, the pressure stays in ``[0, 1]`` and all values are finite.
    """
    solver = GBSB(_COUPLING, n_iter=50, batch_size=2, A=0.2)
    solver.update()
    assert np.abs(solver.x).max() <= 1.0 + 1e-12
    assert solver.p_i.min() >= 0.0
    assert solver.p_i.max() <= 1.0
    assert np.isfinite(solver.x).all()
    assert np.isfinite(solver.y).all()


def test_gbsb_batch_columns_are_independent():
    """
    Feature: Batched evolution of ``GBSB``.

    Description: Start every replica from the same spin configuration.

    Expectation: All replicas follow the same trajectory and stay identical.
    """
    column = np.array([[0.1], [-0.2], [0.3]])
    solver = GBSB(_COUPLING, n_iter=20, batch_size=3, A=0.2, x=np.hstack([column] * 3))
    solver.y = np.zeros_like(solver.x)
    solver.update()
    for index in range(1, 3):
        np.testing.assert_allclose(solver.x[:, index], solver.x[:, 0], atol=1e-12)
    np.testing.assert_allclose(solver.p_i[:, 1], solver.p_i[:, 0], atol=1e-12)


def test_gbsb_is_reproducible_with_a_fixed_seed():
    """
    Feature: Reproducibility of ``GBSB``.

    Description: Build the solver twice under the same NumPy seed.

    Expectation: Both runs produce exactly the same trajectory.
    """
    np.random.seed(1234)
    first = GBSB(_COUPLING, n_iter=15, batch_size=2, A=0.3)
    first.update()

    np.random.seed(1234)
    second = GBSB(_COUPLING, n_iter=15, batch_size=2, A=0.3)
    second.update()

    np.testing.assert_allclose(first.x, second.x)
    np.testing.assert_allclose(first.y, second.y)


@pytest.mark.parametrize('value', [-0.1, -1.0])
def test_gbsb_rejects_negative_bifurcation_parameter(value):
    """
    Feature: Validation of the ``GBSB`` bifurcation parameter.

    Description: Provide a negative ``A``.

    Expectation: A ``ValueError`` reports the lower bound of the parameter.
    """
    with pytest.raises(ValueError, match='should be not less than 0'):
        GBSB(_COUPLING, n_iter=2, A=value)


@pytest.mark.parametrize('value', ['0.2', None, [0.2]])
def test_gbsb_rejects_non_numeric_bifurcation_parameter(value):
    """
    Feature: Validation of the ``GBSB`` bifurcation parameter.

    Description: Provide a non numeric ``A``.

    Expectation: A ``TypeError`` reports the expected type.
    """
    with pytest.raises(TypeError, match='requires a number'):
        GBSB(_COUPLING, n_iter=2, A=value)


def test_gbsb_accepts_zero_bifurcation_parameter():
    """
    Feature: Lower bound of the ``GBSB`` bifurcation parameter.

    Description: Use ``A = 0``, the smallest admissible value.

    Expectation: The solver is constructed and one update stays finite.
    """
    solver = GBSB(_COUPLING, n_iter=5, A=0.0)
    solver.update()
    assert np.isfinite(solver.x).all()
    assert solver.p_i.max() <= 1.0


def test_gbsb_large_bifurcation_parameter_keeps_state_bounded():
    """
    Feature: Upper dynamics of ``GBSB``.

    Description: Use a bifurcation parameter larger than one, which makes the pressure shrink faster.

    Expectation: The trajectory is still bounded and finite.
    """
    solver = GBSB(_COUPLING, n_iter=30, A=2.0)
    solver.update()
    assert np.isfinite(solver.x).all()
    assert np.abs(solver.x).max() <= 1.0 + 1e-12
    assert solver.p_i.min() >= 0.0
